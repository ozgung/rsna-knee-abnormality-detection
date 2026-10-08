"""Standalone contracts, offline packaging and synthetic production-path checks."""
import copy
import csv
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from knee import mirror_input, mirror_submission as runtime
from knee.reference import canonical_bytes, sha256


def builder():
    path=Path(__file__).resolve().parents[1]/'scripts/build_mirror_submission.py'
    spec=importlib.util.spec_from_file_location('mirror_submission_builder',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def csvfile(path, ids=('a','b','c'), *, labels=None, delta=0):
    labels=labels or [f't{i}' for i in range(12)]
    with path.open('w',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['StudyInstanceUID',*labels])
        for i,sid in enumerate(ids):writer.writerow([sid,*[.1+.02*j+.001*i+delta for j in range(12)]])


def input_files(root, ids=('c','a','b')):
    csvfile(root/'sample_submission.csv',ids=('irrelevant',))
    with (root/'test.csv').open('w',newline='') as h:
        w=csv.writer(h);w.writerow(['StudyInstanceUID']);w.writerows([[i] for i in ids])
    with (root/'test_series.csv').open('w',newline='') as h:
        w=csv.writer(h);w.writerow(['StudyInstanceUID','SeriesInstanceUID','Fluid_Sensitive','Fat_Suppression','Anatomical_Plane'])
        w.writerows([[i,'series'+i,'1','1','Sagittal'] for i in ids])


class MirrorSubmissionTests(unittest.TestCase):
    def test_actual_test_order_ignores_sample_coverage_and_rejects_schema_or_invalid_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);input_files(root)
            cols,ids,rows=runtime.read_inputs(root,visible=True)
            self.assertEqual(ids,['c','a','b']);self.assertEqual(list(rows),ids)
            original=(root/'test_series.csv').read_text()
            for change in (original.replace('Sagittal','Unknown'),original.replace('Fat_Suppression','Other'),
                           original.replace('seriesa','seriesc'),original.replace(',1,1,',',2,1,')):
                (root/'test_series.csv').write_text(change)
                with self.assertRaises((ValueError,KeyError)):runtime.read_inputs(root,visible=True)
            (root/'test_series.csv').write_text(original)
            (root/'test.csv').write_text('StudyInstanceUID\n../unsafe\n')
            with self.assertRaises(ValueError):runtime.read_inputs(root,visible=False)

    def test_csv_parity_enforces_order_targets_values_and_probability_tolerance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a,b=root/'a.csv',root/'b.csv';csvfile(a);csvfile(b)
            self.assertEqual(runtime.compare_csv(a,b),0)
            csvfile(b,delta=5e-5);self.assertLess(runtime.compare_csv(a,b),1e-4)
            for kwargs in ({'ids':('b','a','c')},{'delta':2e-4},{'labels':list(reversed([f't{i}' for i in range(12)]))}):
                csvfile(b,**kwargs)
                with self.assertRaises(ValueError):runtime.compare_csv(a,b)
            # A rank flip smaller than the cross-process tolerance is still rejected.
            csvfile(a);csvfile(b)
            text=a.read_text().replace('0.101,','0.10001,')
            a.write_text(text);b.write_text(text.replace('0.10001,','0.09999,'))
            with self.assertRaisesRegex(ValueError,'ranking or tie'):runtime.compare_csv(a,b)
            for value in ('nan','inf','-0.01','1.1'):
                b.write_text(a.read_text().replace('0.1,',value+',',1))
                with self.assertRaises(ValueError):runtime.read_predictions(b)

    def test_visible_receipt_rejects_old_policy_missing_resource_and_csv_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'v.csv';csvfile(p)
            r=self.visible(p,'a'*64)
            runtime.require_visible(r,'a'*64,p)
            for key,value in [('status','failed'),('version','old'),('visible',False),('input_policy',{}),
                              ('elapsed_seconds',32401),('elapsed_seconds',float('nan')),('csv_sha256','b'*64),
                              ('peak_cuda_allocated_bytes',0),('completed_studies',2),('repeat_prediction_max_error',2e-6)]:
                bad={**r,key:value}
                with self.assertRaises(ValueError):runtime.require_visible(bad,'a'*64,p)

    @staticmethod
    def visible(path,fingerprint):
        columns,ids,_=runtime.read_predictions(path)
        return {'status':'PASS','version':runtime.VERSION,'visible':True,'qualification_sha256':fingerprint,
            'test_studies':3,'completed_studies':3,'prediction_fallbacks':0,'input_policy':mirror_input.policy_contract(),
            'production_input_counts':{'resource_errors':0},'repeat_prediction_max_error':0,
            'elapsed_seconds':60,'csv_sha256':sha256(path.read_bytes()),
            'ordered_ids_sha256':sha256(canonical_bytes(ids)),'target_order_sha256':sha256(canonical_bytes(columns[1:])),
            **dict.fromkeys(('peak_cuda_allocated_bytes','peak_cuda_reserved_bytes','peak_host_rss_bytes','disk_free_bytes','output_disk_bytes'),100)}

    @unittest.skipUnless(importlib.util.find_spec('torch'),'optional torch')
    def test_synthetic_production_uses_same_two_views_model_and_no_training_files(self):
        import numpy as np
        import torch
        from contextlib import ExitStack
        labels=[f't{i}' for i in range(12)];calls=[]
        public=SimpleNamespace(LAB=labels,K_EVAL=94)
        def window(v,m,k,res,device,mirror):
            calls.append((k,res,device,mirror));return SimpleNamespace(shape=(94,3,320,320))
        public.gpu_windows=window
        q={'manifest':{'checkpoint_sha256':'a'*64},'input_policy':mirror_input.policy_contract()}
        contract={'version':runtime.VERSION,'qualification':q,'qualification_sha256':sha256(canonical_bytes(q)),
                  'max_seconds':32400,'visible':True}
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
            root=Path(tmp);input_files(root);ck=root/'weight';ck.write_bytes(b'synthetic')
            # A read of any gold/report/training path must fail; none is created.
            stack.enter_context(patch.object(torch.cuda,'is_available',return_value=True))
            for name in ('reset_peak_memory_stats','synchronize'):
                stack.enter_context(patch.object(torch.cuda,name))
            for name in ('max_memory_allocated','max_memory_reserved'):
                stack.enter_context(patch.object(torch.cuda,name,return_value=100))
            stack.enter_context(patch.object(torch.cuda,'get_device_name',return_value='synthetic'))
            stack.enter_context(patch.object(torch,'set_num_threads'))
            stack.enter_context(patch.object(torch,'backends',SimpleNamespace(
                cudnn=SimpleNamespace(benchmark=False),cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)))))
            stack.enter_context(patch.object(runtime.importlib.metadata,'version',return_value='synthetic'))
            stack.enter_context(patch.object(runtime,'load_definitions',return_value=(public,None)))
            model=object();load=stack.enter_context(patch.object(runtime,'load_model',return_value=model))
            prepare=stack.enter_context(patch.object(mirror_input,'prepare_study',return_value=(np.ones(2),np.ones(2),{'resource_errors':0,'valid_slices':96,'missing_slots':2})))
            infer=stack.enter_context(patch.object(runtime,'infer_checked',return_value=np.arange(12)*.02+.1))
            result=runtime.run(root,ck,'source','decoder',contract,root/'out',started=runtime.time.monotonic())
            self.assertEqual(result['status'],'PASS');self.assertEqual(result['completed_studies'],3)
            self.assertEqual(calls,[(94,320,'cuda',m) for _ in range(3) for m in (False,True)])
            self.assertEqual(prepare.call_count,3);self.assertEqual(infer.call_count,6)
            self.assertTrue(all(c.kwargs=={'mirror':True} and c.args[1] is model for c in infer.call_args_list))
            self.assertEqual(runtime.read_predictions(root/'out/visible-predictions.csv')[1],['c','a','b'])
            prepare.side_effect=TimeoutError('sensitive-path-should-not-appear')
            with self.assertRaisesRegex(RuntimeError,'aggregate audit'):
                runtime.run(root,ck,'source','decoder',contract,root/'failed',started=runtime.time.monotonic())
            text=(root/'failed/mirror-audit.json').read_text()
            self.assertNotIn('sensitive-path',text);self.assertIn('TimeoutError',text)
            self.assertFalse((root/'failed/submission.csv').exists())
            self.assertEqual(load.call_count,2)
            load.reset_mock()
            with self.assertRaises(RuntimeError):
                runtime.run(root,ck,'source','decoder',contract,root/'expired',started=runtime.time.monotonic()-32401)
            load.assert_not_called()


@unittest.skipUnless(os.environ.get('KNEE_REFERENCE_PARENT') and os.environ.get('KNEE_REFERENCE_SOURCE'),'archived build/source fixtures')
class MirrorArchiveTests(unittest.TestCase):
    def setUp(self):
        self.root=Path(__file__).resolve().parents[1]
        self.parent=Path(os.environ['KNEE_REFERENCE_PARENT'])
        self.archive=self.parent/'outputs/kaggle/mirror-rehearsal-20261009'
        self.source=Path(os.environ['KNEE_REFERENCE_SOURCE']);self.b=builder()

    def evidence(self,root):
        # Synthetic receipt is explicitly a test fixture, not real qualification.
        from test_mirror_input import fixture
        _,r=fixture()
        _,_,_,c,digest,revision,_=self.b.archived_build(self.archive,self.source,self.root)
        r.update(manifest=c['source'],manifest_sha256=sha256(canonical_bytes(c['source'])),source_bundle_sha256=digest,source_revision=revision)
        csvpath=root/'rehearsal.csv';csvfile(csvpath)
        r['visible_csv_sha256']=sha256(csvpath.read_bytes())
        receipt=root/'rehearsal.json';receipt.write_text(json.dumps(r))
        return dict(rehearsal_build=self.archive,rehearsal_report=receipt,rehearsal_csv=csvpath,public_source=self.source)

    def test_archive_actual_payload_tamper_and_changed_scientific_dependency_fail(self):
        self.b.archived_build(self.archive,self.source,self.root)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);copydir=root/'archive';shutil.copytree(self.archive,copydir,ignore=shutil.ignore_patterns('remote.log','*.log'))
            nbpath=copydir/'notebook.ipynb';nb=json.loads(nbpath.read_text())
            cell=next(c for c in nb['cells'] if c['cell_type']=='code');cell['source'].append("print('unreviewed')\n")
            nbpath.write_text(json.dumps(nb))
            # Even rewriting superficial size/receipt cannot authorize new executable code.
            rp=copydir/'build-receipt.json';r=json.loads(rp.read_text());r['notebook_bytes']=nbpath.stat().st_size;rp.write_text(json.dumps(r))
            with self.assertRaisesRegex(ValueError,'executable'):self.b.archived_build(copydir,self.source,self.root)
            fake=root/'sources';(fake/'scripts').mkdir(parents=True);(fake/'knee').mkdir()
            for p in self.b.SCIENTIFIC+('scripts/build_reference_rehearsal.py','scripts/rehearse_reference_entry.py'):
                shutil.copyfile(self.root/p,fake/p)
            (fake/'knee/reference.py').write_text('changed scientific code')
            with self.assertRaisesRegex(ValueError,'Scientific'):self.b.archived_build(self.archive,self.source,fake)

    def test_build_requires67_then_visible_csv_parity_and_keeps_separate_archive_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=self.evidence(root)
            receipt=self.b.build_mirror(root/'visible','synthetic-mirror',**args)
            self.assertNotEqual(receipt['bundle_sha256'],receipt['rehearsal_bundle_sha256'])
            meta=json.loads((root/'visible/kernel-metadata.json').read_text())
            self.assertFalse(meta['enable_internet']);self.assertTrue(meta['is_private'])
            self.assertEqual(meta['kernel_sources'],[]);self.assertEqual(meta['model_sources'],[])
            self.assertEqual(meta['dataset_sources'],[self.b.WEIGHT_DATASET,self.b.CONVNEXT_DATASET])
            nb=json.loads((root/'visible/notebook.ipynb').read_text());code=''.join(nb['cells'][1]['source'])
            compile(code,'<mirror-synthetic-build>','exec');self.assertLess(len(code),900000)
            self.assertLess(code.index('signal.setitimer'),code.index('sources='))
            self.assertNotIn('stevenleehans',json.dumps(meta))
            with self.assertRaises(ValueError):self.b.build_mirror(root/'full-missing','synthetic-mirror',**args,submission=True)
            visible_csv=root/'visible.csv';shutil.copyfile(args['rehearsal_csv'],visible_csv)
            r=MirrorSubmissionTests.visible(visible_csv,receipt['qualification_sha256'])
            c=json.loads((root/'visible/mirror-contract.json').read_text());q=c['qualification']
            r.update(checkpoint_verified_sha256=q['manifest']['checkpoint_sha256'],checkpoint_bytes=mirror_input.CHECKPOINT_BYTES,
                     codec_hashes=q['codec_hashes'],environment={'python_major_minor':[3,12],'packages':q['environment']})
            rp=root/'visible.json';rp.write_text(json.dumps(r))
            full=self.b.build_mirror(root/'full','synthetic-mirror-full',**args,submission=True,visible_report=rp,visible_csv=visible_csv)
            self.assertEqual(full['qualification_sha256'],receipt['qualification_sha256'])
            csvfile(visible_csv,delta=1e-3);r['csv_sha256']=sha256(visible_csv.read_bytes());rp.write_text(json.dumps(r))
            with self.assertRaisesRegex(ValueError,'probabilities'):
                self.b.build_mirror(root/'bad-parity','synthetic-mirror-full',**args,submission=True,visible_report=rp,visible_csv=visible_csv)
            r=json.loads(Path(args['rehearsal_report']).read_text());r['sampling']['train_count']=24;Path(args['rehearsal_report']).write_text(json.dumps(r))
            with self.assertRaises(ValueError):self.b.build_mirror(root/'old','synthetic-old',**args)
            # Import only the extracted dependency closure in a fresh interpreter.
            import ast
            sources=next(ast.literal_eval(n.value) for n in ast.parse(code).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='sources' for t in n.targets))
            bundle=root/'bundle'
            for name,value in sources.items():
                p=bundle/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value)
            proc=subprocess.run([sys.executable,'-I','-c',f"import sys;sys.path.insert(0,{str(bundle)!r});import knee.mirror_submission;print('isolated import PASS')"],capture_output=True,text=True,check=True)
            self.assertIn('PASS',proc.stdout)


if __name__=='__main__':unittest.main()
