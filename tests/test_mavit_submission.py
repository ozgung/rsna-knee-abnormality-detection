"""Standalone orchestration tests; synthetic outputs are never model evidence."""
import ast
import copy
from contextlib import ExitStack,redirect_stdout
import io
import types
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from knee import mavit_submission as s, mavit_rehearsal as r

ROOT=Path(__file__).resolve().parents[1]
PARENT=Path(os.environ['KNEE_REFERENCE_PARENT']) if os.environ.get('KNEE_REFERENCE_PARENT') else None
BUILD=PARENT/'outputs/kaggle/mavit-image-rehearsal-20261009v1' if PARENT else None
RESULT=PARENT/'outputs/kaggle/mavit-image-rehearsal-20261009v1-result/mavit-rehearsal' if PARENT else None


def builder():
    sys.path.insert(0,str(ROOT/'scripts'));import build_mavit_submission;return build_mavit_submission


def real_args():
    return dict(rehearsal_build=BUILD,rehearsal_report=RESULT/'mavit-audit.json',rehearsal_csv=RESULT/'visible-predictions.csv')


def synthetic_result(out,contract,*,parent=True,ids=None,values=None):
    """Fabricated receipt exclusively for local guard tests; not a CUDA qualification."""
    out=Path(out);out.mkdir(parents=True,exist_ok=False);q=contract['qualification'];ids=ids or ['synthetic-c','synthetic-a','synthetic-b']
    values=values or [[.1+.005*j+.01*i for j in range(12)] for i in range(len(ids))]
    name='visible-predictions.csv' if contract['visible'] else 'submission.csv'
    digest=s.write_probabilities(out/name,['StudyInstanceUID',*q['targets']],ids,values)
    report={'status':'PASS','synthetic_fixture':True,'parent_complete':parent,'version':s.VERSION,'visible':contract['visible'],
       'qualification_sha256':contract['qualification_sha256'],'contract_sha256':s.cp.digest(s.cp.canonical(contract)),
       'test_studies':len(ids),'completed_studies':len(ids),'prediction_fallbacks':0,'backend_flags':r.FLAGS,
       'checkpoint_sha256':r.CHECKPOINT_SHA256,'state_compatibility':{'strict_load':True,'loaded_tensors_equal':True,'actual_schema_sha256':r.STATE_SCHEMA},
       'environment':{'python_major_minor':[3,12],'packages':q['packages']},'codec_hashes':q['rehearsal_contract']['codecs'],
       'input_counts':dict.fromkeys(r.inputs.COUNTERS,0),'repeat_max_error':0.,'elapsed_seconds':1.,'csv_sha256':digest,
       'ordered_ids_sha256':s.cp.digest(s.cp.canonical(ids)),'target_order_sha256':s.cp.digest(s.cp.canonical(q['targets'])),
       'peak_cuda_allocated_bytes':1,'peak_cuda_reserved_bytes':1,'peak_host_rss_bytes':1,'disk_free_bytes':1,'output_disk_bytes':(out/name).stat().st_size}
    (out/'mavit-audit.json').write_text(json.dumps(report));return report,out/name


def fake_contract():
    from knee import mavit_input as inputs
    q={'version':s.VERSION,'cross_run_atol':1e-6,'repeat_atol':1e-6,'targets':[f'target{i}' for i in range(12)],
       'packages':dict(r.PACKAGES,scipy='synthetic-version'),'backend_flags':r.FLAGS,
       'rehearsal_contract':{'checkpoint_sha256':r.CHECKPOINT_SHA256,'model_state_schema_sha256':r.STATE_SCHEMA,'cpu_report_sha256':r.CPU_REPORT_SHA256,
         'input_policy':{'version':inputs.VERSION,'sha256':s.cp.digest(Path(inputs.__file__).read_bytes())},'codecs':{'synthetic-wheel':'a'*64}}}
    return {'version':s.VERSION,'qualification':q,'qualification_sha256':s.cp.digest(s.cp.canonical(q)),
            'visible':True,'max_seconds':32400,'private':True,'internet':False,'visible_evidence':None}


class GuardTests(unittest.TestCase):
    def test_absolute1e6_only_and_exact_order_coverage_bounds(self):
        c=fake_contract()
        with tempfile.TemporaryDirectory() as tmp:
            _,path=synthetic_result(Path(tmp)/'reference',c);cols,ids,values=s.read_predictions(path)
            other=Path(tmp)/'other.csv';values[1][4]+=5e-7;s.write_probabilities(other,cols,ids,values)
            self.assertLess(s.compare_csv(path,other),s.ATOL)
            for mode in ('precision','order','duplicate','missing','target','nan','infinite','range'):
                v=copy.deepcopy(values);names=ids[:];columns=cols[:]
                if mode=='precision':v[1][4]+=1e-5
                if mode=='order':names=names[::-1]
                if mode=='duplicate':names[1]=names[0]
                if mode=='missing':names=names[:-1];v=v[:-1]
                if mode=='target':columns[1:3]=columns[1:3][::-1]
                if mode=='nan':v[0][0]=float('nan')
                if mode=='infinite':v[0][0]=float('inf')
                if mode=='range':v[0][0]=1.1
                import csv
                with other.open('w') as h:writer=csv.writer(h);writer.writerow(columns);writer.writerows([name,*row] for name,row in zip(names,v))
                with self.subTest(mode=mode),self.assertRaises(ValueError):s.compare_csv(path,other)

    def test_receipt_source_environment_completion_and_precision_fail_closed(self):
        c=fake_contract()
        with tempfile.TemporaryDirectory() as tmp:
            report,path=synthetic_result(Path(tmp)/'out',c);s.require_result(report,c,path)
            for key,value in [('parent_complete',False),('qualification_sha256','0'*64),('contract_sha256','0'*64),('completed_studies',2),
                ('repeat_max_error',1.1e-6),('repeat_max_error',float('nan')),('elapsed_seconds',32401),('elapsed_seconds',float('nan')),
                ('prediction_fallbacks',1),('csv_sha256','0'*64),('state_compatibility',{}),('input_counts',{}),('environment',{}),('peak_cuda_allocated_bytes',0)]:
                changed=copy.deepcopy(report);changed[key]=value
                with self.subTest(key=key),self.assertRaises(ValueError):s.require_result(changed,c,path)
            for field,value in [('scipy','other'),('torch','other')]:
                changed=copy.deepcopy(report);changed['environment']['packages'][field]=value
                with self.assertRaises(ValueError):s.require_result(changed,c,path)
            changed=copy.deepcopy(report);changed['input_counts']['resource_errors']=1
            with self.assertRaises(ValueError):s.require_result(changed,c,path)
            changed=copy.deepcopy(c);changed['qualification']['rehearsal_contract']['input_policy']['version']='old';changed['qualification_sha256']=s.cp.digest(s.cp.canonical(changed['qualification']))
            with self.assertRaises(ValueError):s.qualification(changed)

    def test_parent_finish_releases_only_valid_full_csv_and_never_overwrites(self):
        c=fake_contract()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);report,path=synthetic_result(root/'visible',c,parent=False)
            finished=s.finish(root/'visible',c,started=time.monotonic()-1);self.assertTrue(finished['parent_complete']);self.assertFalse((root/'submission.csv').exists())
            full=dict(c,visible=False)
            with self.assertRaises(ValueError):s.qualification(full)
            full['visible_evidence']={'report_sha256':'1'*64,'csv_sha256':'2'*64,'immutable67_max_error':0.,'absolute_tolerance':1e-6}
            for n in (1,2,4,9):
                out=root/f'full{n}';_,csv=synthetic_result(out,full,parent=False,ids=[f'synthetic{i}' for i in range(n)])
                target=root/f'submission{n}.csv';done=s.finish(out,full,started=time.monotonic()-1,target=target)
                self.assertTrue(done['parent_complete']);self.assertEqual(target.read_bytes(),csv.read_bytes())
                with self.assertRaises(ValueError):s.finish(out,full,started=time.monotonic()-1,target=target)
            _,csv=synthetic_result(root/'stale',full,parent=False);target=root/'already.csv';target.write_text('preserve')
            with self.assertRaises(FileExistsError):s.finish(root/'stale',full,started=time.monotonic()-1,target=target)
            self.assertEqual(target.read_text(),'preserve')
            self.assertFalse(json.loads((root/'stale/mavit-audit.json').read_text())['parent_complete'])
            _,csv=synthetic_result(root/'expired',full,parent=False)
            with self.assertRaises(ValueError):s.finish(root/'expired',full,started=time.monotonic()-32401,target=root/'MUST_NOT_WRITE')
            self.assertFalse((root/'MUST_NOT_WRITE').exists())


    def test_finish_cleans_own_output_on_postcopy_deadline_partial_io_and_receipt_failure(self):
        c=dict(fake_contract(),visible=False,visible_evidence={'report_sha256':'1'*64,'csv_sha256':'2'*64,'immutable67_max_error':0.,'absolute_tolerance':1e-6})
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for mode in ('deadline','partial_io','report_write','interrupt'):
                out=root/mode;_,csv=synthetic_result(out,c,parent=False);target=root/f'{mode}.csv';owned=set()
                with self.subTest(mode=mode),ExitStack() as stack:
                    if mode=='deadline':stack.enter_context(patch.object(s.time,'monotonic',side_effect=[1,32401]))
                    if mode in ('partial_io','interrupt'):
                        def fail_copy(src,dst):
                            dst.write(b'partial')
                            if mode=='interrupt':raise KeyboardInterrupt()
                            raise OSError('copy failure')
                        stack.enter_context(patch.object(s.shutil,'copyfileobj',side_effect=fail_copy))
                    if mode=='report_write':stack.enter_context(patch.object(Path,'write_text',side_effect=OSError('receipt failure')))
                    with self.assertRaises(BaseException):s.finish(out,c,started=0 if mode=='deadline' else time.monotonic()-1,target=target,owned_outputs=owned)
                self.assertFalse(target.exists());self.assertEqual(owned,set())
            # Successful creation remains registered for the entry's outer BaseException cleanup.
            out=root/'success';synthetic_result(out,c,parent=False);owned=set();target=root/'success.csv'
            s.finish(out,c,started=time.monotonic()-1,target=target,owned_outputs=owned)
            self.assertEqual(owned,{target});self.assertTrue(target.exists())
            for path in owned:path.unlink()
            self.assertFalse(target.exists())


@unittest.skipUnless(BUILD and RESULT and (RESULT/'mavit-audit.json').is_file(),'authentic MaViT67 archive')
class ArchiveTests(unittest.TestCase):
    def test_real_archive_build_no_fixtures_fixed_scipy_isolated_import_and_full_gate(self):
        b=builder()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);d=root/'visible';receipt=b.build(d,'synthetic-mavit-visible',**real_args())
            contract=json.loads((d/'mavit-contract.json').read_text());q=contract['qualification'];meta=json.loads((d/'kernel-metadata.json').read_text())
            self.assertEqual(q['packages']['scipy'],'1.16.3');self.assertEqual(q['cross_run_atol'],1e-6);self.assertEqual(q['repeat_atol'],1e-6)
            self.assertFalse(meta['enable_internet']);self.assertTrue(meta['is_private']);self.assertEqual(meta['kernel_sources'],[]);self.assertEqual(meta['model_sources'],[])
            self.assertEqual(meta['dataset_sources'],[r.checkpoint.DATASET,b.CONVNEXT_DATASET]);self.assertLess(receipt['notebook_bytes'],900000)
            code=''.join(json.loads((d/'notebook.ipynb').read_text())['cells'][-1]['source']);compile(code,'<standalone>','exec')
            own=ast.literal_eval(next(n.value for n in ast.parse(code).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='sources' for t in n.targets)))
            archive,_,_=b.archived_build(BUILD)
            for name,value in archive.items():self.assertEqual(own[name],value)
            for sid in s.read_predictions(RESULT/'visible-predictions.csv')[1]:self.assertNotIn(sid,code)
            self.assertLess(code.index('signal.setitimer'),code.index('sources='))
            bundle=root/'project'
            for name,value in own.items():p=bundle/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value)
            done=subprocess.run([sys.executable,'-c','import sys;import knee.mavit_submission;import scripts.mavit_submission_entry;assert "torch" not in sys.modules'],cwd=tmp,env=dict(os.environ,PYTHONPATH=str(bundle)),capture_output=True,text=True)
            self.assertEqual(done.returncode,0,done.stderr)
            with self.assertRaises(ValueError):b.build(root/'missing','synthetic-full',**real_args(),submission=True)
            # Deliberately synthetic visible execution receipt, only to exercise packaging guards locally.
            columns,ids,values=s.read_predictions(RESULT/'visible-predictions.csv')
            _,csv=synthetic_result(root/'synthetic-result',contract,ids=ids,values=values)
            kwargs=dict(real_args(),submission=True,visible_report=root/'synthetic-result/mavit-audit.json',visible_csv=csv)
            full=b.build(root/'full','synthetic-full',**kwargs)
            self.assertEqual(full['qualification_sha256'],receipt['qualification_sha256'])
            fullcontract=json.loads((root/'full/mavit-contract.json').read_text());self.assertFalse(fullcontract['visible']);self.assertEqual(fullcontract['visible_evidence']['immutable67_max_error'],0.)
            # Any drift beyond the unchanged threshold fails; no automatic retries or relaxation.
            actual=json.loads(kwargs['visible_report'].read_text());changed=copy.deepcopy(values);changed[0][0]+=1e-4
            csv.unlink();actual['csv_sha256']=s.write_probabilities(csv,columns,ids,changed);kwargs['visible_report'].write_text(json.dumps(actual))
            with self.assertRaisesRegex(ValueError,'cross-run'):b.build(root/'bad-full','synthetic-full',**kwargs)
            self.assertFalse((root/'bad-full').exists())

    @unittest.skipUnless(importlib.util.find_spec('torch'),'optional torch for mocked orchestration')
    def test_actual_test_order_only_metadata_and_frozen_prediction_calls_in_both_modes(self):
        import csv
        import numpy as np
        import torch
        b=builder();assets=PARENT/'outputs/analysis/post-fusion-research-20261009/mavit/dataset-code'
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);b.build(base/'build','synthetic-orchestration',**real_args())
            visible=json.loads((base/'build/mavit-contract.json').read_text())
            for count in (3,9):
                c=copy.deepcopy(visible);c['visible']=count==3
                if not c['visible']:c['visible_evidence']={'report_sha256':'1'*64,'csv_sha256':'2'*64,'immutable67_max_error':0.,'absolute_tolerance':1e-6}
                root=base/f'inputs{count}';root.mkdir();ids=[f'synthetic-{i}' for i in reversed(range(count))];q=c['qualification']
                (root/'test.csv').write_text('StudyInstanceUID\n'+'\n'.join(ids)+'\n')
                # Sample row content is irrelevant and deliberately not actual test coverage.
                (root/'sample_submission.csv').write_text(','.join(['StudyInstanceUID',*q['targets']])+'\n')
                with (root/'test_series.csv').open('w') as handle:
                    writer=csv.DictWriter(handle,fieldnames=r.SERIES_COLUMNS);writer.writeheader()
                    writer.writerows(dict(StudyInstanceUID=sid,SeriesInstanceUID=f'series{i}',Fluid_Sensitive=0,Fat_Suppression=0,Anatomical_Plane='Sagittal') for i,sid in enumerate(ids[::-1]))
                (root/'train.csv').write_text('MUST NOT READ');(root/'train_series.csv').write_text('MUST NOT READ')
                calls=[];prepared=[];model=object();reader=object();opened=[];original_open=Path.open
                def open_file(path,*args,**kwargs):
                    opened.append(path.name)
                    if path.name in ('train.csv','train_series.csv','train_golden58.csv'):raise AssertionError('Forbidden production input')
                    return original_open(path,*args,**kwargs)
                def prepare(module,actual_reader,sid,rows,directory,*,production):
                    self.assertIs(actual_reader,reader);self.assertTrue(production);self.assertEqual(directory,root/'test_series');prepared.append(sid)
                    self.assertTrue(all(row['StudyInstanceUID']==sid for row in rows));return sid,dict.fromkeys(r.inputs.COUNTERS,0)
                def predict(actual_model,sid):
                    self.assertIs(actual_model,model);calls.append(sid);return np.arange(12,dtype=np.float32)*.005+.1+ids.index(sid)*.01
                module=types.SimpleNamespace(KNEE_LABELS=q['targets'],make_knee_dataset_reader=lambda **kwargs:reader)
                audit={'state_compatibility':{'strict_load':True,'loaded_tensors_equal':True,'actual_schema_sha256':r.STATE_SCHEMA}}
                with ExitStack() as stack:
                    stack.enter_context(patch.object(Path,'open',open_file));stack.enter_context(patch.object(s.inputs,'definitions',return_value=module))
                    stack.enter_context(patch.object(s.inputs,'prepare',side_effect=prepare));stack.enter_context(patch.object(r,'load_model',return_value=(model,audit)))
                    stack.enter_context(patch.object(r,'predict',side_effect=predict));stack.enter_context(patch.object(s.importlib.metadata,'version',side_effect=q['packages'].__getitem__))
                    stack.enter_context(patch.object(torch,'set_num_threads'));stack.enter_context(patch.object(torch,'get_num_threads',return_value=4))
                    for attr,value in [('is_available',True),('get_device_name','mock-no-gpu-run'),('synchronize',None),('reset_peak_memory_stats',None),('max_memory_allocated',1),('max_memory_reserved',1)]:
                        stack.enter_context(patch.object(torch.cuda,attr,return_value=value))
                    for owner,attr in [(torch.backends.cudnn,'benchmark'),(torch.backends.cudnn,'deterministic'),(torch.backends.cudnn,'allow_tf32'),(torch.backends.cuda.matmul,'allow_tf32')]:
                        stack.callback(setattr,owner,attr,getattr(owner,attr))
                    output=io.StringIO();stack.enter_context(redirect_stdout(output))
                    report=s.run(root,assets,c,base/f'result{count}',started=time.monotonic())
                self.assertEqual(prepared,ids);self.assertEqual(calls,[sid for sid in ids for _ in range(2)])
                filename='visible-predictions.csv' if c['visible'] else 'submission.csv'
                columns,result_ids,values=s.read_predictions(base/f'result{count}'/filename);self.assertEqual(result_ids,ids)
                self.assertEqual(report['completed_studies'],count);self.assertEqual(report['repeat_max_error'],0.)
                self.assertFalse(report['parent_complete']);self.assertNotIn('train.csv',opened)
                for sid in ids:self.assertNotIn(sid,output.getvalue())

    def test_archived_executable_tamper_is_never_executed_even_if_outer_hash_updated(self):
        b=builder()
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)/'archive';shutil.copytree(BUILD,d);nb=json.loads((d/'notebook.ipynb').read_text())
            nb['cells'][-1]['source'].append("raise RuntimeError('UNTRUSTED ARCHIVE MUST NOT EXECUTE')\n")
            raw=(json.dumps(nb)+'\n').encode();(d/'notebook.ipynb').write_bytes(raw)
            receipt=json.loads((d/'build-receipt.json').read_text());receipt.update(notebook_sha256=s.cp.digest(raw),notebook_bytes=len(raw));(d/'build-receipt.json').write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError,'executable'):b.archived_build(d)
            nb=json.loads((BUILD/'notebook.ipynb').read_text());code=''.join(nb['cells'][-1]['source']);tree=ast.parse(code)
            for node in tree.body:
                if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='sources':node.value=ast.parse('raise_if_executed()',mode='eval').body
            nb['cells'][-1]['source']=ast.unparse(tree).splitlines(True);(d/'notebook.ipynb').write_text(json.dumps(nb))
            with self.assertRaises((ValueError,TypeError)):b.archived_build(d)

    def test_rehearsal_receipt_csv_and_archived_source_mixing_fail_closed(self):
        b=builder()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for field,value in [('parent_complete',False),('source_revision','0'*40),('contract_sha256','0'*64),('studies',66),('repeat_max_error',2e-6)]:
                report=json.loads((RESULT/'mavit-audit.json').read_text());report[field]=value;path=root/'bad-report.json';path.write_text(json.dumps(report))
                kwargs=real_args();kwargs['rehearsal_report']=path
                with self.subTest(field=field),self.assertRaises(ValueError):b.build(root/'reject','synthetic-reject',**kwargs)
            d=root/'archive';shutil.copytree(BUILD,d);c=json.loads((d/'mavit-contract.json').read_text());c['input_policy']['version']='old';(d/'mavit-contract.json').write_text(json.dumps(c))
            with self.assertRaises(ValueError):b.archived_build(d)


if __name__=='__main__':unittest.main()
