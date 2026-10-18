"""Synthetic qualification/packaging fixtures: never execute CUDA or retain a PASS artifact."""
import ast
import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest

from knee import coatnet224_submission as final,public_training_smoke as smoke
from scripts import build_coatnet224_submission as builder


def fixture():
    own=final.own
    members=[{'fold':f,'kernel':own.KERNELS[0 if f<2 else (1 if f<4 else 2)],
              'run_file':f'run_fold{f}.json','run_sha256':'a'*64,
              'checkpoint_file':f'raptor_ft_fold{f}_swa.pt','checkpoint_sha256':'b'*64,
              'selected_epochs':[9,8,7]} for f in range(5)]
    c={**own.POLICY,'version':1,'complete_family':True,'settings':dict(own.SETTINGS),'members':members,
       'model_source_sha256':smoke.SOURCE_SHA,'reader_sha256':own.READER_SHA,
       'model_definitions_sha256':own.MODEL_DEFINITIONS_SHA}
    report={'status':'PASS','contract_sha256':smoke.digest(smoke.canonical(c)),
            'qualified_folds':list(range(5)),'train_studies':5,'test_studies':3,'settings':own.SETTINGS,
            'labels_read':False,'training_parity':[{'slots':[{'exact_hash_match':True,'mask_match':True} for _ in range(5)],'windows_exact':True} for _ in range(5)],
            'models':[{'fold':m['fold'],'checkpoint_sha256':m['checkpoint_sha256'],'strict_load':True,
                       'repeat_max_error':0,'train_probability_max_error':0,'missing_plane_finite':True,'cuda_peak_bytes':123} for m in members],
            'stress_checks':{'cases':dict.fromkeys(final.raptor_input.STRESS_CASES,'PASS')},
            'environment':{'packages':{'torch':'2.10.0+cu128','timm':'1.0.26'}},
            'elapsed_seconds':60,'visible_csv_sha256':'c'*64}
    return c,report


class Coatnet224SubmissionTests(unittest.TestCase):
    def write_csv(self,path,header,rows):
        with path.open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(header);writer.writerows(rows)

    def inputs(self,root,ids):
        self.write_csv(root/'test.csv',['StudyInstanceUID'],[[uid] for uid in ids])
        self.write_csv(root/'sample_submission.csv',['StudyInstanceUID',*smoke.TARGETS],[[uid,*[0]*12] for uid in ids])
        self.write_csv(root/'test_series.csv',final.SERIES_COLUMNS,[[uid,f'series{n}','1','0','Sagittal'] for n,uid in enumerate(ids)])

    def test_authoritative_hidden_ids_order_and_no_training_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ids=['hidden4','hidden1','hidden8','hidden2'];self.inputs(root,ids)
            # Hidden test.csv may differ from the original three-row sample.
            self.write_csv(root/'sample_submission.csv',['StudyInstanceUID',*smoke.TARGETS],[[uid,*[0]*12] for uid in ['visible1','visible2','visible3']])
            columns,actual,records=final.read_inputs(root)
            self.assertEqual(actual,ids);self.assertEqual(list(records),ids)
            self.assertEqual(columns,['StudyInstanceUID',*smoke.TARGETS]);self.assertFalse((root/'train.csv').exists())

    def test_input_duplicates_reordered_schema_and_unknown_series_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ids=['a','b'];self.inputs(root,ids)
            for target in ['duplicates','columns','unknown_series','path','series_duplicate']:
                self.inputs(root,ids)
                if target=='duplicates':self.write_csv(root/'test.csv',['StudyInstanceUID'],[['a'],['a']])
                elif target=='columns':self.write_csv(root/'sample_submission.csv',['StudyInstanceUID',*reversed(smoke.TARGETS)],[['a',*[0]*12]])
                elif target=='unknown_series':self.write_csv(root/'test_series.csv',final.SERIES_COLUMNS,[['extra','s1','1','0','Sagittal']])
                elif target=='path':self.write_csv(root/'test.csv',['StudyInstanceUID'],[['../escape'],['b']])
                else:self.write_csv(root/'test_series.csv',final.SERIES_COLUMNS,[['a','same','1','0','Sagittal'],['b','same','1','0','Sagittal']])
                with self.subTest(target=target),self.assertRaises(ValueError):final.read_inputs(root)

    def test_csv_schema_exact_coverage_finite_and_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'submission.csv';columns=['StudyInstanceUID',*smoke.TARGETS]
            self.write_csv(path,columns,[['a',*[.5]*12],['b',*[.2]*12]])
            final.validate_csv(path,['a','b'])
            with self.assertRaises(ValueError):final.validate_csv(path,['b','a'])
            for value in ['nan','inf','-.1','1.1']:
                self.write_csv(path,columns,[['a',*[value]*12]])
                with self.subTest(value=value),self.assertRaises(ValueError):final.validate_csv(path,['a'])
            self.write_csv(path,columns,[['a',*[.5]*12],['a',*[.5]*12]])
            with self.assertRaises(ValueError):final.validate_csv(path,['a','a'])
            self.write_csv(path,list(reversed(columns)),[['a',*[.5]*12]])
            with self.assertRaises(ValueError):final.validate_csv(path,['a'])

    def test_no_cpu_fallback(self):
        fake=types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda:False))
        with self.assertRaisesRegex(RuntimeError,'no CPU fallback'):final.require_cuda(fake)

    def test_fullfive_qualification_rejects_foreign_family_and_incomplete_evidence(self):
        c,r=fixture();final.require_qualification(c,r)
        for target in ['subset','res','labels','source','model','kernel','status','strict','hash','repeat','slots','stress','contract_hash']:
            cc,rr=copy.deepcopy(c),copy.deepcopy(r)
            if target=='subset':cc['members']=cc['members'][:2];cc['complete_family']=False
            elif target=='res':cc['settings']['res']=384
            elif target=='labels':cc['labels_sha256']='d'*64
            elif target=='source':cc['train_knee_patched_sha256']='d'*64
            elif target=='model':cc['model_definitions_sha256']='d'*64
            elif target=='kernel':cc['members'][0]['kernel']='ozgung/knee-convnext'
            elif target=='status':rr['status']='RUNNING'
            elif target=='strict':rr['models'][1]['strict_load']=False
            elif target=='hash':rr['models'][2]['checkpoint_sha256']='d'*64
            elif target=='repeat':rr['models'][3]['repeat_max_error']=.001
            elif target=='slots':rr['training_parity'][0]['slots'].pop()
            elif target=='stress':rr['stress_checks']['cases'].pop(next(iter(rr['stress_checks']['cases'])))
            else:rr['contract_sha256']='d'*64
            with self.subTest(target=target),self.assertRaises(ValueError):final.require_qualification(cc,rr)

    def archive(self,root):
        model_path=os.environ.get('COATNET224_PUBLIC_SOURCE');reader_path=os.environ.get('COATNET224_READER_SOURCE')
        if not model_path or not reader_path:self.skipTest('Supply pinned source files to verify exact archive/builder')
        model=final.own.model_definitions(Path(model_path).read_text())
        reader=final.parity.recipe(Path(reader_path).read_text())
        sources={name:(builder.ROOT/name).read_text() for name in builder.FILES};c,r=fixture()
        c.update(sources={name:smoke.digest(code.encode()) for name,code in sources.items()},codecs=builder.CODECS)
        r['contract_sha256']=smoke.digest(smoke.canonical(c))
        # Exact archived executable of the rehearsal builder; never execute it.
        code=("import time,sys,hashlib\nfrom pathlib import Path\nSTARTED=time.monotonic()\n"
              f"SOURCES={sources!r}\nCONTRACT={c!r}\nREADER={reader!r}\nMODEL={model!r}\n"
              "base=Path('/kaggle/working/coatnet224_code')\nfor name,source in SOURCES.items():\n"
              "    if hashlib.sha256(source.encode()).hexdigest()!=CONTRACT['sources'][name]:raise ValueError('Packaged sources differ')\n"
              "    p=base/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(source)\n"
              "sys.path.insert(0,str(base))\nfrom knee.coatnet224_rehearsal import run\n"
              "run('/kaggle/input','/kaggle/working/coatnet224_rehearsal',CONTRACT,READER,MODEL,started=STARTED)\n")
        blob=json.dumps({'cells':[{'cell_type':'code','source':code.splitlines(True)}]}).encode()
        receipt={'notebook_sha256':smoke.digest(blob),'notebook_bytes':len(blob),'contract_sha256':smoke.digest(smoke.canonical(c)),
                 'candidate_folds':list(range(5)),'complete_family':True}
        metadata={'enable_internet':False,'enable_gpu':True,'is_private':True,
                  'dataset_sources':[*smoke.CORPORA,'goodpjw2008/rsna-knee-2-5d-convnext-reader'],
                  'kernel_sources':[smoke.IDENTITY_KERNEL,*final.KERNELS],
                  'competition_sources':['rsna-knee-abnormality-detection'],'model_sources':[],'docker_image':builder.DOCKER_IMAGE}
        (root/'notebook.ipynb').write_bytes(blob)
        for name,value in [('build-contract.json',c),('build-receipt.json',receipt),('kernel-metadata.json',metadata)]:
            (root/name).write_text(json.dumps(value))
        self.write_csv(root/'visible.csv',['StudyInstanceUID',*smoke.TARGETS],[[uid,*[.5]*12] for uid in ['visible1','visible2','visible3']])
        r['visible_csv_sha256']=smoke.file_sha(root/'visible.csv');(root/'report.json').write_text(json.dumps(r))
        return c,r,reader,model

    def test_synthetic_builder_closure_nine_hour_alarm_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.archive(root)
            kwargs={'rehearsal_build':root,'rehearsal_report':root/'report.json','rehearsal_csv':root/'visible.csv'}
            built=builder.build(root/'entry','test-coatnet224',**kwargs)
            self.assertTrue(built['build_only']);self.assertFalse(built['production_cuda_tested'])
            metadata=json.loads((root/'entry/kernel-metadata.json').read_text())
            self.assertEqual(metadata['kernel_sources'],final.KERNELS);self.assertFalse(metadata['enable_internet'])
            self.assertEqual(metadata['dataset_sources'],['goodpjw2008/rsna-knee-2-5d-convnext-reader'])
            notebook=json.loads((root/'entry/notebook.ipynb').read_text());code=''.join(notebook['cells'][1]['source'])
            compile(code,'<synthetic-build-only>','exec');self.assertIn('setitimer(signal.ITIMER_REAL,32400)',code)
            self.assertNotIn('visible1',code);self.assertIn('from knee.coatnet224_submission import run',code)
            values={n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(code).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['SOURCES','CONTRACT']}
            bundle=values['CONTRACT'];sources=values['SOURCES'];packed=root/'packed'
            for name,source in sources.items():
                path=packed/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(source)
            result=subprocess.run([sys.executable,'-I','-c','import sys;sys.path.insert(0,sys.argv[1]);import knee.coatnet224_submission',str(packed)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            final.require_bundle(bundle,final.parity.recipe(Path(os.environ['COATNET224_READER_SOURCE']).read_text()),final.own.model_definitions(Path(os.environ['COATNET224_PUBLIC_SOURCE']).read_text()))
            bad=copy.deepcopy(bundle);bad['rehearsal_report']['elapsed_seconds']+=1
            with self.assertRaises(ValueError):final.require_bundle(bad,'reader','model')
            with self.assertRaises(ValueError):builder.build(root/'entry','test-coatnet224',**kwargs)
            (root/'visible.csv').write_text('tampered CSV')
            with self.assertRaises(ValueError):builder.build(root/'badcsv','test-coatnet224',**kwargs)
            self.assertFalse((root/'badcsv').exists())
            (root/'notebook.ipynb').write_bytes((root/'notebook.ipynb').read_bytes()+b' ')
            with self.assertRaises(ValueError):builder.archived(root)

    def test_rehashed_executable_and_scientific_source_drift_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.archive(root)
            def change(mutator):
                nb=json.loads((root/'notebook.ipynb').read_text());code=''.join(nb['cells'][0]['source'])
                nb['cells'][0]['source']=mutator(code).splitlines(True);blob=json.dumps(nb).encode()
                (root/'notebook.ipynb').write_bytes(blob);receipt=json.loads((root/'build-receipt.json').read_text())
                receipt.update(notebook_sha256=smoke.digest(blob),notebook_bytes=len(blob));(root/'build-receipt.json').write_text(json.dumps(receipt))
            change(lambda code:code+'\nraise RuntimeError("extra executable")\n')
            with self.assertRaisesRegex(ValueError,'exact reconstruction'):builder.archived(root)
            self.archive(root);source=(builder.ROOT/'knee/coatnet224_rehearsal.py').read_text()
            change(lambda code:code.replace(repr(source),repr(source+'\n# drift\n')))
            with self.assertRaisesRegex(ValueError,'scientific dependency changed'):builder.archived(root)

    def test_actual_threefold_archive_cannot_build_final_entry(self):
        evidence=os.environ.get('COATNET224_EVIDENCE_ROOT')
        if not evidence:self.skipTest('Supply actual subset archive for rejection check')
        archive=Path(evidence)/'outputs/kaggle/own-coatnet224-rehearsal-fold014-20261010'
        c=json.loads((archive/'build-contract.json').read_text())
        self.assertEqual([m['fold'] for m in c['members']],[0,1,4])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,report=fixture();(root/'report.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'fullfive'):
                builder.build(root/'entry','test-coatnet224',rehearsal_build=archive,
                              rehearsal_report=root/'report.json',rehearsal_csv=root/'absent.csv')
            self.assertFalse((root/'entry').exists())


if __name__=='__main__':unittest.main()
