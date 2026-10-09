"""Aggregate-only diagnostic guard tests; mocked inference is not CUDA evidence."""
import ast
import copy
from contextlib import ExitStack,redirect_stdout
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch
from knee import mavit_diagnostic as d, mavit_rehearsal as r

ROOT=Path(__file__).resolve().parents[1]
PARENT=Path(os.environ['KNEE_REFERENCE_PARENT']) if os.environ.get('KNEE_REFERENCE_PARENT') else None


def builder():
    sys.path.insert(0,str(ROOT/'scripts'));import build_mavit_diagnostic;return build_mavit_diagnostic


def real_args():
    results=PARENT/'outputs/kaggle/mavit-image-rehearsal-v3-20261009-result/mavit-rehearsal'
    return dict(rehearsal_build=PARENT/'outputs/kaggle/mavit-image-rehearsal-v3-20261009',rehearsal_report=results/'mavit-audit.json',
                rehearsal_csv=results/'visible-predictions.csv',plan=PARENT/'outputs/kaggle/train04-cache-full-v2-result/coverage_cache/plan-private.json')


def synthetic_plan(path,*,rows=None):
    rows=rows or [{'uid':f'synthetic-private-{i}','split':'train' if i<3 else 'weak_validation','labels':'FORBIDDEN LABEL VALUE',
                   'group':'FORBIDDEN GROUP','component':'FORBIDDEN COMPONENT'} for i in range(5)]
    raw=json.dumps({'fingerprint':d.PLAN_FINGERPRINT,'records':rows}).encode();path.write_bytes(raw)
    return d.cp.digest(raw)


class MembershipTests(unittest.TestCase):
    def test_pin_then_uid_split_projection_only_and_deterministic_train_order(self):
        class Row(dict):
            def get(self,key,*args):
                if key not in ('uid','split'):raise AssertionError('Supervision field accessed')
                return super().get(key,*args)
            def __iter__(self):raise AssertionError('Do not traverse label-bearing row')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'plan.json';sha=synthetic_plan(path);document=json.loads(path.read_text())
            document['records']=[Row(x) for x in document['records']]
            with patch.multiple(d,PLAN_SHA256=sha,TRAIN_COUNT=3,DEV_COUNT=2,SERIES_COUNT=3),patch.object(d.json,'loads',return_value=document):
                ids,cohort=d.membership(path)
            self.assertEqual(len(ids),3);self.assertEqual(cohort['fields_used'],['uid','split'])
            self.assertEqual(ids,sorted(ids,key=lambda x:(d.cp.digest(f'{d.SALT}:{x}'.encode()),x)))
            self.assertTrue(all(int(x.rsplit('-',1)[1])<3 for x in ids));self.assertNotIn('FORBIDDEN',json.dumps(cohort))
            # Hash failure precedes even JSON deserialization.
            with patch.object(d.json,'loads',side_effect=AssertionError('Must not parse changed file')),self.assertRaises(ValueError):d.membership(path)

    def test_malformed_membership_roles_duplicates_paths_and_counts_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'plan.json';synthetic_plan(path);original=json.loads(path.read_text())['records']
            for mode in ('role','duplicate','unsafe','missing','fingerprint'):
                rows=copy.deepcopy(original)
                if mode=='role':rows[0]['split']='gold'
                if mode=='duplicate':rows[4]['uid']=rows[0]['uid']
                if mode=='unsafe':rows[0]['uid']='../private'
                if mode=='missing':rows.pop()
                sha=synthetic_plan(path,rows=rows)
                if mode=='fingerprint':value=json.loads(path.read_text());value['fingerprint']='0'*64;path.write_text(json.dumps(value));sha=d.cp.digest(path.read_bytes())
                with self.subTest(mode=mode),patch.multiple(d,PLAN_SHA256=sha,TRAIN_COUNT=3,DEV_COUNT=2,SERIES_COUNT=3),self.assertRaises(ValueError):d.membership(path)

    def test_failure_sanitizer_never_exports_exception_text_paths_or_arbitrary_class_names(self):
        marker='PRIVATE UID SECRET REPORT'
        try:raise AssertionError(marker)
        except AssertionError as error:result=d.safe_failure(error)
        self.assertEqual(result['error_class'],'AssertionError');self.assertNotIn(marker,json.dumps(result));self.assertIsNone(result['source_site'])
        error=type('PRIVATE_IDENTIFIER',(Exception,),{})(marker)
        self.assertEqual(d.safe_failure(error)['error_class'],'UnknownError')


@unittest.skipUnless(PARENT and (PARENT/'outputs/kaggle/mavit-image-rehearsal-v3-20261009-result/mavit-rehearsal/mavit-audit.json').is_file(),'authentic v3 MaViT67 fixture')
class AuthenticTests(unittest.TestCase):
    def test_authentic_build_offline_source_identity_no_member_or_prediction_embedding_and_import(self):
        b=builder()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);receipt=b.build(root/'build','synthetic-diagnostic',**real_args())
            c=json.loads((root/'build/mavit-contract.json').read_text());q=d.qualification(c)
            self.assertEqual(q['packages']['scipy'],'1.16.3');self.assertEqual(q['rehearsal_report_sha256'],d.REHEARSAL_REPORT_SHA256)
            ids,cohort=d.membership(real_args()['plan']);self.assertEqual(cohort,c['cohort']);self.assertEqual(len(ids),2000)
            self.assertEqual(cohort['ordered_ids_sha256'],'7a9545f5ae5f79c133f27a20d758f0590da9f04789ef3ca721e8bb44e1a87da9')
            meta=json.loads((root/'build/kernel-metadata.json').read_text())
            self.assertEqual(meta['kernel_sources'],[d.KERNEL]);self.assertFalse(meta['enable_internet']);self.assertTrue(meta['is_private']);self.assertTrue(meta['enable_gpu'])
            code=''.join(json.loads((root/'build/notebook.ipynb').read_text())['cells'][-1]['source']);compile(code,'<diagnostic>','exec')
            self.assertLess(code.index('signal.setitimer'),code.index('sources='));self.assertLess(receipt['notebook_bytes'],900000)
            sources=ast.literal_eval(next(n.value for n in ast.parse(code).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='sources' for t in n.targets)))
            archived,_,_=b.archived.archived_build(real_args()['rehearsal_build'])
            for name,value in archived.items():self.assertEqual(sources[name],value)
            for uid in ids:self.assertNotIn(uid,code)
            project=root/'project'
            for name,value in sources.items():path=project/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(value)
            check=subprocess.run([sys.executable,'-c','import sys;import knee.mavit_diagnostic;import scripts.mavit_diagnostic_entry;assert "torch" not in sys.modules'],cwd=tmp,
                                 env=dict(os.environ,PYTHONPATH=str(project)),capture_output=True,text=True)
            self.assertEqual(check.returncode,0,check.stderr)
            with self.assertRaises(ValueError):b.build(root/'build','synthetic-diagnostic',**real_args())

    def test_old_v2_and_receipt_csv_plan_tampering_rejected_before_build_writes(self):
        b=builder()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for mode in ('old_archive','old_report','report_tamper','csv_tamper','plan_tamper'):
                kwargs=real_args()
                if mode=='old_archive':kwargs['rehearsal_build']=PARENT/'outputs/kaggle/mavit-image-rehearsal-20261009v1'
                if mode=='old_report':kwargs['rehearsal_report']=PARENT/'outputs/kaggle/mavit-image-rehearsal-20261009v1-result/mavit-rehearsal/mavit-audit.json'
                if mode.endswith('tamper'):
                    key={'report_tamper':'rehearsal_report','csv_tamper':'rehearsal_csv','plan_tamper':'plan'}[mode]
                    path=root/mode;path.write_bytes(kwargs[key].read_bytes()+b' ');kwargs[key]=path
                with self.subTest(mode=mode),self.assertRaises(ValueError):b.build(root/f'reject-{mode}','synthetic-reject',**kwargs)
                self.assertFalse((root/f'reject-{mode}').exists())

    def test_exact_production_calls_train_only_no_exports_and_fail_first_with_measured_diagnostics(self):
        import numpy as np
        import torch
        assets=PARENT/'outputs/analysis/post-fusion-research-20261009/mavit/dataset-code'
        b=builder()
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);b.build(base/'build','synthetic-runtime',**real_args());original=json.loads((base/'build/mavit-contract.json').read_text())
            root=base/'inputs';root.mkdir();plan=base/'synthetic-plan.json';sha=synthetic_plan(plan)
            with patch.multiple(d,PLAN_SHA256=sha,TRAIN_COUNT=3,DEV_COUNT=2,SERIES_COUNT=3):ids,cohort=d.membership(plan)
            c=copy.deepcopy(original);c['cohort']=cohort;q=c['production_contract']['qualification']
            (root/'sample_submission.csv').write_text(','.join(['StudyInstanceUID',*q['targets']])+'\n')
            with (root/'train_series.csv').open('w') as handle:
                writer=csv.DictWriter(handle,fieldnames=r.SERIES_COLUMNS);writer.writeheader()
                writer.writerows(dict(StudyInstanceUID=uid,SeriesInstanceUID=f'series-{i}',Fluid_Sensitive=0,Fat_Suppression=0,Anatomical_Plane='Sagittal') for i,uid in enumerate(ids+['forbidden-dev']))
            (root/'train.csv').write_text('FORBIDDEN LABELS');(root/'train_golden58.csv').write_text('FORBIDDEN GOLD')
            for mode in ('pass','decode','infer','memory','timeout','repeat','repeat_nan','resource_snapshot_failure','deadline'):
                calls=[];prepared=[];arrays_seen=[];reader=object();model=object();opened=[];saved_arrays=[];original_open=Path.open
                def open_file(path,*args,**kwargs):
                    opened.append(path.name)
                    if path.name in ('train.csv','train_golden58.csv','test.csv','cache-index-private.json'):raise AssertionError('Forbidden diagnostic data access')
                    return original_open(path,*args,**kwargs)
                def prepare(module,actual_reader,uid,rows,directory,*,production):
                    self.assertIs(actual_reader,reader);self.assertTrue(production);self.assertEqual(directory,root/'train_series');prepared.append(uid)
                    self.assertTrue(all(row['StudyInstanceUID']==uid for row in rows))
                    if mode in ('decode','memory','timeout','resource_snapshot_failure'):
                        cls={'decode':AssertionError,'memory':MemoryError,'timeout':TimeoutError,'resource_snapshot_failure':AssertionError}[mode]
                        error=cls('PRIVATE_UID_REPORT_PATH');error.mavit_counts=dict.fromkeys(r.inputs.COUNTERS,0)
                        error.mavit_counts['resource_errors']=int(mode in ('memory','timeout'));raise error
                    arrays=(np.full((6,8,3,2,2),ids.index(uid)+1,dtype=np.uint8),np.ones(6,dtype=bool),np.zeros((6,8),np.float32))
                    saved_arrays.append(tuple(x.copy() for x in arrays));arrays_seen.append(arrays)
                    counts=dict.fromkeys(r.inputs.COUNTERS,0);counts['selected_views']=6;return arrays,counts
                def predict(actual_model,arrays):
                    self.assertIs(actual_model,model);calls.append(int(arrays[0][0,0,0,0,0]));self.assertIs(arrays,arrays_seen[-1])
                    if mode=='infer':raise AssertionError('PRIVATE_UID_REPORT_PATH')
                    values=np.arange(12,dtype=np.float32)*.005+.1
                    if mode in ('repeat','repeat_nan') and len(calls)==2:values[0]+=1e-4 if mode=='repeat' else float('nan')
                    return values
                module=types.SimpleNamespace(KNEE_LABELS=q['targets'],make_knee_dataset_reader=lambda **kwargs:reader)
                audit={'state_compatibility':{'strict_load':True,'loaded_tensors_equal':True,'actual_schema_sha256':r.STATE_SCHEMA}}
                out=base/f'result-{mode}';output=io.StringIO()
                with self.subTest(mode=mode),ExitStack() as stack:
                    stack.enter_context(patch.multiple(d,PLAN_SHA256=sha,TRAIN_COUNT=3,DEV_COUNT=2,SERIES_COUNT=3))
                    stack.enter_context(patch.object(Path,'open',open_file));stack.enter_context(patch.object(d.inputs,'definitions',return_value=module))
                    stack.enter_context(patch.object(d.inputs,'prepare',side_effect=prepare));loader=stack.enter_context(patch.object(r,'load_model',return_value=(model,audit)))
                    stack.enter_context(patch.object(r,'predict',side_effect=predict));stack.enter_context(patch.object(d.importlib.metadata,'version',side_effect=q['packages'].__getitem__))
                    stack.enter_context(patch.object(torch,'set_num_threads'));stack.enter_context(patch.object(torch,'get_num_threads',return_value=4))
                    for attr,value in [('is_available',True),('get_device_name','mock-only'),('reset_peak_memory_stats',None),('memory_allocated',1),('memory_reserved',1),('max_memory_allocated',1),('max_memory_reserved',1)]:
                        stack.enter_context(patch.object(torch.cuda,attr,return_value=value))
                    if mode=='resource_snapshot_failure':stack.enter_context(patch.object(torch.cuda,'memory_allocated',side_effect=RuntimeError('PRIVATE_RESOURCE_ERROR')))
                    for owner,attr in [(torch.backends.cudnn,'benchmark'),(torch.backends.cudnn,'deterministic'),(torch.backends.cudnn,'allow_tf32'),(torch.backends.cuda.matmul,'allow_tf32')]:stack.callback(setattr,owner,attr,getattr(owner,attr))
                    stack.enter_context(redirect_stdout(output))
                    started=time.monotonic()-(32401 if mode=='deadline' else 0)
                    if mode=='pass':report=d.run(root,assets,plan,c,out,started=started)
                    else:
                        with self.assertRaises((AssertionError,MemoryError,TimeoutError,ValueError)) as raised:d.run(root,assets,plan,c,out,started=started)
                    report=json.loads((out/'diagnostic.json').read_text())
                    if mode=='pass':
                        d.require_result(report,c,parent_complete=False);self.assertEqual(prepared,ids);self.assertEqual(calls,[1,1,2,2,3,3])
                        for field,value in [('completed_studies',2),('finite_prediction_calls',5),('repeat_max_error',2e-6),('phase','repeat'),('elapsed_seconds',32401)]:
                            changed=copy.deepcopy(report);changed[field]=value
                            with self.assertRaises(ValueError):d.require_result(changed,c,parent_complete=False)
                    elif mode=='deadline':loader.assert_not_called();self.assertEqual(prepared,[])
                    else:
                        self.assertEqual(prepared,ids[:1]);self.assertEqual(report['ordinal'],1);self.assertEqual(report['completed_studies'],0)
                        self.assertEqual(report['status'],'FAIL');self.assertEqual(report['phase'],'repeat' if mode.startswith('repeat') else 'inference' if mode=='infer' else 'decode')
                    if mode=='repeat':self.assertGreater(report['current_repeat_max_error'],1e-6);self.assertEqual(report['repeat_max_error'],report['current_repeat_max_error'])
                    if mode=='repeat_nan':self.assertIsNone(report['current_repeat_max_error'])
                    if mode in ('infer','repeat','repeat_nan'):self.assertEqual(report['current_study']['present_views'],6);self.assertEqual(report['cuda_allocated_bytes'],1)
                    if mode=='resource_snapshot_failure':self.assertTrue(report['resource_snapshot_failed']);self.assertIsInstance(raised.exception,AssertionError)
                for actual,expected in zip(arrays_seen,saved_arrays):
                    for x,y in zip(actual,expected):np.testing.assert_array_equal(x,y)
                self.assertEqual({p.name for p in out.iterdir()},{'diagnostic.json'})
                text=(out/'diagnostic.json').read_text()+output.getvalue()
                for marker in [*ids,'PRIVATE_UID_REPORT_PATH','PRIVATE_RESOURCE_ERROR','FORBIDDEN']:self.assertNotIn(marker,text)
                self.assertNotIn('train.csv',opened);self.assertNotIn('train_golden58.csv',opened)


if __name__=='__main__':unittest.main()
