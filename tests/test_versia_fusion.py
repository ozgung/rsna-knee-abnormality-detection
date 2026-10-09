"""Offline orchestration tests: synthetic predictions/processes, pinned source archives."""
import ast
import copy
import importlib.util
import inspect
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

from knee import mirror_input, versia, versia_fusion as f
from knee.reference import canonical_bytes,sha256
from knee.reference_blend import write_probabilities

PARENT=Path(os.environ['KNEE_REFERENCE_PARENT']) if os.environ.get('KNEE_REFERENCE_PARENT') else None
ROOT=Path(__file__).resolve().parents[1]


def builder():
    sys.path.insert(0,str(ROOT/'scripts'))
    import build_versia_fusion
    return build_versia_fusion


def real_args():
    k=PARENT/'outputs/kaggle';r=PARENT/'outputs/research/2026-10-08'
    return dict(mirror_build=k/'mirror-rehearsal-20261009',
      mirror_report=k/'mirror-rehearsal-20261009-result/reference-rehearsal/rehearsal.json',
      mirror_csv=k/'mirror-rehearsal-20261009-result/reference-rehearsal/visible-predictions.csv',
      public_source=r/'mirror-audit/rsna-knee-0949-anatomical-mirror.py',
      versia_build=k/'versia-convnext16-rehearsal-20261009',
      versia_report=k/'versia-convnext16-rehearsal-20261009-result/versia-rehearsal/versia-audit.json',
      versia_csv=k/'versia-convnext16-rehearsal-20261009-result/versia-rehearsal/visible-predictions.csv',
      spec_dir=PARENT/'outputs/analysis/mirror-fusion-spec-20261009')


def evidence(directory,q,*,real=None):
    """Clearly synthetic successful common receipt, never a remote qualification."""
    import numpy as np
    d=Path(directory);d.mkdir();paths={};reports={};ids=['synthetic-a','synthetic-b','synthetic-c'];cols=['StudyInstanceUID',*f.TARGETS]
    for arm in ('mirror','convnext'):
        ad=d/arm;ad.mkdir();paths[arm]=ad/'visible-predictions.csv'
        if real:shutil.copyfile(real[arm],paths[arm]);cols,ids,_=f.read_predictions(paths[arm])
        else:
            p=np.array([[.13+j*.01 for j in range(12)],[.43+j*.01 for j in range(12)],[.73+j*.01 for j in range(12)]])
            write_probabilities(paths[arm],cols,ids,p)
        r={'status':'PASS','fusion_qualification_sha256':sha256(canonical_bytes(q)),
           'csv_sha256':sha256(paths[arm].read_bytes()),'visible':True,'completed_studies':3,'test_studies':3,
           'ordered_ids_sha256':sha256(canonical_bytes(ids)),'target_order_sha256':sha256(canonical_bytes(f.TARGETS)),
           'prediction_fallbacks':0,'codec_hashes':q['codecs'],'repeat_prediction_max_error':0,
           **dict.fromkeys(('component_elapsed_seconds','elapsed_seconds','peak_cuda_allocated_bytes','peak_cuda_reserved_bytes',
                            'peak_host_rss_bytes','disk_free_bytes','output_disk_bytes'),10),
           'environment':{'python_major_minor':[3,12],'packages':q['environment']['packages']},
           'backend_flags':{'cudnn_benchmark':arm=='mirror','cuda_matmul_allow_tf32':arm=='mirror',
                            'cudnn_deterministic':False,'cudnn_allow_tf32':True}}
        if arm=='mirror':
            r.update(checkpoint_verified_sha256=q['mirror']['manifest']['checkpoint_sha256'],checkpoint_bytes=mirror_input.CHECKPOINT_BYTES,
                     input_policy=mirror_input.policy_contract(),production_input_counts={'resource_errors':0})
            name='mirror-audit.json'
        else:
            r.update(checkpoint_sha256=versia.WEIGHTS,policy=versia.policy_contract(),input_counts={'resource_errors':0,'zero_studies':0},
              batch_sizes=[3],execution={k:versia.RECIPE[k] for k in ('batch_size','workers','prefetch_factor','pin_memory','multiprocessing')},
              public_tensor_max_error=0,strict_coverage={'PASS':3},
              batch2_diagnostic={'studies':2,'repeat_max_abs_error':0,'batch3_max_abs_error':.0001,'affects_production_predictions':False})
            name='convnext-audit.json'
        reports[arm]=r;(ad/name).write_text(json.dumps(r))
    return reports,paths


def simple_q():
    return {'policy':f.POLICY,'mirror':{'manifest':{'checkpoint_sha256':'a'*64}},'codecs':{'x':'b'*64},
            'environment':{'packages':{'numpy':'2.0.2'}}}


@unittest.skipUnless(importlib.util.find_spec('numpy') and importlib.util.find_spec('pandas'),'optional numeric dependencies')
class FusionTests(unittest.TestCase):
    def test_exact_ranks_ties_and_reordered_component_alignment(self):
        import numpy as np
        import pandas as pd
        ids=list('abcd');cols=['StudyInstanceUID',*f.TARGETS]
        m=np.repeat(np.array([[.2,.2,.8,.4]]).T,12,axis=1)
        c=np.repeat(np.array([[.8,.4,.4,.2]]).T,12,axis=1)
        expected_m=pd.DataFrame((m.argsort(0).argsort(0)/3).astype(np.float32)).rank(method='average',pct=True)
        expected_c=pd.DataFrame(c).rank(method='average',pct=True)
        expected=((1-np.array(f.WEIGHTS))*expected_m+np.array(f.WEIGHTS)*expected_c).rank(method='average',pct=True).to_numpy()
        actual=f.fuse(cols,ids,m,ids[::-1],c[::-1]);np.testing.assert_array_equal(actual,expected)
        # Q breaks mirror ties; C uses average ties. Replacing Q with average ranks differs.
        wrong=((1-np.array(f.WEIGHTS))*pd.DataFrame(m).rank(pct=True)+np.array(f.WEIGHTS)*expected_c).rank(pct=True)
        self.assertFalse(np.array_equal(actual,wrong))
        for badids in (['a','a','c','d'],['a','b','c','z']):
            with self.assertRaises(ValueError):f.fuse(cols,ids,m,badids,c)
        with self.assertRaises(ValueError):f.fuse(cols,ids,m,ids,np.zeros((4,12)))
        with self.assertRaises(ValueError):f.fuse(cols,['a'],m[:1],['a'],c[:1])
        with self.assertRaises(ValueError):f.fuse(cols[::-1],ids,m,ids,c)
        c[0,0]=float('nan')
        with self.assertRaises(ValueError):f.fuse(cols,ids,m,ids,c)

    def test_legitimate_rank_endpoints_and_both_arms_required_before_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)/'common';q=simple_q();reports,paths=evidence(d,q)
            for arm in ('mirror','convnext'):
                bad=copy.deepcopy(reports);bad[arm]['status']='FAIL'
                with self.assertRaises(ValueError):f.combine(d,q,bad,paths,visible=True,started=time.monotonic()-1)
                self.assertFalse((d/'visible-predictions.csv').exists())
            f.combine(d,q,reports,paths,visible=True,started=time.monotonic()-1)
            self.assertEqual(f.read_rank_csv(d/'visible-predictions.csv')[2][-1].tolist(),[1.]*12)
            f.require_visible(q,d)
            # A stale directory/output cannot become a second successful run.
            with self.assertRaises(FileExistsError):f.combine(d,q,reports,paths,visible=True,started=time.monotonic()-1)
            (d/'fusion-failure.json').write_text('{}')
            with self.assertRaises(ValueError):f.require_visible(q,d)

    def test_receipt_mixing_resources_repeats_flags_and_order_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            q=simple_q();reports,paths=evidence(Path(tmp)/'common',q)
            for arm,key,value in [('mirror','backend_flags',{}),('convnext','input_counts',{'resource_errors':1,'zero_studies':0}),
                ('convnext','batch_sizes',[1,1,1]),('mirror','repeat_prediction_max_error',1.1e-6),
                ('mirror','fusion_qualification_sha256','f'*64),('mirror','elapsed_seconds',float('nan')),
                ('convnext','public_tensor_max_error',1),('convnext','batch2_diagnostic',{})]:
                bad=copy.deepcopy(reports);bad[arm][key]=value
                with self.subTest(key=key),self.assertRaises(ValueError):f.require_components(q,bad,paths,visible=True)
            lines=paths['convnext'].read_text().splitlines();paths['convnext'].write_text('\n'.join([lines[0],*lines[:0:-1]])+'\n')
            bad=copy.deepcopy(reports);bad['convnext']['csv_sha256']=sha256(paths['convnext'].read_bytes())
            bad['convnext']['ordered_ids_sha256']=sha256(canonical_bytes(f.read_predictions(paths['convnext'])[1]))
            with self.assertRaises(ValueError):f.require_components(q,bad,paths,visible=True)

    def test_cross_process_abs_precision_and_tie_guards(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);a=d/'a.csv';b=d/'b.csv';cols=['StudyInstanceUID',*f.TARGETS];ids=list('abc')
            p=np.tile([.1,.4,.8],(12,1)).T+np.arange(12)*.001
            write_probabilities(a,cols,ids,p);write_probabilities(b,cols,ids,p+5e-5)
            self.assertAlmostEqual(f.compare_component(a,b,arm='mirror'),5e-5)
            with self.assertRaises(ValueError):f.compare_component(a,b,arm='convnext')
            b.unlink();p[1,0]=p[0,0];a.unlink();write_probabilities(a,cols,ids,p)
            p[1,0]+=1e-7;write_probabilities(b,cols,ids,p)
            with self.assertRaises(ValueError):f.compare_component(a,b,arm='mirror')


class ProcessTests(unittest.TestCase):
    def test_fresh_processes_nonzero_and_preexpired_deadline(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'state'
            f.run_process([sys.executable,'-c','import os;os.environ["FUSION_CHILD_ONLY"]="yes"'],deadline=time.monotonic()+5)
            f.run_process([sys.executable,'-c',f'import os;from pathlib import Path;Path({str(p)!r}).write_text(str(os.environ.get("FUSION_CHILD_ONLY")))'],deadline=time.monotonic()+5)
            self.assertEqual(p.read_text(),'None')
            with self.assertRaises(RuntimeError):f.run_process([sys.executable,'-c','raise SystemExit(7)'],deadline=time.monotonic()+5)
            with patch.object(subprocess,'Popen') as popen:
                with self.assertRaises(TimeoutError):f.run_process(['unused'],deadline=time.monotonic()-1)
                popen.assert_not_called()

    @unittest.skipUnless(os.name=='posix','POSIX process-group deadline')
    def test_timeout_and_success_cleanup_descendant_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            for mode in ('timeout','success'):
                marker=Path(tmp)/mode
                child=f'import time;from pathlib import Path;time.sleep(1);Path({str(marker)!r}).write_text("leaked")'
                parent=f'import subprocess,sys,time;subprocess.Popen([sys.executable,"-c",{child!r}]);'+('time.sleep(10)' if mode=='timeout' else '')
                if mode=='timeout':
                    with self.assertRaises(subprocess.TimeoutExpired):f.run_process([sys.executable,'-c',parent],deadline=time.monotonic()+.3)
                else:f.run_process([sys.executable,'-c',parent],deadline=time.monotonic()+5)
                time.sleep(1.1);self.assertFalse(marker.exists())


@unittest.skipUnless(PARENT and (PARENT/'outputs/kaggle/versia-convnext16-rehearsal-20261009-result/versia-rehearsal/versia-audit.json').is_file(),
                     'authentic archived component fixtures')
class FusionArchiveTests(unittest.TestCase):
    def test_published_formula_oracle_with_mirror_convnext_and_weighted_ties(self):
        import numpy as np
        import pandas as pd
        source=(real_args()['spec_dir']/'published-fusion.py').read_text();tree=ast.parse(source)
        body=next(n.body for n in tree.body if isinstance(n,ast.Try))
        selected=[];active=False
        for node in body:
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TARGET_WEIGHTS' for t in node.targets):active=True
            if active:
                if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='fused_df.to_csv':break
                selected.append(node)
        for count in (3,4,8,25):
            rng=np.random.default_rng(92+count);m=rng.choice([.1,.2,.4,.7],(count,12));c=rng.choice([.2,.3,.6],(count,12))
            c[0]=.1;c[-1]=.9;ids=[f's{i}' for i in range(count)];cols=['StudyInstanceUID',*f.TARGETS]
            sota=pd.DataFrame((m.clip(0,1).argsort(0).argsort(0)/max(1,count-1)).astype(np.float32),columns=f.TARGETS)
            own=pd.DataFrame(c,columns=f.TARGETS);env={'sota_df':sota,'own_df':own,'labels':f.TARGETS}
            exec(compile(ast.Module(body=selected,type_ignores=[]),'<pinned-public-rank-oracle>','exec'),env)
            np.testing.assert_array_equal(f.fuse(cols,ids,m,ids,c),env['fused_df'][f.TARGETS].to_numpy())

    def test_authentic_archive_build_no_prediction_fixtures_and_isolated_import(self):
        b=builder()
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)/'build';receipt=b.build_fusion(d,'synthetic-visible',**real_args())
            contract=json.loads((d/'fusion-contract.json').read_text());meta=json.loads((d/'kernel-metadata.json').read_text())
            self.assertLess(receipt['notebook_bytes'],900000);self.assertFalse(meta['enable_internet']);self.assertTrue(meta['is_private'])
            self.assertEqual(len(meta['dataset_sources']),2);self.assertEqual(meta['kernel_sources'],[])
            code=''.join(json.loads((d/'notebook.ipynb').read_text())['cells'][-1]['source']);compile(code,'<generated-fusion>','exec')
            own=ast.literal_eval(next(n.value for n in ast.parse(code).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='sources' for t in n.targets)))
            for n in b.SCIENTIFIC:self.assertEqual(own[n],(ROOT/n).read_text())
            for arg in ('mirror_csv','versia_csv'):
                for sid in f.read_predictions(real_args()[arg])[1]:self.assertNotIn(sid,code)
            bundle=Path(tmp)/'bundle'
            for n,v in own.items():
                p=bundle/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(v)
            env=dict(os.environ,PYTHONPATH=str(bundle));r=subprocess.run([sys.executable,'-c','import sys;import knee.versia_fusion;assert "torch" not in sys.modules'],
                     cwd=tmp,env=env,capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr)
            # Synthetic receipts exercise only local full-builder gates, not real qualification.
            q=contract['qualification'];reports,paths=evidence(Path(tmp)/'common',q,real={'mirror':real_args()['mirror_csv'],'convnext':real_args()['versia_csv']})
            f.combine(Path(tmp)/'common',q,reports,paths,visible=True,started=time.monotonic()-1)
            full=b.build_fusion(Path(tmp)/'full','synthetic-full',**real_args(),submission=True,visible_results=Path(tmp)/'common')
            self.assertEqual(full['qualification_sha256'],receipt['qualification_sha256'])
            with self.assertRaises(ValueError):b.build_fusion(Path(tmp)/'missing','synthetic-full',**real_args(),submission=True)
            report=json.loads((Path(tmp)/'common/fusion-audit.json').read_text());report['qualification_sha256']='0'*64
            (Path(tmp)/'common/fusion-audit.json').write_text(json.dumps(report))
            with self.assertRaises(ValueError):b.build_fusion(Path(tmp)/'tamper','synthetic-full',**real_args(),submission=True,visible_results=Path(tmp)/'common')

    def test_archive_receipt_and_source_tamper_rejected(self):
        b=builder();args=real_args()
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)/'archive';shutil.copytree(args['versia_build'],d)
            nb=json.loads((d/'notebook.ipynb').read_text());nb['cells'][-1]['source'].append('\nprint("unreviewed")\n')
            (d/'notebook.ipynb').write_text(json.dumps(nb))
            with self.assertRaises(ValueError):b.archived_versia(d,args['spec_dir'])
            bad=Path(tmp)/'report.json';r=json.loads(args['versia_report'].read_text());r['source_revision']='0'*40;bad.write_text(json.dumps(r))
            with self.assertRaises(ValueError):b.build_fusion(Path(tmp)/'fail','synthetic',**{**args,'versia_report':bad})
            with patch.object(b,'ROOT',Path(tmp)):
                with self.assertRaises((ValueError,FileNotFoundError)):b.archived_versia(args['versia_build'],args['spec_dir'])


@unittest.skipUnless(importlib.util.find_spec('torch'),'optional torch worker fixture')
class BatchLayoutTests(unittest.TestCase):
    def test_actual_dataset_and_four_workers_keep_final_remainders_N1_to9(self):
        import numpy as np
        import torch
        import warnings
        tree=ast.parse(inspect.getsource(f.convnext_run))
        nodes=[next(n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=='TestDataset'),
               next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='dl' for t in n.targets))]
        def arrays(objects,sid,table,path):
            value=int(sid[1:]);return (np.full((6,16,3,2,2),value,dtype=np.uint8),np.ones(6,dtype=bool),np.zeros((6,16),dtype=np.float32)),dict.fromkeys(versia.COUNTERS,0)
        from types import SimpleNamespace
        for n in range(1,10):
            env={'torch':torch,'np':np,'Path':Path,'versia':SimpleNamespace(study_arrays=arrays,strict_eligible=lambda _:True),
                 'protected':None,'plain':None,'visible':False,'root':'synthetic','table':{},'ids':[f's{i}' for i in range(n)]}
            with warnings.catch_warnings():
                warnings.simplefilter('ignore');exec(compile(ast.Module(body=nodes,type_ignores=[]),'<actual-production-loader>','exec'),env)
                sizes=[];order=[]
                for x,mask,pos,counts,eligible in env['dl']:sizes.append(len(x));order.extend(x[:,0,0,0,0,0].tolist())
            self.assertEqual(sizes,[min(4,n-i) for i in range(0,n,4)]);self.assertEqual(order,list(range(n)))
            self.assertEqual(env['dl'].num_workers,4);self.assertEqual(env['dl'].prefetch_factor,2);self.assertFalse(env['dl'].drop_last)
            del env['dl']
