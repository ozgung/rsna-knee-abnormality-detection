"""Selection-B probes: B1 (R7 alone) and B2 (Dread K62 + R7). Synthetic fixtures; build tests
skip unless the pinned evidence is under KNEE_EVIDENCE_ROOT."""
import ast
import base64
import builtins
import copy
import hashlib
import importlib.util
import json
import lzma
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HAS=lambda *m:all(importlib.util.find_spec(x) is not None for x in m)
PROJECT=Path(__file__).resolve().parents[1]
EVIDENCE=Path(os.environ.get('KNEE_EVIDENCE_ROOT',PROJECT))
OUT=EVIDENCE/'outputs'
if HAS('numpy','pandas'):
    import numpy as np
    import pandas as pd
    from knee import public_s6_recovery_entry as E
    from knee import public_r7_recovery_entry as R7
    from knee import public_r7_recovery as RR
    from knee import public_r7_b1_entry as B1
    from knee import public_r7_b2_entry as B2
    from knee import public_s6_recovery_b0 as B0
    from knee import public_r7_stackers as S
    from tests.test_public_r7_recovery import passing_rehearsal as r7_rehearsal,global_loads,scratch
    from tests.test_public_s6_recovery_b0 import passing_rehearsal as b0_rehearsal


@unittest.skipUnless(HAS('numpy','pandas'),'numpy/pandas unavailable')
class StructureTests(unittest.TestCase):
    def test_workers_are_qualified_bytecode_with_resolved_globals(self):
        pairs=[(B1,'r7_worker',R7.r7_worker),(B1,'launch',R7.launch),(B1,'merge',E.merge),(B2,'r7_worker',R7.r7_worker),(B2,'old_worker',B0.old_worker),
               (B2,'launch',R7.launch),(B2,'merge',E.merge),(B2,'run',E.run)]
        for module,name,original in pairs:
            with self.subTest(module=module.__name__,name=name):
                f=getattr(module,name);self.assertIs(f.__code__,original.__code__);self.assertIs(f.__globals__,vars(module))
                self.assertEqual({n for n in global_loads(f.__code__) if not hasattr(builtins,n) and n not in vars(module)},set())
        for name in ('run','projection','recover','require_rehearsal','r7_parity'):
            with self.subTest(b1=name):self.assertEqual({n for n in global_loads(getattr(B1,name).__code__) if not hasattr(builtins,n) and n not in vars(B1)},set())

    def test_frozen_r7_and_b0_bytes(self):
        from scripts.build_public_r7_selection_b import R7_FROZEN_SHA,B0_MODULE_SHA
        for name,pin in {**R7_FROZEN_SHA,**B0_MODULE_SHA}.items():
            with self.subTest(name=name):self.assertEqual(hashlib.sha256((PROJECT/name).read_bytes()).hexdigest(),pin)

    def test_weights_and_slots(self):
        self.assertEqual((B1.METHOD['blend'],B1.METHOD['old'],B1.ARMS),([0.,1.],None,('s6',)));self.assertEqual(B1.METHOD['s6'],R7.R7_RECIPE)
        self.assertIsNone(B1.old_worker)
        self.assertEqual((B2.METHOD['blend'],B2.METHOD['old'],B2.METHOD['old_profile']),([.5,.5],B0.METHOD['old'],'raptor'));self.assertEqual(B2.METHOD['s6'],R7.R7_RECIPE)
        self.assertIs(B2.recover,RR.recover);self.assertEqual(B2.METHOD['recovery_recipe'],RR.RECIPE)
        self.assertEqual({k for k in set(B2.METHOD)|set(R7.METHOD) if B2.METHOD.get(k)!=R7.METHOD.get(k)},{'old','old_profile','variant'})
        self.assertEqual({k for k in set(B1.METHOD)|set(R7.METHOD) if B1.METHOD.get(k)!=R7.METHOD.get(k)},{'old','blend','raptor_per_finding','variant','recovery_recipe'})
        self.assertEqual(len({B1.VERSION,B2.VERSION,R7.VERSION,E.VERSION,B0.VERSION}),5)

    def test_no_nartaa_or_oai_dataset_attached(self):
        from scripts.build_public_r7_selection_b import VARIANTS,no_oai,kernel_title,PROTECTED_SLUGS
        from knee.reference import WEIGHT_DATASET,RAPTOR_WEIGHT_DATASET
        b1,b2=VARIANTS['b1']['datasets'],VARIANTS['b2']['datasets']
        for datasets in (b1,b2):
            self.assertTrue(no_oai(datasets));self.assertNotIn(WEIGHT_DATASET,datasets);self.assertFalse(any(d.startswith('nartaa/') for d in datasets))
            self.assertEqual(datasets[0],'heliosli/rsna-knee-compact-r7cnxfsss/1')
        self.assertNotIn(RAPTOR_WEIGHT_DATASET,b1);self.assertIn(RAPTOR_WEIGHT_DATASET,b2)
        with self.assertRaises(ValueError):no_oai(b2+[WEIGHT_DATASET])
        slugs=[VARIANTS[v][k] for v in VARIANTS for k in ('rehearsal','entry')]
        self.assertEqual(len(set(slugs)),4)
        for slug in slugs:
            with self.subTest(slug=slug):self.assertLessEqual(len(kernel_title(slug)),50);self.assertNotIn(slug,PROTECTED_SLUGS)
        with self.assertRaises(ValueError):kernel_title('knee-public-r7-recovery-entry-20261013')

    def test_launch_selects_variant_modules(self):
        for module,mode,expected in ((B1,'rehearsal','knee.public_r7_b1_rehearsal'),(B1,'entry','knee.public_r7_b1_entry'),(B2,'rehearsal','knee.public_r7_b2_rehearsal'),(B2,'entry','knee.public_r7_b2_entry')):
            with self.subTest(module=module.__name__,mode=mode),tempfile.TemporaryDirectory() as tmp:
                seen=[]
                class Done:
                    returncode=0
                    def poll(self):return 0
                    def wait(self,timeout=None):return 0
                def popen(argv,**kw):
                    seen.append(argv[2]);c=json.loads(Path(argv[-1]).read_bytes());(Path(c['out'])/c['arm']).mkdir();(Path(c['out'])/c['arm']/'receipt.json').write_text('{"board":%d}'%c['board']);return Done()
                with patch.object(module.subprocess,'Popen',side_effect=popen):module.launch({'mode':mode,'assignment':[['a'],['b']],'boards':[{},{}]},Path(tmp),'s6')
                self.assertEqual(seen,[expected,expected]);self.assertTrue((PROJECT/(expected.replace('.','/')+'.py')).is_file())

    def test_b1_run_launches_only_r7(self):
        src=ast.parse(Path(B1.__file__).read_text());run=next(n for n in src.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        self.assertNotIn("'old'",ast.unparse(run));self.assertIn('for arm in ARMS',ast.unparse(run))


@unittest.skipUnless(HAS('numpy','pandas'),'numpy/pandas unavailable')
class B1RecoveryTests(unittest.TestCase):
    def test_weight_one_rank_of_nine_digit_r7(self):
        s=np.random.default_rng(4).random((50,12));r=B1.recover(s,np.zeros(50,bool))
        self.assertTrue(np.array_equal(r['blend'],pd.DataFrame(S.quantize9(s)).rank(method='average',pct=True).to_numpy()))
        self.assertEqual((r['cap'],r['union_recovered'],r['recipe']),(1,0,'r7-only-capped-arm-mean-rank-v1'))

    def test_cap_and_mean_fill(self):
        s=np.random.default_rng(5).random((100,12));m=np.zeros(100,bool);m[[2,40,77]]=True;s2=s.copy();s2[m]=np.nan
        r=B1.recover(s2,m);mean=S.quantize9(S.quantize9(s[~m]).mean(0)[None])[0]
        self.assertTrue(np.array_equal(r['s6'][m],np.tile(mean,(3,1))));self.assertEqual(r['statuses']['s6'].count('recovered'),3)
        m[5]=True
        with self.assertRaises(ValueError):B1.recover(s2,m)
        for bad in ((s*2,np.zeros(100,bool)),(s,np.zeros(100)),(s[:,:11],np.zeros(100,bool))):
            with self.subTest(),self.assertRaises(ValueError):B1.recover(*bad)

    def test_projection_is_r7_phase_only(self):
        d=[500]*870;reports={'s6':[{'records':[{'windows':500,'production_seconds':10.}]*20},{'records':[{'windows':500,'production_seconds':11.}]*19}]}
        p=B1.projection({'development_windows':d},{'s6':{'wall_seconds':400.}},reports,10.,1.)
        self.assertEqual(set(p['measured_production_wall_seconds']),{'s6'});self.assertAlmostEqual(p['scenarios'][0]['seconds'],10+1+(400-209)+209*(1300*500/(39*500)))


def b1_receipt(tmp):
    r,binding,path=r7_rehearsal(tmp)
    binding.update(version=B1.VERSION,method=json.loads(json.dumps(B1.METHOD)));r.update(version=B1.VERSION,binding=binding)
    r['workers'].pop('old');r['phases'].pop('old');r['sentinel_ids'].pop('old')
    return r,binding,path


def b2_receipt(tmp):
    r,binding,path=r7_rehearsal(tmp);o,ob,_=b0_rehearsal(tmp)
    binding.update(version=B2.VERSION,method=json.loads(json.dumps(B2.METHOD)),old_profile=json.loads(json.dumps(B0.OLD_PROFILE)),old_qualification=ob['old_qualification'],
                   qualified_state={'old_state':B0.DREAD_STATE_SHA256,'old_grid':B0.DREAD_GRID_SHA256,'s6_state':None})
    binding['sentinel_probabilities']['old']=ob['sentinel_probabilities']['old'];binding['sentinel_inputs']['old']=ob['sentinel_inputs']['old']
    r.update(version=B2.VERSION,binding=binding);r['workers']['old']=json.loads(json.dumps(o['workers']['old']))
    return r,binding,path


@unittest.skipUnless(HAS('numpy','pandas'),'numpy/pandas unavailable')
class ContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)

    def test_b1_contract(self):
        r,b,p=b1_receipt(self.tmp.name);B1.require_rehearsal(r,b,p)
        def with_old(x):x['workers']['old']=[{}];x['phases']['old']={}
        def wrong_method(x):x['binding']['method']=R7.METHOD
        def fallback(x):x['workers']['s6'][0]['compile']['active']=False
        def slow(x):x['projection']['conservative_seconds']=8.4*3600
        def cross_board(x):
            s=x['workers']['s6'][1]['sentinels'][0];s['probabilities'][0]+=2e-3;s['reloaded_probabilities'][0]+=2e-3
        for mutate in (with_old,wrong_method,fallback,slow,cross_board):
            with self.subTest(mutate=mutate.__name__):
                x=copy.deepcopy(r);mutate(x)
                with self.assertRaises((ValueError,KeyError)):B1.require_rehearsal(x,x['binding'] if mutate is not wrong_method else b,p)
        with self.assertRaises(ValueError):B1.require_rehearsal(r,{**b,'method':R7.METHOD},p)

    def test_b2_contract(self):
        r,b,p=b2_receipt(self.tmp.name);B2.require_rehearsal(r,b,p)
        def nartaa_state(x):x['workers']['old'][0]['state_before']=x['workers']['old'][0]['state_after']={'state':'1d8594d77f7dd592bd1ef1b2090d74eb8ea09580fe689f78dbab273c8ae8a219'}
        def mirror_stress(x):x['workers']['old'][0]['partial_input_stress']['case_count']=17
        def old_cross_board(x):x['workers']['old'][1]['sentinels'][0]['probabilities'][0]+=2e-6
        def raptor62_out(x):
            for w in x['workers']['old']:
                for row in w['records']:
                    if 'raptor62' in row:row['raptor62'].update(max_abs=.02,within_atol=False)
        def r7_fallback(x):x['workers']['s6'][1]['compile']['fallback']='compile check failed'
        def no_r7(x):x['workers'].pop('s6')
        for mutate in (nartaa_state,mirror_stress,old_cross_board,raptor62_out,r7_fallback,no_r7):
            with self.subTest(mutate=mutate.__name__):
                x=copy.deepcopy(r);mutate(x)
                with self.assertRaises((ValueError,KeyError)):B2.require_rehearsal(x,b,p)


BUILD_INPUTS=[OUT/'kaggle/public-s6-evaluation-20261011-reviewed',OUT/'analysis/s6-capped-recovery-20261012/protocol-private.json',EVIDENCE/'data/raw/test.csv',
              OUT/'analysis/next-direction-20261013/kernels/heliosli-0957/0-957-latest-dataset-score-in-one-click.ipynb',OUT/'analysis/r7-recovery-20261013/r7-entry-v2-visible.json',
              OUT/'research/2026-10-08/raptor-audit/knee-mri-twelve-findings-from-a-single-model.py',OUT/'kaggle/raptor62-rehearsal-result',OUT/'analysis/b0-dread-swap-20261012/cpu-qualification-b0.json']


@unittest.skipUnless(all(p.exists() for p in BUILD_INPUTS) and HAS('numpy','pandas'),'selection-B build evidence unavailable')
class BuildTests(unittest.TestCase):
    def args(self,variant,destination):
        a=json.loads((OUT/'analysis/s6-capped-recovery-20261012/parent-build-arguments.json').read_bytes())
        a={k:str(v).replace('~/Dev/rsna-knee-abnormality-detection',str(EVIDENCE)) for k,v in a.items() if k not in ('control_build','control_result','destination','slug')}
        a.update(variant=variant,destination=str(destination),slug='knee-r7-only-b1-rehearsal-20261013' if variant=='b1' else 'knee-r7-dread-b2-rehearsal-20261013',
                 heliosli_notebook=str(BUILD_INPUTS[3]),r7_asset_dir=str(OUT/'analysis/r7-recovery-20261013/r7-v1-assets'),heliosli_output=str(OUT/'analysis/r7-recovery-20261013/heliosli-0957-output'),
                 r7_reference=str(BUILD_INPUTS[4]),dread_source=str(BUILD_INPUTS[5]),raptor62_result=str(BUILD_INPUTS[6]),dread_cpu_receipt=str(BUILD_INPUTS[7]))
        return a

    def pack(self,root):
        code=''.join(json.loads((root/'notebook.ipynb').read_bytes())['cells'][0]['source'])
        value=next(ast.literal_eval(n.value) for n in ast.parse(code).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='encoded' for t in n.targets))
        return code,json.loads(lzma.decompress(base64.b64decode(value,validate=True)))

    def test_rehearsal_builds(self):
        from scripts.build_public_r7_selection_b import build,verify_build,VARIANTS,R7_FROZEN_SHA,B0_MODULE_SHA
        root=scratch(self)
        for variant in ('b1','b2'):
            with self.subTest(variant=variant):
                r=build(**self.args(variant,root/variant));self.assertEqual(verify_build(root/variant,variant),r)
                meta=json.loads((root/variant/'kernel-metadata.json').read_bytes());self.assertEqual(meta['dataset_sources'],VARIANTS[variant]['datasets'])
                self.assertTrue(meta['is_private'] and not meta['enable_internet'] and meta['machine_shape']=='NvidiaTeslaT4')
                code,pack=self.pack(root/variant);cfg=pack['cfg']
                self.assertNotIn('nartaa',json.dumps(meta));self.assertNotIn("mount('nartaa",code);self.assertNotIn('rsna-knee-publication-swa-weights-20261007\',[',code);self.assertIn('NO_NARTAA_ATTACHMENT',code)
                self.assertNotIn('control_checkpoint',json.dumps(cfg));self.assertTrue(all('nartaa' not in json.dumps(cfg.get(k,'')) for k in ('public_manifest','public_source')))
                for name,pin in R7_FROZEN_SHA.items():
                    if name.startswith('knee/public_r7_recovery') and 'rehearsal' not in name or name=='knee/public_r7_stackers.py':self.assertEqual(hashlib.sha256(pack['sources'][name].encode()).hexdigest(),pin)
                self.assertEqual(sorted(cfg['r7_entry_reference']),sorted(cfg['binding']['visible_ids']))
                if variant=='b1':
                    self.assertNotIn('public_source',cfg);self.assertNotIn('knee/public_s6_recovery_b0.py',pack['sources']);self.assertIn('from knee.public_r7_b1_entry import run',code)
                else:
                    self.assertEqual((cfg['public_manifest']['profile'],cfg['control_state_sha256']),('raptor',B0.DREAD_STATE_SHA256));self.assertIn('dreaddevelopment/raptor-knee-widedense',code)
                    self.assertEqual(hashlib.sha256(pack['sources']['knee/public_s6_recovery_b0.py'].encode()).hexdigest(),B0_MODULE_SHA['knee/public_s6_recovery_b0.py'])
                with self.assertRaises(ValueError):build(**{**self.args(variant,root/(variant+'-entry')),'entry':True})


if __name__=='__main__':
    unittest.main()
