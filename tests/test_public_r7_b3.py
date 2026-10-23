"""B3 (B2 + dr-full at 0.2 of the Dread slot). Synthetic fixtures; build/source tests skip unless
the pinned evidence is under KNEE_EVIDENCE_ROOT."""
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
import types
import unittest
from unittest.mock import patch

HAS=lambda *m:all(importlib.util.find_spec(x) is not None for x in m)
PROJECT=Path(__file__).resolve().parents[1]
EVIDENCE=Path(os.environ.get('KNEE_EVIDENCE_ROOT',PROJECT))
OUT=EVIDENCE/'outputs'
DREAD_SOURCE=OUT/'research/2026-10-08/raptor-audit/knee-mri-twelve-findings-from-a-single-model.py'
if HAS('numpy','pandas'):
    import numpy as np
    import pandas as pd
    from knee import public_s6_recovery_entry as E
    from knee import public_r7_recovery_entry as R7
    from knee import public_r7_recovery as RR
    from knee import public_r7_b2_entry as B2
    from knee import public_r7_b3_entry as B3
    from knee import public_s6_recovery_b0 as B0
    from knee import public_r7_stackers as S
    from tests.test_public_r7_recovery import global_loads,scratch
    from tests.test_public_r7_selection_b import b2_receipt


def consts(code):
    out=[]
    for c in code.co_consts:
        out.append(consts(c) if isinstance(c,types.CodeType) else c)
    return out


def codes(code):
    yield code
    for c in code.co_consts:
        if isinstance(c,types.CodeType):yield from codes(c)


@unittest.skipUnless(HAS('numpy','pandas'),'numpy/pandas unavailable')
class StructureTests(unittest.TestCase):
    def test_workers_are_qualified_bytecode(self):
        for f,original in ((B3.old_worker,B0.old_worker),(B3.r7_worker,R7.r7_worker),(B3.merge,E.merge)):
            with self.subTest(name=f.__name__):
                self.assertIs(f.__code__,original.__code__);self.assertIs(f.__globals__,vars(B3))
                self.assertEqual({n for n in global_loads(f.__code__) if not hasattr(builtins,n) and n not in vars(B3)},set())
        self.assertIs(B3.WORKERS['old'],B3.old_worker);self.assertIs(B3.WORKERS['s6'],B3.r7_worker);self.assertIs(B3.WORKERS['dr'],B3.dr_worker)

    def test_dr_worker_is_b0_bytecode_with_only_arm_constant_swapped(self):
        swapped,original=B3._dr_old_worker.__code__,B0.old_worker.__code__
        pairs=list(zip(codes(swapped),codes(original)));self.assertEqual(len(pairs),len(list(codes(original))))
        changed=0
        for a,b in pairs:
            self.assertEqual(a.co_code,b.co_code);self.assertEqual(a.co_names,b.co_names);self.assertEqual(a.co_varnames,b.co_varnames)
            for x,y in zip(a.co_consts,b.co_consts):
                if isinstance(y,types.CodeType):continue
                if x!=y:self.assertEqual((x,y),('dr','old'));changed+=1
        self.assertEqual(changed,1)  # Python stores the repeated 'old' constant once per code object
        self.assertNotIn('old',[c for code in codes(swapped) for c in code.co_consts if isinstance(c,str)])
        self.assertIn('old',original.co_consts);self.assertIn('old_sentinels',swapped.co_consts)
        self.assertIs(B3._dr_old_worker.__globals__,B3.DR)
        self.assertEqual({n for n in global_loads(original) if not hasattr(builtins,n) and n not in B3.DR},set())

    def test_dr_namespace_pins(self):
        D=B3.DR
        self.assertEqual((D['DREAD_CHECKPOINT_SHA256'],D['DREAD_CHECKPOINT_BYTES'],D['DREAD_STATE_SHA256'],D['DREAD_GRID_SHA256'],D['RAPTOR_WINDOWS']),
                         (B3.DRFULL_CHECKPOINT_SHA256,292826627,B3.DRFULL_STATE_SHA256,B3.DRFULL_GRID_SHA256,42))
        self.assertIs(D['raptor_profile_gate'],B3.dr_profile_gate);self.assertIsNone(D['raptor62_gate']('x',None,{},{}))
        for name in ('raptor_windows','raptor_forward'):
            with self.subTest(name=name):self.assertIs(D[name].__code__,getattr(B0,name).__code__);self.assertIs(D[name].__globals__,D)
        self.assertEqual(B0.RAPTOR_WINDOWS,62);self.assertEqual(B0.DREAD_STATE_SHA256,'e0f0e7e1b10b03a4611283be8da4d95fb69d37f8e6d91929e5fd33d8291b62f2')
        with patch.object(B3,'rehearsal_hook','hook'):
            with patch.object(B3,'_dr_old_worker',lambda path:(D['rehearsal_hook'],D['VERSION'])):self.assertEqual(B3.dr_worker('x'),('hook',B3.VERSION))

    def test_weights_slots_and_method(self):
        self.assertEqual(B3.OLD_SLOT,{'old':.8,'dr':.2});self.assertEqual(B3.ARM_WEIGHTS,{'old':.4,'dr':.1,'s6':.5});self.assertAlmostEqual(sum(B3.ARM_WEIGHTS.values()),1.)
        self.assertEqual(B3.ARMS,('old','dr','s6'));self.assertEqual(B3.METHOD['old'],B2.METHOD['old']);self.assertEqual(B3.METHOD['s6'],R7.R7_RECIPE)
        self.assertEqual({k for k in set(B3.METHOD)|set(B2.METHOD) if B3.METHOD.get(k)!=B2.METHOD.get(k)},{'dr','dr_profile','old_slot','arm_weights','recovery_recipe','variant'})
        self.assertEqual(B3.DR_PROFILE['windows'],42);self.assertEqual({k for k in B3.DR_PROFILE if B3.DR_PROFILE[k]!=B0.OLD_PROFILE.get(k)},{'checkpoint_sha256','checkpoint_bytes','windows','model','export_k'})
        self.assertEqual(B3.CROSS_BOARD_TOLERANCE,{'old':1e-6,'dr':1e-6,'s6':1e-3})

    def test_datasets_and_slugs(self):
        from scripts.build_public_r7_b3 import DATASETS,REHEARSAL_SLUG,ENTRY_SLUG
        from scripts.build_public_r7_selection_b import no_oai,kernel_title
        self.assertTrue(no_oai(DATASETS));self.assertEqual(DATASETS[-1],'ozgung/knee-raptor64-dr-full-fold0-final-20261013/1')
        self.assertIn('dreaddevelopment/raptor-knee-widedense',DATASETS);self.assertFalse(any(d.startswith('nartaa/') for d in DATASETS))
        for slug in (REHEARSAL_SLUG,ENTRY_SLUG):self.assertLessEqual(len(kernel_title(slug)),50)

    def test_launch_overlays_dr_config_and_selects_modules(self):
        for mode,expected in (('rehearsal','knee.public_r7_b3_rehearsal'),('entry','knee.public_r7_b3_entry')):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                seen=[]
                class Done:
                    returncode=0
                    def poll(self):return 0
                    def wait(self,timeout=None):return 0
                def popen(argv,**kw):
                    c=json.loads(Path(argv[-1]).read_bytes());seen.append((argv[2],c['arm'],c['public_manifest']));(Path(c['out'])/c['arm']).mkdir();(Path(c['out'])/c['arm']/'receipt.json').write_text('{"board":%d}'%c['board']);return Done()
                cfg={'mode':mode,'assignment':[['a'],['b']],'boards':[{},{}],'public_manifest':'dread','arm_overlays':{'dr':{'public_manifest':'drfull'}}}
                with patch.object(B3.subprocess,'Popen',side_effect=popen):
                    B3.launch(cfg,Path(tmp),'dr');(Path(tmp)/'o').mkdir();B3.launch(cfg,Path(tmp)/'o','old')
                self.assertEqual(seen,[(expected,'dr','drfull')]*2+[(expected,'old','dread')]*2)

    @unittest.skipUnless(DREAD_SOURCE.is_file(),'pinned Dread source unavailable')
    def test_dr_profile_gate(self):
        from scripts.qualify_b3_drfull_cpu import drfull_manifest
        from knee.reference import audit_source
        text=DREAD_SOURCE.read_text();_,_,m=drfull_manifest(text);self.assertEqual(B3.dr_profile_gate(None,m)['windows'],42)
        _,_,dread=audit_source(text,profile='raptor',checkpoint_sha256=B0.DREAD_CHECKPOINT_SHA256,raptor_windows=62)
        for bad in (dread,{**m,'checkpoint_file':'raptor_ft_coatnet_v4_full.pt'},{**m,'recipe':{**m['recipe'],'windows':62}},{**m,'weight_dataset':'nartaa/x'}):
            with self.subTest(),self.assertRaises(ValueError):B3.dr_profile_gate(None,bad)


@unittest.skipUnless(HAS('numpy','pandas'),'numpy/pandas unavailable')
class BlendTests(unittest.TestCase):
    def arrays(self,n=60,seed=8):
        rng=np.random.default_rng(seed);return rng.random((n,12)),rng.random((n,12)),rng.random((n,12))

    def test_fixed_forty_ten_fifty_on_arm_ranks(self):
        o,d,s=self.arrays();z=np.zeros(len(o),bool);r=B3.recover(o,d,s,z,z,z)
        rank=lambda a:pd.DataFrame(a).rank(method='average',pct=True).to_numpy()
        self.assertTrue(np.array_equal(r['blend'],.4*rank(o)+.1*rank(d)+.5*rank(S.quantize9(s))))
        self.assertTrue(np.allclose(r['blend'],.5*(.8*rank(o)+.2*rank(d))+.5*rank(S.quantize9(s)),atol=1e-15))
        same=B3.recover(o,o,s,z,z,z)['blend'];self.assertTrue(np.allclose(same,RR.recover(o,s,z,z)['blend'],atol=1e-15))
        self.assertEqual((r['recipe'],r['weights']),('r7-dread-drfull-fixed40-10-50-capped-arm-mean-v1',B3.ARM_WEIGHTS))

    def test_union_cap_over_three_arms_and_mean_fill(self):
        o,d,s=self.arrays(n=100);z=np.zeros(100,bool);a,b,c=z.copy(),z.copy(),z.copy();a[1]=b[2]=c[3]=True
        r=B3.recover(o,d,s,a,b,c);self.assertEqual((r['cap'],r['union_recovered']),(3,3))
        self.assertTrue(np.array_equal(r['dr'][2],d[~b].mean(0)))
        c[4]=True
        with self.assertRaises(ValueError):B3.recover(o,d,s,a,b,c)
        for args in ((o,d,s*2,z,z,z),(o,d[:,:11],s,z,z,z),(o,d,s,z,np.zeros(100),z)):
            with self.subTest(),self.assertRaises(ValueError):B3.recover(*args)


def b3_receipt(tmp):
    r,binding,path=b2_receipt(tmp)
    binding.update(version=B3.VERSION,method=json.loads(json.dumps(B3.METHOD)),dr_profile=json.loads(json.dumps(B3.DR_PROFILE)))
    r.update(version=B3.VERSION,binding=binding,recovery={'weights':B3.ARM_WEIGHTS,'recipe':B3.RECOVERY_RECIPE_B3})
    for w in r['workers']['old']:w.update(arm='old',version=B3.VERSION)
    dr=json.loads(json.dumps(r['workers']['old']))
    for w in dr:
        w.update(arm='dr',state_before={'state':B3.DRFULL_STATE_SHA256},state_after={'state':B3.DRFULL_STATE_SHA256},constructor_grid_sha256=B3.DRFULL_GRID_SHA256,old_profile=json.loads(json.dumps(B3.DR_PROFILE)))
        for row in w['records']:row.pop('raptor62',None)
    r['workers']['dr']=dr;r['phases']['dr']=r['phases']['old'];r['sentinel_ids']['dr']=r['sentinel_ids']['old']
    return r,binding,path


@unittest.skipUnless(HAS('numpy','pandas'),'numpy/pandas unavailable')
class ContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.r,self.b,self.p=b3_receipt(self.tmp.name)

    def test_synthetic_receipt_passes(self):
        B3.require_rehearsal(self.r,self.b,self.p)

    def test_rejections(self):
        def dr_dread_state(x):x['workers']['dr'][0]['state_before']=x['workers']['dr'][0]['state_after']={'state':B0.DREAD_STATE_SHA256}
        def dr_dread_profile(x):x['workers']['dr'][1]['old_profile']=json.loads(json.dumps(B0.OLD_PROFILE))
        def dr_cross_board(x):x['workers']['dr'][1]['sentinels'][0]['probabilities'][0]+=2e-6
        def dr_missing(x):x['workers'].pop('dr')
        def dr_wrong_arm(x):x['workers']['dr'][0]['arm']='old'
        def weights(x):x['recovery']['weights']={'old':.5,'dr':0.,'s6':.5}
        def old_raptor62(x):
            for w in x['workers']['old']:
                for row in w['records']:
                    if 'raptor62' in row:row['raptor62'].update(within_atol=False,max_abs=.02)
        def r7_fallback(x):x['workers']['s6'][0]['compile']['active']=False
        def slow(x):x['projection']['conservative_seconds']=8.31*3600
        def dr_no_strict(x):
            for w in x['workers']['dr']:
                for row in w['records']+w['sentinels']:
                    if 'strict_source_comparison' in row:row['strict_source_comparison']='ineligible_partial_input'
        for mutate in (dr_dread_state,dr_dread_profile,dr_cross_board,dr_missing,dr_wrong_arm,weights,old_raptor62,r7_fallback,slow,dr_no_strict):
            with self.subTest(mutate=mutate.__name__):
                x=copy.deepcopy(self.r);mutate(x)
                with self.assertRaises((ValueError,KeyError)):B3.require_rehearsal(x,self.b,self.p)
        with self.assertRaises(ValueError):B3.require_rehearsal(self.r,{**self.b,'method':B2.METHOD},self.p)


BUILD_INPUTS=[OUT/'kaggle/public-s6-evaluation-20261011-reviewed',OUT/'analysis/s6-capped-recovery-20261012/protocol-private.json',EVIDENCE/'data/raw/test.csv',DREAD_SOURCE,
              OUT/'analysis/next-direction-20261013/kernels/heliosli-0957/0-957-latest-dataset-score-in-one-click.ipynb',OUT/'analysis/r7-recovery-20261013/r7-entry-v2-visible.json',
              OUT/'analysis/r7-recovery-20261013/cpu-qualification-b3-drfull.json',OUT/'kaggle/raptor62-rehearsal-result',OUT/'analysis/b0-dread-swap-20261012/cpu-qualification-b0.json']


@unittest.skipUnless(all(p.exists() for p in BUILD_INPUTS) and HAS('numpy','pandas'),'B3 build evidence unavailable')
class BuildTests(unittest.TestCase):
    def test_rehearsal_build(self):
        from scripts.build_public_r7_b3 import build,verify_build,DATASETS,REHEARSAL_SLUG
        a=json.loads((OUT/'analysis/s6-capped-recovery-20261012/parent-build-arguments.json').read_bytes())
        a={k:str(v).replace('~/Dev/rsna-knee-abnormality-detection',str(EVIDENCE)) for k,v in a.items() if k not in ('control_build','control_result','destination','slug')}
        root=scratch(self)
        a.update(destination=str(root/'b3'),slug=REHEARSAL_SLUG,heliosli_notebook=str(BUILD_INPUTS[4]),r7_asset_dir=str(OUT/'analysis/r7-recovery-20261013/r7-v1-assets'),
                 heliosli_output=str(OUT/'analysis/r7-recovery-20261013/heliosli-0957-output'),r7_reference=str(BUILD_INPUTS[5]),dread_source=str(DREAD_SOURCE),
                 raptor62_result=str(BUILD_INPUTS[7]),dread_cpu_receipt=str(BUILD_INPUTS[8]),drfull_cpu_receipt=str(BUILD_INPUTS[6]))
        r=build(**a);self.assertEqual(verify_build(root/'b3'),r)
        meta=json.loads((root/'b3/kernel-metadata.json').read_bytes());self.assertEqual(meta['dataset_sources'],DATASETS);self.assertNotIn('nartaa',json.dumps(meta))
        code=''.join(json.loads((root/'b3/notebook.ipynb').read_bytes())['cells'][0]['source'])
        value=next(ast.literal_eval(n.value) for n in ast.parse(code).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='encoded' for t in n.targets))
        pack=json.loads(lzma.decompress(base64.b64decode(value,validate=True)));cfg=pack['cfg']
        self.assertEqual(cfg['arm_overlays']['dr']['public_manifest']['checkpoint_sha256'],B3.DRFULL_CHECKPOINT_SHA256);self.assertEqual(cfg['arm_overlays']['dr']['raptor62_reference'],{})
        self.assertEqual(cfg['public_manifest']['checkpoint_sha256'],B0.DREAD_CHECKPOINT_SHA256);self.assertIn("mount('ozgung/knee-raptor64-dr-full-fold0-final-20261013'",code)
        self.assertIn('knee/public_r7_b3_rehearsal.py',pack['sources']);self.assertNotIn("mount('nartaa",code)
        with self.assertRaises(ValueError):build(**{**a,'destination':str(root/'entry'),'entry':True})


if __name__=='__main__':
    unittest.main()
