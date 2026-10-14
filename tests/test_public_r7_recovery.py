"""R7 entry: arm swap only, ported stackers, 0.5/0.5 blend, schema and unchanged recovery guards.

Synthetic fixtures throughout. Tests that read pinned evidence (downloaded R7 v1
assets, heliosli's public notebook and its visible-3 output, the qualified S6
build inputs) skip unless it is present under KNEE_EVIDENCE_ROOT (default: repo).
"""
import ast
import base64
import builtins
import copy
import csv
import dis
import hashlib
import importlib.util
import json
import lzma
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from knee import public_s6_recovery_entry as E
from knee import public_s6_recovery_runtime as R
from knee import public_r7_stackers as S
from knee.public_s6_qualification import LABELS,UID

HAS=lambda *m:all(importlib.util.find_spec(x) is not None for x in m)
if HAS('numpy','pandas'):
    import numpy as np
    from knee import public_r7_recovery_entry as R7
    from knee import public_r7_recovery as RR
    from knee import public_s6_recovery as FROZEN
PROJECT=Path(__file__).resolve().parents[1]
EVIDENCE=Path(os.environ.get('KNEE_EVIDENCE_ROOT',PROJECT))
OUT=EVIDENCE/'outputs'
R7_EVIDENCE=OUT/'analysis/r7-recovery-20261013'
HELIOSLI_NOTEBOOK=OUT/'analysis/next-direction-20261013/kernels/heliosli-0957/0-957-latest-dataset-score-in-one-click.ipynb'
DEFAULT_MODULES={'knee/public_s6_recovery_entry.py':'76217a402ae3bb4c8c9517c76875691f3565b84bf0b2e548295cd9f726a90be8',
                 'knee/public_s6_recovery_runtime.py':'d67ceb738930c85b19e1866dfeb0d524e015a68ff321e039f00105c590ca6d08',
                 'knee/public_s6_recovery_rehearsal.py':'8424d87b5656ccdf449c69c5ee06c09744116f185f086861b7feee8bd5864502',
                 'knee/public_s6_recovery.py':'2fbc950cdcc54dddd9b6722ced524943b784dc0ae3a0df185aad10146f6a9ce8',
                 'scripts/build_public_s6_recovery_entry.py':'d8915ef31345f452ca8a8ea0a2cbeb12ad717363d9e7b54b3cd714ef3ed3109c'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def global_loads(code):
    names=set()
    for ins in dis.get_instructions(code):
        if ins.opname in ('LOAD_GLOBAL','LOAD_NAME'):names.add(ins.argval)
    for const in code.co_consts:
        if hasattr(const,'co_code'):names|=global_loads(const)
    return names


def scratch(test):
    (PROJECT/'outputs').mkdir(exist_ok=True)
    tmp=tempfile.TemporaryDirectory(dir=PROJECT/'outputs',prefix='test-r7-');test.addCleanup(tmp.cleanup);return Path(tmp.name)


# ---- synthetic stacker fixtures (two categories per class, one binary for Baker's) ----
def spec():
    return {'categories':{l:['none','present'] for l in LABELS},'prior':{l:[.5,.5] for l in LABELS},'binaries':{'acute':["Baker's"]}}


def fluid_json(**change):
    cls={'Effusion':{'adopt':True,'volume_column':'eff_sag_p99_open','view_id':4,'b0':-3.9,'b_z':.32,'b_lv':.76,'b_miss':2.13},
         "Baker's":{'adopt':False,'volume_column':'bak_sag_p99_open','view_id':5,'b0':0.,'b_z':1.,'b_lv':0.,'b_miss':0.}}
    doc={'feature':{'combine':'logit','weights':[1.0],'lines':[{'run':'synthetic'}]},'classes':cls};doc.update(change)
    return json.dumps(doc).encode()


def state_json(layout,md5,**change):
    cls={"Baker's":{'adopt':True,'features':['z','state:none','state:present','bin:acute'],'coef':[.8,-1.5,2.5,.4],'b0':-.7,'dropped':[]},
         'ACL':{'adopt':False,'features':['z','state:none','state:present'],'coef':[1.,0.,0.],'b0':0.}}
    doc={'kind':'state_stacker_v1','feature':{'weights':[1.0],'lines':[{'run':'synthetic'}]},'layout':layout,'state_spec_md5':md5,'classes':cls};doc.update(change)
    return json.dumps(doc).encode()


def loaded():
    sp=spec();raw=json.dumps(sp).encode();md5=hashlib.md5(raw).hexdigest();layout=S.state_layout(sp,LABELS)
    return S.load_fluid(fluid_json(),LABELS),S.load_state(state_json(layout,md5),LABELS,layout,md5),layout,md5


@unittest.skipUnless(HAS('numpy','pandas'),'numpy/pandas unavailable')
class ArmSwapStructureTests(unittest.TestCase):
    def test_default_modules_byte_identical_and_pinned_in_builder(self):
        from scripts.build_public_r7_recovery_entry import DEFAULT_MODULE_SHA
        self.assertEqual(DEFAULT_MODULE_SHA,DEFAULT_MODULES)
        for name,pin in DEFAULT_MODULES.items():
            with self.subTest(name=name):self.assertEqual(sha(PROJECT/name),pin)

    def test_old_arm_merge_run_share_exact_default_bytecode(self):
        for name in ('old_worker','merge','run'):
            with self.subTest(name=name):
                self.assertIs(getattr(R7,name).__code__,getattr(E,name).__code__);self.assertIs(getattr(R7,name).__globals__,vars(R7))
                missing={n for n in global_loads(getattr(E,name).__code__) if not hasattr(builtins,n) and n not in vars(R7)}
                self.assertEqual(missing,set())
                swapped={'VERSION','METHOD','rehearsal_hook','launch','merge','require_rehearsal','recover'}
                self.assertEqual({n for n in global_loads(getattr(E,name).__code__) if n not in swapped and n in vars(R7) and n in vars(E) and vars(R7)[n] is not vars(E)[n]},set())

    def test_r7_worker_resolves_every_global(self):
        missing={n for n in global_loads(R7.r7_worker.__code__) if not hasattr(builtins,n) and n not in vars(R7)}
        self.assertEqual(missing,set())

    def test_method_differs_only_in_compact_slot_blend_and_recovery(self):
        self.assertNotEqual(R7.VERSION,E.VERSION)
        changed={k for k in set(R7.METHOD)|set(E.METHOD) if R7.METHOD.get(k)!=E.METHOD.get(k)}
        self.assertEqual(changed,{'s6','blend','raptor_per_finding','s6_quantization','slot','recovery_recipe'})
        self.assertEqual(R7.METHOD['old'],E.METHOD['old']);self.assertEqual(R7.METHOD['old'],'central320 K94 equal original/mirror probabilities')
        self.assertIs(R7.INPUT_POLICY,E.INPUT_POLICY);self.assertEqual(R7.METHOD['assignment'],E.METHOD['assignment'])
        recipe=R7.METHOD['s6']
        for key in ('backbone','size','crop','views','slices','phys_mm','eval_stride','folds','amp'):self.assertEqual(recipe[key],E.METHOD['s6'][key])
        self.assertEqual((recipe['chunk'],recipe['bundle_id'],recipe['dataset'],recipe['compile']['backend']=='inductor' and recipe['compile']['tol_prob']==1e-3,recipe['flip_tta']),(192,'8c7b666e27dc0e8f','heliosli/rsna-knee-compact-r7cnxfsss/1',True,False))
        self.assertEqual((recipe['fluid_stacker'],recipe['state_stacker']),(['Effusion'],["Baker's"]))

    def test_launch_selects_r7_worker_modules(self):
        for mode,module in (('rehearsal','knee.public_r7_recovery_rehearsal'),('entry','knee.public_r7_recovery_entry')):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                seen=[]
                class Done:
                    returncode=0
                    def poll(self):return 0
                    def wait(self,timeout=None):return 0
                def popen(argv,**kw):
                    seen.append(argv[2]);c=json.loads(Path(argv[-1]).read_bytes());(Path(c['out'])/c['arm']).mkdir();(Path(c['out'])/c['arm']/'receipt.json').write_text('{"board":%d}'%c['board']);return Done()
                cfg={'mode':mode,'assignment':[['a'],['b']],'boards':[{'uuid':'0'},{'uuid':'1'}]}
                with patch.object(R7.subprocess,'Popen',side_effect=popen):reports,_=R7.launch(cfg,Path(tmp),'s6')
                self.assertEqual(seen,[module,module]);self.assertEqual([r['board'] for r in reports],[0,1])
                self.assertTrue((PROJECT/(module.replace('.','/')+'.py')).is_file())

    def test_recovery_guards_are_the_default_runtime_objects(self):
        for name in ('install_source_guards','install_old_guards','cleanup_study','save_ledger','DataLocalFailure','FatalSourceError'):
            with self.subTest(name=name):self.assertIs(getattr(R7,name),getattr(R,name))
        for name in ('stage','verify_staged','assignments','probe_gate','projection','board_environment','native_availability_stress','mirror_stress','old_decode_gate','MAX_SECONDS','SENTINELS'):
            with self.subTest(name=name):self.assertIs(getattr(R7,name),getattr(E,name))
        self.assertIsNone(R7.rehearsal_hook)

    def test_rehearsal_module_uses_default_injections(self):
        tree=ast.parse((PROJECT/'knee/public_r7_recovery_rehearsal.py').read_text())
        imports={(n.module,a.name) for n in ast.walk(tree) if isinstance(n,ast.ImportFrom) for a in n.names}
        self.assertIn(('knee.public_s6_recovery_rehearsal','injection_hook'),imports)


@unittest.skipUnless(HAS('numpy'),'numpy unavailable')
class StackerTests(unittest.TestCase):
    def setUp(self):
        self.fluid,self.state,self.layout,self.md5=loaded()
        self.p=np.array([.1+.05*i for i in range(12)]);self.z=np.log(self.p/(1-self.p));self.stv=np.linspace(.05,.95,self.layout['width'])

    def test_layout_and_loaded_adopted_classes(self):
        self.assertEqual(self.layout['width'],25);self.assertEqual(self.layout['columns'][-1],{'binary':'acute','class':"Baker's",'column':24})
        self.assertEqual(list(self.fluid['classes']),['Effusion']);self.assertEqual(list(self.state['classes']),["Baker's"])
        self.assertEqual(self.state['classes']["Baker's"]['columns'],[None,18,19,24]);self.assertFalse(self.state['needs_volume'])
        self.assertEqual(S.check_pair(self.fluid,self.state),{'fluid':['Effusion'],'state':["Baker's"]})

    def oracle(self,record):
        v=None if record is None else record.get('eff_sag_p99_open')
        miss=v is None or not math.isfinite(v) or v<0;lv=0. if miss else math.log1p(v)
        ze=-3.9+.32*self.z[7]+.76*lv+2.13*(1. if miss else 0.)
        zb=-.7+.8*self.z[9]-1.5*self.stv[18]+2.5*self.stv[19]+.4*self.stv[24]
        return ze,zb

    def test_application_matches_formula_and_only_adopted_classes_change(self):
        for record in ({'eff_sag_p99_open':12.5},{'eff_sag_p99_open':0.},None,{'eff_sag_p99_open':None},{'eff_sag_p99_open':-1.},{'eff_sag_p99_open':float('nan')},{'status':'error: x'}):
            with self.subTest(record=record):
                out,receipt=S.apply(self.p,self.z,self.stv,record,self.fluid,self.state,LABELS);ze,zb=self.oracle(record)
                self.assertAlmostEqual(receipt['Effusion']['z_out'],ze,places=12);self.assertAlmostEqual(receipt["Baker's"]['z_out'],zb,places=12)
                self.assertEqual(out[7],1/(1+math.exp(-receipt['Effusion']['z_out'])) if receipt['Effusion']['z_out']>=0 else math.exp(receipt['Effusion']['z_out'])/(1+math.exp(receipt['Effusion']['z_out'])))
                self.assertAlmostEqual(out[9],1/(1+math.exp(-zb)),places=14)
                untouched=[j for j in range(12) if j not in (7,9)];self.assertTrue(np.array_equal(out[untouched],self.p[untouched]))
                self.assertEqual(receipt['Effusion']['volume_missing'],record is None or not (isinstance(record.get('eff_sag_p99_open'),float) and math.isfinite(record['eff_sag_p99_open']) and record['eff_sag_p99_open']>=0))
                self.assertEqual(receipt["Baker's"]['features'],[self.z[9],self.stv[18],self.stv[19],self.stv[24]])
                self.assertTrue(np.array_equal(self.p,np.array([.1+.05*i for i in range(12)])),'input probabilities must not be mutated')

    def test_stable_sigmoid_both_branches_and_nonfinite_keeps_model(self):
        self.assertEqual(S.sigmoid(0.),.5);self.assertAlmostEqual(S.sigmoid(-800.),0.);self.assertEqual(S.sigmoid(800.),1.)
        stv=self.stv.copy();stv[19]=float('nan')
        out,receipt=S.apply(self.p,self.z,stv,{'eff_sag_p99_open':3.},self.fluid,self.state,LABELS)
        self.assertEqual(out[9],self.p[9]);self.assertIsNone(receipt["Baker's"]['z_out']);self.assertEqual(receipt["Baker's"]['fallback'],'non-finite stacker logit')

    def test_rejected_fixtures(self):
        sp=spec();layout=self.layout;md5=self.md5
        bad_fluid=[fluid_json(feature={'combine':'prob','weights':[1.0],'lines':[{}]}),fluid_json(feature={'combine':'logit','weights':[.5,.5],'lines':[{},{}]}),
                   fluid_json(classes={'Effusion':{'adopt':True,'volume_column':'eff_sag_p99_open','view_id':5,'b0':0,'b_z':1,'b_lv':0,'b_miss':0}}),
                   fluid_json(classes={'Effusion':{'adopt':True,'volume_column':'eff_sag_p99_open','view_id':4,'b0':float('inf'),'b_z':1,'b_lv':0,'b_miss':0}}),
                   fluid_json(classes={'Unknown':{'adopt':True,'volume_column':'eff_sag_p99_open','view_id':4,'b0':0,'b_z':1,'b_lv':0,'b_miss':0}})]
        for raw in bad_fluid:
            with self.subTest(raw=raw[:60]),self.assertRaises(ValueError):S.load_fluid(raw,LABELS)
        other=json.loads(json.dumps(layout));other['columns'][0]['categories']=['x','y']
        bad_state=[(state_json(layout,md5,kind='other'),md5),(state_json(other,md5),md5),(state_json(layout,'0'*32),md5),
                   (state_json(layout,md5,classes={"Baker's":{'adopt':True,'features':['z','state:present'],'coef':[1.,1.],'b0':0.}}),md5),
                   (state_json(layout,md5,classes={"Baker's":{'adopt':True,'features':['z','state:none','state:present','bin:acute'],'coef':[1.,1.],'b0':0.}}),md5)]
        for raw,m in bad_state:
            with self.subTest(raw=raw[:60]),self.assertRaises(ValueError):S.load_state(raw,LABELS,layout,m)
        both=S.load_state(state_json(layout,md5,classes={'Effusion':{'adopt':True,'features':['z','state:none','state:present'],'coef':[1.,0.,0.],'b0':0.}}),LABELS,layout,md5)
        with self.assertRaises(ValueError):S.check_pair(self.fluid,both)
        with self.assertRaises(ValueError):S.check_pair(self.fluid,self.state,size=288)

    def test_dropped_features_and_volume_terms_follow_source_layout(self):
        raw=state_json(self.layout,self.md5,classes={"Baker's":{'adopt':True,'volume_column':'bak_sag_p99_open','dropped':['state:none'],'features':['z','state:present','bin:acute','lv','miss'],'coef':[1.,1.,1.,1.,1.],'b0':0.}})
        st=S.load_state(raw,LABELS,self.layout,self.md5);self.assertTrue(st['needs_volume'])
        self.assertEqual(st['classes']["Baker's"]['columns'],[None,19,24,None,None])
        _,rc=S.apply(self.p,self.z,self.stv,{'bak_sag_p99_open':2.},self.fluid,st,LABELS)
        self.assertEqual(rc["Baker's"]['features'],[self.z[9],self.stv[19],self.stv[24],math.log1p(2.),0.])

    def test_quantize9_source_precision(self):
        q=S.quantize9([[1.43506907123e-06,0.99999773512,1-1e-12]]);self.assertEqual(q.tolist(),[[1.43506907e-06,0.999997735,1.0]])

    @unittest.skipUnless((R7_EVIDENCE/'r7-v1-assets/manifest.json').is_file(),'pinned R7 v1 assets unavailable')
    def test_pinned_r7_v1_stackers(self):
        from scripts.build_public_r7_recovery_entry import r7_assets
        manifest,fluid,state=r7_assets(R7_EVIDENCE/'r7-v1-assets')
        self.assertEqual(manifest['bundle_id'],'8c7b666e27dc0e8f');self.assertEqual(list(fluid['classes']),['Effusion']);self.assertEqual(list(state['classes']),["Baker's"])
        self.assertEqual(state['classes']["Baker's"]['features'][0],'z');self.assertEqual(len(state['classes']["Baker's"]['coef']),7)
        self.assertEqual(fluid['classes']['Effusion']['volume_column'],'eff_sag_p99_open')
        for change in ({'compile':False},{'bundle_id':'209a32c8b50a6b1c'},{'flip_tta':True},{'trainer_md5':'0'*32},{'crop':384}):
            with self.subTest(change=change),self.assertRaises(ValueError):R7.r7_manifest_gate({**manifest,**change})


@unittest.skipUnless(HAS('numpy','pandas'),'numpy/pandas unavailable')
class BlendRecoveryTests(unittest.TestCase):
    def arrays(self,n=40,seed=3):
        rng=np.random.default_rng(seed);return rng.random((n,12)),rng.random((n,12))

    def test_parameterized_recovery_reproduces_frozen_policy_bit_for_bit(self):
        c,s=self.arrays();n=len(c)
        for cm,sm in ((np.zeros(n,bool),np.zeros(n,bool)),(np.eye(n,dtype=bool)[0],np.zeros(n,bool)),(np.zeros(n,bool),np.eye(n,dtype=bool)[5])):
            with self.subTest(missing=(cm.sum(),sm.sum())):
                a=FROZEN.recover(c,s,cm,sm);b=RR.recover_with(c,s,cm,sm,weights=(.7,.3),quantizer=FROZEN.quantize,recipe=FROZEN.RECIPE)
                self.assertEqual(set(a),set(b))
                for k in a:
                    if isinstance(a[k],np.ndarray):self.assertTrue(np.array_equal(a[k],b[k]),k)
                    else:self.assertEqual(a[k],b[k],k)

    def test_r7_blend_is_fixed_half_half_ranks_with_nine_digit_r7(self):
        import pandas as pd
        c,s=self.arrays();n=len(c);r=RR.recover(c,s,np.zeros(n,bool),np.zeros(n,bool))
        rank=lambda a:pd.DataFrame(a).rank(method='average',pct=True).to_numpy()
        self.assertTrue(np.array_equal(r['blend'],.5*rank(c)+.5*rank(S.quantize9(s))))
        self.assertEqual((r['recipe'],R7.METHOD['blend'],R7.METHOD['raptor_per_finding']),('r7-fixed50-50-capped-arm-mean-v1',[.5,.5],{'Medial OA':.5,'Lateral OA':.5,'PF OA':.5}))
        self.assertIs(R7.recover,RR.recover)

    def test_nine_digit_quantization_preserves_source_ties_and_order(self):
        c=np.full((4,12),.5);s=np.full((4,12),.5);s[:,0]=[1-1e-11,1-2e-11,1e-6,1.43506907e-06]
        r=RR.recover(c,s,np.zeros(4,bool),np.zeros(4,bool))
        self.assertEqual(r['s6'][:,0].tolist(),[1.,1.,1e-6,1.43506907e-06]);self.assertEqual(r['s6_rank'][:,0].tolist(),[.875,.875,.25,.5])

    def test_cap_masks_and_values_unchanged(self):
        c,s=self.arrays(n=100);cap=np.zeros(100,bool);cap[:3]=True;over=np.zeros(100,bool);over[:4]=True
        self.assertEqual(RR.recover(c,s,cap,np.zeros(100,bool))['union_recovered'],3)
        for args in ((c,s,over,np.zeros(100,bool)),(c,s,np.zeros(100,bool),over),(c,s,np.zeros(100),np.zeros(100,bool)),
                     (np.where(np.eye(100,12,dtype=bool),np.nan,c),s,np.zeros(100,bool),np.zeros(100,bool)),(c,s*2,np.zeros(100,bool),np.zeros(100,bool)),(c[:,:11],s[:,:11],np.zeros(100,bool),np.zeros(100,bool))):
            with self.subTest(),self.assertRaises(ValueError):RR.recover(*args)
        with self.assertRaises(ValueError):RR.recover(c,s,np.zeros(100,bool),np.zeros(100,bool),fatal_errors=['x'])
        with self.assertRaises(ValueError):RR.recover_with(c,s,np.zeros(100,bool),np.zeros(100,bool),weights=(.6,.6),quantizer=S.quantize9,recipe='x')

    def test_submission_schema_order_and_finite_range(self):
        from knee.public_fullfield_entry import write_csv
        ids=['1.2.3','1.2.4','1.2.5'];c=np.random.default_rng(0).random((3,12));r=RR.recover(c,c,np.zeros(3,bool),np.zeros(3,bool))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'submission.csv';write_csv(path,[UID]+LABELS,ids,r['blend'])
            rows=list(csv.reader(path.read_text().splitlines()));self.assertEqual(rows[0],[UID]+LABELS);self.assertEqual([x[0] for x in rows[1:]],ids)
            values=np.array([[float(v) for v in x[1:]] for x in rows[1:]]);self.assertTrue(np.isfinite(values).all() and (values>=0).all() and (values<=1).all())
            for bad in (np.full((3,12),np.nan),np.full((3,12),1.5),r['blend'][:2]):
                with self.subTest(),self.assertRaises(ValueError):write_csv(Path(tmp)/'bad.csv',[UID]+LABELS,ids,bad)

    @unittest.skipUnless((R7_EVIDENCE/'heliosli-0957-output/submission.csv').is_file(),'heliosli visible output unavailable')
    def test_heliosli_published_visible_blend_reproduced(self):
        read=lambda n:[[float(v) for v in r[1:]] for r in list(csv.reader((R7_EVIDENCE/'heliosli-0957-output'/n).read_text().splitlines()))[1:]]
        raw,ours,sub=map(np.array,(read('raptor_raw.csv'),read('ours_sub.csv'),read('submission.csv')))
        blend=RR.recover(raw,ours,np.zeros(3,bool),np.zeros(3,bool))['blend']
        self.assertLessEqual(float(np.max(np.abs(blend-sub))),1e-7);self.assertTrue(np.array_equal(blend.astype(np.float32),sub.astype(np.float32)))


def passing_rehearsal(tmp):
    """Synthetic receipt satisfying the R7 contract; mutated by rejection tests."""
    fluid,state,layout,md5=loaded()
    s6ids=[f's6-{i}' for i in range(32)];old=[f'old-{i}' for i in range(4)];visible=['v0','v1','v2'];s6sent=[s6ids[i] for i in E.SENTINELS]
    p=np.array([.25+.01*i for i in range(12)]);logits=np.log(p/(1-p)).astype(np.float32);p_model=(1/(1+np.exp(-logits.astype(np.float64)))).tolist()
    z=np.asarray(p_model);zf=np.log(z/(1-z));stv=np.linspace(.1,.9,layout['width'])
    out,stack=S.apply(np.asarray(p_model),zf,stv,{'eff_sag_p99_open':4.},fluid,state,LABELS)
    for k in stack:stack[k]['z_feature']=float(zf[LABELS.index(k)])
    env={'torch':'2.10.0+cu128','packages':{'timm':'1.0.26'}}
    binding={'version':R7.VERSION,'method':R7.METHOD,'input_policy':R7.INPUT_POLICY,'environment':env,
             'qualified_state':{'old_state':'old','old_grid':'grid','s6_state':None},
             'sentinel_inputs':{'old':{s:{'volume_sha256':'v'} for s in old},'s6':{s:{'render_windows':'x'} for s in s6sent}},
             'sentinel_probabilities':{'old':{s:p.tolist() for s in old},'s6':dict.fromkeys(s6sent)},
             's6_probe_ids':s6ids,'old_historical_ids':old,'r7':{'recipe':R7.R7_RECIPE,'fluid':fluid,'state':state,'visible_atol':5e-3}}
    path=Path(tmp)/'visible.csv'
    with path.open('w',newline='') as h:w=csv.writer(h);w.writerow([UID]+LABELS);w.writerows([[v]+p.tolist() for v in visible])
    ids=s6ids+old+visible
    row7=lambda s:{'id':s,'fluid':{'status':'ok','eff_sag_p99_open':4.},'model_probabilities':p_model,'probabilities':out.tolist(),'stackers':copy.deepcopy(stack),
                   **({'pilot_input_parity':True} if s in s6ids else {}),**({'heliosli_reference':{'max_abs':1e-3,'atol':5e-3,'within_atol':True}} if s in visible else {})}
    def worker(arm,board):
        x={'status':'PASS','board':board,'assigned':ids[board::2],'completed':len(ids[board::2]),'peak_cuda_bytes':1,'peak_rss_bytes':1,'zero_failed':True,'zero_imputed':True,'fp32_recoveries':0,
           'environment':{**env,'physical_uuid':f'GPU-{board}','cuda_device_count':1},'recovery_injection':{'status':'PASS','failure_status':'data_local_failure','injected_rows_genuine':False,'state_healthy':True}}
        if arm=='old':
            x.update(state_before={'state':'old'},state_after={'state':'old'},constructor_grid_sha256='grid',
                     sentinels=[{'id':s,'input':{'volume_sha256':'v'},'probabilities':p.tolist(),'reloaded_probabilities':p.tolist(),'strict_reload':True} for s in old])
            x['records']=[{'id':s,**({'source_reference_parity':True,'strict_reload':True,'reload_max_abs':0.} if s in s6ids else {})} for s in ids[board::2]]
            if board==0:
                from knee.mirror_input import STRESS_CASES
                x.update(partial_input_stress={'status':'PASS','cases':dict.fromkeys(STRESS_CASES,'PASS'),'case_count':17,'uses_raw_data':False,'uses_labels':False},
                         guarded_partial_input_stress={'status':'PASS','guarded':True,'case_count':10,'cases':dict.fromkeys(['healthy','missing_iop','missing_ipp','missing_spacing','empty_series','missing_series','bad_header','bad_pixel','black_frame','short_series'],'PASS')})
        else:
            x.update(slot='r7',r7_manifest={'bundle_id':R7.R7_BUNDLE_ID},fluid_selfcheck={'status':'PASS'},fluid_guard={'site':'compact_infer.py:fluid_d2','unknown':'fatal'},compile={'requested':True,'active':True,'fallback':None,'runtime_fallback_study':None,'decision':'compiled == eager within 0.001 (max |dp| 0.0002)'},
                     stackers={'adopted':{'fluid':['Effusion'],'state':["Baker's"]},'fluid_sha256':R7.R7_ASSET_SHA['fluid_stacker.json'],'state_sha256':R7.R7_ASSET_SHA['state_stacker.json']},
                     state_before='r7state',state_after='r7state',records=[row7(s) for s in ids[board::2]],
                     sentinels=[{**row7(s),'input':{'render_windows':'x'},'reloaded_probabilities':out.tolist(),'reloaded_model_probabilities':p_model,'strict_reload':True,'logits':[logits.tolist()]*4,'reloaded_logits':[logits.tolist()]*4,'pilot_input_parity':True} for s in s6sent])
            if board==0:
                x.update(native_availability_stress={'status':'PASS','case_count':2,'uses_raw_data':False,'uses_labels':False,'cases':{k:{'status':'PASS','usable_series':1,'slices':4} for k in ('absent','empty')}},
                         guarded_native_availability_stress={'status':'PASS','guarded':True,'case_count':2,'cases':{'absent':{},'empty':{}}})
        return x
    phase={'wall_seconds':1.,'peaks':{'host_rss_bytes':1,'output_disk_bytes':1},'minimum_disk_free_bytes':1}
    r={'status':'PASS','version':R7.VERSION,'mode':'rehearsal','binding':binding,'completed':39,'completed_visible':3,'completed_probes':36,'completed_s6_probes':32,'completed_old_historical':4,
       'all_input_state_probability_gates':True,'zero_failed':True,'zero_imputed':True,'boards':[{'uuid':'GPU-0','name':'Tesla T4'},{'uuid':'GPU-1','name':'Tesla T4'}],
       'projection':{'conservative_seconds':3600.},'elapsed_seconds':1800.,'csv_sha256':sha(path),'visible_ids':visible,'phases':{'old':phase,'s6':phase},
       'workers':{arm:[worker(arm,0),worker(arm,1)] for arm in ('old','s6')},'sentinel_ids':{'old':old,'s6':s6sent}}
    return json.loads(json.dumps(r)),json.loads(json.dumps(binding)),path


@unittest.skipUnless(HAS('numpy','pandas'),'numpy/pandas unavailable')
class R7RehearsalContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.r,self.binding,self.csv=passing_rehearsal(self.tmp.name)

    def test_synthetic_receipt_passes(self):
        R7.require_rehearsal(self.r,self.binding,self.csv)

    def test_rejections(self):
        def visible_out_of_atol(r):
            for w in r['workers']['s6']:
                for row in w['records']:
                    if 'heliosli_reference' in row:row['heliosli_reference'].update(max_abs=.02,within_atol=False)
        def no_pilot_input_parity(r):r['workers']['s6'][0]['records'][0].pop('pilot_input_parity')
        def cross_board_state(r):r['workers']['s6'][1]['state_before']=r['workers']['s6'][1]['state_after']='other'
        def tampered_stacker(r):r['workers']['s6'][0]['records'][0]['probabilities'][9]=.5
        def wrong_stacker_feature(r):r['workers']['s6'][1]['records'][0]['stackers']["Baker's"]['features'][1]+=.1
        def fluid_missing_flag(r):r['workers']['s6'][0]['records'][0]['stackers']['Effusion']['volume_missing']=True
        def r7_reload_drift(r):r['workers']['s6'][0]['sentinels'][0]['reloaded_probabilities'][0]+=1e-3
        def default_version(r):r['version']=E.VERSION
        def wrong_slot(r):r['workers']['s6'][0]['slot']='s6'
        def no_fluid_selfcheck(r):r['workers']['s6'][1]['fluid_selfcheck']={'status':'FAIL'}
        def old_sentinel_drift(r):r['workers']['old'][0]['sentinels'][0]['probabilities'][0]+=1e-3
        def projection_over_nine_hours(r):r['projection']['conservative_seconds']=9*3600.
        def fp32_rescue(r):r['workers']['s6'][0]['fp32_recoveries']=1
        def missing_visible(r):r['completed_visible']=2
        def no_compile_receipt(r):r['workers']['s6'][1].pop('compile')
        def no_fluid_boundary(r):r['workers']['s6'][0].pop('fluid_guard')
        def fluid_error_status(r):r['workers']['s6'][1]['records'][0]['fluid']['status']="error: ValueError('x')"
        def compile_fell_back(r):r['workers']['s6'][0]['compile'].update(active=False,fallback='compile check failed: TypeError')
        def runtime_fallback(r):r['workers']['s6'][1]['compile']['runtime_fallback_study']='v1'
        def projection_over_helios_horizon(r):r['projection']['conservative_seconds']=8.31*3600
        for mutate in (visible_out_of_atol,no_pilot_input_parity,cross_board_state,tampered_stacker,wrong_stacker_feature,fluid_missing_flag,r7_reload_drift,default_version,wrong_slot,
                       no_fluid_selfcheck,old_sentinel_drift,projection_over_nine_hours,fp32_rescue,missing_visible,no_compile_receipt,no_fluid_boundary,fluid_error_status,compile_fell_back,runtime_fallback,projection_over_helios_horizon):
            with self.subTest(mutate=mutate.__name__):
                r=copy.deepcopy(self.r);mutate(r)
                with self.assertRaises(ValueError):R7.require_rehearsal(r,self.binding,self.csv)
        with self.assertRaises(ValueError):R7.require_rehearsal(self.r,{**self.binding,'method':E.METHOD},self.csv)

    def test_cross_board_tolerance_is_arm_specific(self):
        self.assertEqual(R7.CROSS_BOARD_TOLERANCE,{'old':1e-6,'s6':1e-3})
        def shift(arm,delta):
            r=copy.deepcopy(self.r);s=r['workers'][arm][1]['sentinels'][0]
            if arm=='old':
                s['probabilities'][0]+=delta;s['reloaded_probabilities'][0]+=delta;return r
            s['model_probabilities'][0]+=delta;s['reloaded_model_probabilities']=list(s['model_probabilities'])
            z=np.log(np.asarray(s['model_probabilities'])/(1-np.asarray(s['model_probabilities']))).astype(np.float32).tolist();s['logits']=[z]*4;s['reloaded_logits']=[z]*4
            s['probabilities'][0]+=delta;s['reloaded_probabilities']=list(s['probabilities']);return r
        R7.require_rehearsal(shift('s6',5e-4),self.binding,self.csv)
        for arm,delta in (('s6',1.1e-3),('old',2e-6),('old',1e-5)):
            with self.subTest(arm=arm,delta=delta),self.assertRaises(ValueError):R7.require_rehearsal(shift(arm,delta),self.binding,self.csv)
        r=copy.deepcopy(self.r);r['workers']['s6'][0]['sentinels'][0]['reloaded_probabilities'][0]+=2e-6
        with self.assertRaises(ValueError):R7.require_rehearsal(r,self.binding,self.csv)

    def test_visible_blend_rank_difference(self):
        from scripts.build_public_r7_recovery_entry import visible_rank_difference
        ref={v:[float(x) for x in np.float32([.25+.01*i for i in range(12)])] for v in self.r['visible_ids']}
        self.assertEqual(visible_rank_difference(self.csv,ref,self.r['visible_ids']),0.)
        ref['v1'][3]+=1/6
        self.assertGreater(visible_rank_difference(self.csv,ref,self.r['visible_ids']),.16)
        with open(self.csv,'a') as h:h.write('extra,'+','.join(['.5']*12)+'\n')
        with self.assertRaises(ValueError):R7.require_rehearsal(self.r,self.binding,self.csv)


@unittest.skipUnless(HAS('numpy','pandas'),'numpy/pandas unavailable')
class CompileReceiptTests(unittest.TestCase):
    def test_check_maxima_share_key_names_without_collision(self):
        gw={'synthetic_dlogit':.0045,'synthetic_dprob':.00056,'study_dlogit':.0082,'study_dprob':.0019}
        r=R7.compile_receipt({'requested':True,'active':True,'fallback':None},range(32,193,32),'compiled == eager within 0.001 (max |dp| 0.000232)',.000232,None,None,gw)
        self.assertEqual((r['study_dprob'],r['max_abs_dprob'],r['max_abs_dlogit']),(.000232,.0019,.0082));self.assertEqual(r['per_group'][0]['study_dprob'],.0019)
        self.assertEqual(r['shapes'],[32,64,96,128,160,192]);self.assertTrue(r['active']);self.assertIsNone(r['fallback']);json.dumps(r,allow_nan=False)
        r=R7.compile_receipt({},[32],'tie-break',.002,.0011,.0009,{});self.assertEqual((r['fp32_err_compiled'],r['fp32_err_eager'],r['max_abs_dprob']),(.0011,.0009,0.))

    def test_compile_encoders_records_without_exception_on_stand_in_models(self):
        """CPU stand-in for the receipt path of compile_encoders: real torch modules, eager backend
        substituted for inductor and cpu for cuda; the check must finish with compile active."""
        import torch
        class Enc(torch.nn.Module):
            def __init__(self):super().__init__();self.lin=torch.nn.Linear(3,4)
            def forward(self,x):return self.lin(x.mean((-1,-2)))
        class Net(torch.nn.Module):
            def __init__(self):super().__init__();self.enc=Enc()
            def forward(self,img,meta,mask,chunk=0,return_aux=False):
                x=img[mask].float().div(255.);f=torch.cat([self.enc(c) for c in x.split(chunk or len(x))]);return f.mean(0,keepdim=True)[:, :1].repeat(1,12)
        models=[Net().eval(),Net().eval()];item={'images':torch.randint(0,256,(40,3,384,384),dtype=torch.uint8),'meta':torch.zeros((40,9))}
        real=torch.Tensor.to
        def to(self,*a,**k):
            a=tuple('cpu' if x=='cuda' else x for x in a);k={kk:('cpu' if v=='cuda' else v) for kk,v in k.items()};return real(self,*a,**k)
        compile_=torch.nn.Module.compile
        autocast_=torch.autocast
        with patch.object(torch.Tensor,'to',to),patch.object(torch.nn.Module,'compile',lambda self,**kw:compile_(self,backend='eager',dynamic=False)),\
             patch.object(torch,'ones',lambda *a,**k:torch.full(a[0],True,dtype=k.get('dtype')) if k.get('device')=='cuda' else torch.full(a[0],1.,dtype=k.get('dtype'))),\
             patch.object(torch,'zeros',lambda *a,**k:torch.full(a[0],0.,dtype=k.get('dtype'))),patch.object(torch.cuda,'synchronize',lambda:None),\
             patch.object(torch,'autocast',lambda *a,**k:autocast_('cpu',dtype=torch.bfloat16)):
            receipt,set_eager=R7.compile_encoders(models,item,torch)
        self.assertTrue(receipt['active'],receipt.get('fallback'));self.assertIsNone(receipt['fallback']);self.assertTrue(receipt['decision'].startswith('compiled'))
        self.assertIn('study_dprob',receipt['per_group'][0]);self.assertEqual(sorted(models[0].state_dict()),['enc.lin.bias','enc.lin.weight'])
        set_eager('test');self.assertIsNone(models[0].enc._compiled_call_impl);self.assertFalse(receipt['active'])


@unittest.skipUnless(HAS('numpy','pandas'),'numpy/pandas unavailable')
class ProductionContractTests(unittest.TestCase):
    def test_fluid_status_gate(self):
        for status in ('ok','no fluid-sensitive sagittal series','no region view for eff_sag_p99_open','no region view for eff_sag_p99_open,bak_sag_p99_open'):
            with self.subTest(status=status):self.assertEqual(R7.fluid_gate({'status':status,'eff_sag_p99_open':None}),status)
        for record in ({'status':"error: ImportError('cv2')"},{'status':'error: MemoryError()'},{'status':'surprise'},{'status':'no region view for other_column'},None,{'eff_sag_p99_open':3.}):
            with self.subTest(record=record),self.assertRaises(ValueError):R7.fluid_gate(record)

    SYNTHETIC_FLUID='''
def fluid_d2(px, meta, blocks):
    rec = {"status": "ok", "eff_sag_p99_open": None, "bak_sag_p99_open": None}
    try:
        hook(px)
        if blocks == "none":
            rec["status"] = "no fluid-sensitive sagittal series"
            return rec
        rec["eff_sag_p99_open"] = 1.5
    except Exception as exc:
        rec.update(status="error: " + repr(exc)[:200], eff_sag_p99_open=None, bak_sag_p99_open=None)
    return rec
'''

    def guarded(self,error):
        def hook(px):
            if error is not None:raise error
        ns={'hook':hook};f,sha=R7.guard_fluid_source(self.SYNTHETIC_FLUID,ns);self.assertIs(ns['fluid_d2'],f);return f

    def test_fluid_boundary_allowlist(self):
        geometry=type('_S6GeometryData124',(ValueError,),{});decode=type('_S6DecodedPixelData',(ValueError,),{})
        with patch.object(R,'REGISTERED_DATA_CLASSES',{geometry,decode}),patch.object(R,'REGISTERED_DICOM_CLASS',type('InvalidDicomError',(Exception,),{})):
            self.assertEqual(self.guarded(None)(1,2,'x'),{'status':'ok','eff_sag_p99_open':1.5,'bak_sag_p99_open':None})
            self.assertEqual(R7.fluid_gate(self.guarded(None)(1,2,'none')),'no fluid-sensitive sagittal series')
            for error in (geometry('bad slab'),decode('bad pixels'),R.REGISTERED_DICOM_CLASS('not dicom'),R.AuditedPixelBufferError('short buffer'),R.DataLocalFailure('x')):
                with self.subTest(error=type(error).__name__),self.assertRaises(R.DataLocalFailure) as caught:self.guarded(error)(1,2,'x')
                self.assertEqual((str(caught.exception),caught.exception.site),('r7_fluid_measurement_error','compact_infer.py:fluid_d2'))
            wrapped=geometry('wrapped');wrapped.__cause__=MemoryError()
            for error in (RuntimeError('cuda'),MemoryError(),ValueError('cv2 shape'),TypeError('bug'),KeyError(4),ImportError('cv2'),OSError('disk'),Exception('cv2.error'),wrapped):
                with self.subTest(error=repr(error)),self.assertRaises(R.FatalSourceError):self.guarded(error)(1,2,'x')
        with self.assertRaises(ValueError):R7.guard_fluid_source(self.SYNTHETIC_FLUID.replace('except Exception as exc','except ValueError as exc'),{})

    @unittest.skipUnless(HELIOSLI_NOTEBOOK.is_file(),'heliosli notebook source unavailable')
    def test_pinned_r7_fluid_catch_site_shape(self):
        from scripts.build_public_r7_recovery_entry import heliosli_sources
        src=heliosli_sources(HELIOSLI_NOTEBOOK)['SOURCES']['compact_infer.py']
        node=next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name=='fluid_d2')
        handlers=[h for h in ast.walk(node) if isinstance(h,ast.ExceptHandler)]
        self.assertEqual([(h.type.id,h.name) for h in handlers],[('Exception','exc')])
        self.assertIn('fluid_d2(px, meta, item["blocks"])',src)  # CStudies resolves the module global at call time

    def test_fluid_failures_count_toward_union_cap(self):
        n=100;c,s=np.random.default_rng(1).random((2,n,12));old=np.zeros(n,bool)
        fluid=np.zeros(n,bool);fluid[[3,50,99]]=True
        r=RR.recover(c,s,old,fluid);self.assertEqual((r['cap'],r['union_recovered']),(3,3));self.assertEqual(r['statuses']['s6'].count('recovered'),3)
        fluid[7]=True
        with self.assertRaises(ValueError):RR.recover(c,s,old,fluid)
        old[3]=True;fluid[7]=False;self.assertEqual(RR.recover(c,s,old,fluid)['union_recovered'],3)
        src=inspect_source(R7.r7_worker)
        self.assertIn("GLOBAL_UNION_CAP_ALREADY_EXCEEDED",src);self.assertLess(src.index('fluid_gate(item.get('),src.index('compile_encoders(models,item,torch)'))

    def test_production_requires_compiled_runtime(self):
        ok={'requested':True,'active':True,'fallback':None,'runtime_fallback_study':None,'decision':'compiled == eager within 0.001 (max |dp| 0.0002)'}
        self.assertTrue(R7.require_compiled(ok))
        for change in ({'active':False,'fallback':"compile check failed: TypeError('x')"},{'fallback':'compiled forward failed'},{'runtime_fallback_study':'1.2'},
                       {'decision':None},{'requested':False},{'active':False,'fallback':'compile check: compiled vs eager 0.002 > tol; fp32 error compiled 0.003 eager 0.001'}):
            with self.subTest(change=change),self.assertRaises(ValueError):R7.require_compiled({**ok,**change})
        src=inspect_source(R7.r7_worker)
        self.assertIn('require_compiled(compiled)',src);self.assertNotIn('set_eager(',src);self.assertNotIn('compiled forward failed',src)
        self.assertLess(src.index('require_compiled(compiled)'),src.index('return_aux=True'))


def inspect_source(function):
    import inspect
    return inspect.getsource(function)


class BuilderUnitTests(unittest.TestCase):
    @unittest.skipUnless(HAS('numpy','pandas'),'numpy/pandas unavailable')
    def test_kernel_slugs_at_most_fifty_and_protected_kernels_rejected(self):
        from scripts.build_public_r7_recovery_entry import kernel_title,REHEARSAL_SLUG,ENTRY_SLUG,PROTECTED_SLUGS,DATASETS
        for slug in (REHEARSAL_SLUG,ENTRY_SLUG):
            with self.subTest(slug=slug):self.assertLessEqual(len(slug),50);self.assertLessEqual(len(kernel_title(slug)),50)
        for slug in (*PROTECTED_SLUGS,'x'*51,'Upper-Case','a'):
            with self.subTest(slug=slug),self.assertRaises(ValueError):kernel_title(slug)
        self.assertEqual(DATASETS[0],'heliosli/rsna-knee-compact-r7cnxfsss/1');self.assertNotIn('heliosli/rsna-knee-compact-v1',DATASETS)

    @unittest.skipUnless(HELIOSLI_NOTEBOOK.is_file() and HAS('numpy','pandas'),'heliosli notebook source unavailable')
    def test_heliosli_literals_parsed_without_execution(self):
        from scripts.build_public_r7_recovery_entry import heliosli_sources
        v=heliosli_sources(HELIOSLI_NOTEBOOK)
        self.assertEqual((v['BUNDLE_ID'],v['W_RAPTOR'],v['W_OURS'],v['DEADLINE_HOURS']),('8c7b666e27dc0e8f',.5,.5,8.3))
        for name,pin in R7.R7_SOURCE_SHA.items():self.assertEqual(hashlib.sha256(v['SOURCES'][name].encode()).hexdigest(),pin)


BUILD_INPUTS=[OUT/'kaggle/public-s6-evaluation-20261011-reviewed',OUT/'analysis/s6-capped-recovery-20261012/protocol-private.json',EVIDENCE/'data/raw/test.csv',HELIOSLI_NOTEBOOK,R7_EVIDENCE/'r7-v1-assets/manifest.json',R7_EVIDENCE/'heliosli-0957-output/ours_sub.csv']


@unittest.skipUnless(all(p.exists() for p in BUILD_INPUTS) and HAS('numpy','pandas'),'qualified S6/R7 build evidence unavailable')
class R7BuildTests(unittest.TestCase):
    def args(self,destination):
        args=json.loads((OUT/'analysis/s6-capped-recovery-20261012/parent-build-arguments.json').read_bytes())
        args={k:str(Path(v)).replace('/Users/ozgun/Dev/rsna-knee-abnormality-detection',str(EVIDENCE)) for k,v in args.items()}
        args.update(destination=str(destination),slug='knee-public-r7-recovery-rehearsal-20261013',heliosli_notebook=str(HELIOSLI_NOTEBOOK),
                    r7_asset_dir=str(R7_EVIDENCE/'r7-v1-assets'),heliosli_output=str(R7_EVIDENCE/'heliosli-0957-output'))
        return args

    def test_rehearsal_build_contract_and_entry_gate(self):
        from scripts.build_public_r7_recovery_entry import build,verify_build,DATASETS
        root=scratch(self);r=build(**self.args(root/'r7'));self.assertEqual(verify_build(root/'r7'),r)
        meta=json.loads((root/'r7/kernel-metadata.json').read_bytes())
        self.assertEqual(meta['dataset_sources'],DATASETS);self.assertTrue(meta['is_private'] and meta['enable_gpu'] and not meta['enable_internet'] and meta['machine_shape']=='NvidiaTeslaT4')
        nb=json.loads((root/'r7/notebook.ipynb').read_bytes());code=''.join(nb['cells'][0]['source'])
        value=next(ast.literal_eval(n.value) for n in ast.parse(code).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='encoded' for t in n.targets))
        pack=json.loads(lzma.decompress(base64.b64decode(value,validate=True)));cfg=pack['cfg']
        for name,pin in R7.R7_SOURCE_SHA.items():self.assertEqual(hashlib.sha256(pack['sources'][name].encode()).hexdigest(),pin)
        for name,pin in DEFAULT_MODULES.items():
            if name.startswith('knee/public_s6_recovery_') and name!='knee/public_s6_recovery_rehearsal.py':self.assertEqual(hashlib.sha256(pack['sources'][name].encode()).hexdigest(),pin)
        self.assertEqual(cfg['public_manifest']['profile'],'mirror');self.assertEqual(cfg['manifest']['bundle_id'],'8c7b666e27dc0e8f');self.assertIsNone(cfg['pilot_state_sha256'])
        self.assertEqual(cfg['guard_source_sha256']['compact_infer.py'],R7.R7_SOURCE_SHA['compact_infer.py'])
        self.assertEqual(len(cfg['probes']),36);self.assertTrue(all(x.get('s6',{}).get('probabilities') is None for x in cfg['probes'].values()))
        self.assertEqual(sorted(cfg['heliosli_reference']),['r7','raptor_raw','submission']);self.assertIn("from knee.public_r7_recovery_entry import run",code)
        self.assertIn("'heliosli/rsna-knee-compact-r7cnxfsss'",code)
        with self.assertRaises(ValueError):build(**{**self.args(root/'entry'),'entry':True})


if __name__=='__main__':
    unittest.main()
