"""B0 (Dread raptor62 old arm) regressions and default-entry byte identity.

Evidence-dependent tests read ignored outputs/ under KNEE_EVIDENCE_ROOT (default:
project root) and skip explicitly when absent. Real-checkpoint tests additionally
need torch, timm, cv2 and pydicom.
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
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from knee import public_s6_recovery_entry as E
from knee import public_s6_recovery_b0 as B0
from knee import public_s6_recovery_runtime as R
from knee.public_s6_qualification import LABELS,UID

PROJECT=Path(__file__).resolve().parents[1]
EVIDENCE=Path(os.environ.get('KNEE_EVIDENCE_ROOT',PROJECT))
OUT=EVIDENCE/'outputs'
DREAD_SOURCE=OUT/'research/2026-10-08/raptor-audit/knee-mri-twelve-findings-from-a-single-model.py'
DREAD_CHECKPOINT=OUT/'research/2026-10-08/raptor-audit/raptor_ft_coatnet_v4_full.pt'
RAPTOR62=OUT/'kaggle/raptor62-rehearsal-result'
S6=OUT/'analysis/s6-capped-recovery-20261012'
DEFAULT_REHEARSAL=OUT/'kaggle/public-s6-recovery-rehearsal-20261012-reviewed-v1'
DEFAULT_RESULT=OUT/'kaggle/public-s6-recovery-rehearsal-20261012-result-v1/s6-recovery-entry'
# Qualified default pins: 0.954 entry v1 and its rehearsal v1 (STATUS.md).
DEFAULT_REHEARSAL_FILES={'notebook.ipynb':'78b3be843d0e6a8b7f5827449ce43fa4fcf53bbfb1655586e1a7c940a97f1ec1','kernel-metadata.json':'4a17feb8570e68cd175b841141ddcb15d19092838a7d6d6e8333b1292d7a29c0'}
DEFAULT_ENTRY_FILES={'notebook.ipynb':'bd1d1d37899ca40130d4298f12361539a80414cacf0a23663252927cc888125e','kernel-metadata.json':'35e450bbf4fe9787ed1db39854d3e6942827e729a8524cc65ad8e56887a2024c'}
DEFAULT_MODULES={'knee/public_s6_recovery_entry.py':'76217a402ae3bb4c8c9517c76875691f3565b84bf0b2e548295cd9f726a90be8',
                 'knee/public_s6_recovery_runtime.py':'d67ceb738930c85b19e1866dfeb0d524e015a68ff321e039f00105c590ca6d08',
                 'knee/public_s6_recovery_rehearsal.py':'8424d87b5656ccdf449c69c5ee06c09744116f185f086861b7feee8bd5864502',
                 'scripts/build_public_s6_recovery_entry.py':'d8915ef31345f452ca8a8ea0a2cbeb12ad717363d9e7b54b3cd714ef3ed3109c'}
HAS=lambda *m:all(importlib.util.find_spec(x) is not None for x in m)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def manifest(**change):
    from knee.reference import RAPTOR_RECIPE,RAPTOR_WEIGHT_DATASET,RAPTOR_WEIGHT_NAME
    m={'profile':'raptor','checkpoint_sha256':B0.DREAD_CHECKPOINT_SHA256,'checkpoint_file':RAPTOR_WEIGHT_NAME,'weight_dataset':RAPTOR_WEIGHT_DATASET,
       'decoded_fastread_sha256':None,'recipe':{**RAPTOR_RECIPE,'windows':62}}
    m.update(change);return m


def payload(directory):
    nb=json.loads((Path(directory)/'notebook.ipynb').read_bytes());tree=ast.parse(''.join(nb['cells'][0]['source']))
    value=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='encoded' for t in n.targets))
    return json.loads(lzma.decompress(base64.b64decode(value,validate=True)))


def global_loads(code):
    names=set()
    for ins in dis.get_instructions(code):
        if ins.opname in ('LOAD_GLOBAL','LOAD_NAME'):names.add(ins.argval)
    for const in code.co_consts:
        if hasattr(const,'co_code'):names|=global_loads(const)
    return names


def scratch(test):
    (PROJECT/'outputs').mkdir(exist_ok=True)
    tmp=tempfile.TemporaryDirectory(dir=PROJECT/'outputs',prefix='test-b0-');test.addCleanup(tmp.cleanup);return Path(tmp.name)


class DefaultEntryUnchangedTests(unittest.TestCase):
    def test_default_modules_byte_identical_to_qualified_pins(self):
        from scripts.build_public_s6_recovery_b0 import DEFAULT_MODULE_SHA
        self.assertEqual(DEFAULT_MODULE_SHA,DEFAULT_MODULES)
        for name,pin in DEFAULT_MODULES.items():
            with self.subTest(name=name):self.assertEqual(sha(PROJECT/name),pin)

    def test_default_constants_unchanged(self):
        self.assertEqual(E.VERSION,'s6-dual-capped-recovery-v1');self.assertEqual(E.METHOD['old'],'central320 K94 equal original/mirror probabilities')
        self.assertIsNone(E.rehearsal_hook);self.assertEqual(E.METHOD['blend'],[.7,.3])

    @unittest.skipUnless((S6/'parent-build-arguments.json').is_file() and DEFAULT_RESULT.is_dir() and DEFAULT_REHEARSAL.is_dir() and HAS('numpy'),'default qualified evidence/numpy unavailable')
    def test_default_rehearsal_and_entry_rebuild_byte_identical(self):
        from scripts.build_public_s6_recovery_entry import build,verify_build
        args=json.loads((S6/'parent-build-arguments.json').read_bytes());root=scratch(self)
        rehearsal=build(**{**args,'destination':str(root/'rehearsal')})
        self.assertEqual(rehearsal['files'],DEFAULT_REHEARSAL_FILES);self.assertEqual(verify_build(root/'rehearsal'),rehearsal)
        self.assertEqual(rehearsal['binding'],json.loads((DEFAULT_REHEARSAL/'build-receipt.json').read_bytes())['binding'])
        entry=build(**{**args,'destination':str(root/'entry'),'slug':'knee-public-s6-recovery-entry-20261012','entry':True,'rehearsal_build':str(DEFAULT_REHEARSAL),
                       'rehearsal_receipt':str(DEFAULT_RESULT/'receipt.json'),'rehearsal_csv':str(DEFAULT_RESULT/'visible-predictions.csv')})
        self.assertEqual(entry['files'],DEFAULT_ENTRY_FILES)
        cfg=payload(root/'entry')['cfg'];self.assertEqual(cfg['public_manifest']['profile'],'mirror');self.assertEqual(json.loads((root/'entry/kernel-metadata.json').read_bytes())['dataset_sources'][-1],'nartaa/rsna-knee-publication-swa-weights-20261007')


class B0StructureTests(unittest.TestCase):
    def test_rebound_s6_merge_run_share_exact_default_bytecode(self):
        for name in ('s6_worker','merge','run'):
            with self.subTest(name=name):
                self.assertIs(getattr(B0,name).__code__,getattr(E,name).__code__);self.assertIs(getattr(B0,name).__globals__,vars(B0))
                missing={n for n in global_loads(getattr(E,name).__code__) if not hasattr(builtins,n) and n not in vars(B0)}
                self.assertEqual(missing,set())
                self.assertEqual({n for n in global_loads(getattr(E,name).__code__) if n not in ('VERSION','METHOD','rehearsal_hook','launch','merge','require_rehearsal') and n in vars(B0) and n in vars(E) and vars(B0)[n] is not vars(E)[n]},set())

    def test_version_method_only_old_arm_differs(self):
        self.assertNotEqual(B0.VERSION,E.VERSION)
        self.assertEqual({k for k in set(B0.METHOD)|set(E.METHOD) if B0.METHOD.get(k)!=E.METHOD.get(k)},{'old','old_profile'})
        self.assertEqual(B0.METHOD['blend'],[.7,.3]);self.assertEqual(B0.METHOD['recovery_recipe'],E.METHOD['recovery_recipe']);self.assertIs(B0.INPUT_POLICY,E.INPUT_POLICY)

    def test_launch_selects_b0_worker_modules(self):
        for mode,module in (('rehearsal','knee.public_s6_recovery_b0_rehearsal'),('entry','knee.public_s6_recovery_b0')):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                seen=[]
                class Done:
                    returncode=0
                    def poll(self):return 0
                    def wait(self,timeout=None):return 0
                def popen(argv,**kw):
                    seen.append(argv[2]);c=json.loads(Path(argv[-1]).read_bytes());(Path(c['out'])/c['arm']).mkdir();(Path(c['out'])/c['arm']/'receipt.json').write_text('{"board":%d}'%c['board']);return Done()
                cfg={'mode':mode,'assignment':[['a'],['b']],'boards':[{'uuid':'0'},{'uuid':'1'}]}
                with patch.object(B0.subprocess,'Popen',side_effect=popen):reports,_=B0.launch(cfg,Path(tmp),'old')
                self.assertEqual(seen,[module,module]);self.assertEqual([r['board'] for r in reports],[0,1])
                self.assertTrue((PROJECT/(module.replace('.','/')+'.py')).is_file())

    def test_profile_gate_explicit_raptor62_only(self):
        self.assertEqual(B0.raptor_profile_gate(None,manifest())['windows'],62)
        from knee.reference import RECIPE
        bad={'mirror':manifest(profile='mirror'),'nartaa_sha':manifest(checkpoint_sha256='7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef'),
             'k42':manifest(recipe={**manifest()['recipe'],'windows':42}),'mirror_recipe':manifest(recipe=RECIPE),'fastread':manifest(decoded_fastread_sha256='0'*64),
             'weights':manifest(weight_dataset='nartaa/rsna-knee-publication-swa-weights-20261007')}
        for name,m in bad.items():
            with self.subTest(name=name),self.assertRaises(ValueError):B0.raptor_profile_gate(None,m)

    def test_raptor62_gate_counts_exact_probability_bounded(self):
        counts=dict.fromkeys(B0.RAPTOR62_COUNT_KEYS,1);cfg={'raptor62_reference':{'v':{'probabilities':[.5]*12,'counts':counts}},'raptor62_visible_atol':5e-3}
        self.assertIsNone(B0.raptor62_gate('other',[.5]*12,counts,cfg))
        g=B0.raptor62_gate('v',[.502]+[.5]*11,counts,cfg);self.assertTrue(g['within_atol'] and g['decode_counts_equal']);self.assertAlmostEqual(g['max_abs'],.002,places=6)
        self.assertFalse(B0.raptor62_gate('v',[.51]+[.5]*11,counts,cfg)['within_atol'])
        with self.assertRaises(ValueError):B0.raptor62_gate('v',[.5]*12,{**counts,'valid_slices':2},cfg)

    def test_header_scope_value_type_errors_only_become_typed_data_local(self):
        def order_and_meta(error):B0.public_guard(error)   # header catch frame name -> old_public:_hdr
        def build_study(error):B0.public_guard(error)      # pixel catch frame name -> old_public:_px
        def other(error):B0.public_guard(error)
        for error in (ValueError('could not convert string to float'),TypeError("'DSfloat' object is not subscriptable")):
            with self.subTest(error=type(error).__name__,scope='hdr'),self.assertRaises(R.DataLocalFailure) as caught:order_and_meta(error)
            self.assertEqual((str(caught.exception),caught.exception.site,caught.exception.detail),('old_header_geometry_malformed','dread:_make_reader.order_and_meta',type(error).__name__))
            self.assertIs(caught.exception.__cause__,error)
            for function in (build_study,other):
                with self.subTest(error=type(error).__name__,scope=function.__name__),self.assertRaises(R.FatalSourceError):function(error)
        fatal=[AttributeError('x'),KeyError('x'),RuntimeError('x'),MemoryError('x'),TimeoutError('x'),IndexError('x'),OSError('x'),AssertionError('x'),
               UnicodeDecodeError('utf-8',b'\xff',0,1,'subclass'),type('Custom',(ValueError,),{})('subclass'),ZeroDivisionError('x'),OverflowError('x')]
        for error in fatal:
            with self.subTest(error=type(error).__name__,scope='hdr'),self.assertRaises(R.FatalSourceError):order_and_meta(error)
        caused=ValueError('wrapped');caused.__cause__=MemoryError('oom')
        with self.assertRaises(R.FatalSourceError) as wrapped:order_and_meta(caused)
        self.assertIsInstance(wrapped.exception.error,MemoryError)
        context=TypeError('during');context.__context__=KeyError('schema')
        with self.assertRaises(R.FatalSourceError):order_and_meta(context)
        if HAS('pydicom'):
            from pydicom.errors import InvalidDicomError
            previous=R.REGISTERED_DICOM_CLASS;R.REGISTERED_DICOM_CLASS=InvalidDicomError;self.addCleanup(setattr,R,'REGISTERED_DICOM_CLASS',previous)
            self.assertIsNone(order_and_meta(InvalidDicomError('not dicom')))  # default acceptance unchanged
            self.assertIsNone(build_study(InvalidDicomError('not dicom')))
        self.assertEqual(B0.OLD_PROFILE['header_data_local'],['ValueError','TypeError'])
        # The default (nartaa) policy is untouched: the same classes at its scopes stay fatal.
        for error in (ValueError('x'),TypeError('x')):
            for scope in ('old_public:_hdr','old_public:_px','old_decoder:_hdr'):
                with self.subTest(default=scope),self.assertRaises(R.FatalSourceError):R.guard_source_error(error,scope)

    def test_kernel_slug_and_title_at_most_fifty_characters(self):
        from scripts.build_public_s6_recovery_b0 import kernel_title,build,MAX_KERNEL_SLUG,REHEARSAL_SLUG,ENTRY_SLUG
        failed='knee-public-s6-recovery-dread-b0-rehearsal-20261013'  # actual 400 SaveKernel push
        self.assertEqual((len(failed),MAX_KERNEL_SLUG),(51,50))
        for slug in (failed,'x'*51,'abc','Upper-case','bad_slug','-lead'):
            with self.subTest(slug=slug),self.assertRaisesRegex(ValueError,'SLUG'):kernel_title(slug)
        for slug in (REHEARSAL_SLUG,ENTRY_SLUG,'a'*50):
            with self.subTest(slug=slug):self.assertLessEqual(len(kernel_title(slug)),50)
        self.assertEqual((REHEARSAL_SLUG,ENTRY_SLUG),('knee-s6-dread-b0-rehearsal-20261013','knee-s6-dread-b0-entry-20261013'))
        missing=str(PROJECT/'outputs'/'never-created')  # rejected before any evidence or destination is touched
        with self.assertRaisesRegex(ValueError,'SLUG'):build(*[missing]*9,missing,failed,cpu_evidence=missing)
        self.assertFalse(Path(missing).exists())

    def test_unknown_catch_function_fatal_without_guards_installed_globally(self):
        class Public:pass
        public=Public();public._raptor_protected_source='def other():\n    try:\n        pass\n    except Exception:\n        pass\n';public._raptor_propagate_fatal=lambda e:None
        with self.assertRaises(ValueError):B0.install_raptor_old_guards(public)


@unittest.skipUnless(HAS('numpy','torch'),'torch/numpy unavailable')
class StandInModelTests(unittest.TestCase):
    """Fixture public module and stand-in model: window schema, finite plain outputs."""
    def public(self,shape=(62,3,384,384),value=0.):
        import torch
        class Public:
            def eval_windows(self,v,m,k,res):
                self.calls=(k,res);return torch.full(shape,value,dtype=torch.float32)
        return Public()

    def model(self,bias=0.):
        import torch
        class Head(torch.nn.Module):
            def __init__(self):super().__init__();self.fc=torch.nn.Linear(3,12)
            def forward(self,x):
                assert x.shape[:3]==(1,62,3);return self.fc(x.mean((-1,-2)).mean(1))+bias
        torch.manual_seed(0);return Head().eval()

    def test_k62_384_plain_single_pass(self):
        import numpy as np
        public=self.public();w=B0.raptor_windows(public,None,None);self.assertEqual(public.calls,(62,384));self.assertEqual(tuple(w.shape),(62,3,384,384))
        p=B0.raptor_forward(public,self.model(),w,'cpu');self.assertEqual(p.shape,(12,));self.assertTrue(np.isfinite(p).all())
        self.assertTrue(np.array_equal(p,B0.raptor_forward(public,self.model(),w,'cpu')))

    def test_wrong_window_schema_and_nonfinite_rejected(self):
        for shape in ((42,3,384,384),(62,3,320,320),(94,3,320,320)):
            with self.subTest(shape=shape),self.assertRaises(ValueError):B0.raptor_windows(self.public(shape),None,None)
        with self.assertRaises(ValueError):B0.raptor_windows(self.public(value=float('nan')),None,None)
        w=B0.raptor_windows(self.public(),None,None)
        with self.assertRaises(ValueError):B0.raptor_forward(self.public(),self.model(float('nan')),w,'cpu')


@unittest.skipUnless(DREAD_SOURCE.is_file() and HAS('numpy','torch','cv2','pydicom'),'pinned Dread source or cv2/pydicom/torch unavailable')
class RaptorGuardTests(unittest.TestCase):
    """Actual audited Dread definitions through raptor_input with B0 guards."""
    def setUp(self):
        from knee.reference import audit_source,load_definitions
        import knee.raptor_input as RI
        module,decoder,m=audit_source(DREAD_SOURCE.read_text(),profile='raptor',checkpoint_sha256=B0.DREAD_CHECKPOINT_SHA256,raptor_windows=62)
        self.public,_=load_definitions(module,decoder,m);self.manifest=m;self.RI=RI
        previous=(RI.validate_production_volume,R.REGISTERED_DICOM_CLASS)
        def restore():RI.validate_production_volume,R.REGISTERED_DICOM_CLASS=previous
        self.addCleanup(restore)

    def test_manifest_matches_raptor62_and_profile(self):
        self.assertEqual(self.manifest['extracted_module_sha256'],'977d94c22deebd4f50d183bbdf5cd222136db785e0cae59dc334e34bc329e263')
        if (RAPTOR62/'reference-contract.json').is_file():self.assertEqual(json.loads((RAPTOR62/'reference-contract.json').read_bytes())['source'],self.manifest)
        self.assertEqual(B0.raptor_profile_gate(self.public,self.manifest),B0.OLD_PROFILE);self.assertEqual(list(self.public.LAB),LABELS)

    def test_baseline_stress_then_guarded_ten(self):
        from knee.public_s6_recovery_b0_rehearsal import guarded_raptor_partial
        self.assertEqual(B0.raptor_stress(self.public)['case_count'],16)
        g=B0.install_raptor_old_guards(self.public)
        self.assertEqual(g['all_empty_guard_sha256'],'226dd444bd32408f8869182961536871906d44c4a37344ac9d447659c76a43e6')  # equals default T4 receipts
        self.assertEqual(g['pixel_buffer_guard_sha256'],'811b458d0a94cca3a0899f30a7b02defdee0a2ff5cac99d8634e5cd855350dc7')
        r=guarded_raptor_partial(self.public);self.assertEqual(r['cases'],dict.fromkeys(B0.GUARDED_PARTIAL_CASES,'PASS'))
        self.assertEqual(B0.install_raptor_old_guards(self.public)['all_empty_guard_sha256'],g['all_empty_guard_sha256'])  # reinstall from source function

    def test_all_empty_typed_unknown_header_and_scope_fatal(self):
        import numpy as np
        import pydicom
        from knee.public_s6_recovery_b0_rehearsal import _dataset
        B0.install_raptor_old_guards(self.public)
        plan=[(None,18),({'SeriesInstanceUID':'x'},14),(None,12),(None,8),(None,12)]
        with self.assertRaises(R.DataLocalFailure) as caught:self.RI.validate_production_volume(np.zeros((64,336,336),np.uint8),np.zeros(64,np.uint8),plan,image_size=336)
        self.assertEqual((str(caught.exception),caught.exception.site),('old_no_usable_image',B0.ALL_EMPTY_SITE))
        with self.assertRaises(R.FatalSourceError):self.public._raptor_propagate_fatal(ValueError('unknown catch function'))
        with tempfile.TemporaryDirectory() as tmp:
            series=Path(tmp)/'s'/'sag';series.mkdir(parents=True)
            for i in range(4):pydicom.dcmwrite(series/f'{i}.dcm',_dataset(i),enforce_file_format=True)
            records={'s':[dict(SeriesInstanceUID='sag',Anatomical_Plane='Sagittal',Fluid_Sensitive=1)]}
            v,m,c=self.RI.prepare_study(self.public,'s',records,Path(tmp));self.assertEqual(v.shape,(64,336,336))
            original=pydicom.dcmread
            def broken(*a,**kw):
                if kw.get('stop_before_pixels'):raise KeyError('programming error inside header catch')
                return original(*a,**kw)
            with patch.object(pydicom,'dcmread',side_effect=broken),self.assertRaises(R.FatalSourceError):self.RI.prepare_study(self.public,'s',records,Path(tmp))

    def test_malformed_readable_header_is_data_local_and_filled_under_cap(self):
        import types
        import numpy as np
        import pydicom
        from knee.public_s6_recovery import recover
        from knee.public_s6_recovery_b0_rehearsal import _dataset
        B0.install_raptor_old_guards(self.public)
        records={'s':[dict(SeriesInstanceUID='sag',Anatomical_Plane='Sagittal',Fluid_Sensitive=1)]}
        failures=[]
        with tempfile.TemporaryDirectory() as tmp:
            series=Path(tmp)/'s'/'sag';series.mkdir(parents=True)
            for i in range(4):pydicom.dcmwrite(series/f'{i}.dcm',_dataset(i),enforce_file_format=True)
            healthy=self.RI.prepare_study(self.public,'s',records,Path(tmp))
            # Real file: single-valued PixelSpacing is readable; Dread's ps[0] raises TypeError.
            ds=_dataset(0);ds.PixelSpacing=0.5;pydicom.dcmwrite(series/'0.dcm',ds,enforce_file_format=True)
            with self.assertRaises(R.DataLocalFailure) as caught:self.RI.prepare_study(self.public,'s',records,Path(tmp))
            self.assertEqual((str(caught.exception),type(caught.exception.__cause__)),('old_header_geometry_malformed',TypeError));failures.append(caught.exception)
            pydicom.dcmwrite(series/'0.dcm',_dataset(0),enforce_file_format=True)
            original=pydicom.dcmread
            for field,value in (('ImageOrientationPatient',['x']*6),('ImagePositionPatient',['x','0','0'])):
                def malformed(path,*a,**kw):
                    if kw.get('stop_before_pixels') and Path(path).name=='1.dcm':
                        h=types.SimpleNamespace(ImageOrientationPatient=[1,0,0,0,1,0],ImagePositionPatient=[0,0,1],PixelSpacing=[.5,.5],InstanceNumber=1);setattr(h,field,value);return h
                    return original(path,*a,**kw)
                with self.subTest(field=field),patch.object(pydicom,'dcmread',side_effect=malformed),self.assertRaises(R.DataLocalFailure) as caught:
                    self.RI.prepare_study(self.public,'s',records,Path(tmp))
                self.assertEqual((str(caught.exception),type(caught.exception.__cause__)),('old_header_geometry_malformed',ValueError));failures.append(caught.exception)
            again=self.RI.prepare_study(self.public,'s',records,Path(tmp))  # next healthy study unaffected
            self.assertTrue(all(np.array_equal(a,b) for a,b in zip(healthy[:2],again[:2])))
        # Worker classification -> genuine-only ledger -> unchanged capped arm-mean fill.
        n=100;rng=np.random.default_rng(3);old=rng.uniform(size=(n,12)).astype(np.float32);s6=rng.uniform(size=(n,12))
        statuses=['genuine']*n;statuses[5]='data_local_failure';old[5]=np.nan
        R.save_ledger(Path(self.enterContext(tempfile.TemporaryDirectory()))/'old.npz',[f'id{i}' for i in range(n)],old,statuses)
        missing=np.array([s=='data_local_failure' for s in statuses]);r=recover(old,s6,missing,np.zeros(n,bool))
        np.testing.assert_array_equal(r['control'][5],old[~missing].astype(float).mean(0));self.assertEqual(r['statuses']['old320'][5],'recovered');self.assertEqual(r['union_recovered'],1)
        over=missing.copy();over[:4]=True
        with self.assertRaisesRegex(ValueError,'UNION_RECOVERY_CAP'):recover(np.where(over[:,None],np.nan,old),s6,over,np.zeros(n,bool))
        self.assertEqual(len(failures),3)


@unittest.skipUnless(DREAD_SOURCE.is_file() and DREAD_CHECKPOINT.is_file() and HAS('numpy','torch','timm','cv2','pydicom'),'local Dread checkpoint/source or timm/cv2/pydicom unavailable')
class DreadCheckpointCpuTests(unittest.TestCase):
    def test_sha_strict_load_labels_pins_windows_finite(self):
        from scripts.qualify_b0_dread_cpu import qualify
        contract=RAPTOR62/'reference-contract.json'
        r=qualify(DREAD_SOURCE,DREAD_CHECKPOINT,raptor62_contract=contract if contract.is_file() else None,forward_windows=8)
        self.assertEqual(r['checkpoint_sha256'],'89606f05849838529e1b4658d28fb049623205d9853371548403bca460361ded')
        self.assertTrue(r['strict_state_load']);self.assertEqual(r['labels'],LABELS);self.assertEqual(r['windows_shape'],[62,3,384,384])
        self.assertEqual((r['state_sha256'],r['constructor_grid_sha256']),(B0.DREAD_STATE_SHA256,B0.DREAD_GRID_SHA256))
        self.assertEqual(len(r['cpu_probabilities']),12);self.assertEqual(r['cpu_repeat_max_abs'],0.)


def passing_rehearsal(tmp):
    """Synthetic receipt satisfying the B0 contract; mutated by rejection tests."""
    import numpy as np
    s6ids=[f's6-{i}' for i in range(32)];old=[f'old-{i}' for i in range(4)];visible=['v0','v1','v2'];s6sent=[s6ids[i] for i in E.SENTINELS]
    p=[.25+.01*i for i in range(12)];logit=np.log(np.asarray(p)/(1-np.asarray(p))).astype(np.float32).tolist()
    env={'torch':'2.10.0+cu128','packages':{'timm':'1.0.26'}}
    binding={'version':B0.VERSION,'method':B0.METHOD,'input_policy':B0.INPUT_POLICY,'old_profile':B0.OLD_PROFILE,'environment':env,
             'qualified_state':{'old_state':B0.DREAD_STATE_SHA256,'old_grid':B0.DREAD_GRID_SHA256,'s6_state':'s6'},
             'sentinel_inputs':{'old':{s:{} for s in old},'s6':{s:{'render_windows':'x'} for s in s6sent}},'sentinel_probabilities':{'old':dict.fromkeys(old),'s6':{s:p for s in s6sent}},
             's6_probe_ids':s6ids,'old_historical_ids':old,'old_qualification':{'raptor62':{'visible_atol':5e-3}}}
    path=Path(tmp)/'visible.csv'
    with path.open('w',newline='') as h:w=csv.writer(h);w.writerow([UID]+LABELS);w.writerows([[v]+p for v in visible])
    ids=s6ids+old+visible
    def worker(arm,board):
        x={'status':'PASS','board':board,'assigned':ids[board::2],'completed':len(ids[board::2]),'peak_cuda_bytes':1,'peak_rss_bytes':1,'zero_failed':True,'zero_imputed':True,'fp32_recoveries':0,
           'environment':{**env,'physical_uuid':f'GPU-{board}','cuda_device_count':1},'recovery_injection':{'status':'PASS','failure_status':'data_local_failure','injected_rows_genuine':False,'state_healthy':True}}
        if arm=='old':
            x.update(state_before={'state':B0.DREAD_STATE_SHA256},state_after={'state':B0.DREAD_STATE_SHA256},constructor_grid_sha256=B0.DREAD_GRID_SHA256,old_profile=B0.OLD_PROFILE,source_guard_receipt={'profile':'raptor'},
                     sentinels=[{'id':s,'input':{},'probabilities':p,'reloaded_probabilities':p,'strict_reload':True,'strict_source_comparison':'PASS'} for s in old])
            x['records']=[{'id':s,**({'source_reference_parity':True,'strict_reload':True,'reload_max_abs':0.,'strict_source_comparison':'PASS'} if s not in visible else
                                     {'raptor62':{'decode_counts_equal':True,'atol':5e-3,'within_atol':True,'max_abs':1e-3}})} for s in ids[board::2]]
            if board==0:
                import knee.raptor_input as RI
                x.update(partial_input_stress={'status':'PASS','cases':dict.fromkeys(RI.STRESS_CASES,'PASS'),'case_count':16,'uses_raw_data':False,'uses_labels':False},
                         guarded_partial_input_stress={'status':'PASS','guarded':True,'case_count':10,'cases':dict.fromkeys(B0.GUARDED_PARTIAL_CASES,'PASS')})
        else:
            x.update(state_before='s6',state_after='s6',sentinels=[{'id':s,'input':{'render_windows':'x'},'probabilities':p,'reloaded_probabilities':p,'strict_reload':True,'logits':[logit]*4,'reloaded_logits':[logit]*4} for s in s6sent],
                     records=[{'id':s,'qualified_error':0. if s in s6ids else None} for s in ids[board::2]])
            if board==0:
                x.update(native_availability_stress={'status':'PASS','case_count':2,'uses_raw_data':False,'uses_labels':False,'cases':{k:{'status':'PASS','usable_series':1,'slices':4} for k in ('absent','empty')}},
                         guarded_native_availability_stress={'status':'PASS','guarded':True,'case_count':2,'cases':{'absent':{},'empty':{}}})
        return x
    phase={'wall_seconds':1.,'peaks':{'host_rss_bytes':1,'output_disk_bytes':1},'minimum_disk_free_bytes':1}
    r={'status':'PASS','version':B0.VERSION,'mode':'rehearsal','binding':binding,'completed':39,'completed_visible':3,'completed_probes':36,'completed_s6_probes':32,'completed_old_historical':4,
       'all_input_state_probability_gates':True,'zero_failed':True,'zero_imputed':True,'boards':[{'uuid':'GPU-0','name':'Tesla T4'},{'uuid':'GPU-1','name':'Tesla T4'}],
       'projection':{'conservative_seconds':3600.},'elapsed_seconds':1800.,'csv_sha256':sha(path),'visible_ids':visible,'phases':{'old':phase,'s6':phase},
       'workers':{arm:[worker(arm,0),worker(arm,1)] for arm in ('old','s6')},'sentinel_ids':{'old':old,'s6':s6sent}}
    return r,binding,path


@unittest.skipUnless(HAS('numpy'),'numpy unavailable')
class B0RehearsalContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.r,self.binding,self.csv=passing_rehearsal(self.tmp.name)

    def test_synthetic_receipt_passes(self):
        B0.require_rehearsal(self.r,self.binding,self.csv)

    def test_rejections(self):
        def visible_out_of_atol(r):
            for w in r['workers']['old']:
                for row in w['records']:
                    if 'raptor62' in row:row['raptor62'].update(max_abs=.02,within_atol=False)
        def default_mirror_stress(r):r['workers']['old'][0]['partial_input_stress']['case_count']=17
        def no_strict_pass(r):
            for w in r['workers']['old']:
                for row in w['records']+w['sentinels']:
                    if 'strict_source_comparison' in row:row['strict_source_comparison']='ineligible_partial_input'
        def nartaa_state(r):r['workers']['old'][1]['state_before']=r['workers']['old'][1]['state_after']={'state':'1d8594d77f7dd592bd1ef1b2090d74eb8ea09580fe689f78dbab273c8ae8a219'}
        def old_reload_drift(r):r['workers']['old'][0]['sentinels'][0]['reloaded_probabilities']=[x+1e-3 for x in r['workers']['old'][0]['sentinels'][0]['probabilities']]
        def default_version(r):r['version']=E.VERSION
        def missing_repeat(r):r['workers']['old'][0]['records'][0].pop('strict_reload')
        for mutate in (visible_out_of_atol,default_mirror_stress,no_strict_pass,nartaa_state,old_reload_drift,default_version,missing_repeat):
            with self.subTest(mutate=mutate.__name__):
                r=copy.deepcopy(self.r);mutate(r)
                with self.assertRaises(ValueError):B0.require_rehearsal(r,self.binding,self.csv)
        with self.assertRaises(ValueError):B0.require_rehearsal(self.r,{**self.binding,'method':E.METHOD},self.csv)


B0_BUILD_INPUTS=[OUT/'kaggle/public-s6-evaluation-20261011-reviewed',RAPTOR62,DREAD_SOURCE,OUT/'analysis/b0-dread-swap-20261012/cpu-qualification-b0.json',S6/'protocol-private.json',EVIDENCE/'data/raw/test.csv']


@unittest.skipUnless(all(p.exists() for p in B0_BUILD_INPUTS) and HAS('numpy'),'B0 build evidence/numpy unavailable')
class B0BuildTests(unittest.TestCase):
    def args(self,destination):
        return dict(evaluation_build=str(OUT/'kaggle/public-s6-evaluation-20261011-reviewed'),evaluation_result=str(OUT/'kaggle/public-s6-evaluation-fold0-20261011-result/s6-evaluation'),
                    audit_path=str(OUT/'analysis/public-s6-evaluation-20261011/independent-result-audit.json'),pilot_build=str(OUT/'kaggle/public-s6-qualification-20261011-reviewed-v2'),
                    pilot_result=str(OUT/'kaggle/public-s6-qualification-20261011-result-v2/s6-qualification'),data_root=str(EVIDENCE/'data/raw'),cpu_evidence=str(S6),
                    dread_source=str(DREAD_SOURCE),raptor62_result=str(RAPTOR62),dread_cpu_receipt=str(OUT/'analysis/b0-dread-swap-20261012/cpu-qualification-b0.json'),
                    destination=str(destination),slug='knee-s6-dread-b0-rehearsal-20261013')

    def test_rehearsal_build_contract_and_entry_gate(self):
        from scripts.build_public_s6_recovery_b0 import build,verify_build,DATASETS
        from knee.reference import RAPTOR_WEIGHT_DATASET
        root=scratch(self);r=build(**self.args(root/'b0'));self.assertEqual(verify_build(root/'b0'),r)
        meta=json.loads((root/'b0/kernel-metadata.json').read_bytes())
        self.assertEqual(meta['dataset_sources'],DATASETS);self.assertIn(RAPTOR_WEIGHT_DATASET,DATASETS);self.assertFalse(any(d.startswith('nartaa/') for d in DATASETS))
        self.assertTrue(meta['is_private'] and meta['enable_gpu'] and not meta['enable_internet'])
        pack=payload(root/'b0');cfg=pack['cfg']
        self.assertEqual((cfg['public_manifest']['profile'],cfg['public_manifest']['recipe']['windows'],cfg['public_manifest']['checkpoint_sha256']),('raptor',62,B0.DREAD_CHECKPOINT_SHA256))
        self.assertEqual((cfg['control_state_sha256'],cfg['control_grid_sha256'],cfg['decoder_source']),(B0.DREAD_STATE_SHA256,B0.DREAD_GRID_SHA256,''))
        for name,pin in DEFAULT_MODULES.items():
            if name.startswith('knee/'):self.assertEqual(hashlib.sha256(pack['sources'][name].encode()).hexdigest(),pin)
        self.assertIn('knee/public_s6_recovery_b0_rehearsal.py',pack['sources']);self.assertIn("from knee.public_s6_recovery_b0 import run",''.join(json.loads((root/'b0/notebook.ipynb').read_bytes())['cells'][0]['source']))
        self.assertEqual(len(cfg['probes']),36);self.assertTrue(all(x['old']=={'input':{},'probabilities':None} for x in cfg['probes'].values()))
        self.assertEqual(len(cfg['raptor62_reference']),3);self.assertEqual(cfg['binding']['qualified_state']['s6_state'],cfg['pilot_state_sha256'])
        with self.assertRaises(ValueError):build(**{**self.args(root/'entry'),'entry':True})


if __name__=='__main__':
    unittest.main()
