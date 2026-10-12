import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from knee import public_s6_recovery_runtime as R
from knee import public_s6_recovery_entry as E
try:
    import numpy as np
except ImportError:
    np=None

class RuntimeBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)

    def test_unknown_programming_and_resource_errors_fatal(self):
        for error in (MemoryError('oom'),TimeoutError('deadline'),RuntimeError('CUDA'),FileNotFoundError('path'),PermissionError('path'),ValueError('source_state'),TypeError('bug'),KeyError('schema'),IndexError('bug')):
            with self.subTest(error=type(error).__name__),self.assertRaises(R.FatalSourceError):R.guard_source_error(error,'native:prepare:87')

    def test_geometry_name_alone_never_whitelisted(self):
        cls=type('GeometryError',(RuntimeError,),{'__module__':'training.dicom_geometry'})
        with self.assertRaises(R.FatalSourceError):R.guard_source_error(cls('bad header'),'native:prepare:87')

    def test_forged_marker_and_wrapped_resource_fatal(self):
        forged=type('GeometryData',(RuntimeError,),{'_s6_data_site':323})
        with self.assertRaises(R.FatalSourceError):R.guard_source_error(forged('forged'),'native')
        tagged=type('Registered',(RuntimeError,),{})
        R.REGISTERED_DATA_CLASSES.add(tagged)
        try:
            error=tagged('wrapped');error.__cause__=MemoryError('oom')
            with self.assertRaises(R.FatalSourceError):R.guard_source_error(error,'native')
        finally:R.REGISTERED_DATA_CLASSES.remove(tagged)

    def test_invalid_dicom_unknown_scope_and_decorator_preserved(self):
        from pydicom.errors import InvalidDicomError
        with patch.object(R,'REGISTERED_DICOM_CLASS',InvalidDicomError):
            with self.assertRaises(R.FatalSourceError):R.guard_source_error(InvalidDicomError('bad'),'unknown')
            R.guard_source_error(InvalidDicomError('bad'),'training.native_cache:prepare:87')
        source='@decorate\ndef f(value):\n    try: return value()\n    except Exception as exc: raise\n'
        code,_,_=R.transformed_function(source,name='f',scope='test')
        self.assertIn('@decorate',code)

    def test_pixel_buffer_unknown_scope_and_causal_resource_fatal(self):
        error=R.AuditedPixelBufferError('bytes')
        R.guard_source_error(error,'old_public:_px')
        with self.assertRaises(R.FatalSourceError):R.guard_source_error(error,'unknown')
        error.__cause__=MemoryError('oom')
        with self.assertRaises(R.FatalSourceError):R.guard_source_error(error,'old_public:_px')
        local=R.DataLocalFailure('wrapped');local.__cause__=MemoryError('oom')
        with self.assertRaises(R.FatalSourceError):R.guard_source_error(local,'native')

    def test_pixel_buffer_changed_package_and_unrecognized_function_fail(self):
        import pydicom.pixels.decoders.base as B
        R.install_pixel_buffer_guard()
        with patch.object(R.Path,'read_bytes',return_value=b'changed'),self.assertRaisesRegex(ValueError,'SOURCE'):R.install_pixel_buffer_guard()
        def unknown(self):pass
        unknown._s6_buffer_guarded=True
        with patch.object(B.DecodeRunner,'_validate_buffer',unknown),self.assertRaisesRegex(ValueError,'INSTALLED'):R.install_pixel_buffer_guard()

    def test_primitive_conversion_allowlist_exact_scope(self):
        R.guard_source_error(ValueError('bad int'),'training.dicom_geometry:header_from_dataset:322')
        with self.assertRaises(R.FatalSourceError):R.guard_source_error(ValueError('bad int'),'training.dicom_geometry:read_pixels:648')
        with self.assertRaises(R.FatalSourceError):R.guard_source_error(MemoryError('oom'),'training.dicom_geometry:header_from_dataset:322')

    def test_data_local_propagates_source_catches(self):
        with self.assertRaises(R.DataLocalFailure):R.guard_source_error(R.DataLocalFailure('typed'),'any')

    def test_source_catch_guard_prevents_wrapped_unknown(self):
        source='def f(value):\n    try: return value()\n    except Exception as exc: raise GeometryError("wrapped") from exc\n'
        code,count,_=R.transformed_function(source,name='f',scope='native:f',source_start=1)
        ns={'_s6_guard_source_error':R.guard_source_error,'GeometryError':RuntimeError};exec(code,ns)
        with self.assertRaises(R.FatalSourceError):ns['f'](lambda:(_ for _ in ()).throw(MemoryError('oom')))
        self.assertEqual(count,1)

    def test_getter_unwrap_propagates_baseexception_and_unknown(self):
        source='def f(value):\n    try: return value()\n    except Exception as exc: return str(exc)\n    finally: cleanup()\n'
        code,_,_=R.transformed_function(source,name='f',scope='getter',unwrap=True)
        calls=[];ns={'cleanup':lambda:calls.append(True)};exec(code,ns)
        for error in (ValueError('bug'),KeyboardInterrupt(),SystemExit()):
            with self.assertRaises(type(error)):ns['f'](lambda:(_ for _ in ()).throw(error))
        self.assertEqual(len(calls),3)

    def test_cleanup_exact_current_only_and_side(self):
        sid='synthetic';exact=self.root/('s_'+hashlib.sha256(sid.encode()).hexdigest()[:16]);exact.mkdir();(exact/'a').write_text('a')
        neighbor=self.root/'s_unrelated';neighbor.mkdir();forced={sid:'R','other':'L'}
        self.assertEqual(R.cleanup_study(self.root,sid,forced),dict(scratch_removed=True,side_override_removed=True))
        self.assertTrue(neighbor.exists());self.assertEqual(forced,{'other':'L'})

    def test_cleanup_failure_is_fatal_and_side_not_silently_success(self):
        sid='synthetic';exact=self.root/('s_'+hashlib.sha256(sid.encode()).hexdigest()[:16]);exact.mkdir()
        with patch.object(R.shutil,'rmtree',side_effect=PermissionError('denied')),self.assertRaises(PermissionError):R.cleanup_study(self.root,sid,{sid:'R'})

    def test_missing_generated_artifacts_are_fatal(self):
        self.assertIsInstance(R.empty_native_failure({'series':[]}),R.DataLocalFailure)
        self.assertIsInstance(R.empty_native_failure({'series':[{'path':'gone'}]}),R.FatalSourceError)

    def test_old_strict_receipt_cannot_qualify_recovery(self):
        from knee.public_s6_entry import VERSION as strict
        with self.assertRaisesRegex(ValueError,'EXACT_FROZEN'):E.require_rehearsal({'version':strict},{'version':strict},self.root/'none')

    @unittest.skipIf(np is None,'optional numpy unavailable')
    def test_genuine_only_ledger_missing_rows_nan(self):
        p=np.ones((2,12))*.3;p[1]=np.nan
        R.save_ledger(self.root/'ledger.npz',['a','b'],p,['genuine','data_local_failure'])
        with np.load(self.root/'ledger.npz') as a:self.assertTrue(np.isnan(a['probabilities'][1]).all())
        p[1]=.5
        with self.assertRaisesRegex(ValueError,'VALUES'):R.save_ledger(self.root/'bad.npz',['a','b'],p,['genuine','data_local_failure'])

    @unittest.skipIf(np is None,'optional numpy unavailable')
    def test_merge_bothboards_state_cleanup_and_global_fill(self):
        from knee.public_s6_recovery import recover
        ids=['s'+str(i) for i in range(100)];shards=[]
        for board,assigned in enumerate(E.assignments(ids)):
            p=np.full((len(assigned),12),.2+board*.2);status=['genuine']*len(assigned)
            if board==0:p[0]=np.nan;status[0]='data_local_failure'
            path=self.root/f'{board}.npz';R.save_ledger(path,assigned,p,status)
            records=[{'id':sid,'status':s,'cleanup':{'scratch_removed':True,'side_override_removed':True},'state_healthy':True} for sid,s in zip(assigned,status)]
            r={'version':E.VERSION,'status':'PASS','board':board,'arm':'old','assigned':assigned,'completed':len(assigned),'state_before':'x','state_after':'x','zero_imputed':True,'fp32_recoveries':0,'ledger_sha256':E.file_sha(path),'genuine':status.count('genuine'),'failed':status.count('data_local_failure'),'records':records}
            shards.append((r,path))
        c,cm=E.merge(ids,shards,'old');r=recover(c,np.full_like(c,.4),cm,np.zeros(100,bool))
        self.assertAlmostEqual(r['control'][0,0],(49*.2+50*.4)/99)
        bad=copy.deepcopy(shards);bad[0][0]['state_after']='changed'
        with self.assertRaisesRegex(ValueError,'STATE'):E.merge(ids,bad,'old')
        bad=copy.deepcopy(shards);bad[0][0]['records'][0]['cleanup']['scratch_removed']=False
        with self.assertRaisesRegex(ValueError,'CLEANUP'):E.merge(ids,bad,'old')

@unittest.skipUnless(os.environ.get('KNEE_GUARD_SOURCE'),'set KNEE_GUARD_SOURCE for actual pinned CPU source tests')
class ActualPinnedSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source=Path(os.environ['KNEE_GUARD_SOURCE']).resolve();sys.path.insert(0,str(source));os.environ['RSNA_CODE_ROOT']=str(source)
        import compact_infer as CI
        import training.dicom_geometry as G
        cls.G=G;cls.CI=CI
        files=['compact_infer.py','training/native_cache.py','training/dicom_geometry.py','knee_anchor_bank_v2.py','px_cache.py']
        cls.cfg={'code':str(source),'guard_source_sha256':{f:hashlib.sha256((source/f).read_bytes()).hexdigest() for f in files}}
        cls.receipt=R.install_source_guards(CI,None,cls.cfg)

    def test_actual_source_inventory_and_lean(self):
        self.assertEqual(set(x for k,v in self.receipt.items() if k.startswith('training/dicom_geometry.py:') for x in v['data_raise_sites']),R.GEOMETRY_DATA_RAISES)
        self.assertEqual(self.receipt['training/native_cache.py:prepare']['catches'],3)

    def test_actual_header_nested_oom_and_programming_errors(self):
        for error in (MemoryError('oom'),RuntimeError('CUDA'),KeyError('bug')):
            class Dataset:
                @property
                def InstanceNumber(self):raise error
            with self.assertRaises(R.FatalSourceError):self.G.header_from_dataset(Dataset(),'synthetic')

    def test_actual_geometry_unknown_plane_artifacts_fatal(self):
        with self.assertRaises(Exception) as caught:self.G.canonical_normal('unknown')
        with self.assertRaises(R.FatalSourceError):R.guard_source_error(caught.exception,'native')
        with self.assertRaises(R.FatalSourceError):R.guard_source_error(self.G.GeometryError('missing NumPy artifact: synthetic'),'native')

    def test_actual_getter_no_window_and_unknown_not_serialized(self):
        import torch
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            ds=self.CI.CStudies(tmp,'test',['synthetic'],tmp,self.cfg['code'],384,views='v1')
            ds.native=object();ds.bank=None
            ds.builder=SimpleNamespace(build=lambda uid:{'px':torch.zeros((0,384,384),dtype=torch.uint8),'meta':torch.zeros((0,12)),'blocks':[]})
            ds.window_rows=lambda *a,**kw:[]
            ds._native_and_geometry=lambda uid,work:work.mkdir(exist_ok=True)
            with self.assertRaises(R.DataLocalFailure) as caught:ds[0]
            self.assertEqual(caught.exception.site,'compact_infer.py:CStudies.__getitem__:210')
            ds._native_and_geometry=lambda *a:(_ for _ in ()).throw(MemoryError('oom'))
            with self.assertRaises(MemoryError):ds[0]
            self.assertFalse(list(Path(tmp).glob('s_*')))

    def test_actual_px_empty_versus_missing_generated_artifact(self):
        import px_cache
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(px_cache.study_json(tmp,'synthetic'));path.parent.mkdir(parents=True)
            builder=px_cache.StudySlices(tmp,[],384,'v1')
            path.write_text(json.dumps({'series':[]}))
            with self.assertRaises(R.DataLocalFailure):builder.build('synthetic')
            path.write_text(json.dumps({'series':[{'path':str(Path(tmp)/'gone.npy'),'SeriesInstanceUID':'syntheticseries'}]}))
            with self.assertRaises(R.FatalSourceError):builder.build('synthetic')

    def test_actual_guard_reset_native_reinstall_and_shared_globals(self):
        import probe_infer as PI
        import training.native_cache as native
        cfg=dict(self.cfg,source_pins={'training/native_cache.py':self.cfg['guard_source_sha256']['training/native_cache.py']})
        for _ in range(2):
            E.reset_native_renderer(cfg);PI.install_native(True)
            receipt=R.install_source_guards(self.CI,None,cfg)
            self.assertEqual(receipt['training/native_cache.py:prepare']['catches'],3)
            self.assertIs(self.G.header_from_dataset.__globals__,self.G.__dict__)
        class Dataset:
            InstanceNumber=1
            class Scalar:
                def __array__(self,*a,**kw):raise MemoryError('nested vector oom')
            ImagePositionPatient=Scalar()
        with self.assertRaises(R.FatalSourceError):self.G.header_from_dataset(Dataset(),'synthetic')

    def test_actual_old_all_empty_typed_and_unknown_callbacks_fatal(self):
        from types import SimpleNamespace
        import knee.mirror_input as MI
        from unittest.mock import patch
        original=MI.validate_production_volume
        public=SimpleNamespace();decoder=SimpleNamespace(__name__='audited_fastread',NeedMore=type('NeedMore',(Exception,),{'__module__':'audited_fastread'}))
        try:
            R.install_old_guards(public,decoder)
            plan=[(None,1)]
            with self.assertRaises(R.DataLocalFailure) as caught:MI.validate_production_volume(np.zeros((1,4,4),np.uint8),np.zeros(1,np.uint8),plan,image_size=4)
            self.assertEqual(caught.exception.site,'raptor_input.py:validate_production_volume:164')
            for callback in (public._raptor_propagate_fatal,decoder._mirror_propagate_fatal):
                with self.assertRaises(R.FatalSourceError):callback(KeyError('programming'))
        finally:MI.validate_production_volume=original

    def test_actual_malformed_header_typed_and_nested_scalar_oom_fatal(self):
        from pydicom.dataset import Dataset
        with self.assertRaises(self.G.GeometryError) as caught:self.G.header_from_dataset(Dataset(),'synthetic')
        self.assertEqual(type(caught.exception)._s6_data_site,323)
        R.guard_source_error(caught.exception,'native')
        class Scalar:
            def __float__(self):raise MemoryError('oom')
        with self.assertRaises(R.FatalSourceError):self.G._finite_float(Scalar(),0)
        self.assertIsNone(self.G._float_vector(['bad','tag'],2))

if __name__=='__main__':unittest.main()
