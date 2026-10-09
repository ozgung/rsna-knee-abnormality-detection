"""Synthetic/source-only MaViT reader qualification; no weights or real MRI required."""
import ast
from contextlib import ExitStack
import copy
import csv
import errno
import importlib.util
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

from knee import mavit_input as m, mavit_rehearsal as r

ROOT=Path(__file__).resolve().parents[1]
PARENT=Path(os.environ['KNEE_REFERENCE_PARENT']) if os.environ.get('KNEE_REFERENCE_PARENT') else None
SOURCE=PARENT/'outputs/analysis/post-fusion-research-20261009/mavit/dataset-code' if PARENT else None
CPU=PARENT/'outputs/kaggle/mavit-checkpoint-audit-20261009-result/mavit-checkpoint-audit/checkpoint-audit.json' if PARENT else None


def script(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def dicom(array,*,implicit=False,signed=False,bits=16,stored=12,mono1=False,geometry=True,position=0,instance=0,laterality='L',spacing=1.,slope=1.5,intercept=-10):
    import numpy as np
    from pydicom.dataset import FileDataset,FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian,ImplicitVRLittleEndian,MRImageStorage
    meta=FileMetaDataset();meta.TransferSyntaxUID=ImplicitVRLittleEndian if implicit else ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID=MRImageStorage;meta.MediaStorageSOPInstanceUID='1.2.3.4'
    d=FileDataset(None,{},file_meta=meta,preamble=b'\0'*128)
    d.SOPClassUID=MRImageStorage;d.SOPInstanceUID='1.2.3.4';d.Rows=array.shape[-2];d.Columns=array.shape[-1]
    d.SamplesPerPixel=1;d.PhotometricInterpretation='MONOCHROME1' if mono1 else 'MONOCHROME2'
    d.BitsAllocated=bits;d.BitsStored=stored;d.HighBit=stored-1;d.PixelRepresentation=int(signed)
    d.InstanceNumber=instance;d.RescaleSlope=slope;d.RescaleIntercept=intercept
    if geometry:d.ImageOrientationPatient=[1,0,0,0,1,0];d.ImagePositionPatient=[0,0,position];d.PixelSpacing=[spacing,spacing]
    if laterality:d.ImageLaterality=laterality
    if array.ndim==3:d.NumberOfFrames=array.shape[0]
    d.PixelData=np.asarray(array,dtype='<u2' if bits==16 else 'u1').tobytes()
    out=io.BytesIO();d.save_as(out,enforce_file_format=True);return out.getvalue()


def rows(series='series',plane='Sagittal'):
    return [dict(StudyInstanceUID='synthetic',SeriesInstanceUID=series,Anatomical_Plane=plane,Fluid_Sensitive=0,Fat_Suppression=0)]


def valid_evidence():
    c={'version':r.VERSION,'checkpoint_sha256':r.CHECKPOINT_SHA256,'cpu_report_sha256':r.CPU_REPORT_SHA256,
       'model_state_schema_sha256':r.STATE_SCHEMA,'max_seconds':r.MAX_SECONDS,'private':True,'internet':False,'codecs':{'wheel':'a'*64},
       'input_policy':{'version':m.VERSION,'sha256':r.checkpoint.digest(Path(m.__file__).read_bytes()),
                       'source_sha256':r.checkpoint.SOURCE_HASHES['my_fastread_knee.py']}}
    a={'status':'PASS','parent_complete':True,'contract_sha256':r.checkpoint.digest(r.checkpoint.canonical(c)),
       'version':r.VERSION,'studies':67,'test_studies':3,'train_studies':64,'checkpoint_sha256':r.CHECKPOINT_SHA256,
       'prediction_fallbacks':0,'input_counts':dict.fromkeys(m.COUNTERS,0),'backend_flags':r.FLAGS,
       'stress':{'status':'PASS','cases':dict.fromkeys(r.STRESS_CASES,'PASS'),'uses_labels':False,'uses_raw_data':False},
       'tensor_max_error':0,'probability_public_max_error':0.,'repeat_max_error':0.,'elapsed_seconds':12.,
       'state_compatibility':{'strict_load':True,'loaded_tensors_equal':True,'actual_schema_sha256':r.STATE_SCHEMA},
       'visible_csv_sha256':'1'*64,'prediction_order_sha256':'2'*64,'strict_coverage':{'PASS':66,'ineligible_malformed_input':1},
       'sampling':{'seed':20261008,'uses_labels':False,'selection':'SHA256(seed:StudyInstanceUID) order','ordered_ids_sha256':'a'*64},
       'environment':{'python_major_minor':[3,12],'packages':dict(r.PACKAGES,scipy='first-observed-test-version')},'codec_hashes':c['codecs'],
       'peak_cuda_allocated_bytes':1,'peak_cuda_reserved_bytes':1,'peak_host_rss_bytes':1,'disk_free_bytes':1}
    return c,a


class ContractTests(unittest.TestCase):
    def test_requires_parent_completion_full_coverage_exact_recipe_environment_and_stress(self):
        c,a=valid_evidence();r.require_rehearsal(c,a)
        changes=[('parent_complete',False),('studies',66),('checkpoint_sha256','0'*64),('elapsed_seconds',32401),
                 ('tensor_max_error',1),('repeat_max_error',1.1e-6),('probability_public_max_error',float('nan')),
                 ('prediction_fallbacks',1),('visible_csv_sha256','bad'),('strict_coverage',{'PASS':0,'ineligible_malformed_input':67}),
                 ('stress',{}),('input_counts',{}),('state_compatibility',{}),('backend_flags',dict(r.FLAGS,cudnn_benchmark=True))]
        for key,value in changes:
            changed=copy.deepcopy(a);changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):r.require_rehearsal(c,changed)
        old=copy.deepcopy(c);old['input_policy']['version']='mavit-active-reader-v1'
        changed=copy.deepcopy(a);changed['contract_sha256']=r.checkpoint.digest(r.checkpoint.canonical(old))
        with self.assertRaisesRegex(ValueError,'input policy'):r.require_rehearsal(old,changed)
        a['parent_complete']=False;r.require_rehearsal(c,a,parent_complete=False)
        a['sampling']['uses_labels']=True
        with self.assertRaises(ValueError):r.require_rehearsal(c,a,parent_complete=False)

    def test_metadata_order_schema_and_no_label_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);data=rows('z')+rows('a');data[1]['Fluid_Sensitive']=data[1]['Fat_Suppression']=1
            with (p/'train_series.csv').open('w') as h:
                w=csv.DictWriter(h,fieldnames=r.SERIES_COLUMNS);w.writeheader();w.writerows(data)
            (p/'train.csv').write_text('NEVER READ LABELS');original=Path.open
            def no_labels(path,*args,**kw):
                self.assertNotEqual(path.name,'train.csv');return original(path,*args,**kw)
            with patch.object(Path,'open',no_labels):loaded=r.read_series(p,'train')
            self.assertEqual([x['SeriesInstanceUID'] for x in loaded['synthetic']],['z','a'])
            for bad in ('duplicate','unsafe_id','view','schema'):
                v=copy.deepcopy(data);fields=r.SERIES_COLUMNS.copy()
                if bad=='duplicate':v[1]['SeriesInstanceUID']='z'
                if bad=='unsafe_id':v[0]['StudyInstanceUID']='../bad'
                if bad=='view':v[0]['Fat_Suppression']=1
                if bad=='schema':fields=fields[::-1]
                with (p/'train_series.csv').open('w') as h:
                    w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(v)
                with self.subTest(bad=bad),self.assertRaises(ValueError):r.read_series(p,'train')

    def test_parent_timeout_and_success_both_clean_descendants(self):
        entry=script('rehearse_mavit_entry')
        with tempfile.TemporaryDirectory() as tmp:
            for mode in ('timeout','success'):
                marker=Path(tmp)/mode
                child=f'import time;from pathlib import Path;time.sleep(.6);Path({str(marker)!r}).write_text("leaked")'
                parent=f'import subprocess,sys,time;subprocess.Popen([sys.executable,"-c",{child!r}]);'+('time.sleep(10)' if mode=='timeout' else '')
                if mode=='timeout':
                    with self.assertRaises(subprocess.TimeoutExpired):entry.run_process([sys.executable,'-c',parent],time.monotonic()+.2)
                else:entry.run_process([sys.executable,'-c',parent],time.monotonic()+5)
                time.sleep(.7);self.assertFalse(marker.exists())
            with patch.object(subprocess,'Popen') as popen:
                with self.assertRaises(TimeoutError):entry.run_process(['unused'],time.monotonic()-1)
                popen.assert_not_called()


@unittest.skipUnless(importlib.util.find_spec('torch'),'optional torch')
class InferenceContractTests(unittest.TestCase):
    def test_infer_only_batch1_sigmoid_inside_fp16_context_and_no_device_fallback(self):
        import numpy as np
        import torch
        from contextlib import contextmanager
        active=[];devices=[]
        class Model(torch.nn.Module):
            def forward(self,*args):raise AssertionError('forward augmentation path forbidden')
            def infer(self,x,mask,pos):
                if not active or x.dtype!=torch.uint8 or mask.dtype!=torch.bool or pos.dtype!=torch.float32:raise AssertionError('Input/autocast recipe differs')
                if tuple(x.shape)!=(1,6,8,3,8,8):raise AssertionError('Batch/slot recipe differs')
                return torch.linspace(-2,2,12,dtype=torch.float16).reshape(1,12)
        @contextmanager
        def autocast(device,*,dtype):
            self.assertEqual((device,dtype),('cuda',torch.float16));active.append(True)
            try:yield
            finally:active.pop()
        original_to=torch.Tensor.to;original_sigmoid=torch.sigmoid
        def fake_to(value,*args,**kw):
            if args==('cuda',):devices.append('cuda');return value  # local call-contract check only; no CUDA execution claim
            return original_to(value,*args,**kw)
        def sigmoid(value):self.assertTrue(active);return original_sigmoid(value)
        arrays=(np.zeros((6,8,3,8,8),np.uint8),np.ones(6,bool),np.zeros((6,8),np.float32))
        model=Model().eval()
        with patch.object(torch.Tensor,'to',fake_to),patch.object(torch.amp,'autocast',autocast),patch.object(torch,'sigmoid',sigmoid):
            actual=r.predict(model,arrays)
        self.assertEqual(devices,['cuda']*3);np.testing.assert_array_equal(actual,original_sigmoid(torch.linspace(-2,2,12,dtype=torch.float16)).float().numpy())
        model.train()
        with self.assertRaises(ValueError):r.predict(model,arrays)

    def test_checkpoint_hash_precedes_deserialization_and_unavailable_cuda_fails(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/r.checkpoint.CHECKPOINT).write_bytes(b'not approved weights')
            with patch.object(torch,'load') as load:
                with self.assertRaises(ValueError):r.load_model(p)
                load.assert_not_called()
            c={'source_revision':'synthetic'}
            with patch.object(torch.cuda,'is_available',return_value=False),patch.object(r,'load_model') as load:
                with self.assertRaisesRegex(RuntimeError,'aggregate'):r.rehearse(p,p,c,p/'out',started=time.monotonic())
                load.assert_not_called()
            report=json.loads((p/'out/mavit-audit.json').read_text());self.assertEqual(report['status'],'FAIL');self.assertFalse(report['parent_complete'])
            self.assertEqual(report['studies'],0);self.assertNotIn(str(p),json.dumps(report))


@unittest.skipUnless(SOURCE and SOURCE.is_dir() and importlib.util.find_spec('pydicom') and importlib.util.find_spec('scipy'),'pinned reader/pydicom/scipy fixtures')
class ReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.source=(SOURCE/'my_fastread_knee.py').read_text()

    def module(self,production=True):return m.definitions(self.source,production=production)

    def test_inverse_ast_proves_scientific_expressions_unchanged_and_comparator_independent(self):
        original=ast.parse(m.extract(self.source))
        for production in (False,True):
            tree=ast.parse(m.transform(m.extract(self.source),production=production));counts={'catch':0,'check':0,'view':0,'workers':0}
            class Undo(ast.NodeTransformer):
                def visit_Assign(self,node):
                    if ast.unparse(node.value)=='_mavit_collect(ex, _one, jobs)':
                        counts['workers']+=1;node.value=ast.parse('list(ex.map(_one, jobs))',mode='eval').body
                    return node
                def visit_Try(self,node):
                    if len(node.handlers)==1 and node.handlers[0].name=='_mavit_view_error':counts['view']+=1;return node.body
                    return self.generic_visit(node)
                def visit_ExceptHandler(self,node):
                    if node.name=='_mavit_error':
                        self.assert_prefix=node.body.pop(0);node.name=None;counts['catch']+=1
                    return node
                def visit_Expr(self,node):
                    if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id in ('_mavit_records','_mavit_spacing','_mavit_percentiles'):
                        counts['check']+=1;return None
                    return node
            restored=Undo().visit(tree)
            self.assertEqual(ast.dump(restored),ast.dump(original));self.assertEqual(counts,{'catch':5,'check':3 if production else 0,'view':int(production),'workers':1})
        with self.assertRaises(ValueError):m.extract(self.source+'\n#changed')
        self.assertNotIn('make_reader_v2',m.extract(self.source));self.assertNotIn('build_study_batched',m.extract(self.source))

    def test_actual_dicom_raw_lut_equals_pydicom_signed_highbits_rescale_mono_and_implicit(self):
        import numpy as np
        mod=self.module();_,px=mod._make_pyd()
        for implicit,signed,mono in [(False,False,False),(True,True,False),(False,True,True),(True,False,True)]:
            raw=np.arange(63,dtype=np.uint16).reshape(7,9)*61
            if signed:raw=(raw+2050)&4095
            raw|=0xa000  # unused high bits must follow pinned pydicom3 correction semantics
            data=dicom(raw,implicit=implicit,signed=signed,mono1=mono)
            decoded=mod.fast_raw(data,mod._scan_full(data));self.assertIsInstance(decoded,mod.RawSlice)
            np.testing.assert_array_equal(decoded.to_float(),px(data))
            other=mod.fast_raw(dicom(raw//2,implicit=implicit,signed=signed,mono1=mono),mod._scan_full(dicom(raw//2,implicit=implicit,signed=signed,mono1=mono)))
            lo,hi,ctx=mod._knee_percentile_context([decoded,other]);expect=np.percentile(np.concatenate([decoded.to_float().ravel(),other.to_float().ravel()]),[1,99])
            np.testing.assert_allclose([lo,hi],expect,atol=1e-10,rtol=0)
            for item in (decoded,other):
                baseline=(np.clip((item.to_float()-lo)/max(hi-lo,1e-6),0,1)*255).astype(np.uint8)
                np.testing.assert_array_equal(mod._knee_to_u8(item,lo,hi,ctx),baseline)
        data=dicom(np.arange(63).reshape(7,9),bits=8,stored=8)
        self.assertIsNone(mod.fast_raw(data,mod._scan_full(data)))
        np.testing.assert_array_equal(mod._Ctx(mod.Cfg(fast=True,lut=True)).px(data,mod._scan_full(data))[0],px(data))

    def test_real_dicom_order_missing_geometry_laterality_and_public_tensor_parity(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);directory=root/'synthetic/series';directory.mkdir(parents=True)
            base=np.arange(64).reshape(8,8)
            for laterality,plane,geometry in [('L','Sagittal',True),('R','Sagittal',True),('R','Axial',True),('','Coronal',False)]:
                mats=[]
                # Filename order differs from geometry/InstanceNumber; no random ordering.
                for file,depth in enumerate([3,0,4,1,2]):
                    raw=base+100*depth;data=dicom(raw,position=depth,instance=depth,geometry=geometry,laterality=laterality,slope=1,intercept=0)
                    (directory/f'{file}.dcm').write_bytes(data);mats.append((depth,raw.astype(np.float32)))
                ordered=[v for _,v in sorted(mats)];lo,hi=np.percentile(np.stack(ordered),[1,99]);u8=[(np.clip((v-lo)/(hi-lo),0,1)*255).astype(np.uint8) for v in ordered]
                prod=self.module();public=self.module(False);pr=prod.make_knee_dataset_reader(out_size=8,io_threads=2);rr=public.make_knee_dataset_reader(out_size=8,io_threads=2)
                actual,count=m.prepare(prod,pr,'synthetic',rows(plane=plane),root,production=True)
                expected,_=m.prepare(public,rr,'synthetic',rows(plane=plane),root,production=False)
                for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)
                slot=prod.KNEE_VIEW[(plane,0,0)];centers=prod._knee_centers(5)
                self.assertEqual(actual[1].sum(),1);np.testing.assert_array_equal(actual[2][slot],(centers/5).astype(np.float32))
                indices=np.stack([centers-1,centers,centers+1],1)
                if laterality=='R' and plane=='Sagittal':indices=4-indices
                image=np.stack([[u8[i] for i in sample] for sample in indices])
                if laterality=='R' and plane!='Sagittal':image=image[...,::-1]
                np.testing.assert_array_equal(actual[0][slot],image)
                self.assertEqual(count['decoded_files'],5)
                if not geometry:self.assertGreater(count['header_geometry_default_frames'],0);self.assertEqual(count['laterality_unknown_views'],1)

    def test_uint8_before_zoom_square_bypass_nonsquare_crop_boundaries_and_short_centers(self):
        import numpy as np
        import scipy.ndimage as ndi
        mod=self.module();image=(np.arange(40*60).reshape(40,60)%256).astype(np.uint8)
        for size,spacing in [(20,6.5),(20,130/16),(20,130/40)]:
            expected=image[10:30,20:40] if spacing==6.5 else ndi.zoom(image,(20/40,20/60),order=1)
            np.testing.assert_array_equal(mod._knee_crop_resize_u8(image,spacing,size),expected)
        square=image[:40,:40];np.testing.assert_array_equal(mod._knee_crop_resize_u8(square,6.5,20),ndi.zoom(square,(.5,.5),order=1))
        self.assertEqual(mod._knee_centers(3).tolist(),[1]*8)
        with self.assertRaises(ValueError):mod._knee_centers(2)
        # Exact public geometric fallback and first-explicit precedence.
        records=[{'data':b'','info':None}];h=dict(image_laterality='',laterality='',iop=[1,0,0,0,1,0],ipp=[-100,0,0],ps=[1,1],rows=8,cols=8)
        with patch.object(mod,'_knee_header_values',return_value=h):self.assertEqual(mod._knee_get_laterality(records),'R')
        with patch.object(mod,'_knee_header_values',return_value=dict(h,image_laterality='L')):self.assertEqual(mod._knee_get_laterality(records),'L')

    def test_real_unreadable_first_series_mask_no_retry_black_and_hook_restoration(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=np.arange(64).reshape(8,8)
            for name in ('bad','good'):
                d=root/'synthetic'/name;d.mkdir(parents=True)
                for i in range(4):(d/f'{i}.dcm').write_bytes(dicom(base+i,position=i))
            (root/'synthetic/bad/1.dcm').write_bytes(b'not dicom')
            mod=self.module();reader=mod.make_knee_dataset_reader(out_size=8,io_threads=2);original=mod._mavit_caught
            selected=rows('bad')+rows('good')+rows('good','Axial')
            arr,c=m.prepare(mod,reader,'synthetic',selected,root,production=True)
            self.assertEqual(arr[1].tolist(),[False,False,True,False,False,False]);self.assertEqual(c['masked_views'],1)
            self.assertEqual(c['pixel_decode_failures'],1);self.assertIs(mod._mavit_caught,original);self.assertNotIn('_mavit_mask',vars(mod))
            text=json.dumps(c);self.assertNotIn('synthetic',text);self.assertNotIn(str(root),text)
            # Unknown errors inside the actual original pixel catch must escape, not become None/masked.
            with patch.object(reader.ctx,'px_b',side_effect=AssertionError('source bug')),patch.object(mod,'fast_raw',return_value=None):
                with self.assertRaises(AssertionError):m.prepare(mod,reader,'synthetic',rows('good'),root,production=True)
            self.assertIs(mod._mavit_caught,original)

    def test_six_slots_unique_modal_shape_mixed_decoder_and_resource_counts(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);prod=self.module();public=self.module(False);metadata=[]
            for slot,(plane,fluid,fat) in enumerate(prod.KNEE_VIEW_KEYS):
                d=root/'synthetic'/f'v{slot}';d.mkdir(parents=True)
                metadata.append(dict(StudyInstanceUID='synthetic',SeriesInstanceUID=f'v{slot}',Anatomical_Plane=plane,Fluid_Sensitive=fluid,Fat_Suppression=fat))
                for i in range(4):
                    # One smaller outlier; the remaining 3 include an8bit pydicom fallback.
                    shape=(7,7) if i==3 else (8,8);raw=np.arange(np.prod(shape)).reshape(shape)+slot+i
                    (d/f'{i}.dcm').write_bytes(dicom(raw,position=i,bits=8 if i==1 else 16,stored=8 if i==1 else 12))
            reader=prod.make_knee_dataset_reader(out_size=8,io_threads=1);other=public.make_knee_dataset_reader(out_size=8,io_threads=1)
            actual,c=m.prepare(prod,reader,'synthetic',metadata,root,production=True)
            expected,_=m.prepare(public,other,'synthetic',metadata,root,production=False)
            for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)
            self.assertTrue(actual[1].all());self.assertEqual(c['mixed_shape_views'],6);self.assertEqual(c['shape_discarded_frames'],6)
            self.assertEqual(c['pixel_fallbacks'],6);np.testing.assert_array_equal(actual[2],np.full((6,8),1/3,dtype=np.float32))
            for error in (MemoryError,TimeoutError):
                original=prod._mavit_caught
                with patch.object(reader.ctx,'px_b',side_effect=error()),patch.object(prod,'fast_raw',return_value=None):
                    with self.assertRaises(error) as caught:m.prepare(prod,reader,'synthetic',metadata,root,production=True)
                self.assertEqual(caught.exception.mavit_counts['resource_errors'],1);self.assertIs(prod._mavit_caught,original)

    def test_all_five_actual_catches_propagate_fatal_errors_in_both_paths(self):
        import numpy as np
        import pydicom
        data=dicom(np.arange(64).reshape(8,8));record=[{'data':data,'info':None}]
        for production in (False,True):
            for error in (MemoryError,TimeoutError,lambda:OSError(errno.EMFILE,'too many files')):
                mod=self.module(production);hdr,_=mod._make_pyd();ctx=mod._Ctx(mod.Cfg(fast=True,lut=True));kind=type(error())
                with self.subTest(production=production,error=kind.__name__):
                    with patch.object(pydicom,'dcmread',side_effect=error()):
                        with self.assertRaises(kind):hdr('private',data)
                    # This is inside the whole header catch after dcmread; not just a callback guard.
                    with patch.object(mod.np,'cross',side_effect=error()):
                        with self.assertRaises(kind):hdr('private',data)
                    class Version:
                        def split(self,*args):raise error()
                    with patch.object(pydicom,'__version__',Version()):
                        with self.assertRaises(kind):mod._pydicom_major()
                    with patch.object(ctx,'px_b',side_effect=error()):
                        with self.assertRaises(kind):ctx.px(data,None)
                    with patch.object(mod,'_knee_header_values',side_effect=error()):
                        with self.assertRaises(kind):mod._knee_get_laterality(record)
                    values=dict(image_laterality='',laterality='',iop=[1,0,0,0,1,0],ipp=[0,0,0],ps=[1,1],rows=8,cols=8)
                    with patch.object(mod,'_knee_header_values',return_value=values),patch.object(mod.np,'asarray',side_effect=error()):
                        with self.assertRaises(kind):mod._knee_get_laterality(record)
        CvError=type('error',(Exception,),{'__module__':'cv2','code':-4})
        self.assertTrue(m.fatal(CvError()));self.assertFalse(m.fatal(ValueError()))

    def test_concurrent_early_declared_error_cannot_hide_later_fatal_or_unknown(self):
        import numpy as np
        import threading
        from concurrent.futures import ThreadPoolExecutor
        for production in (False,True):
            for error_type in (MemoryError,TimeoutError,AssertionError):
                with self.subTest(production=production,error=error_type.__name__),tempfile.TemporaryDirectory() as tmp:
                    root=Path(tmp);base=np.arange(64).reshape(8,8)
                    for name in ('bad','good'):
                        directory=root/'synthetic'/name;directory.mkdir(parents=True)
                        for index in range(4):(directory/f'{index}.dcm').write_bytes(dicom(base+index,position=index))
                    mod=self.module(production);reader=mod.make_knee_dataset_reader(out_size=8,io_threads=32)
                    original_read=mod._read_whole;barrier=threading.Barrier(4);finished=[];futures=[];executors=[]
                    class ObservedExecutor(ThreadPoolExecutor):
                        def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);executors.append(self)
                        def submit(self,*args,**kwargs):
                            future=super().submit(*args,**kwargs);futures.append(future);return future
                    def read(path):
                        if '/bad/' not in path:return original_read(path)
                        index=int(Path(path).stem);barrier.wait(timeout=3)
                        try:
                            if index==0:raise FileNotFoundError('Synthetic first file disappeared')
                            if index==1:raise error_type('Synthetic later worker failure')
                            return original_read(path)
                        finally:finished.append(index)
                    with patch.object(mod,'_read_whole',side_effect=read),patch.object(mod,'ThreadPoolExecutor',ObservedExecutor):
                        with self.assertRaises(error_type) as caught:
                            m.prepare(mod,reader,'synthetic',rows('bad')+rows('good','Axial'),root,production=production)
                    self.assertEqual(sorted(finished),[0,1,2,3]);self.assertTrue(all(f.done() for f in futures))
                    self.assertTrue(all(not thread.is_alive() for ex in executors for thread in ex._threads))
                    self.assertEqual(caught.exception.mavit_counts['masked_views'],0)
                    self.assertEqual(caught.exception.mavit_counts['resource_errors'],int(error_type!=AssertionError))
        # Ordinary malformed errors may still mask; successful ordered results stay exact.
        with ThreadPoolExecutor(max_workers=4) as executor:
            self.assertEqual(m.collect_workers(executor,lambda i:i*i,[3,0,2,1]),[9,0,4,1])
        with ThreadPoolExecutor(max_workers=4) as executor:
            def one(i):
                if i==0:raise FileNotFoundError()
                return i
            with self.assertRaises(FileNotFoundError):m.collect_workers(executor,one,[0,1,2,3])
        class Interrupted:
            def result(self):raise TimeoutError('main thread alarm')
            def done(self):return False
        fake=types.SimpleNamespace(submit=lambda *_:Interrupted())
        with self.assertRaises(TimeoutError):m.collect_workers(fake,lambda _:None,[0])

    def test_runtime_production_stress_and_descriptor_close(self):
        import numpy as np
        result=r.stress_inputs(self.source);self.assertEqual(result['cases'],dict.fromkeys(r.STRESS_CASES,'PASS'))
        mod=self.module()
        with tempfile.NamedTemporaryFile() as file:
            file.write(b'nonempty');file.flush()
            with patch.object(mod,'_pread',side_effect=TimeoutError('deadline')),patch.object(os,'close',wraps=os.close) as close:
                with self.assertRaises(TimeoutError):mod._read_whole(file.name)
                self.assertEqual(close.call_count,1)

    def test_authentic_cpu_binding_build_import_and_tamper(self):
        b=script('build_mavit_rehearsal')
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)/'build';receipt=b.build(d,'synthetic-mavit67',source_dir=SOURCE,cpu_report=CPU)
            meta=json.loads((d/'kernel-metadata.json').read_text());contract=json.loads((d/'mavit-contract.json').read_text())
            self.assertTrue(meta['is_private']);self.assertFalse(meta['enable_internet']);self.assertTrue(meta['enable_gpu'])
            self.assertEqual(meta['dataset_sources'],[r.checkpoint.DATASET,b.CONVNEXT_DATASET]);self.assertEqual(meta['competition_sources'],['rsna-knee-abnormality-detection'])
            self.assertEqual(meta['model_sources'],[]);self.assertEqual(meta['kernel_sources'],[])
            self.assertLess(receipt['notebook_bytes'],900000);self.assertEqual(contract['input_policy']['fatal_catches'],m.CATCHES)
            code=''.join(json.loads((d/'notebook.ipynb').read_text())['cells'][-1]['source']);compile(code,'<mavit-notebook>','exec')
            own=ast.literal_eval(next(n.value for n in ast.parse(code).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='sources' for t in n.targets)))
            bundle=Path(tmp)/'bundle'
            for n,v in own.items():p=bundle/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(v)
            result=subprocess.run([sys.executable,'-c','import sys;import knee.mavit_rehearsal;assert "torch" not in sys.modules'],cwd=tmp,env=dict(os.environ,PYTHONPATH=str(bundle)),capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertNotIn('train_golden58.csv',code);self.assertNotIn('train.csv',code)
            self.assertLess(code.index('signal.setitimer'),code.index('sources='))
            with self.assertRaises(ValueError):b.build(d,'synthetic-mavit67',source_dir=SOURCE,cpu_report=CPU)
            bad=Path(tmp)/'changed.json';raw=json.loads(CPU.read_text());raw['elapsed_seconds']=0;bad.write_text(json.dumps(raw))
            with self.assertRaises(ValueError):b.build(Path(tmp)/'bad','bad',source_dir=SOURCE,cpu_report=bad)


if __name__=='__main__':unittest.main()
