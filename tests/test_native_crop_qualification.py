import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from knee.native_crop_qualification import geometry,physical_box,native_series,triplets,CropInvalid,comparison_crop,select_inputs
try:
    import numpy as np
    import cv2
except ImportError:
    np=None

@unittest.skipIf(np is None,'numpy/cv2 required')
class GeometryTests(unittest.TestCase):
    def test_anisotropic_oblique_translation_and_roundtrip(self):
        u=np.array([1.,1.,0.])/np.sqrt(2); v=np.array([0.,0.,1.]); normal=np.cross(u,v)
        g=geometry([*u,*v],[4.,7.,2.],[.6,.3],(400,600))
        moved=geometry([*u,*v],g[1]+normal*13+u*3+v*6,[.6,.3],(400,600))
        box=physical_box(moved,g[3],g[0],(400,600)); y0,y1,x0,x1=box
        self.assertAlmostEqual((y1-y0)*.6,90.,places=10); self.assertAlmostEqual((x1-x0)*.3,90.,places=10)
        point=moved[1]+u*((x0+x1)/2)*.3+v*((y0+y1)/2)*.6
        projected=g[3]+normal*np.dot(moved[1]-g[3],normal)
        self.assertLess(np.linalg.norm(point-projected),1e-6)
        self.assertAlmostEqual((y0+y1)/2,189.5); self.assertAlmostEqual((x0+x1)/2,289.5)
    def test_flipped_axes(self):
        g=geometry([-1,0,0,0,-1,0],[0,0,0],[1,2],(200,100))
        self.assertEqual(physical_box(g,g[3],g[0],(200,100)),(54.5,144.5,27.,72.))
    def test_invalid_and_unclipped_bounds(self):
        for meta in [(None,[0,0,0],[1,1]),([1,0,0,1,0,0],[0,0,0],[1,1]),([1,0,0,0,1,0],[0,0,float('nan')],[1,1])]:
            with self.assertRaises(CropInvalid): geometry(*meta,(200,200))
        g=geometry([1,0,0,0,1,0],[0,0,0],[1,1],(50,200))
        with self.assertRaisesRegex(CropInvalid,'out_of_bounds'): physical_box(g,g[3],g[0],(50,200))
        g=geometry([1,0,0,0,1,0],[0,0,0],[1,1],(200,200))
        with self.assertRaisesRegex(CropInvalid,'inconsistent_orientation'): physical_box(g,g[3],-g[0],(200,200))
    def test_lower_median_and_full_series_normalization(self):
        arrays=[np.arange(200*200,dtype=np.float32).reshape(200,200)+offset for offset in (0,100000)]
        meta=[([1,0,0,0,1,0],[0,0,i],[1,1]) for i in (0,5)]
        crops,boxes,reasons=native_series(arrays,meta)
        self.assertEqual(reasons,[None,None]); self.assertEqual(boxes[0],boxes[1]); self.assertLess(crops[0].max(),100)
        self.assertGreater(crops[1].min(),150)
        shifted=[meta[0],([1,0,0,0,1,0],[10,0,5],[1,1])]
        _,boxes,_=native_series(arrays,shifted)
        self.assertAlmostEqual(boxes[1][2],boxes[0][2]-10)
    def test_native_array_metadata_lengths_required(self):
        for arrays,metadata in (([],[]),([np.ones((200,200),np.float32)],[]),([],[None])):
            with self.assertRaisesRegex(ValueError,'equal nonzero length'): native_series(arrays,metadata)
    def test_missing_anchor_and_unsupported_pixels(self):
        arrays=[np.ones((200,200),np.float32)]*2
        meta=[(None,None,None),([1,0,0,0,1,0],[0,0,0],[1,1])]
        _,_,reasons=native_series(arrays,meta)
        self.assertEqual(reasons,['missing_geometry','missing_anchor_geometry'])
        with self.assertRaises(CropInvalid): native_series([np.full((2,2),np.nan)],[(None,None,None)])
    def test_exact_slot_and_missing_fallback_global_immutability(self):
        v=np.arange(96,dtype=np.uint8)[:,None,None]*np.ones((96,384,384),np.uint8)
        before=v.tobytes(); native=[np.full((320,320),123,np.uint8)]*96; reasons=[None]*96
        slots=[0]*26+[1]*22+[2]*18+[3]*12+[4]*18; reasons[5]='missing_geometry'
        windows,counts=triplets(v,native,reasons,slots)
        self.assertEqual(counts['cross_slot_triplet'],8); self.assertEqual(counts['missing_geometry'],3)
        np.testing.assert_array_equal(windows[24],v[24:27,32:352,32:352]); self.assertEqual(v.tobytes(),before)
        np.testing.assert_array_equal(windows[3],v[3:6,32:352,32:352]); self.assertEqual(windows[10].min(),123)
    def test_public384_same_box_mapping(self):
        frame=np.tile(np.arange(384,dtype=np.float32),(384,1))
        box=(54.5,144.5,54.5,144.5)
        result=comparison_crop(frame,(200,200),1.,box)
        self.assertAlmostEqual(float(result.mean()),191.5,places=5)
        self.assertIsNone(comparison_crop(frame,(200,200),1.,(-.5,90.,-.5,90.)))

class SamplingTests(unittest.TestCase):
    def test_train_gold_exclusions_deterministic(self):
        targets=['t']; records={f's{i}':[] for i in range(50)}; records['gold']=[]; records['quarantine']=[]
        with tempfile.TemporaryDirectory() as d:
            Path(d,'train.csv').write_text('StudyInstanceUID,t,Report\n'+''.join(f's{i},,secret\n' for i in range(50))+'gold,1,secret\nquarantine,,secret\n')
            folds={s:i%5 for i,s in enumerate(records)}
            with patch('knee.public_rich_cache.select_inputs',return_value=([],records,{})):
                a,_,_=select_inputs(d,targets,{'quarantined_study_ids':['quarantine']},folds)
                b,_,_=select_inputs(d,targets,{'quarantined_study_ids':['quarantine']},dict(reversed(list(folds.items()))))
                self.assertEqual(a,b); self.assertNotIn('gold',a); self.assertNotIn('quarantine',a)
                self.assertTrue(all(folds[s]!=0 for s in a))
                with self.assertRaises(ValueError): select_inputs(d,targets,{'quarantined_study_ids':['quarantine']},{})
    def test_immutable_build_tamper(self):
        from scripts.build_native_crop_qualification import verify_build
        from knee.reference import sha256
        with tempfile.TemporaryDirectory() as d:
            path=Path(d,'notebook.ipynb'); path.write_text('{}')
            Path(d,'build-receipt.json').write_text(json.dumps({'files':{'notebook.ipynb':sha256(path.read_bytes())}}))
            verify_build(d); path.write_text('tampered')
            with self.assertRaises(ValueError): verify_build(d)

@unittest.skipIf(np is None,'numpy/cv2 required')
class CaptureTests(unittest.TestCase):
    def test_synthetic_dicom_parity_masks_mirror_repeat(self):
        import os
        source=Path(os.environ.get('KNEE_TEST_PUBLIC_SOURCE','~/Dev/rsna-knee-abnormality-detection/outputs/research/2026-10-08/mirror-audit/rsna-knee-0949-anatomical-mirror.py'))
        if not source.exists(): self.skipTest('audited private source fixture unavailable')
        import pydicom
        from knee.reference import audit_source,load_definitions
        from knee.native_crop_qualification import capture_study,mirrored_triplets
        public,decoder=load_definitions(*audit_source(source.read_text()))
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); series=root/'synthetic'/'sag'; series.mkdir(parents=True)
            for i in range(4):
                meta=pydicom.Dataset(); meta.TransferSyntaxUID=pydicom.uid.ExplicitVRLittleEndian
                meta.MediaStorageSOPClassUID=pydicom.uid.MRImageStorage
                meta.MediaStorageSOPInstanceUID='1.2.826.0.1.3680043.8.498.'+str(i+1)
                ds=pydicom.FileDataset(None,{},file_meta=meta,preamble=b'\0'*128)
                ds.SOPClassUID=meta.MediaStorageSOPClassUID; ds.SOPInstanceUID=meta.MediaStorageSOPInstanceUID
                ds.Rows=240;ds.Columns=300;ds.SamplesPerPixel=1;ds.BitsAllocated=16;ds.BitsStored=12;ds.HighBit=11;ds.PixelRepresentation=0
                ds.PhotometricInterpretation='MONOCHROME2';ds.PixelSpacing=[.5,.4]
                ds.ImageOrientationPatient=[1,0,0,0,1,0];ds.ImagePositionPatient=[0,0,i];ds.InstanceNumber=i
                ds.PixelData=((np.arange(72000,dtype=np.uint16).reshape(240,300)+i*11)%3000).tobytes()
                pydicom.dcmwrite(series/f'{i}.dcm',ds,enforce_file_format=True)
            rows={'synthetic':[{'SeriesInstanceUID':'sag','Anatomical_Plane':'Sagittal','Fluid_Sensitive':1}]}
            a=capture_study(public,decoder,'synthetic',rows,root)
            b=capture_study(public,decoder,'synthetic',rows,root)
            for x,y in zip(a[:3],b[:3]): np.testing.assert_array_equal(x,y)
            # Public mask-selected K94 repeats centers in first26; no missing-slot windows invented.
            self.assertLessEqual(max(a[7]['window_centers']),24)
            self.assertEqual(a[2].shape,(94,3,320,320))
            mirrored=mirrored_triplets(public,a[2],a[7]['window_centers'])
            np.testing.assert_array_equal(mirrored,a[2][:,::-1])
            self.assertEqual(public.SLOTS,[('Sagittal',1,26),('Sagittal',0,22),('Coronal',1,18),('Coronal',0,12),('Axial',-1,18)])

    def test_partial_leading_and_trailing_exact_fallback_normalization(self):
        import torch
        from knee.reference import load_definitions,audit_source
        from knee.native_crop_qualification import mirrored_triplets
        source=Path('~/Dev/rsna-knee-abnormality-detection/outputs/research/2026-10-08/mirror-audit/rsna-knee-0949-anatomical-mirror.py')
        if not source.exists(): self.skipTest('audited source fixture unavailable')
        public,_=load_definitions(*audit_source(source.read_text()))
        volume=np.zeros((96,384,384),np.uint8)
        volume[48:66]=np.random.default_rng(5).integers(1,255,(18,384,384),dtype=np.uint8)
        mask=(volume.reshape(96,-1).any(1)).astype(np.uint8); centers=public._eval_centers(mask,96,94)
        slots=[0]*26+[1]*22+[2]*18+[3]*12+[4]*18
        windows,counts=triplets(volume,[None]*96,['missing_geometry']*96,slots,centers)
        self.assertGreaterEqual(min(centers),49); self.assertLessEqual(max(centers),64)
        self.assertEqual(counts,{'missing_geometry':94})
        mirrored=mirrored_triplets(public,windows,centers)
        np.testing.assert_array_equal(mirrored,windows[:,:,:,::-1])
        for mirror,triplet in ((False,windows),(True,mirrored)):
            actual=(torch.from_numpy(triplet.astype(np.float32)/255)-public._MEAN)/public._STD
            self.assertTrue(torch.equal(actual,public.eval_windows(volume,mask,94,320,mirror=mirror)))
        np.testing.assert_array_equal(mask[:48],0); np.testing.assert_array_equal(mask[66:],0)
    def test_actual_capture_rethrows_fatal_read_and_metadata(self):
        from knee.reference import audit_source,load_definitions
        from knee.native_crop_qualification import capture_study
        source=Path('~/Dev/rsna-knee-abnormality-detection/outputs/research/2026-10-08/mirror-audit/rsna-knee-0949-anatomical-mirror.py')
        if not source.exists(): self.skipTest('audited source fixture unavailable')
        public,decoder=load_definitions(*audit_source(source.read_text()))
        _,_,resize=public._make_reader()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); path=root/'synthetic'/'sag'/'0.dcm';path.parent.mkdir(parents=True);path.write_bytes(b'synthetic')
            rows={'synthetic':[{'SeriesInstanceUID':'sag','Anatomical_Plane':'Sagittal','Fluid_Sensitive':1}]}
            for location in ('pixels','metadata'):
                for error_type in (MemoryError,TimeoutError):
                    error=error_type('injected fatal')
                    def read(_):
                        if location=='pixels': raise error
                        return np.ones((240,300),np.float32)
                    reader=(lambda _ : ([(str(path),.5)],.5),read,resize)
                    with self.subTest(location=location,error=error_type.__name__),patch.object(public,'_make_reader',return_value=reader),patch('pydicom.dcmread',side_effect=error),patch('knee.mirror_input.prepare_study',side_effect=AssertionError('fatal must stop before fast decode')):
                        with self.assertRaises(error_type) as caught: capture_study(public,decoder,'synthetic',rows,root)
                        self.assertIs(caught.exception,error)

    def test_runtime_receipt_privacy_and_frozen_manifest(self):
        import copy
        import cv2
        from knee.reference import load_definitions,audit_source,canonical_bytes,sha256
        from knee.native_crop_qualification import run
        source=Path('~/Dev/rsna-knee-abnormality-detection/outputs/research/2026-10-08/mirror-audit/rsna-knee-0949-anatomical-mirror.py')
        if not source.exists(): self.skipTest('audited source fixture unavailable')
        public_source,decoder_source,manifest=audit_source(source.read_text())
        v=np.ones((96,384,384),np.uint8);m=np.ones(96,np.uint8);c=np.ones((94,3,320,320),np.uint8)
        counts={'window_centers':list(range(1,95)),'global_window_hashes':{'normal':'fixture','mirrored':'fixture'},'private_sources':[{'source_path':'sensitive-study/series/file'}]}
        output=(v,m,c,[None]*96,[None]*96,[0]*96,[(None,None)]*96,counts,{})
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'train.csv').write_text('synthetic table');(root/'train_series.csv').write_text('synthetic series')
            binding={'selected_order_sha256':sha256(canonical_bytes(['sensitive-study'])),'train_csv_sha256':sha256((root/'train.csv').read_bytes())}
            with patch('knee.native_crop_qualification.select_inputs',return_value=(['sensitive-study'],{},{})),patch('knee.native_crop_qualification.capture_study',side_effect=lambda *args:copy.deepcopy(output)):
                report=run(public_source,decoder_source,manifest,root,root/'result',{}, {},binding)
            self.assertEqual(report['status'],'PASS');self.assertEqual(report['anatomical_acceptance'],'UNVERIFIED')
            self.assertNotIn('sensitive-study',(root/'result'/'receipt.json').read_text())
            self.assertIn('sensitive-study',(root/'result'/'sample-private.json').read_text())
            images=list((root/'result').glob('*.png'));self.assertEqual(len(images),3)
            self.assertTrue(all('sensitive' not in p.name for p in images));self.assertEqual(cv2.imread(str(images[0])).shape,(960,960,3))
            with patch('knee.native_crop_qualification.select_inputs',return_value=(['different'],{},{})):
                with self.assertRaises(RuntimeError):run(public_source,decoder_source,manifest,root,root/'bad',{}, {},binding)
            self.assertEqual(json.loads((root/'bad'/'receipt.json').read_text())['status'],'FAIL')
            self.assertFalse((root/'bad'/'sample-private.json').exists())
