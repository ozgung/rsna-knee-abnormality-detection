import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SOURCE=Path(os.environ.get('KNEE_S6_TEST_SOURCE','/Users/ozgun/Dev/rsna-knee-abnormality-detection/outputs/kaggle/knee-s6-dread-b0-rehearsal-20261013-result-v1/s6-recovery-entry-project/source'))
try:
    import numpy as np
    import cv2
    import torch
except ImportError:
    np=None

from knee.native_joint_roi_pilot import joint_view,qualify_frame_references,read_pair,paired_triplets


@unittest.skipIf(np is None or not SOURCE.is_dir(),'numpy/cv2/torch and pinned S6 fixture required')
class JointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0,str(SOURCE))
        from training.input_provenance import view_geometry
        import knee_anchor_bank_v2 as KB
        cls.view=staticmethod(view_geometry);cls.project=staticmethod(KB.project)

    def fixture(self,spacing=(.6,.3),shape=(3,240,420),reverse=False):
        sy,sx=spacing;sg=-1 if reverse else 1
        aff=np.array([[[sg*sx,0.,0.],[0.,sy,0.],[0.,0.,z*3.]] for z in range(shape[0])])
        s={'SeriesInstanceUID':'s','native_shape':list(shape),'spacing_mm':[list(spacing)]*shape[0]}
        g={'affines_col_row_origin_lps_mm':aff.tolist(),'quality':{'aligned':True},'frame_of_reference_uid':'frame'}
        point=(aff[1]@np.array([(shape[2]-1)/2,(shape[1]-1)/2,1])).tolist()
        return s,g,{'joint_lps':point,'why':[]}

    def test_anisotropic_reversed_projection_fixed100(self):
        for reverse in (False,True):
            s,g,a=self.fixture(reverse=reverse)
            center,r=joint_view(s,g,a,self.view,self.project)
            np.testing.assert_allclose(center,(.5,.5),atol=1e-10);self.assertEqual(r['nearest_native_slice'],1)
            self.assertLess(r['projection_error_mm'],1e-6)
            self.assertLessEqual(abs(r['actual_field_mm'][0]-100),.3)
            self.assertLessEqual(abs(r['actual_field_mm'][1]-100),.15)
            self.assertEqual(r['out_of_field_fraction'],0)
            self.assertGreater(r['native_support_pixels'],50000)

    def test_oblique_projection(self):
        s,g,a=self.fixture();theta=.35
        rotate=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1.]])
        g['affines_col_row_origin_lps_mm']=(rotate@np.asarray(g['affines_col_row_origin_lps_mm'])).tolist()
        a['joint_lps']=(rotate@np.asarray(a['joint_lps'])).tolist()
        _,r=joint_view(s,g,a,self.view,self.project)
        self.assertLess(r['projection_error_mm'],1e-6)

    def test_boundary_no_source_clipping_or_padding(self):
        s,g,a=self.fixture();a['joint_lps']=[0,0,3]
        c,r=joint_view(s,g,a,self.view,self.project)
        self.assertIsNone(c);self.assertEqual(r['reason'],'out_of_bounds_100mm')
        self.assertGreater(r['out_of_field_fraction'],0)
        s,g,a=self.fixture(shape=(3,80,100));c,r=joint_view(s,g,a,self.view,self.project)
        self.assertIsNone(c);self.assertEqual(r['reason'],'out_of_bounds_100mm')

    def test_missing_anchor_geometry_alignment_fallback(self):
        s,g,a=self.fixture()
        for geo,anchor,reason in ((g,{**a,'joint_lps':None},'missing_anchor'),(None,a,'missing_geometry'),({**g,'quality':{'aligned':False}},a,'unaligned_geometry')):
            _,r=joint_view(s,geo,anchor,self.view,self.project);self.assertEqual(r['reason'],reason)

    def test_contract_errors_are_not_fallback(self):
        s,g,a=self.fixture()
        with self.assertRaisesRegex(ValueError,'swallowed localizer'):
            joint_view(s,g,{**a,'why':['error:bad']},self.view,self.project)
        s['spacing_mm'][1]=[.8,.3]
        with self.assertRaisesRegex(ValueError,'Changing in-plane'):
            joint_view(s,g,a,self.view,self.project)
        s,g,a=self.fixture();g['affines_col_row_origin_lps_mm'][0][0][0]=1
        with self.assertRaisesRegex(ValueError,'affine/spacing'):
            joint_view(s,g,a,self.view,self.project)

    def test_incompatible_frame_reference_is_recorded_global_fallback(self):
        _,g,a=self.fixture();self.assertIsNone(qualify_frame_references({'s':g},a))
        for geos in ({'s':{**g,'frame_of_reference_uid':None}}, {'s':g,'other':{**g,'frame_of_reference_uid':'other'}}):
            self.assertEqual(qualify_frame_references(geos,a),'incompatible_frame_of_reference')
        self.assertIsNone(qualify_frame_references({'s':None},{**a,'joint_lps':None}))

    def test_actual_pinned_source_slab_intensity_and_fallback_parity(self):
        import px_cache as PX
        s,g,a=self.fixture(spacing=(1.,1.),shape=(9,140,160))
        # Descending physical positions and0.8mm depth exercise source stable
        # ordering + thin-stack averaging, rather than just a resized toy crop.
        s.update(positions=list(np.arange(8,-1,-1)*.8),plane='Sagittal',side='R',slot=0,
                 fat_suppression=1,fluid_sensitive=1,intensity={'display_window_uint16':[100,60000]})
        with tempfile.TemporaryDirectory() as d:
            cache=Path(d);volume=np.arange(9*140*160,dtype=np.uint16).reshape(9,140,160)
            np.save(cache/'s.npy',volume);s['path']=str(cache/'s.npy')
            path=Path(PX.study_json(d,'uid'));path.parent.mkdir();path.write_text(json.dumps({'StudyInstanceUID':'uid','series':[s]}))
            selector=PX.series_views
            with patch.object(PX,'series_views',return_value=[(0,None,(.5,.5))]):
                original=PX.StudySlices(d,[],384,'v1').build('uid')
            global_item,local,receipts=read_pair(cache,'uid',{'s':g},a)
            self.assertIs(PX.series_views,selector)
            self.assertTrue(torch.equal(original['px'],global_item['px']))
            self.assertLess(global_item['blocks'][0]['n_slices'],9)
            self.assertEqual(receipts['s']['reason'],None)
            control,treatment,plan=paired_triplets(global_item,local)
            self.assertEqual(control.shape,(24,3,384,384))
            np.testing.assert_array_equal(control[:12],treatment[:12])
            self.assertTrue(all(len(set(p['indices']))<=3 for p in plan))
            fallback_global,fallback,rr=read_pair(cache,'uid',{'s':g},{**a,'joint_lps':None})
            self.assertTrue(torch.equal(fallback_global['px'],fallback['px']))
            c,t,_=paired_triplets(fallback_global,fallback);np.testing.assert_array_equal(c,t)
            self.assertEqual(rr['s']['reason'],'missing_anchor')
            frame_global,frame_fallback,frame_receipts=read_pair(cache,'uid',{'s':{**g,'frame_of_reference_uid':None}},a)
            self.assertTrue(torch.equal(frame_global['px'],frame_fallback['px']))
            self.assertEqual(frame_receipts['s']['reason'],'incompatible_frame_of_reference')
            # Shared train/inference adapter is deterministic and preserves raw
            # source arrays. This is not real-DICOM reader qualification.
            again=read_pair(cache,'uid',{'s':g},a)
            self.assertTrue(torch.equal(local['px'],again[1]['px']))
            np.testing.assert_array_equal(np.load(cache/'s.npy'),volume)

    def test_selector_restored_on_source_exception(self):
        import px_cache as PX
        s,g,a=self.fixture()
        with tempfile.TemporaryDirectory() as d:
            s['path']=str(Path(d)/'s.npy');np.save(s['path'],np.zeros(s['native_shape'],np.uint16))
            p=Path(PX.study_json(d,'uid'));p.parent.mkdir();p.write_text(json.dumps({'StudyInstanceUID':'uid','series':[s]}))
            original=PX.series_views
            with patch.object(PX.StudySlices,'build',side_effect=RuntimeError('source failure')):
                with self.assertRaises(RuntimeError):read_pair(d,'uid',{'s':g},a)
            self.assertIs(PX.series_views,original)


class BuildTests(unittest.TestCase):
    def test_build_tamper(self):
        from scripts.build_native_joint_roi_pilot import verify_build,digest_text
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'notebook.ipynb';p.write_text('{}')
            (Path(d)/'build-receipt.json').write_text(json.dumps({'files':{p.name:digest_text('{}')}}))
            verify_build(d);p.write_text('changed')
            with self.assertRaises(ValueError):verify_build(d)
