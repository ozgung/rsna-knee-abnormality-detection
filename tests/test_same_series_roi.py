"""Synthetic mechanics only; never import or execute extracted public source."""
import copy
import json
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

try:
    import numpy as np
except ImportError:
    np = None

from knee.same_series_roi import read_pair, series_candidate, triplet_plan, retained_support, source_render


def project(aff, point):
    normal = np.cross(aff[0, :, 0], aff[0, :, 1]); normal /= np.linalg.norm(normal)
    distances = (aff[:, :, 2] - point) @ normal
    z = int(np.argmin(abs(distances)))
    col, row = np.linalg.lstsq(aff[z, :, :2], point-aff[z, :, 2], rcond=None)[0]
    return z, float(col), float(row), float(abs(distances[z]))


def geometry(shape, spacing, size, crop_mm=None, center=(.5, .5)):
    # Independent synthetic geometry implementation. No public runtime imports.
    h, w = shape; sy, sx = spacing
    ch, cw = (h, w) if crop_mm is None else (round(crop_mm/sy), round(crop_mm/sx))
    y, x = round(center[0]*(h-1)-(ch-1)/2), round(center[1]*(w-1)-(cw-1)/2)
    scale = size/max(ch*sy, cw*sx)
    oh, ow = max(1, round(ch*sy*scale)), max(1, round(cw*sx*scale))
    return SimpleNamespace(crop_xyxy=(x,y,x+cw,y+ch),native_shape=shape,spacing=spacing,
                           resized_shape=(oh,ow),valid_xyxy=(0,0,ow,oh))


class Bank:
    def __init__(self): self.calls = []; self.mass = 700
    def candidate(self, s, g):
        self.calls.append((s, g))
        h,w = s['native_shape'][-2:]; aff=np.asarray(g['affines_col_row_origin_lps_mm'])
        return {'sid':s['SeriesInstanceUID'],'X':aff[1]@np.array([(w-1)/2,(h-1)/2,1]),'mass':self.mass}, 'ok'
    def record(self, *args): raise AssertionError('study anchor forbidden')
    def slab_prior(self, *args): raise AssertionError('cross-series prior forbidden')


def fixture(sid='s', plane='Sagittal', shift=0., spacing=(.6,.3), shape=(9,240,420), flip=False, frame='frame'):
    sy,sx=spacing; sign=-1 if flip else 1
    # Sagittal col along Y; row along Z; slice normal X, with optional reversal.
    aff=np.array([[[0.,0.,shift+z*3.],[sign*sx,0.,0.],[0.,sy,0.]] for z in range(shape[0])])
    s={'SeriesInstanceUID':sid,'plane':plane,'native_shape':list(shape),'spacing_mm':[list(spacing)]*shape[0],
       'positions':list(range(shape[0])),'side':'L'}
    g={'StudyInstanceUID':'u','SeriesInstanceUID':sid,'affines_col_row_origin_lps_mm':aff.tolist(),
       'quality':{'aligned':True},'frame_of_reference_uid':frame}
    return s,g


def renderer(cache, uid, study, centers):
    blocks=[];px=[];meta=[]
    for ordinal,s in enumerate(study['series']):
        off=len(px);n=s['native_shape'][0]
        blocks.append({'series':s['SeriesInstanceUID'],'plane':s['plane'],'n_slices':n,'views':[[0,off]]})
        for k in range(n):
            px.append(np.full((384,384),ordinal*20+k+(50 if centers[s['SeriesInstanceUID']] is not None else 0),np.uint8))
            row=np.zeros(12,np.float32);row[[0,9,10]]=[ordinal,k,n];row[6]=1 if centers[s['SeriesInstanceUID']] is None else 2
            meta.append(row)
    return {'px':np.stack(px),'meta':np.stack(meta),'blocks':blocks}


@unittest.skipIf(np is None, 'numpy required')
class SameSeriesTests(unittest.TestCase):
    def candidate(self,s,g,bank=None):
        return series_candidate('u',s,g,bank or Bank(),geometry,project)

    def pair(self, series_geos, bank=None, render=renderer):
        study={'StudyInstanceUID':'u','series':[s for s,g in series_geos]}
        return read_pair(None,'u',study,{s['SeriesInstanceUID']:g for s,g in series_geos},bank or Bank(),
                         view_geometry=geometry,project=project,renderer=render)

    def test_own_candidate_origin_mixed_and_missing_frames(self):
        fixtures=[fixture('a',shift=0,frame=None),fixture('b',shift=300,flip=True,frame='foreign'),fixture('c','Coronal',frame='third')]
        bank=Bank();c,t,plan,receipts,masks=self.pair(fixtures,bank)
        self.assertEqual([s['SeriesInstanceUID'] for s,g in bank.calls],['a','b'])
        self.assertTrue(all(s is fixtures[i][0] and g is fixtures[i][1] for i,(s,g) in enumerate(bank.calls)))
        self.assertEqual(receipts['a']['origin_series'],'a');self.assertEqual(receipts['b']['origin_series'],'b')
        self.assertAlmostEqual(receipts['b']['candidate_lps'][0]-receipts['a']['candidate_lps'][0],300)
        self.assertNotEqual(receipts['a']['geometry_sha256'],receipts['b']['geometry_sha256'])
        self.assertEqual(receipts['c']['reason'],'unsupported_plane_no_independent_segmenter')
        np.testing.assert_array_equal(c[:12],t[:12]);self.assertTrue(masks.all())
        self.assertEqual({p['series'] for p in plan[12:]},{'a','b'})
        self.assertEqual(c.shape,(24,3,384,384));self.assertTrue(np.isfinite(t).all())

    def test_geometry_and_candidate_binding_are_fatal(self):
        s,g=fixture();wrong={**g,'SeriesInstanceUID':'other'}
        bank=Bank()
        with self.assertRaisesRegex(ValueError,'Geometry origin'):self.candidate(s,wrong,bank)
        self.assertEqual(bank.calls,[])
        with self.assertRaisesRegex(ValueError,'Geometry origin'):self.candidate(s,{**g,'StudyInstanceUID':'other'})
        bank=Mock();bank.candidate.return_value=({'sid':'foreign','X':np.zeros(3),'mass':1000},'ok')
        with self.assertRaisesRegex(ValueError,'Candidate origin'):self.candidate(s,g,bank)

    def test_anisotropic_and_reversed_orientation_full100mm(self):
        for flip in (False,True):
            s,g=fixture(flip=flip);center,r=self.candidate(s,g)
            np.testing.assert_allclose(center,(.5,.5));self.assertLess(r['projection_error_mm'],1e-6)
            self.assertEqual(r['out_of_field_fraction'],0)
            self.assertEqual(r['retained_box_xyxy'],r['requested_box_xyxy'])
            self.assertLessEqual(abs(r['actual_field_mm'][0]-100),.3)
            self.assertLessEqual(abs(r['actual_field_mm'][1]-100),.15)
            self.assertGreater(r['native_support_pixels'],50000)

    def test_rotated_affine_uses_own_projection(self):
        s,g=fixture();theta=.3
        rotate=np.array([[np.cos(theta),-np.sin(theta),0.],[np.sin(theta),np.cos(theta),0.],[0.,0.,1.]])
        g['affines_col_row_origin_lps_mm']=(rotate@np.asarray(g['affines_col_row_origin_lps_mm'])).tolist()
        center,r=self.candidate(s,g);np.testing.assert_allclose(center,(.5,.5),atol=1e-12)
        self.assertLess(r['projection_error_mm'],1e-6)

    def test_boundary_missing_geometry_alignment_exact_fallback(self):
        s,g=fixture();bank=Mock();bank.candidate.return_value=({'sid':'s','X':np.array([3.,0.,0.]),'mass':1000},'ok')
        c,t,p,r,m=self.pair([(s,g)],bank);np.testing.assert_array_equal(c,t)
        self.assertEqual(r['s']['reason'],'out_of_bounds_100mm');self.assertGreater(r['s']['out_of_field_fraction'],0)
        for geo,why in ((None,'missing_geometry'),({**g,'quality':{'aligned':False}},'unaligned_geometry')):
            b=Bank();c,t,p,r,m=self.pair([(s,geo)],b)
            self.assertEqual(b.calls,[]);np.testing.assert_array_equal(c,t);self.assertEqual(r['s']['reason'],why)
            self.assertEqual([x['indices'] for x in p[:12]],[x['indices'] for x in p[12:]])
            self.assertEqual(int(m.sum()),24)
        s,g=fixture(shape=(9,80,100));center,r=self.candidate(s,g)
        self.assertIsNone(center);self.assertEqual(r['reason'],'out_of_bounds_100mm')

    def test_known_absent_candidate_statuses_exact_fallback(self):
        for status in ('oblique','mass<250(0)','mass<250(249)','implausible(rf=0.05,cf=0.50)'):
            s,g=fixture();bank=Mock();bank.candidate.return_value=(None,status)
            c,t,p,r,m=self.pair([(s,g)],bank);np.testing.assert_array_equal(c,t)
            self.assertEqual(r['s']['reason'],'no_same_series_candidate');self.assertEqual(r['s']['candidate_reason'],status)

    def test_unknown_programming_resource_and_invalid_contract_fatal(self):
        s,g=fixture()
        for error in (RuntimeError('unknown'),MemoryError('resource'),TypeError('programming')):
            bank=Mock();bank.candidate.side_effect=error
            with self.assertRaises(type(error)):self.candidate(s,g,bank)
        for status in ('error:bad','unknown','ok',None):
            bank=Mock();bank.candidate.return_value=(None,status)
            with self.assertRaises((ValueError,TypeError)):self.candidate(s,g,bank)
        s,g=fixture();s['spacing_mm'][3]=[.8,.3]
        with self.assertRaisesRegex(ValueError,'Changing in-plane'):self.candidate(s,g)
        s,g=fixture();g['affines_col_row_origin_lps_mm'][0][0][0]=2.
        with self.assertRaisesRegex(ValueError,'Affine/spacing'):self.candidate(s,g)
        s,g=fixture();bank=Mock();bank.candidate.return_value=({'sid':'s','X':[np.nan,1,2],'mass':999},'ok')
        with self.assertRaisesRegex(ValueError,'Candidate point'):self.candidate(s,g,bank)

    def test_roundrobin_source_order_per_series_even_spacing_clamping(self):
        blocks=[{'series':str(i),'plane':'Sagittal','n_slices':101,'views':[[0,101*i]]} for i in range(5)]
        p=triplet_plan(blocks,{str(i):True for i in range(5)})
        self.assertEqual([x['series'] for x in p[12:]],[str(i%5) for i in range(12)])
        for i in range(5):
            entries=[x for x in p[12:] if x['series']==str(i)];count=len(entries)
            self.assertEqual([x['retained_indices'][1] for x in entries],[round(100*(k+.5)/count) for k in range(count)])
        # Existing context keeps its original shared-ceiling denominator.
        self.assertEqual([p[i]['retained_indices'][1] for i in range(5)],[17]*5)
        single=[{'series':'only','plane':'Sagittal','n_slices':1,'views':[[0,0]]}]
        self.assertTrue(all(x['indices']==[0,0,0] for x in triplet_plan(single,{'only':True})))

    def test_missing_planes_repeat_masks_and_immutable_inputs(self):
        fixtures=[fixture()];before=copy.deepcopy(fixtures);bank=Bank()
        first=self.pair(fixtures,bank);again=self.pair(fixtures,bank)
        for a,b in ((first[0],again[0]),(first[1],again[1]),(first[4],again[4])):np.testing.assert_array_equal(a,b)
        self.assertEqual(first[2:4],again[2:4]);self.assertEqual(fixtures,before);self.assertEqual(bank.mass,700)
        c,t,p,r,m=self.pair([fixture(plane='Axial')]);np.testing.assert_array_equal(c,t)
        self.assertEqual(r['s']['reason'],'unsupported_plane_no_independent_segmenter');self.assertTrue(m.all())

    def test_fallback_and_ordering_discrepancy_are_fatal(self):
        def bad(*args):
            item=renderer(*args)
            if bad.calls%2: item['px'][0,0,0]+=1
            bad.calls+=1;return item
        bad.calls=0
        with self.assertRaisesRegex(ValueError,'Fallback differs'):self.pair([fixture(plane='Coronal')],render=bad)
        def wrong(*args):
            item=renderer(*args)
            if wrong.calls%2:item['meta'][0,9]+=1
            wrong.calls+=1;return item
        wrong.calls=0
        with self.assertRaisesRegex(ValueError,'ordering differs'):self.pair([fixture()],render=wrong)

    def test_native_support_thin_slabs_and_right_side_order(self):
        s,g=fixture(shape=(9,240,420));s['positions']=list(np.arange(8,-1,-1)*.8)
        self.assertEqual(retained_support(s),[[8,7,6,5],[4,3,2,1]])
        s['side']='R';self.assertEqual(retained_support(s),[[0,1,2,3],[4,5,6,7]])
        s['positions']=[float('nan')]*9
        with self.assertRaisesRegex(ValueError,'depth mapping'):retained_support(s)

    def test_source_selector_restored_on_fatal_without_public_execution(self):
        # Synthetic source API, not imported extracted public code.
        import sys
        s,g=fixture();study={'StudyInstanceUID':'u','series':[s]}
        original=lambda *args:[(0,None,(.5,.5))]
        px=SimpleNamespace(series_views=original,StudySlices=Mock())
        px.StudySlices.return_value.build.side_effect=MemoryError('fatal')
        resolver=SimpleNamespace(ExactDuplicateResolver=Mock())
        resolver.ExactDuplicateResolver.return_value.resolve.return_value={'representative_indices':[0]}
        with patch.dict(sys.modules,{'px_cache':px,'training':SimpleNamespace(),'training.exact_duplicates':resolver}):
            with self.assertRaises(MemoryError):source_render(None,'u',study,{'s':(.5,.5)})
        self.assertIs(px.series_views,original)


    def test_every_triplet_channel_native_support_and_frozen_plan_evidence(self):
        try:
            import cv2
        except ImportError:
            self.skipTest('cv2 required for synthetic artifact rendering')
        from knee.same_series_roi import render_evidence
        s,g=fixture(spacing=(1.,1.),shape=(9,140,160))
        s['positions']=list(np.arange(8,-1,-1)*.8)
        s['intensity']={'display_window_uint16':[0,50000]}
        _,r=self.candidate(s,g)
        plan=triplet_plan([{'series':'s','plane':'Sagittal','n_slices':2,'views':[[0,0]]}],{'s':True})
        before=copy.deepcopy(plan)
        control=np.stack([np.stack([np.full((384,384),channel+slot,np.uint8) for channel in range(3)]) for slot in range(24)])
        treatment=control+50
        with tempfile.TemporaryDirectory() as d:
            s['path']=str(Path(d)/'s.npy')
            volume=np.arange(9*140*160,dtype=np.uint16).reshape(9,140,160);np.save(s['path'],volume)
            support=render_evidence(d,0,{'series':[s]},plan,{'s':r},control,treatment)
            self.assertEqual(plan,before);self.assertEqual(len(support),24)
            self.assertEqual(len(list(Path(d).glob('*paired-triplet*.png'))),24)
            self.assertTrue((Path(d)/'000-native-series-00.png').is_file())
            first=cv2.imread(str(Path(d)/'000-paired-triplet-00.png'))
            self.assertEqual(first.shape[0],3*384)
            for channel in range(3):
                np.testing.assert_array_equal(first[channel*384:(channel+1)*384,:384,0],control[0,channel])
                np.testing.assert_array_equal(first[channel*384:(channel+1)*384,384:768,0],treatment[0,channel])
            for slot,item in enumerate(support):
                self.assertEqual(item['native_support_indices'],[retained_support(s)[k] for k in plan[slot]['retained_indices']])
            np.testing.assert_array_equal(np.load(s['path']),volume)


class BuilderTests(unittest.TestCase):
    def test_immutable_build_tamper(self):
        from scripts.build_same_series_roi import verify_build,digest_text
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'notebook.ipynb';p.write_text('{}')
            (Path(d)/'build-receipt.json').write_text(json.dumps({'files':{p.name:digest_text('{}')}}))
            verify_build(d);p.write_text('changed')
            with self.assertRaises(ValueError):verify_build(d)

    def test_fresh_destination_and_short_slug_guards_before_source_reads(self):
        from scripts.build_same_series_roi import build
        root=Path(__file__).resolve().parents[1]
        args={k:'missing' for k in ('embedded','closure','closure_receipt','contract_path','manifest_path','sample_path','folds_path','exclusion_path','data_root')}
        (root/'outputs').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root/'outputs') as d:
            with self.assertRaisesRegex(ValueError,'Fresh immutable'):build(**args,destination=d,slug='same-series-roi-20261013')
            with self.assertRaisesRegex(ValueError,'<50 chars'):build(**args,destination=Path(d)/'fresh',slug='a'*50)
            with self.assertRaisesRegex(ValueError,'Invalid slug'):build(**args,destination=Path(d)/'fresh',slug='bad_slug')

    @unittest.skipUnless(os.environ.get('KNEE_EVIDENCE_ROOT'), 'explicit audited input fixture required')
    def test_real_fixture_build_source_binding_frozen_sample_and_host_policy(self):
        import ast
        import base64
        import hashlib
        import zlib
        from scripts.build_same_series_roi import build, verify_build
        from knee.public_s6_qualification import canonical, digest
        from knee.same_series_roi import PROTOCOL
        root=Path(__file__).resolve().parents[1]
        evidence=Path(os.environ['KNEE_EVIDENCE_ROOT'])
        args=json.loads((evidence/'outputs/analysis/native-joint-roi-qualification-20261013/same-series-builder-inputs.json').read_bytes())
        (root/'outputs').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root/'outputs') as d:
            dest=Path(d)/'build'
            receipt=build(**args,destination=dest,slug='same-series-roi-reader-synthetic-check')
            self.assertEqual(receipt['protocol'],PROTOCOL);verify_build(dest)
            nb=json.loads((dest/'notebook.ipynb').read_bytes())
            source=''.join(nb['cells'][1]['source']);tree=ast.parse(source)
            encoded=next(ast.literal_eval(n.value.args[0].args[0].args[0]) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='sources' for t in n.targets))
            sources=json.loads(zlib.decompress(base64.b64decode(encoded,validate=True)))
            self.assertEqual(digest(sources),receipt['binding']['source_bundle_sha256'])
            for name in ('knee/same_series_roi.py','knee/native_joint_roi_pilot.py'):
                self.assertEqual(sources[name],(root/name).read_text())
            self.assertTrue(all(hashlib.sha256(sources[name].encode()).hexdigest()==pin for name,pin in receipt['source_pins'].items()))
            metadata=json.loads((dest/'kernel-metadata.json').read_bytes())
            self.assertTrue(metadata['is_private']);self.assertFalse(metadata['enable_gpu']);self.assertFalse(metadata['enable_internet'])
            self.assertIn("device='cpu'",source);self.assertIn('from knee.same_series_roi import run',source)
            self.assertNotIn('from knee.native_joint_roi_pilot import run',source)
            self.assertNotIn('bank.record(',sources['knee/same_series_roi.py'])
            self.assertLess((dest/'notebook.ipynb').stat().st_size,900000)
            self.assertEqual(receipt['status'],'BUILT_UNLAUNCHED')
            bad_sample=Path(d)/'bad-sample.json';bad_sample.write_text(json.dumps({'ids':['synthetic']*32}))
            changed=dict(args,sample_path=bad_sample)
            with self.assertRaisesRegex(ValueError,'unique studies'):build(**changed,destination=Path(d)/'badbuild',slug='same-series-roi-reader-synthetic-check')
            self.assertFalse((Path(d)/'badbuild').exists())
