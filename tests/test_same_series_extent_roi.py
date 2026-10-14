"""Synthetic/source-only acceptance; never execute extracted public pixels."""
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
try:
    import numpy as np
except ImportError:
    np=None
from knee.same_series_extent_roi import (series_extent, read_pair, source_render,
                                         render_evidence, triplet_plan, retained_support, PROTOCOL)


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
    def __init__(self, change=None, status='ok'):
        self.calls=[];self.change=change;self.status=status
    def segment(self, s, g, n_slices):
        self.calls.append((s,g,n_slices))
        if self.status != 'ok': return None,self.status
        depth,h,w=s['native_shape'];aff=np.asarray(g['affines_col_row_origin_lps_mm'])
        dz=float(np.linalg.norm(aff[1,:,2]-aff[0,:,2])) if depth>1 else 4.
        step=max(1,round(4./max(dz,1e-3)))
        zs=sorted({min(depth-1,max(0,depth//2+k*step)) for k in range(-3,4)})
        result=[]
        for z in zs:
            p=np.zeros((4,h,w),np.float32)
            p[1,h//2-10:h//2+10,w//2-15:w//2+15]=.9
            p[2,h//2-12:h//2+12,w//2+15:w//2+35]=.8
            result.append((z,p))
        return self.change(result) if self.change else result, 'ok'
    def candidate(self,*args): raise AssertionError('centroid forbidden')
    def record(self,*args): raise AssertionError('study anchor forbidden')
    def slab_prior(self,*args): raise AssertionError('prior forbidden')


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
class ExtentTests(unittest.TestCase):
    def extent(self,s=None,g=None,bank=None,geo=geometry):
        if s is None:s,g=fixture()
        return series_extent('u',s,g,bank or Bank(),geo)
    def pair(self,fixtures=None,bank=None,render=renderer,support=None):
        fixtures=fixtures or [fixture()]
        study={'StudyInstanceUID':'u','series':[s for s,g in fixtures]}
        return read_pair(None,'u',study,{s['SeriesInstanceUID']:g for s,g in fixtures},bank or Bank(),
                         view_geometry=geometry,renderer=render,support_store=support)
    def test_optional_p0_union_mass_and_threshold(self):
        selector,r,support=self.extent()
        self.assertIsNotNone(selector);self.assertEqual(r['accepted_class_pixels'][0],0)
        self.assertGreaterEqual(r['accepted_joint_union_pixels'],250)
        def overlap(segs):
            for z,p in segs:p[2]=p[1]
            return segs
        _,r,_=self.extent(bank=Bank(overlap))
        self.assertEqual(r['accepted_joint_union_pixels'],r['accepted_class_pixels'][1])
        def weak(segs):
            for z,p in segs:
                p[1:]=0;p[1,10:15,10:18]=.5;p[2,20:25,20:28]=.50001
            return segs
        selector,r,support=self.extent(bank=Bank(weak))
        self.assertEqual(r['reason'],'partial_joint_channel_support')
        self.assertTrue(support['accepted_class_slice'][:,2].all())
        self.assertFalse(support['accepted_class_slice'][:,1].any())
    def test_missing_and_sub40_support_global(self):
        for size in (0,39):
            def weak(segs):
                for z,p in segs:p[:]=0;p[1,20,20:20+size]=.9;p[2,21,20:20+size]=.9
                return segs
            c,t,plan,r,m=self.pair(bank=Bank(weak));np.testing.assert_array_equal(c,t)
            self.assertEqual(r['s']['reason'],'insufficient_joint_support')
            self.assertTrue(m.all());self.assertEqual([p['indices'] for p in plan[:12]],[p['indices'] for p in plan[12:]])
    def test_partial_channel_exact_fallback(self):
        def partial(segs):
            for z,p in segs:p[2]=0
            return segs
        c,t,p,r,m=self.pair(bank=Bank(partial));np.testing.assert_array_equal(c,t)
        self.assertEqual(r['s']['reason'],'partial_joint_channel_support')
    def test_border_uses_accepted_support_raw_noise_recorded(self):
        def noise(segs):
            for z,p in segs:p[0,0,0]=.9
            return segs
        selector,r,_=self.extent(bank=Bank(noise));self.assertIsNotNone(selector)
        self.assertEqual(r['raw_border_class_slice_pixels'][0][0],1)
        self.assertEqual(r['accepted_border_class_slice_pixels'][0][0],0)
        def border(segs):
            for z,p in segs:p[0,0,10:50]=.9
            return segs
        c,t,p,r,m=self.pair(bank=Bank(border));np.testing.assert_array_equal(c,t)
        self.assertEqual(r['s']['reason'],'foreground_touches_source_border')
    def test_cap_and_nonfit_global(self):
        def wide(segs):
            for z,p in segs:p[0,30:60,10:410]=.9
            return segs
        self.assertEqual(self.extent(bank=Bank(wide))[1]['reason'],'required_side_exceeds_140mm')
        s,g=fixture(shape=(9,120,240),spacing=(.6,.3))
        self.assertEqual(self.extent(s,g)[1]['reason'],'support_crop_margin_does_not_fit_source')
        def near(segs):
            for z,p in segs:p[:]=0;p[1,10:30,20:40]=.9;p[2,20:40,30:50]=.9
            return segs
        self.assertEqual(self.extent(bank=Bank(near))[1]['reason'],'support_crop_margin_does_not_fit_source')
    def test_extent_expansion_wholemm_rounding_and_four_margins(self):
        s,g=fixture(spacing=(.61,.37),shape=(9,300,500))
        def expanded(segs):
            for z,p in segs:p[0,65:235,230:270]=.9
            return segs
        selector,r,support=self.extent(s,g,Bank(expanded))
        self.assertIsNotNone(selector);self.assertEqual(selector[0],r['side_mm_requested'])
        self.assertGreater(selector[0],100);self.assertLessEqual(selector[0],140)
        self.assertIsInstance(selector[0],int);self.assertGreaterEqual(min(r['actual_margin_mm']),10-1e-9)
        x,y,x1,y1=r['requested_box_xyxy'];self.assertEqual((x1-x,y1-y),(round(selector[0]/.37),round(selector[0]/.61)))
        self.assertEqual(r['requested_box_xyxy'],r['retained_box_xyxy'])
        side=selector[0]-1;row,col=selector[1][0]*299,selector[1][1]*499
        ch,cw=round(side/.61),round(side/.37);y,x=round(row-(ch-1)/2),round(col-(cw-1)/2)
        a,b,c,d=r['support_box_xyxy']
        self.assertLess(min((b-y)*.61,(y+ch-d)*.61,(a-x)*.37,(x+cw-c)*.37),10)
    def test_source_rounding_can_require_one_extra_mm(self):
        s,g=fixture(spacing=(.6,.3),shape=(9,300,500))
        def elongated(segs):
            for z,p in segs:p[0,75:210,245:255]=.9
            return segs
        selector,r,_=self.extent(s,g,Bank(elongated))
        self.assertEqual(r['required_side_mm'],101.)
        self.assertEqual(selector[0],102)
        self.assertGreaterEqual(min(r['actual_margin_mm']),10)
    def test_normalized_center_roundtrip_uses_source_half_even_rounding(self):
        # Observed native dimensions/spacing, synthetic segmentation only.
        # 242/479*479 is 241.99999999999997: source rounding starts at
        # row101, whereas rounding the pre-normalization center starts102.
        spacing=.35416665673255
        s,g=fixture(spacing=(spacing,spacing),shape=(9,480,480))
        def centered(segs):
            for z,p in segs:
                p[:]=0
                p[1,232:253,230:250]=.9
                p[2,232:253,250:270]=.9
            return segs
        selector,r,_=self.extent(s,g,Bank(centered))
        self.assertIsNotNone(selector)
        self.assertEqual(selector[0],100)
        self.assertEqual(r['requested_box_xyxy'][1],101)
        self.assertEqual(r['requested_box_xyxy'],r['retained_box_xyxy'])
        self.assertGreaterEqual(min(r['actual_margin_mm']),10)
    def test_exact250_joint_union_and140mm_cap_acceptance(self):
        s,g=fixture(shape=(1,240,420))
        def exact(segs):
            z,p=segs[0];p[:]=0;p[1,115:120,185:210]=.9;p[2,120:125,185:210]=.9
            return segs
        selector,r,_=self.extent(s,g,Bank(exact));self.assertIsNotNone(selector)
        self.assertEqual(r['accepted_joint_union_pixels'],250)
        def below(segs):
            segs=exact(segs);segs[0][1][2,120,185]=0;return segs
        self.assertEqual(self.extent(s,g,Bank(below))[1]['reason'],'insufficient_joint_support')
        def overlap(segs):
            segs=exact(segs);segs[0][1][2]=segs[0][1][1];return segs
        self.assertEqual(self.extent(s,g,Bank(overlap))[1]['accepted_joint_union_pixels'],125)
        self.assertEqual(self.extent(s,g,Bank(overlap))[1]['reason'],'insufficient_joint_support')
        s,g=fixture(spacing=(.5,.5),shape=(9,400,400))
        def cap(segs):
            for z,p in segs:p[0,190:210,80:320]=.9
            return segs
        selector,r,_=self.extent(s,g,Bank(cap));self.assertEqual(selector[0],140)
        self.assertEqual(min(r['actual_margin_mm']),10.)
    def test_reversed_anisotropic_own_geometry_and_origin(self):
        bank=Bank();fixtures=[fixture('a'),fixture('b',flip=True,shift=300,frame=None),fixture('c','Coronal')]
        c,t,p,r,m=self.pair(fixtures,bank)
        self.assertEqual([x[0]['SeriesInstanceUID'] for x in bank.calls],['a','b'])
        self.assertTrue(all(x[2]==7 for x in bank.calls));self.assertEqual(r['a']['requested_box_xyxy'],r['b']['requested_box_xyxy'])
        self.assertEqual({x['series'] for x in p[12:]},{'a','b'});np.testing.assert_array_equal(c[:12],t[:12])
        self.assertTrue(m.all());self.assertEqual(r['b']['origin_series'],'b')
    def test_missing_unaligned_unsupported_and_oblique_fallback(self):
        s,g=fixture()
        for geo,reason in ((None,'missing_geometry'),({**g,'quality':{'aligned':False}},'unaligned_geometry')):
            b=Bank();c,t,p,r,m=self.pair([(s,geo)],b);np.testing.assert_array_equal(c,t)
            self.assertFalse(b.calls);self.assertEqual(r['s']['reason'],reason)
        c,t,p,r,m=self.pair([fixture(plane='Axial')]);np.testing.assert_array_equal(c,t)
        self.assertEqual(r['s']['reason'],'unsupported_plane_no_independent_segmenter')
        self.assertEqual(self.extent(bank=Bank(status='oblique'))[1]['reason'],'oblique')
    def test_bad_origin_spacing_shape_affine_are_fatal(self):
        for field in ('StudyInstanceUID','SeriesInstanceUID'):
            s,g=fixture();g[field]='wrong'
            with self.assertRaisesRegex(ValueError,'Geometry origin'):self.extent(s,g)
        s,g=fixture();s['spacing_mm'][1]=[.7,.3]
        with self.assertRaisesRegex(ValueError,'Changing in-plane'):self.extent(s,g)
        s,g=fixture();s['spacing_mm'][0][0]=0
        with self.assertRaisesRegex(ValueError,'Spacing/native'):self.extent(s,g)
        s,g=fixture();s['native_shape'][0]=1.5
        with self.assertRaisesRegex(ValueError,'Native shape'):self.extent(s,g)
        s,g=fixture();g['affines_col_row_origin_lps_mm'][0][0][0]=2
        with self.assertRaisesRegex(ValueError,'Affine/spacing'):self.extent(s,g)
        s,g=fixture();s['positions'][0]=float('nan')
        with self.assertRaisesRegex(ValueError,'depth mapping'):self.extent(s,g)
    def test_bad_segment_depth_shape_probability_are_fatal(self):
        changes=[lambda a:a[::-1],lambda a:a+a[:1],lambda a:[(100,p) for z,p in a],
                 lambda a:[(float(z),p) for z,p in a],lambda a:[(z,p[:3]) for z,p in a],
                 lambda a:[(z,p.astype(np.uint8)) for z,p in a],lambda a:[],
                 lambda a:[(z,p*float('nan')) for z,p in a],lambda a:[(z,p+1.) for z,p in a],
                 lambda a:[(z,p-1.) for z,p in a]]
        for change in changes:
            with self.subTest(change=change),self.assertRaises(ValueError):self.extent(bank=Bank(change))
    def test_unknown_programming_resource_errors_are_fatal(self):
        for status in ('unknown','ok',None,'error:failure'):
            b=Mock();b.segment.return_value=(None,status)
            with self.assertRaises(ValueError):self.extent(bank=b)
        for error in (RuntimeError('unknown'),TypeError('programming'),MemoryError('resource')):
            b=Mock();b.segment.side_effect=error
            with self.assertRaises(type(error)):self.extent(bank=b)
    def test_source_geometry_clipping_is_fatal(self):
        def clipped(*a,**kw):
            g=geometry(*a,**kw);g.crop_xyxy=(0,0,2,2);return g
        with self.assertRaisesRegex(ValueError,'clipped/shrank'):self.extent(geo=clipped)
    def test_repeated_reader_receipts_masks_arrays_and_inputs(self):
        fixtures=[fixture()];before=copy.deepcopy(fixtures);one={};two={}
        a=self.pair(fixtures,support=one);b=self.pair(fixtures,support=two)
        for i in (0,1,4):np.testing.assert_array_equal(a[i],b[i])
        self.assertEqual(a[2:4],b[2:4]);self.assertEqual(fixtures,before)
        for k,v in one['s'].items():np.testing.assert_array_equal(v,two['s'][k])
        self.assertEqual(a[0].shape,(24,3,384,384));self.assertTrue(np.isfinite(a[1]).all())
    def test_fallback_and_ordering_errors_are_fatal(self):
        def bad(*a):
            item=renderer(*a)
            if bad.n%2:item['px'][0,0,0]+=1
            bad.n+=1;return item
        bad.n=0
        with self.assertRaisesRegex(ValueError,'Fallback differs'):self.pair([fixture(plane='Coronal')],render=bad)
        def badmeta(*a):
            item=renderer(*a)
            if badmeta.n%2:item['meta'][0,9]+=1
            badmeta.n+=1;return item
        badmeta.n=0
        with self.assertRaisesRegex(ValueError,'ordering differs'):self.pair(render=badmeta)
    def test_source_selector_side_center_and_restoration(self):
        import sys
        s,g=fixture();study={'StudyInstanceUID':'u','series':[s]};original=lambda *args:[]
        px=SimpleNamespace(series_views=original,StudySlices=Mock())
        resolver=SimpleNamespace(ExactDuplicateResolver=Mock())
        resolver.ExactDuplicateResolver.return_value.resolve.return_value={'representative_indices':[0]}
        def build(uid):
            self.assertEqual(px.series_views(),[(0,121,(.4,.6))]);raise MemoryError('fatal')
        px.StudySlices.return_value.build.side_effect=build
        with patch.dict(sys.modules,{'px_cache':px,'training':SimpleNamespace(),'training.exact_duplicates':resolver}):
            with self.assertRaises(MemoryError):source_render(None,'u',study,{'s':(121,(.4,.6))})
        self.assertIs(px.series_views,original)
    def test_every_selected_constituent_native_support_and_saved_masks(self):
        try:import cv2
        except ImportError:self.skipTest('cv2 required for synthetic artifact rendering')
        s,g=fixture(spacing=(1.,1.),shape=(9,160,180));s['positions']=list(np.arange(8,-1,-1)*.8)
        s['intensity']={'display_window_uint16':[0,50000]};selector,r,support=self.extent(s,g)
        plan=triplet_plan([{'series':'s','plane':'Sagittal','n_slices':2,'views':[[0,0]]}],{'s':True});before=copy.deepcopy(plan)
        c=np.zeros((24,3,384,384),np.uint8);t=c+50
        with tempfile.TemporaryDirectory() as d:
            s['path']=str(Path(d)/'s.npy');volume=np.arange(9*160*180,dtype=np.uint16).reshape(9,160,180);np.save(s['path'],volume)
            evidence=render_evidence(d,0,{'series':[s]},plan,{'s':r},c,t,{'s':support})
            self.assertEqual(plan,before);self.assertEqual(len(evidence),24)
            self.assertEqual(len(list(Path(d).glob('*paired-triplet*.png'))),24)
            self.assertEqual(len(list(Path(d).glob('*segmentation*-native*.png'))),len(support['depth_indices']))
            with np.load(Path(d)/'000-segmentation-series-00.npz',allow_pickle=False) as stored:
                for k,v in support.items():np.testing.assert_array_equal(stored[k],v)
            for slot,entry in enumerate(evidence):self.assertEqual(entry['native_support_indices'],[retained_support(s)[k] for k in plan[slot]['retained_indices']])
            np.testing.assert_array_equal(np.load(s['path']),volume)
            overlay=cv2.imread(str(Path(d)/'000-segmentation-series-00-native-z0001.png'))
            self.assertTrue(np.any(overlay[:,:,0]!=overlay[:,:,1]))


class BuilderTests(unittest.TestCase):
    def test_immutable_build_tamper(self):
        from scripts.build_same_series_extent_roi import verify_build,digest_text
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'notebook.ipynb';p.write_text('{}')
            (Path(d)/'build-receipt.json').write_text(json.dumps({'files':{p.name:digest_text('{}')}}))
            verify_build(d);p.write_text('changed')
            with self.assertRaises(ValueError):verify_build(d)

    def test_fresh_destination_and_short_slug_guards_before_source_reads(self):
        from scripts.build_same_series_extent_roi import build
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
        from scripts.build_same_series_extent_roi import build, verify_build
        from knee.public_s6_qualification import canonical, digest
        from knee.same_series_extent_roi import PROTOCOL
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
            for name in ('knee/same_series_roi.py','knee/same_series_extent_roi.py','knee/native_joint_roi_pilot.py'):
                self.assertEqual(sources[name],(root/name).read_text())
            self.assertTrue(all(hashlib.sha256(sources[name].encode()).hexdigest()==pin for name,pin in receipt['source_pins'].items()))
            metadata=json.loads((dest/'kernel-metadata.json').read_bytes())
            self.assertTrue(metadata['is_private']);self.assertFalse(metadata['enable_gpu']);self.assertFalse(metadata['enable_internet'])
            self.assertIn("device='cpu'",source);self.assertIn('from knee.same_series_extent_roi import run',source)
            self.assertNotIn('from knee.native_joint_roi_pilot import run',source)
            self.assertNotIn('bank.record(',sources['knee/same_series_extent_roi.py'])
            self.assertNotIn('bank.candidate(',sources['knee/same_series_extent_roi.py'])
            self.assertFalse(receipt['binding']['protocol']['advancement_allowed'])
            self.assertEqual(receipt['binding']['protocol']['anatomical_containment'],'UNVERIFIED')
            self.assertEqual(receipt['binding']['protocol']['classifier_models'],0)
            self.assertNotIn('train.csv',sources['knee/same_series_extent_roi.py'])
            self.assertEqual(receipt['binding']['protocol']['margin_mm'],10)
            self.assertIn('<=30%',receipt['binding']['protocol']['advancement'])
            compile(source,'<source-only-audit>','exec')
            self.assertLess((dest/'notebook.ipynb').stat().st_size,900000)
            self.assertEqual(receipt['status'],'BUILT_UNLAUNCHED')
            bad_sample=Path(d)/'bad-sample.json';bad_sample.write_text(json.dumps({'ids':['synthetic']*32}))
            changed=dict(args,sample_path=bad_sample)
            with self.assertRaisesRegex(ValueError,'unique studies'):build(**changed,destination=Path(d)/'badbuild',slug='same-series-roi-reader-synthetic-check')
            self.assertFalse((Path(d)/'badbuild').exists())
