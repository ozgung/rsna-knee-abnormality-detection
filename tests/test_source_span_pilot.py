import ast
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from knee import source_span_pilot as pilot

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = Path(os.environ.get('SOURCE_SPAN_EVIDENCE_ROOT','/nonexistent'))
PUBLIC = EVIDENCE/'outputs/research/2026-10-08/raptor-audit/knee-mri-twelve-findings-from-a-single-model.py'
TRAINING = EVIDENCE/'outputs/kaggle/trackA-coatnet224-fold01-20261009/train_knee_original.py'
CACHE = EVIDENCE/'outputs/kaggle/own224-window-samerun-20261012-thread1-v1/build-contract-private.json'


class SpanContractTests(unittest.TestCase):
    def test_source_and_structural_drift_rejected(self):
        with self.assertRaises(ValueError): pilot.recipes('unreviewed')
        narrow='def build_study():\n return (.15, .85)\n'
        wide='def build_study():\n return (.02, .98)\n'
        pilot.validate_recipes(narrow,wide)
        with self.assertRaises(ValueError): pilot.validate_recipes(narrow,wide.replace('return','raise'))
        with self.assertRaises(ValueError): pilot.validate_recipes(narrow,wide.replace('.98','.97'))
        with self.assertRaises(ValueError): pilot.sampler_source('unreviewed')

    @unittest.skipUnless(importlib.util.find_spec('numpy'), 'NumPy unavailable')
    def test_mask_empty_and_roundtrip_cleanup(self):
        import numpy as np
        v=np.zeros((44,336,336),np.uint8);m=np.zeros(44,np.uint8)
        with self.assertRaisesRegex(ValueError,'Whole-empty'):pilot.validate_arrays(v,m)
        v[0,0,0]=255;m[0]=1;pilot.validate_arrays(v,m)
        with tempfile.TemporaryDirectory() as tmp,patch('tempfile.tempdir',tmp):
            result=pilot.roundtrip(v,m)
            self.assertTrue(result['save_reload_equal']);self.assertFalse(list(Path(tmp).iterdir()))
        bad=m.copy();bad[1]=1
        with self.assertRaisesRegex(ValueError,'consistency'):pilot.validate_arrays(v,bad)
        with self.assertRaises(ValueError):pilot.validate_arrays(v.astype(float),m)


@unittest.skipUnless(PUBLIC.is_file() and TRAINING.is_file(), 'Explicit pinned source fixtures required')
class PinnedSpanTests(unittest.TestCase):
    def test_actual_source_two_constants_and_sampler_only(self):
        codes=pilot.recipes(PUBLIC.read_text())
        pilot.validate_recipes(**codes)
        a,b=ast.parse(codes['narrow']),ast.parse(codes['wide'])
        self.assertEqual([n.name for n in a.body],['_make_reader','_pick_series_for_slot','build_study'])
        self.assertEqual(ast.dump(a.body[0]),ast.dump(b.body[0]))
        self.assertEqual(ast.dump(a.body[1]),ast.dump(b.body[1]))
        sampler=pilot.sampler_source(TRAINING.read_text())
        self.assertEqual([n.name for n in ast.parse(sampler).body],['StudyWindows'])
        with self.assertRaises(ValueError):pilot.recipes(PUBLIC.read_text()+'\n# drift')

    def test_real_builder_immutable_private_without_ids(self):
        spec=importlib.util.spec_from_file_location('span_builder',ROOT/'scripts/build_source_span_pilot.py')
        builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)/'build'
            kwargs=dict(evidence_root=EVIDENCE,public_source=PUBLIC,training_source=TRAINING,cache_contract=CACHE)
            receipt=builder.build(dest,'source-span-test',**kwargs)
            self.assertTrue(receipt['build_only'])
            meta=json.loads((dest/'kernel-metadata.json').read_text())
            self.assertTrue(meta['is_private']);self.assertFalse(meta['enable_gpu']);self.assertFalse(meta['enable_internet'])
            nb=json.loads((dest/'notebook.ipynb').read_text());code=''.join(nb['cells'][1]['source']);compile(code,'draft','exec')
            identity=json.loads((EVIDENCE/'outputs/kaggle/sealed-identity-full-result/sealed_identity/sealed-manifest-private.json').read_text())
            self.assertFalse(any(r['uid'] in code for r in identity['records']))
            with self.assertRaises(ValueError):builder.build(dest,'source-span-test',**kwargs)

    @unittest.skipUnless(all(importlib.util.find_spec(p) for p in ('torch','cv2')), 'CPU backend required')
    def test_missing_canonical_raw_writes_sanitized_failure_receipt(self):
        import time
        spec=importlib.util.spec_from_file_location('span_failure_builder',ROOT/'scripts/build_source_span_pilot.py')
        builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);builder.build(root/'build','source-span-failure-test',evidence_root=EVIDENCE,
                public_source=PUBLIC,training_source=TRAINING,cache_contract=CACHE)
            contract=json.loads((root/'build/build-contract.json').read_text())
            identity=EVIDENCE/'outputs/kaggle/sealed-identity-full-result/sealed_identity'
            with patch('knee.fresh_reader_setup.notebook_root',return_value=identity), \
                 patch('knee.fresh_reader_setup.exact_root',return_value=EVIDENCE/'data/raw'):
                with self.assertRaisesRegex(RuntimeError,'sanitized aggregate'):
                    pilot.run(root,root/'output',contract,pilot.recipes(PUBLIC.read_text()),
                        pilot.sampler_source(TRAINING.read_text()),started=time.monotonic())
            text=(root/'output/receipt.json').read_text();receipt=json.loads(text)
            self.assertEqual(receipt['status'],'ERROR');self.assertEqual(receipt['failure']['type'],'ValueError')
            items=json.loads((identity/'sealed-manifest-private.json').read_text())['records']
            self.assertFalse(any(r['uid'] in text for r in items))
            self.assertEqual(list((root/'output').iterdir()),[root/'output/receipt.json'])


@unittest.skipUnless(PUBLIC.is_file() and TRAINING.is_file() and all(importlib.util.find_spec(p) for p in ('numpy','torch','cv2','pydicom')), 'Pinned sources and CPU codec/backend required')
class SpanBackendTests(unittest.TestCase):
    def setUp(self):
        import torch,cv2
        torch.set_num_threads(1);cv2.setNumThreads(1)
        self.codes=pilot.recipes(PUBLIC.read_text())

    def make_inputs(self,root,n=20):
        import numpy as np
        from pydicom.dataset import Dataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian
        directory=root/'study/series';directory.mkdir(parents=True)
        datasets={}
        # Reverse filenames: physical positions must drive ordering.
        for i in range(n):
            path=directory/f'{n-i:03d}.dcm';path.touch()
            d=Dataset();d.file_meta=FileMetaDataset();d.file_meta.TransferSyntaxUID=ExplicitVRLittleEndian
            d.Rows=d.Columns=8;d.SamplesPerPixel=1;d.PhotometricInterpretation='MONOCHROME2'
            d.BitsAllocated=d.BitsStored=16;d.HighBit=15;d.PixelRepresentation=0
            d.ImageOrientationPatient=[1,0,0,0,1,0];d.ImagePositionPatient=[0,0,i]
            d.PixelSpacing=[.5,.5];d.InstanceNumber=n-i
            d.PixelData=(np.arange(64,dtype=np.uint16).reshape(8,8)+i*100).tobytes()
            datasets[str(path)]=d
        records={'study':[{'StudyInstanceUID':'study','SeriesInstanceUID':'series','Fluid_Sensitive':'1','Anatomical_Plane':'Sagittal'}]}
        return records,datasets

    def test_actual_reader_order_span_short_partial_and_repeated(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);records,ds=self.make_inputs(root)
            def read(path,**kwargs):return ds[str(path)]
            with patch('pydicom.dcmread',side_effect=read):
                arms={}
                for arm,code in self.codes.items():
                    public=pilot.parity.reader_module(code)
                    v,m,a=pilot.decode(public,'study',records,root,pilot.SPANS[arm])
                    v2,m2,_=pilot.decode(public,'study',records,root,pilot.SPANS[arm])
                    self.assertTrue(np.array_equal(v,v2));self.assertTrue(np.array_equal(m,m2))
                    self.assertEqual(int(m.sum()),12);self.assertEqual(a['reader_counts']['missing_slots'],4)
                    self.assertEqual(a['reader_counts']['headers'],20)
                    arms[arm]=(v,m,a)
                self.assertLess(arms['wide'][2]['ordinal_depth'][0]['first_fraction'],arms['narrow'][2]['ordinal_depth'][0]['first_fraction'])
                self.assertGreater(arms['wide'][2]['ordinal_depth'][0]['last_fraction'],arms['narrow'][2]['ordinal_depth'][0]['last_fraction'])
                # Pixel transformations identical: direct narrow builder must agree exactly.
                p=pilot.parity.reader_module(self.codes['narrow'])
                v,m=p.build_study('study',records,str(root),p._make_reader())
                slots=pilot.parity.compare(arms['narrow'][0],arms['narrow'][1],v,m)
                self.assertTrue(all(s['exact_hash_match'] and s['mask_match'] for s in slots))
                # One recoverable pixel failure affects selected frame(s), keeps other input.
                bad=next(k for k,d in ds.items() if float(d.ImagePositionPatient[2])==3)
                def partial(path,**kwargs):
                    if str(path)==bad and not kwargs.get('stop_before_pixels'):raise ValueError('synthetic failure')
                    return ds[str(path)]
                with patch('pydicom.dcmread',side_effect=partial):
                    v,m,a=pilot.decode(p,'study',records,root,pilot.SPANS['narrow'])
                    self.assertEqual(a['reader_counts']['partial_slots'],1)
                    self.assertGreater(a['reader_counts']['decode_failures'],0)
            with tempfile.TemporaryDirectory() as short:
                records,ds=self.make_inputs(Path(short),n=1)
                with patch('pydicom.dcmread',side_effect=lambda path,**kw:ds[str(path)]):
                    v,m,a=pilot.decode(pilot.parity.reader_module(self.codes['wide']),'study',records,Path(short),pilot.SPANS['wide'])
                    self.assertEqual(a['ordinal_depth'][0]['repeated_picks'],11)

    def test_96_is_same_wide_functions_declared_allocation_only(self):
        import numpy as np
        wide=pilot.arm_module(self.codes,'wide');wide96=pilot.arm_module(self.codes,'wide96')
        self.assertEqual(wide.build_study.__code__.co_code,wide96.build_study.__code__.co_code)
        self.assertEqual(wide._make_reader.__code__.co_code,wide96._make_reader.__code__.co_code)
        self.assertEqual(wide96.SLOTS,pilot.SLOTS96);self.assertEqual(wide96.MAXS,96)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);records,ds=self.make_inputs(root,n=20)
            with patch('pydicom.dcmread',side_effect=lambda path,**kw:ds[str(path)]):
                v,m,a=pilot.decode(wide96,'study',records,root,pilot.SPANS['wide96'])
                self.assertEqual(v.shape,(96,336,336));self.assertEqual(int(m.sum()),26)
                self.assertEqual(a['ordinal_depth'][0]['selected_frames'],26)
                sampler=pilot.sampler_class(pilot.sampler_source(TRAINING.read_text()))
                windows=pilot.window_checks(v,m,sampler,slots=wide96.SLOTS)
                self.assertTrue(all(w['finite'] for w in windows.values()))
                v2,m2,_=pilot.decode(wide96,'study',records,root,pilot.SPANS['wide96'])
                self.assertTrue(np.array_equal(v,v2));self.assertTrue(np.array_equal(m,m2))
                self.assertTrue(pilot.roundtrip(v,m)['save_reload_equal'])

    def test_guard_fatal_invalid_and_empty(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);records,ds=self.make_inputs(root)
            p=pilot.parity.reader_module(self.codes['narrow'])
            original=p._make_reader
            for error in (MemoryError(),TimeoutError()):
                with patch('pydicom.dcmread',side_effect=error),self.assertRaises(type(error)):
                    pilot.decode(p,'study',records,root,pilot.SPANS['narrow'])
                self.assertIs(p._make_reader,original)
            def read(path,**kw):return ds[str(path)]
            with patch('pydicom.dcmread',side_effect=read):
                factory=p._make_reader
                def invalid_factory():
                    order,pixel,resize=factory()
                    return order,lambda path:np.full((8,8),np.nan),resize
                p._make_reader=invalid_factory
                with self.assertRaisesRegex(ValueError,'finiteness'):pilot.decode(p,'study',records,root,pilot.SPANS['narrow'])
                p._make_reader=factory
                for d in ds.values():d.PixelData=np.zeros((8,8),np.uint16).tobytes()
                with self.assertRaisesRegex(ValueError,'Whole-empty'):pilot.decode(p,'study',records,root,pilot.SPANS['narrow'])

    def test_pinned_sampler_k8_k16_keeps_legacy_mask_artifacts(self):
        import numpy as np
        v=np.zeros((44,336,336),np.uint8);v[0:12]=20;v[22:30]=30;v[43]=10
        m=v.reshape(44,-1).any(1).astype(np.uint8)
        sampler=pilot.sampler_class(pilot.sampler_source(TRAINING.read_text()))
        result=pilot.window_checks(v,m,sampler)
        for k in ('8','16'):
            self.assertTrue(result[k]['finite']);self.assertGreater(result[k]['masked_slice_windows'],0)
        self.assertEqual(result,pilot.window_checks(v,m,sampler))
