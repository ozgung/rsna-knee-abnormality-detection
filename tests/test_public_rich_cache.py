import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from knee.public_rich_cache import select_inputs, verify_cache, SERIES_FIELDS

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'outputs/research/2026-10-08/mirror-audit/rsna-knee-0949-anatomical-mirror.py'

class RichCacheTests(unittest.TestCase):
    def test_selection_presence_only_and_exclusions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (root/'train_series.csv').open('w') as h:
                w=csv.writer(h); w.writerow(SERIES_FIELDS)
                for i in range(100): w.writerow([str(i), 's'+str(i), 1, 1, 'Sagittal'])
            def labels(value):
                with (root/'train.csv').open('w') as h:
                    w=csv.writer(h); w.writerow(['StudyInstanceUID','a','b','report'])
                    for i in range(100): w.writerow([str(i),value if i<58 else '',value if i<58 else '', 'DO NOT READ'])
            ex={'quarantined_study_ids':['58','59','60','61']}
            labels('0'); a=select_inputs(root,['a','b'],ex)[0]
            labels('9999'); b=select_inputs(root,['a','b'],ex)[0]
            self.assertEqual(a,b); self.assertEqual(len(a),32)
            self.assertTrue(all(int(i)>=62 for i in a))
            with self.assertRaises(ValueError): select_inputs(root,['a','b'],{})

    def test_build_private_cpu_immutable_and_source_pin(self):
        spec=importlib.util.spec_from_file_location('rich_builder',ROOT/'scripts/build_public_rich_cache_pilot.py')
        mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); ex=root/'exclude.json'
            ex.write_text(json.dumps({'quarantined_study_ids':['a','b','c','d']}))
            fixture=root/'source.py'; fixture.write_text('mocked audited source')
            manifest={'attribution':'fixture','source_url':'https://example.test/fixture'}
            dest=root/'build'
            with patch.object(mod,'audit_source',return_value=('', '', manifest)):
                receipt=mod.build_pilot(fixture,dest,'rich-pilot-test',ex)
            meta=json.loads((dest/'kernel-metadata.json').read_text())
            self.assertFalse(meta['enable_gpu']); self.assertFalse(meta['enable_internet']); self.assertTrue(meta['is_private'])
            self.assertEqual(meta['dataset_sources'],['goodpjw2008/rsna-knee-2-5d-convnext-reader'])
            self.assertEqual(meta['kernel_sources'],[])
            self.assertEqual(len(receipt['exclusion_sha256']),64)
            with self.assertRaises(ValueError): mod.build_pilot(SOURCE,dest,'rich-pilot-test',ex)
            bad=root/'bad.py'; bad.write_text('unreviewed source')
            with self.assertRaises(ValueError): mod.build_pilot(bad,root/'badbuild','rich-pilot-test',ex)


@unittest.skipUnless(SOURCE.is_file() and all(importlib.util.find_spec(p) for p in ('torch','numpy','cv2','pydicom')),
                     'Requires installed ML/codec packages and ignored audited public snapshot')
class RichCacheBackendTests(unittest.TestCase):
    def test_cache_exact_and_invalid(self):
        import numpy as np
        from knee.reference import audit_source, load_definitions
        public,_=load_definitions(*audit_source(SOURCE.read_text()))
        with tempfile.TemporaryDirectory() as tmp:
            v=np.random.default_rng(1).integers(0,256,(96,384,384),dtype=np.uint8)
            m=np.ones(96,np.float32)
            result=verify_cache(public,v,m,Path(tmp)/'cache')
            self.assertEqual(len(result['volume_sha256']),64)
            with self.assertRaises(ValueError): verify_cache(public,v,m*0,Path(tmp)/'bad')
            with self.assertRaises(ValueError): verify_cache(public,v.astype(np.float32),m,Path(tmp)/'bad')
            m[0]=0
            with self.assertRaisesRegex(ValueError,'consistency'):
                verify_cache(public,v,m,Path(tmp)/'inconsistent')

    def test_real_synthetic_decoder_stresses(self):
        from knee.reference import audit_source,load_definitions
        from knee.mirror_input import stress_inputs,STRESS_CASES
        public,decoder=load_definitions(*audit_source(SOURCE.read_text()))
        result=stress_inputs(public,decoder)
        self.assertEqual(result['status'],'PASS')
        self.assertEqual(result['cases'],dict.fromkeys(STRESS_CASES,'PASS'))
