import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from knee import raptor64_cache as cache

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = Path(os.environ.get('SOURCE_SPAN_EVIDENCE_ROOT', '/nonexistent'))
PUBLIC = Path(os.environ.get('KNEE_RAPTOR_SOURCE') or EVIDENCE / 'outputs/research/2026-10-08/raptor-audit/knee-mri-twelve-findings-from-a-single-model.py')
HAVE_NUMPY = importlib.util.find_spec('numpy') is not None
HAVE_READER = PUBLIC.is_file() and all(importlib.util.find_spec(p) for p in ('numpy', 'torch', 'cv2', 'pydicom'))


def load_builder():
    spec = importlib.util.spec_from_file_location('raptor64_builder', ROOT / 'scripts/build_raptor64_cache.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class SlugAndConfigTests(unittest.TestCase):
    def test_slug_guard(self):
        self.assertEqual(cache.check_slug('knee-raptor64-cache-shard0-20261013'), 'knee-raptor64-cache-shard0-20261013')
        for bad in ('x' * 51, 'Upper-case', 'trailing-', '-lead', 'a--b', 'abc', 'has_underscore', 7):
            with self.assertRaises(ValueError):
                cache.check_slug(bad)
        self.assertEqual(len('knee-raptor64-cache-shard3-20261013'), 35)
        cache.check_slug('a' * 50)

    def test_v4_contract_constants(self):
        self.assertEqual([s[2] for s in cache.SLOTS], [18, 14, 12, 8, 12])
        self.assertEqual(sum(s[2] for s in cache.SLOTS), 64)
        self.assertEqual(cache.SPAN, [0.06, 0.94]); self.assertEqual(cache.STORAGE, 336)
        self.assertEqual(cache.CONFIG['model_input_resolution'], 384)
        from knee.reference import RAPTOR_RECIPE
        self.assertEqual(RAPTOR_RECIPE['slots'], [list(s) for s in cache.SLOTS])
        self.assertEqual(RAPTOR_RECIPE['slice_span'], cache.SPAN); self.assertEqual(RAPTOR_RECIPE['image_size'], cache.STORAGE)

    def test_geometry_guard_rejects_non_v4_module(self):
        good = SimpleNamespace(SLOTS=list(cache.SLOTS), MAXS=64, IMG=336, CROP_MM=140.0, K_EVAL=42)
        self.assertIs(cache.require_public(good), good)
        for change in ({'MAXS': 44}, {'IMG': 384}, {'K_EVAL': 62}, {'SLOTS': [('Sagittal', 1, 12)]}):
            with self.assertRaises(ValueError):
                cache.require_public(SimpleNamespace(**{**vars(good), **change}))


@unittest.skipUnless(HAVE_NUMPY, 'NumPy required')
class MetadataTests(unittest.TestCase):
    def slots(self):
        return [{'plane': 'Sagittal', 'selected': True, 'fluid_sensitive': 1, 'frames': 30, 'count': 18},
                {'plane': 'Sagittal', 'selected': True, 'fluid_sensitive': 0, 'frames': 1, 'count': 14},
                {'plane': 'Coronal', 'selected': True, 'fluid_sensitive': 0, 'frames': 9, 'count': 12},
                {'plane': 'Coronal', 'selected': False, 'fluid_sensitive': None, 'frames': 0, 'count': 8},
                {'plane': 'Axial', 'selected': True, 'fluid_sensitive': 1, 'frames': 40, 'count': 12}]

    def test_slice_metadata_mirrors_public_picks(self):
        import numpy as np
        codes, depth = cache.slice_metadata(self.slots())
        self.assertEqual(codes.shape, (64,)); self.assertEqual(depth.dtype, np.float32)
        self.assertEqual(codes[:18].tolist(), [1] * 18); self.assertEqual(codes[18:32].tolist(), [0] * 14)
        self.assertEqual(codes[32:44].tolist(), [2] * 12); self.assertEqual(codes[44:52].tolist(), [2] * 8)
        self.assertEqual(codes[52:].tolist(), [5] * 12)
        n, k = 30, 18; lo, hi = int(n * .06), max(int(n * .94) - 1, int(n * .06))
        expected = np.linspace(lo, hi, k).round().astype(int) / (n - 1)
        np.testing.assert_allclose(depth[:18], expected.astype(np.float32))
        self.assertTrue((depth[18:32] == 0).all()); self.assertTrue((depth[44:52] == 0).all())
        self.assertTrue(0 <= depth.min() and depth.max() <= 1)
        with self.assertRaises(ValueError):
            cache.slice_metadata(self.slots()[:4])

    def test_window_metadata_uses_clamped_centres(self):
        codes, depth = cache.slice_metadata(self.slots())
        meta = cache.window_metadata(codes, depth, [0, 1, 17, 18, 63])
        self.assertEqual(meta.shape, (5, 2))
        self.assertEqual(meta[:, 0].tolist(), [1, 1, 1, 0, 5])  # 0 -> 1 and 63 -> 62 clamped
        self.assertEqual(float(meta[4, 1]), float(depth[62]))


@unittest.skipUnless(HAVE_NUMPY, 'NumPy required')
class FileTests(unittest.TestCase):
    def arrays(self):
        import numpy as np
        v = np.zeros((64, 336, 336), np.uint8); v[0, 0, 0] = 1; m = np.zeros(64, np.uint8); m[0] = 1
        return v, m

    def test_atomic_roundtrip_corruption_and_raw_equality(self):
        v, m = self.arrays()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / '000000.npz'; result = cache.atomic_npz(path, v, m)
            self.assertEqual(cache.verify_file(path, result), result)
            self.assertEqual(cache.atomic_npz(path, v, m), result)
            wrong = v.copy(); wrong[0, 0, 0] = 2
            with self.assertRaises(ValueError):
                cache.atomic_npz(path, wrong, m)
            with self.assertRaises(ValueError):
                cache.verify_file(path, result | {'file_sha256': '0' * 64})
            path.write_bytes(b'corrupt compressed archive')
            with self.assertRaises(Exception):
                cache.verify_file(path, result)

    def test_rejects_source96_depth_and_bad_masks(self):
        import numpy as np
        v, m = self.arrays()
        with self.assertRaises(ValueError):
            cache.array_hashes(np.zeros((96, 336, 336), np.uint8), np.zeros(96, np.uint8))
        with self.assertRaises(ValueError):
            cache.array_hashes(v, m.astype(float))
        with self.assertRaises(ValueError):
            cache.array_hashes(v, np.zeros(64, np.uint8))  # pixel/mask inconsistent

    def test_bounded_header_rejected_before_allocation(self):
        import io, zipfile
        import numpy as np
        volume = io.BytesIO(); np.lib.format.write_array_header_1_0(volume, {'descr': '|u1', 'fortran_order': False, 'shape': (10**12,)})
        mask = io.BytesIO(); np.save(mask, np.zeros(64, np.uint8), allow_pickle=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / '000000.npz'
            with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
                z.writestr('volume.npy', volume.getvalue()); z.writestr('mask.npy', mask.getvalue())
            with patch('numpy.load', side_effect=AssertionError('must reject before allocation')), self.assertRaises(ValueError):
                cache.verify_file(path)

    def test_index_membership(self):
        v, m = self.arrays()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); saved = cache.atomic_npz(root / '000000.npz', v, m)
            row = {'ordinal': 0, 'StudyInstanceUID': 'fixture', 'file': '000000.npz', **saved}
            doc = {'version': cache.VERSION, 'contract_sha256': 'contract', 'entries': [row]}
            self.assertEqual(cache.index_rows(root, doc, 'contract', [0], ['fixture']), {0: row})
            for broken in (doc | {'version': 'own96-wide336-sharded-v1'}, doc | {'entries': [row, row]},
                           doc | {'entries': [row | {'StudyInstanceUID': 'other'}]}, doc | {'entries': [row | {'file': '../x.npz'}]}):
                with self.assertRaises(ValueError):
                    cache.index_rows(root, broken, 'contract', [0], ['fixture'])

    def test_disk_projection(self):
        self.assertEqual(cache.disk_required(3), 5 * cache.MAX_FILE_BYTES + cache.DISK_RESERVE)
        with tempfile.TemporaryDirectory() as tmp, patch('shutil.disk_usage', return_value=SimpleNamespace(free=1)):
            with self.assertRaises(OSError):
                cache.require_disk(tmp, 3)


def write_series(root, study, series, n, plane, *, seed):
    """Synthetic DICOM series with geometry out of file-name order."""
    import numpy as np
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid
    rng = np.random.default_rng(seed)
    directory = Path(root) / study / series; directory.mkdir(parents=True)
    normal = {'Sagittal': ([0, 1, 0, 0, 0, -1], 0), 'Coronal': ([1, 0, 0, 0, 0, -1], 1), 'Axial': ([1, 0, 0, 0, 1, 0], 2)}[plane]
    order = rng.permutation(n)
    for i in range(n):
        meta = FileMetaDataset(); meta.MediaStorageSOPClassUID = MRImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid(); meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = Dataset(); ds.file_meta = meta; ds.SOPClassUID = MRImageStorage; ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.Modality = 'MR'; ds.Rows = 48; ds.Columns = 40; ds.SamplesPerPixel = 1; ds.PhotometricInterpretation = 'MONOCHROME2'
        ds.BitsAllocated = 16; ds.BitsStored = 16; ds.HighBit = 15; ds.PixelRepresentation = 0
        ds.ImageOrientationPatient = normal[0]; position = [0.0, 0.0, 0.0]; position[normal[1]] = float(order[i]) * 3.0
        ds.ImagePositionPatient = position; ds.PixelSpacing = [0.5, 0.5]; ds.InstanceNumber = int(order[i])
        pixels = (rng.integers(0, 900, (48, 40)) + 40 * order[i]).astype(np.uint16); ds.PixelData = pixels.tobytes()
        ds.save_as(directory / f'{i:03d}.dcm', enforce_file_format=True)


@unittest.skipUnless(HAVE_READER, 'Pinned Raptor source plus numpy/torch/cv2/pydicom required')
class ReaderParityTests(unittest.TestCase):
    """Cache decode must equal the raptor-profile reader (knee.raptor_input / knee.reference) exactly."""
    @classmethod
    def setUpClass(cls):
        builder = load_builder()
        cls.module, cls.manifest = builder.reader(PUBLIC.read_text())
        cls.public = cache.load_public(cls.module, cls.manifest)

    def study(self, root):
        series = [('s1', 'Sagittal', 1, 23), ('s0', 'Sagittal', 0, 5), ('c1', 'Coronal', 1, 1), ('ax', 'Axial', 0, 31),
                  ('ax2', 'Axial', 1, 7)]
        rows = []
        for seed, (uid, plane, fs, n) in enumerate(series):
            write_series(root, 'study', uid, n, plane, seed=seed)
            rows.append({'StudyInstanceUID': 'study', 'SeriesInstanceUID': uid, 'Anatomical_Plane': plane,
                         'Fluid_Sensitive': str(fs), 'Fat_Suppression': str(fs)})
        return {'study': rows}

    def test_decode_equals_public_build_study_and_strict_reference(self):
        import numpy as np
        from knee.reference import reference_reader
        with tempfile.TemporaryDirectory() as tmp:
            records = self.study(tmp)
            original = self.public._make_reader
            volume, mask, audit = cache.decode(self.public, 'study', records, tmp)
            self.assertIs(self.public._make_reader, original)
            direct, direct_mask = self.public.build_study('study', records, tmp, self.public._make_reader())
            self.assertTrue(np.array_equal(volume, direct)); self.assertTrue(np.array_equal(mask, direct_mask))
            strict, strict_failures = reference_reader(self.public)
            ref, ref_mask = self.public.build_study('study', records, tmp, strict)
            self.assertEqual(strict_failures, []); self.assertTrue(np.array_equal(volume, ref)); self.assertTrue(np.array_equal(mask, ref_mask))
            slots = audit['slots']
            self.assertEqual([s['frames'] for s in slots], [23, 5, 1, 0, 31])
            self.assertEqual([s['fluid_sensitive'] for s in slots], [1, 0, 1, None, 0])
            self.assertEqual([s['selected'] for s in slots], [True, True, True, False, True])
            self.assertEqual(mask[44:52].tolist(), [0] * 8)  # missing coronal non-FS slot is masked, not substituted
            self.assertTrue(mask[:44].all() and mask[52:].all())
            self.assertGreater(audit['reader_counts']['decoded_unique_slices'], 0)
            checks = cache.window_checks(self.public, volume, mask)
            self.assertEqual(checks['shape'], [42, 3, 384, 384])
            again, again_mask, _ = cache.decode(self.public, 'study', records, tmp)
            self.assertTrue(np.array_equal(volume, again) and np.array_equal(mask, again_mask))

    def test_reader_restored_after_failure_and_empty_study_rejected(self):
        original = self.public._make_reader
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                cache.decode(self.public, 'study', {'study': []}, tmp)
        self.assertIs(self.public._make_reader, original)

    def test_production_stress_parity(self):
        from knee import raptor_input
        report = raptor_input.stress_inputs(self.public)
        self.assertEqual(set(report['cases']), raptor_input.STRESS_CASES)
        self.assertTrue(all(v == 'PASS' for v in report['cases'].values()))


@unittest.skipUnless(HAVE_READER, 'Pinned Raptor source plus numpy/torch/cv2/pydicom required')
class StreamingTests(unittest.TestCase):
    arrays = FileTests.arrays

    def test_failure_preserves_completed_and_resume(self):
        import time
        public = SimpleNamespace(SLOTS=list(cache.SLOTS), MAXS=64, IMG=336, CROP_MM=140.0, K_EVAL=42,
                                 _eval_centers=lambda m, D, k: list(range(1, 1 + k)),
                                 eval_windows=lambda v, m, k, r: __import__('torch').zeros(k, 3, r, r))
        env = {'python': __import__('sys').version.split()[0], 'device': 'cpu', 'torch_threads': 1, 'opencv_threads': 1,
               'packages': dict.fromkeys(('torch', 'numpy', 'pydicom', 'opencv-python', 'pylibjpeg', 'pylibjpeg-libjpeg', 'pylibjpeg-openjpeg'), 'test')}
        ids = ['fixture-identity-one', 'fixture-identity-two']; v, m = self.arrays(); m[:] = 1; v[:, 0, 0] = 1
        contract = {'version': cache.VERSION, 'config': cache.CONFIG, 'shard_count': 1, 'shard_index': 0, 'membership': {'eligible': 2},
                    'partitions': [{'studies': 2}], 'code_revision': 'fixture', 'sources': {}, 'input_pins': {}, 'qualified_environment': env}
        slots = [{'plane': p, 'selected': True, 'fluid_sensitive': 1, 'frames': 20, 'count': c} for p, _, c in cache.SLOTS]

        def failing(public, uid, records, raw):
            if uid == ids[1]:
                raise ValueError('Private error ' + uid)
            return v.copy(), m.copy(), {'reader_counts': {'decoded_unique_slices': 1}, 'slots': slots}
        with tempfile.TemporaryDirectory() as tmp, patch('importlib.metadata.version', return_value='test'), patch('cv2.getNumThreads', return_value=1):
            root = Path(tmp) / 'cache'
            with patch.object(cache, 'decode', side_effect=failing), self.assertRaisesRegex(RuntimeError, 'sanitized aggregate'):
                cache.stream_cache(root, contract, ids, {}, Path(tmp), public, started=time.monotonic())
            text = (root / 'receipt.json').read_text(); report = json.loads(text)
            self.assertEqual(report['status'], 'ERROR'); self.assertEqual(report['completed_studies'], 1)
            self.assertFalse(any(uid in text for uid in ids))
            private = json.loads((root / 'index-private.json').read_text())
            self.assertEqual(private['entries'][0]['slots'], slots)
            with patch.object(cache, 'decode', side_effect=lambda *a: (v.copy(), m.copy(), {'reader_counts': {}, 'slots': slots})) as decode:
                done = cache.stream_cache(root, contract, ids, {}, Path(tmp), public, started=time.monotonic())
                self.assertEqual(decode.call_count, 1); self.assertEqual(done['status'], 'PASS'); self.assertEqual(done['resumed_studies'], 1)
            with self.assertRaises(ValueError):
                cache.stream_cache(root, contract | {'shard_index': 1}, ids, {}, Path(tmp), public)


@unittest.skipUnless(HAVE_READER and (EVIDENCE / 'data/raw/train.csv').is_file(), 'Pinned evidence root required')
class ActualBuildTests(unittest.TestCase):
    def test_actual_population_partitions_and_private_offline_cpu_kernel(self):
        builder = load_builder()
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / 'build'
            receipt = builder.build(dest, 'knee-raptor64-cache-shard0-20261013', evidence_root=EVIDENCE, public_source=PUBLIC, shard_index=0)
            c = json.loads((dest / 'build-contract.json').read_text())
            self.assertEqual([p['studies'] for p in c['partitions']], [1087, 1086, 1086, 1086])
            self.assertEqual((c['membership']['eligible'], c['membership']['training'], c['membership']['development']), (4345, 3475, 870))
            self.assertEqual(c['reader']['slots'], [list(s) for s in cache.SLOTS]); self.assertTrue(receipt['build_only'])
            meta = json.loads((dest / 'kernel-metadata.json').read_text())
            self.assertTrue(meta['is_private']); self.assertFalse(meta['enable_gpu']); self.assertFalse(meta['enable_internet'])
            self.assertEqual(meta['competition_sources'], ['rsna-knee-abnormality-detection'])
            self.assertLessEqual(len(meta['id'].split('/')[1]), 50)
            code = ''.join(json.loads((dest / 'notebook.ipynb').read_text())['cells'][1]['source']); compile(code, 'nb', 'exec')
            identity = json.loads((EVIDENCE / 'outputs/kaggle/sealed-identity-full-result/sealed_identity/sealed-manifest-private.json').read_text())
            self.assertFalse(any(r['uid'] in code for r in identity['records']))
            with self.assertRaises(ValueError):
                builder.build(dest, 'knee-raptor64-cache-shard0-20261013', evidence_root=EVIDENCE, public_source=PUBLIC, shard_index=0)
            with self.assertRaises(ValueError):
                builder.build(Path(tmp) / 'long', 'k' * 51, evidence_root=EVIDENCE, public_source=PUBLIC, shard_index=0)
            with self.assertRaises(ValueError):
                builder.build(Path(tmp) / 'bad', 'knee-raptor64-test', evidence_root=EVIDENCE, public_source=PUBLIC, shard_index=4)


if __name__ == '__main__':
    unittest.main()
