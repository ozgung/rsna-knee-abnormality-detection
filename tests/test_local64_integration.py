"""Recovery and metadata trust-boundary regressions; no real images or labels."""
import copy
import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

from knee import local64_cache as cache, local64_runtime as runtime
from scripts import collect_local64_cache_metadata as collector

HAVE_NUMPY = importlib.util.find_spec('numpy') is not None


def entry_fixture():
    slots = [{'plane': p, 'fluid_sensitive': f, 'count': n, 'selected': False,
              'series_key': None, 'frame_count': 0} for p, f, n in cache.SLOTS]
    slots[0].update(selected=True, series_key='a' * 24, frame_count=3,
                    source_fluid_sensitive='1', source_fat_suppression='FS')
    return dict(ordinal=0, StudyInstanceUID='fixture-only', file=cache.filename(0),
                slots=slots, anchor={'path': 'learned'}, side={'flipped': False},
                joint_mask={'schema': cache.MASK_SCHEMA},
                **{k: 'a' * 64 for k in ('file_sha256', 'volume_sha256', 'mask_sha256', 'joint_mask_sha256')})


class MetadataTests(unittest.TestCase):
    def test_identity_binding_rejects_changed_duplicate_or_reordered_studies(self):
        rows = [{'StudyInstanceUID': 'fixture-a'}, {'StudyInstanceUID': 'fixture-b'}]
        expected = cache.digest(['fixture-a', 'fixture-b'])
        self.assertEqual(collector.verify_identities(rows, expected), {'fixture-a', 'fixture-b'})
        for bad in (rows[::-1], [rows[0], rows[0]], [{'StudyInstanceUID': 'fixture-gold'}, rows[1]]):
            with self.assertRaises(ValueError):
                collector.verify_identities(bad, expected)

    def test_metadata_mode_requires_bound_hashes_without_local_cache_pixels(self):
        row = entry_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            collector.verify_cache_files(root, [row], verify_pixels=False)
            with self.assertRaises(ValueError):
                collector.verify_cache_files(root, [row], verify_pixels=True)
            bad = copy.deepcopy(row); bad.pop('file_sha256')
            with self.assertRaises(ValueError):
                collector.verify_cache_files(root, [bad], verify_pixels=False)

    def test_resume_rebuilds_only_derived_staging_and_preserves_raw_pixels(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / 'raw'; raw.mkdir()
            images = raw / 'train_series' / 'fixture-only' / 'fixture-series'; images.mkdir(parents=True)
            frame = images / 'frame.dcm'; frame.write_bytes(b'synthetic-not-dicom')
            with (raw / 'train_series.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=runtime.SERIES_COLUMNS)
                writer.writeheader(); writer.writerow(dict(zip(runtime.SERIES_COLUMNS,
                    ['fixture-only', 'fixture-series', '1', 'FS', 'Sagittal'])))
            stage = Path(tmp) / 'derived'
            first = runtime._stage_image_only_inputs(raw, stage, ['fixture-only'])
            second = runtime._stage_image_only_inputs(raw, stage, ['fixture-only'])
            self.assertEqual(first, second)
            self.assertEqual(frame.read_bytes(), b'synthetic-not-dicom')
            self.assertEqual((stage / 'train.csv').read_text().splitlines(), ['StudyInstanceUID', 'fixture-only'])
            (stage / 'unexpected').write_text('preserve')
            with self.assertRaises(ValueError):
                runtime._stage_image_only_inputs(raw, stage, ['fixture-only'])
            self.assertTrue((stage / 'unexpected').exists())

    @unittest.skipUnless(HAVE_NUMPY, 'NumPy required')
    def test_resume_rejects_npz_byte_tampering_with_unchanged_arrays(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = root / cache.filename(0)
            volume = np.ones((64,352,352), np.uint8); mask = np.ones(64,np.uint8)
            volume[18:] = 0; mask[18:] = 0
            row = entry_fixture(); row.update(cache.save_npz(path, volume, mask, np.zeros((352,352),np.uint8)))
            index = {'version': cache.VERSION, 'contract_sha256': 'fixture-contract', 'entries': [row]}
            runtime._index_rows(root, index, 'fixture-contract', [0], ['fixture-only'])
            # ZIP archive comments change file identity while leaving all three arrays intact.
            import zipfile
            with zipfile.ZipFile(path, 'a') as z:
                z.comment = b'changed artifact'
            self.assertTrue(np.array_equal(cache.read_npz(path)[0], volume))
            with self.assertRaisesRegex(ValueError, 'file identity'):
                runtime._index_rows(root, index, 'fixture-contract', [0], ['fixture-only'])
            with self.assertRaisesRegex(ValueError, 'file identity'):
                collector.verify_cache_files(root, [row], verify_pixels=True)


if __name__ == '__main__':
    unittest.main()
