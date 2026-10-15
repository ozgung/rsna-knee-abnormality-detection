"""Recovery and metadata trust-boundary regressions; no real images or labels."""
import copy
import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

from knee import local64_cache as cache, local64_runtime as runtime
from scripts import collect_local64_cache_metadata as collector
from scripts import build_local64_cache as builder
from knee.fresh_reader_setup import dataset_root

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
    def test_all_four_bootstrap_configs_validate_gold_identity_without_staging_it_elsewhere(self):
        import hashlib
        ids = [f'fixture-train-{i}' for i in range(cache.TRAIN_COUNT)]
        gold = [f'fixture-gold-{i}' for i in range(cache.GOLD_COUNT)]
        parts = cache.make_shards(ids, 4)
        common = dict(version=cache.VERSION, shard_count=4,
            training_membership_sha256=cache.digest(ids), gold_membership_sha256=cache.digest(gold),
            frozen32={'uid_sha256': [hashlib.sha256(u.encode()).hexdigest() for u in ids[:32]]},
            partitions=[{'studies': len(ns), 'ordinals_sha256': cache.digest(ns),
                         'ordered_ids_sha256': cache.digest([ids[n] for n in ns])} for ns in parts])
        for i in range(4):
            contract = dict(common, shard_index=i)
            cfg = runtime.make_runtime_config(contract=contract, training_ids=ids, all_gold_ids=gold,
                                               data_root='/fixture', code='/fixture', out='/fixture')
            got_ids, cached_gold, index, members = runtime.validate_runtime_population(cfg)
            self.assertEqual(got_ids, ids)
            self.assertEqual(index, i)
            self.assertEqual(members, parts[i])
            self.assertEqual(cached_gold, gold if i == 0 else [])
            self.assertEqual(cfg['all_gold_ids'], gold)
            broken = dict(cfg); broken.pop('all_gold_ids')
            with self.assertRaises(ValueError):
                runtime.validate_runtime_population(broken)
            if i:
                broken = dict(cfg, gold_ids=gold)
                with self.assertRaises(ValueError):
                    runtime.validate_runtime_population(broken)

    def test_notebook_exec_exception_preserves_private_diagnostic(self):
        source = ("import os\nseen=[]\n"
                  "def private_failure(kind,error,tb): seen.append((kind.__name__,str(error)))\n"
                  "os.environ.update()\nraise ValueError('fixture private detail')\n")
        namespace = {}
        with self.assertRaisesRegex(RuntimeError, 'private diagnostics preserved'):
            exec(builder.guard_bootstrap(source), namespace)
        self.assertEqual(namespace['seen'], [('ValueError', 'fixture private detail')])

    def test_bootstrap_attachment_manifest_includes_all_required_mounts(self):
        metadata = builder.kernel_metadata('fixture-local64')
        self.assertEqual(metadata['dataset_sources'], list(builder.DATASET_REQUIREMENTS))
        self.assertIn('heliosli/rsna-knee-compact-v1', metadata['dataset_sources'])
        self.assertEqual(metadata['title'], 'fixture-local64')
        self.assertFalse(metadata['enable_gpu'])
        self.assertFalse(metadata['enable_internet'])
        with tempfile.TemporaryDirectory() as tmp:
            inputs = Path(tmp)
            for ref, names in builder.DATASET_REQUIREMENTS.items():
                if ref == 'heliosli/rsna-knee-compact-v1':
                    continue  # Original failed packets omitted this attachment.
                root = inputs / ref.split('/')[1]; root.mkdir()
                for name in names:
                    (root / name).write_bytes(b'fixture-only')
            with self.assertRaises(ValueError):
                {ref: dataset_root(inputs, ref, names) for ref, names in builder.DATASET_REQUIREMENTS.items()}
            compact = inputs / 'rsna-knee-compact-v1'; compact.mkdir()
            for name in builder.DATASET_REQUIREMENTS['heliosli/rsna-knee-compact-v1']:
                (compact / name).write_bytes(b'fixture-only')
            resolved = {ref: dataset_root(inputs, ref, names) for ref, names in builder.DATASET_REQUIREMENTS.items()}
            self.assertEqual(set(resolved), set(metadata['dataset_sources']))

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
