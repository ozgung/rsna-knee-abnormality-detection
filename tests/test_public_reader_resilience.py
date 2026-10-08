"""Synthetic production-function checks; no model weights, labels or raw images."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from knee.public_reader_contract import STRESS_CASES
from knee.reference import sha256
from knee.reference_blend import CONVNEXT_ENTRY_SHA256, CONVNEXT_SOURCE_HASHES, extract_reader

ROOT = Path(__file__).resolve().parents[1]


def definitions(source, names):
    nodes = [node for node in ast.parse(source).body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    return compile(ast.Module(body=nodes, type_ignores=[]), '<audited-functions>', 'exec')


@unittest.skipUnless(importlib.util.find_spec('numpy'), 'optional numpy unavailable')
class ResilienceTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.np = np
        class OpenCVError(Exception):
            def __init__(self, code):
                self.code = code
        self.cv_error = OpenCVError
        self.volume = np.arange(8*4*4, dtype=np.uint8).reshape(8, 4, 4)
        self.calls = []
        def healthy(path):
            self.calls.append(Path(path).name)
            return self.volume.copy(), {'n_bad': 0}
        self.healthy = healthy
        self.prep = SimpleNamespace(SIZE=4, load_series=healthy, _read=lambda path: [])
        self.audit = {}
        self.centers = lambda n, k, train: np.linspace(1, n-2, k).round().astype(int)
        self.ns = dict(np=np, json=json, root=Path('/synthetic'), prep=self.prep,
                       cv2=SimpleNamespace(error=OpenCVError), audit=self.audit,
                       model_code=SimpleNamespace(N_SLOTS=6, window_centres=self.centers))
        source = (ROOT/'scripts/public_reader_entry.py').read_text()
        self.assertEqual(sha256(source.encode()), CONVNEXT_ENTRY_SHA256)
        extract_reader(source)
        exec(definitions(source, {'count_event', 'fatal_series_error', 'load_selected_series', 'prepare_study', 'stress_inputs'}), self.ns)
        self.table = {'study': [['selected', 'never-retry'], ['other'], [], [], [], []]}
        self.missing = {'study': [[], ['other'], [], [], [], []]}

    def prepare(self, table=None):
        return self.ns['prepare_study']('study', 'test', self.table if table is None else table)

    def assert_arrays_equal(self, first, second):
        for a, b in zip(first, second):
            self.np.testing.assert_array_equal(a, b)

    def test_partial_unreadable_item_retains_identical_pixels_masks_and_positions(self):
        healthy = self.prepare()
        self.prep.load_series = lambda path: (self.volume.copy(), {'n_bad': int(Path(path).name == 'selected')})
        self.assert_arrays_equal(healthy, self.prepare())
        self.assertEqual(self.audit['decode_failures'], 1)
        self.assertEqual(self.audit['partial_series'], 1)

    def test_selected_exception_or_short_masks_only_that_slot_without_retry(self):
        expected = self.prepare(self.missing)
        for mode in ('exception', 'short'):
            self.calls.clear()
            def loader(path):
                self.calls.append(Path(path).name)
                if Path(path).name == 'selected':
                    if mode == 'exception':
                        raise ValueError('synthetic selected-series failure')
                    return self.volume[:2], {'n_bad': 1}
                return self.volume, {'n_bad': 0}
            self.prep.load_series = loader
            with self.subTest(mode=mode):
                self.assert_arrays_equal(expected, self.prepare())
                self.assertEqual(self.calls, ['selected', 'other'])
        self.assertEqual(self.audit['failed_series'], 1)
        self.assertEqual(self.audit['short_series'], 1)
        self.assertEqual(self.audit['series_error_types'], {'ValueError': 1})

    def test_all_empty_or_all_zero_fails_with_aggregate_not_identifier(self):
        for table in ({}, {'study': [['selected'], [], [], [], [], []]}):
            self.prep.load_series = lambda path: (self.np.zeros_like(self.volume), {'n_bad': 0})
            with self.assertRaisesRegex(RuntimeError, 'no usable image content.*missing_slots=.*no fallback') as error:
                self.prepare(table)
            self.assertNotIn('/synthetic', str(error.exception))
        self.assertEqual(self.audit['zero_studies'], 2)

    def test_valid_all_zero_slot_with_healthy_other_plane_preserves_source_mask(self):
        self.prep.load_series = lambda path: (self.np.zeros_like(self.volume) if Path(path).name == 'selected' else self.volume, {'n_bad': 0})
        arrays = self.prepare()
        self.assertEqual(arrays[1].tolist(), [True, True, False, False, False, False])
        self.assertEqual(self.audit['zero_decoded_series'], 1)

    def test_fatal_direct_and_internally_caught_read_errors_propagate_and_restore(self):
        for error in (MemoryError('synthetic'), TimeoutError('synthetic'), self.cv_error(-4)):
            for internal in (False, True):
                def read(path):
                    raise error
                def loader(path):
                    if not internal:
                        raise error
                    try:
                        self.prep._read(path)
                    except Exception:
                        pass
                    return self.volume, {'n_bad': 1}
                self.prep._read, self.prep.load_series = read, loader
                with self.subTest(error=type(error).__name__, internal=internal), self.assertRaises(type(error)):
                    self.prepare()
                self.assertIs(self.prep._read, read)
        self.assertEqual(self.audit['resource_errors'], 6)
        self.assertNotIn('failed_series', self.audit)

    def test_stress_checks_exercise_production_path_and_restore_loader_and_audit(self):
        before = self.prep.load_series, self.prep._read
        self.audit['real-counter'] = 123
        result = self.ns['stress_inputs']()
        self.assertEqual(result['cases'], dict.fromkeys(STRESS_CASES, 'PASS'))
        self.assertFalse(result['uses_raw_data']); self.assertFalse(result['uses_labels'])
        self.assertEqual(self.audit, {'real-counter': 123})
        self.assertEqual((self.prep.load_series, self.prep._read), before)

    @unittest.skipUnless(os.environ.get('KNEE_REFERENCE_PARENT'), 'requires pinned source snapshot')
    def test_pinned_public_loader_partial_item_and_fatal_read_with_public_inference_parity(self):
        source_dir = Path(os.environ['KNEE_REFERENCE_PARENT'])/'outputs/research/2026-10-08/convnext-audit'
        preprocess = (source_dir/'preprocess.py').read_text()
        self.assertEqual(sha256(preprocess.encode()), CONVNEXT_SOURCE_HASHES['preprocess.py'])
        ns = dict(np=self.np, os=os, SIZE=4, MAX_SLICES=64, PLANES=['Sagittal','Coronal','Axial'],
                  _canon_slice=lambda image, iop, ps, plane: (image, ps), _resample=lambda image, ps: image)
        exec(definitions(preprocess, {'load_series'}), ns)
        class PublicPrep:
            SIZE = 4
            load_series = staticmethod(ns['load_series'])
            @property
            def _read(self): return ns['_read']
            @_read.setter
            def _read(self, value): ns['_read'] = value
        self.prep = PublicPrep(); self.ns['prep'] = self.prep
        with tempfile.TemporaryDirectory() as directory:
            series = Path(directory)
            for name in ('0', '1', '2', '3', 'sidecar'): (series/name).touch()
            def read(path):
                if Path(path).name == 'sidecar': raise ValueError('synthetic unreadable item')
                index = int(Path(path).name)
                return [(self.np.arange(16).reshape(4,4).astype(float) + index*10, None, None, None, index)]
            self.prep._read = read
            expected, info = self.prep.load_series(str(series))
            self.assertEqual(info['n_bad'], 1); self.assertEqual(len(expected), 4)
            actual, info2 = self.ns['load_selected_series'](str(series))
            self.np.testing.assert_array_equal(actual, expected); self.assertEqual(info2, info)
            for error_type in (MemoryError, TimeoutError):
                def fatal(path): raise error_type('synthetic fatal')
                self.prep._read = fatal
                with self.assertRaises(error_type): self.ns['load_selected_series'](str(series))
                self.assertIs(self.prep._read, fatal)
        # Compare our actual preparation with the inspected author's Dataset method.
        fake_torch = SimpleNamespace(utils=SimpleNamespace(data=SimpleNamespace(Dataset=object)), from_numpy=lambda x:x)
        reference = dict(np=self.np, torch=fake_torch, N_SLOTS=6, SIZE=4, window_centres=self.centers)
        exec(definitions((source_dir/'infer.py').read_text(), {'DicomStudyDataset'}), reference)
        self.ns['prep'] = SimpleNamespace(SIZE=4, _read=lambda path: [], load_series=self.healthy)
        for table in (self.table, self.missing):
            reference['load_series'] = self.healthy
            expected = reference['DicomStudyDataset'](['study'], table, '/synthetic/test_series', 12)[0]
            self.assert_arrays_equal(self.prepare(table), expected)


if __name__ == '__main__': unittest.main()
