"""Synthetic A‴ fixed-profile, checkpoint-binding and recovery regressions."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from knee import public_nafull_aprime as A


class NaFullAdapterTests(unittest.TestCase):
    def test_cpu_receipt_binds_actual_checkpoint_state_and_k62_grid(self):
        receipt = {'checkpoint_sha256': A.CHECKPOINT['sha256'],
            'checkpoint_bytes': A.CHECKPOINT['bytes'], 'state_sha256': A.CHECKPOINT['state_sha256'],
            'strict_state_load': True, 'device': 'cpu', 'windows_shape': [62, 3, 384, 384],
            'forward_windows': 62, 'uses_raw_data': False, 'uses_labels': False,
            'dread_pin_method_check': True, 'constructor_grid_sha256': 'a' * 64}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'cpu.json'
            path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, 'NAFULL_EXACT_CPU_RECEIPT'):
                A.checkpoint_manifest(path)

    def test_profile_requires_dedicated_k62_recipe_and_target_order(self):
        cpu = {'checkpoint': dict(A.CHECKPOINT), 'constructor_grid_sha256': 'b' * 64}
        manifest = {'checkpoint': dict(A.CHECKPOINT), 'constructor_grid_sha256': 'b' * 64,
                    'profile': A.RAPTOR_PROFILE, 'labels': list(A.LABELS)}
        self.assertTrue(A.require_profile(manifest, cpu))
        for key, value, code in [('checkpoint', {'sha256': 'old320'}, 'NAFULL_PROFILE_CHECKPOINT'),
                ('profile', {'windows': 42}, 'NAFULL_PROFILE_RECIPE'),
                ('labels', list(A.LABELS[::-1]), 'NAFULL_PROFILE_LABEL_ORDER')]:
            bad = {**manifest, key: value}
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, code):
                A.require_profile(bad, cpu)

    @classmethod
    def setUpClass(cls):
        try:
            import numpy as np
            import pandas as pd
        except ImportError as e:
            raise unittest.SkipTest(f'optional rank dependencies unavailable: {e}')
        cls.np, cls.pd = np, pd

    def arrays(self, n):
        np = self.np
        x = np.tile(np.linspace(.05, .95, n)[:, None], (1, 12))
        return x.copy(), x[::-1].copy(), (x * .7 + .1).copy()

    def test_fixed_ordered_weights_ranks_and_r224_csv_precision(self):
        np, pd = self.np, self.pd
        na, r224, r7 = self.arrays(4)
        r224[:, 0] = [.2, .2, .8, .8]
        zero = np.zeros(4, dtype=bool)
        result = A.recover(na, r224, r7, zero, zero, zero)
        self.assertEqual(result['weights'], A.WEIGHTS)
        self.assertTrue(np.array_equal(result['ranks']['r224'][:, 0], [.375, .375, .875, .875]))
        expected = .5 * (.65 * result['ranks']['na_full'] + .35 * result['ranks']['r224']) + .5 * result['ranks']['r7']
        self.assertTrue(np.array_equal(result['blend'], expected))
        ordered = A.recover_ws2_order(na, r7, r224, zero, zero, zero)
        self.assertTrue(np.array_equal(ordered['blend'], expected))
        self.assertEqual(ordered['weights'], {'old320': .325, 'crop224': .175, 'r7': .5})

    def test_all_population_global_mean_recovery_and_fatal_gates(self):
        np = self.np
        na, r224, r7 = self.arrays(100)
        masks = [np.zeros(100, dtype=bool) for _ in range(3)]
        masks[0][[0, 1, 2]] = True; masks[1][[1, 2]] = True; masks[2][[2]] = True
        for values, mask in zip((na, r224, r7), masks): values[mask] = np.nan
        result = A.recover(na, r224, r7, *masks)
        self.assertEqual(result['population'], 100)
        self.assertEqual(result['cap'], 3)
        self.assertEqual(result['union_recovered'], 3)
        self.assertTrue(np.isfinite(result['blend']).all())
        self.assertTrue(np.allclose(result['filled']['na_full'][2], np.nanmean(na, axis=0)))
        masks[2][3] = True; r7[3] = np.nan
        with self.assertRaisesRegex(ValueError, 'UNION_RECOVERY_CAP'):
            A.recover(na, r224, r7, *masks)
        masks[2][3] = False; r7[3] = .5
        with self.assertRaisesRegex(ValueError, 'FATAL_ERRORS_REQUIRE_ABORT'):
            A.recover(na, r224, r7, *masks, fatal_errors=('timeout',))

    def test_runtime_facades_dispatch_new_base_and_archived_r224_mergers(self):
        import hashlib
        from knee import public_r7_recovery_entry as R7
        from knee import public_s6_recovery_entry as S6
        from knee import public_s6_recovery_b0prime as B0P
        from knee import public_ws2_entry as WS2
        from knee.public_nafull_aprime_runtime import make_base_facade, make_ws2_facade

        base = make_base_facade({}, rehearsal=True)
        ws2 = make_ws2_facade(base)
        self.assertIs(base.old_worker.__code__, B0P.old_worker.__code__)
        self.assertIs(base.r7_worker.__code__, R7.r7_worker.__code__)
        self.assertIs(base.merge.__code__, R7.merge.__code__)
        self.assertIs(ws2.R7, base)
        self.assertIs(ws2.run.__code__, WS2.run.__code__)
        self.assertEqual(base.METHOD['old'], A.METHOD_OLD)

        def fixture(root, arm, version):
            ids = ['fixture-0', 'fixture-1']
            assignments = S6.assignments(ids)
            reports = []
            for board, assigned in enumerate(assignments):
                folder = root / f'board{board}-{arm}' / arm
                folder.mkdir(parents=True)
                p = self.np.full((len(assigned), 12), .4 + board / 10, dtype=self.np.float64)
                arrays = {'study_ids': self.np.asarray(assigned, dtype='U16'),
                    'probabilities': p, 'statuses': self.np.asarray(['genuine'] * len(assigned), dtype='U20')}
                if arm == 's6': arrays['logits'] = self.np.zeros((len(assigned), 4, 12), dtype=self.np.float32)
                self.np.savez(folder / 'predictions.npz', **arrays)
                ledger = folder / 'predictions.npz'
                rows = [{'id': sid, 'status': 'genuine', 'cleanup': {'scratch_removed': True,
                    'side_override_removed': True}, 'state_healthy': True} for sid in assigned]
                reports.append({'version': version, 'status': 'PASS', 'board': board, 'arm': arm,
                    'assigned': assigned, 'completed': len(assigned), 'state_before': 'state',
                    'state_after': 'state', 'zero_imputed': True, 'fp32_recoveries': 0,
                    'ledger_sha256': hashlib.sha256(ledger.read_bytes()).hexdigest(), 'records': rows,
                    'genuine': len(assigned), 'failed': 0})
            return ids, reports

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids, reports = fixture(root, 's6', A.VERSION)
            merged, missing = ws2.merged_arm(ids, reports, root, 's6')
            self.assertEqual(merged.shape, (2, 12)); self.assertFalse(missing.any())
            ids, reports = fixture(root, 'crop224', R7.VERSION)
            merged, missing = ws2.merged_arm(ids, reports, root, 'crop224')
            self.assertEqual(merged.shape, (2, 12)); self.assertFalse(missing.any())

        adapted = base.require_rehearsal
        self.assertEqual(adapted.__closure__[0].cell_contents.__globals__['VERSION'], A.VERSION)
        ws2_checker = ws2.require_rehearsal
        archived_checker = next(cell.cell_contents for cell in ws2_checker.__closure__
            if callable(cell.cell_contents) and cell.cell_contents.__name__ == '_ws2_nafull_require_rehearsal')
        self.assertIs(archived_checker.__globals__['R7'], base)




class NaFullReaderIsolationTests(unittest.TestCase):
    def test_projection_capture_reaches_cloned_run_and_restores_original(self):
        """Execute the base orchestrator with synthetic workers and a WS2 hook."""
        import time
        from unittest.mock import patch
        try:
            import numpy as np
        except ImportError:
            self.skipTest('optional numpy unavailable')
        from knee import public_nafull_aprime_runtime as runtime
        from knee import public_r7_recovery_entry as r7
        base = runtime.make_base_facade({})
        probes = ['synthetic-probe-' + str(i) for i in range(36)]
        cfg = {'mode': 'rehearsal', 'binding': {}, 'nafull_base_binding': {}, 'probes': dict.fromkeys(probes, {}),
               'probe_order': probes, 'development_windows': [400] * 870}
        reports = [{'records': [{'production_seconds': 1., 'windows': 10}] * 20},
                   {'records': [{'production_seconds': 1., 'windows': 10}] * 19}]
        values = np.full((39, 12), .5)
        missing = np.zeros(39, dtype=bool)
        original = base.projection
        captured = {}
        def capture(project_cfg, phases, workers, setup, merge):
            captured.update(base_setup_seconds=setup, base_merge_seconds=merge,
                            development_windows=project_cfg['development_windows'])
            return original(project_cfg, phases, workers, setup, merge)
        # Stub only infrastructure in the actual execution globals, so the
        # detached legacy facade reaches projection too and fails this test.
        with tempfile.TemporaryDirectory() as folder, patch.dict(base.launch.__globals__,
                test_contract=lambda *a, **k: (['StudyInstanceUID'] + A.LABELS,
                                               ['synthetic-visible-' + str(i) for i in range(3)], {}),
                read_series=lambda *a: dict.fromkeys(probes, []),
                stage=lambda *a: {},
                launch=lambda *a: (reports, {'wall_seconds': 30.}),
                merge=lambda *a: (values, missing),
                write_csv=lambda *a: 'synthetic-csv',
                require_rehearsal=lambda *a: None), patch.object(base.subprocess,
                    'check_output', return_value='0, synthetic-gpu0, Tesla T4\n1, synthetic-gpu1, Tesla T4'):
            base.projection = capture
            try:
                receipt = base.run(Path(folder), cfg, Path(folder) / 'base', started=time.monotonic())
            finally:
                base.projection = original
        self.assertEqual(receipt['status'], 'PASS')
        self.assertEqual(set(captured), {'base_setup_seconds', 'base_merge_seconds', 'development_windows'})
        self.assertEqual(captured['base_setup_seconds'], receipt['setup_seconds'])
        self.assertEqual(captured['development_windows'], cfg['development_windows'])
        self.assertGreaterEqual(captured['base_merge_seconds'], 0)
        self.assertIs(base.projection, original)
        self.assertIs(r7.projection, original)

    def test_old_worker_reader_is_isolated_and_restored_after_failure(self):
        import sys
        from unittest.mock import patch
        import knee
        from knee import public_nafull_aprime_runtime as runtime
        from knee import public_nafull_raptor_input as full_reader
        from knee import raptor_input as shared_reader
        from knee import reference as shared_reference
        from knee import public_nafull_reference as full_reference
        cfg={'arm':'old','control_state_sha256':A.CHECKPOINT['state_sha256'],
             'control_grid_sha256':A.CONSTRUCTOR_GRID_SHA256,
             'binding':{'qualified_state':{'old_grid':A.CONSTRUCTOR_GRID_SHA256}}}
        def worker(path):
            from knee import raptor_input as active
            self.assertIs(active,full_reader)
            self.assertIs(sys.modules['knee.raptor_input'],full_reader)
            from knee import reference as active_reference
            self.assertIs(active_reference,full_reference)
            self.assertTrue(callable(active_reference.infer_checked))
            self.assertTrue(active_reference.RAPTOR_WEIGHT_DATASET)
            raise RuntimeError('synthetic worker failure')
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'worker.json';path.write_text(json.dumps(cfg))
            with patch.object(runtime,'NA_OLD_WORKER',side_effect=worker):
                with self.assertRaisesRegex(RuntimeError,'synthetic worker failure'):
                    runtime.run_worker(path,rehearsal=True)
        self.assertIs(knee.raptor_input,shared_reader)
        self.assertIs(sys.modules['knee.raptor_input'],shared_reader)
        self.assertIs(knee.reference,shared_reference)
        self.assertIs(sys.modules['knee.reference'],shared_reference)

    def test_new_recovery_receipt_is_json_serializable_like_archived_ws2(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest('optional numpy unavailable')
        from knee import public_nafull_aprime_recovery_adapter as rec
        values=np.full((39,12),.5);missing=np.zeros(39,dtype=bool)
        result=rec.recover(values,values,values,missing,missing,missing)
        # Exactly the unchanged WS2 runtime's receipt selection.
        record={k:v for k,v in result.items() if k not in ('filled','ranks','source224_csv_values','raptor_slot','blend')}
        json.dumps(record,allow_nan=False)


if __name__ == '__main__':
    unittest.main()
