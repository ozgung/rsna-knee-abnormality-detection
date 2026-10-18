import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.summarize_native_joint_roi import summarize


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.binding = {'protocol': {'study_count': 2, 'treatment_joint_views': 3}}
        self.build = self.root / 'build.json'
        self.build.write_text(json.dumps({'binding': self.binding}))
        self.report = dict(binding=self.binding, status='PASS_READER_MECHANICS',
                           completed_studies=2, train_inference_parity='PASS_SHARED_ADAPTER_REPEAT',
                           localizer_state_before='frozen', localizer_state_after='frozen',
                           advancement_allowed=False, elapsed_seconds=12., peak_rss_bytes=1024,
                           studies=[], fallback_series={'missing_anchor': 2})
        for i in range(2):
            crops = {'private-a': {'reason': 'missing_anchor'},
                     'private-b': {'reason': None if i == 0 else 'missing_anchor'}}
            private = dict(study={'series': [dict(SeriesInstanceUID='private-a', plane='Sagittal'),
                                             dict(SeriesInstanceUID='private-b', plane='Coronal')]},
                           series_crops=crops,
                           triplet_plan=[{'series': sid} for sid in ('private-a', 'private-b', 'private-b')])
            (self.root / f'{i:03d}-source-private.json').write_text(json.dumps(private))
            self.report['studies'].append(dict(ordinal=i, series=list(reversed(list(crops.values()))),
                                              fallback_treatment_triplets=1 if i == 0 else 3,
                                              shared_reader_repeat_parity=True))
        self.report['fallback_series']['missing_anchor'] = 3

    def run_summary(self):
        (self.root / 'receipt.json').write_text(json.dumps(self.report))
        return summarize(self.root, self.build)

    def test_mixed_planes_and_repeated_slots(self):
        result = self.run_summary()
        self.assertEqual(result['series']['fraction'], .75)
        self.assertEqual(result['treatment_joint_slots']['fraction'], 4 / 6)
        self.assertEqual(result['studies_with_any_slot_fallback']['fraction'], 1)
        self.assertEqual(result['studies_with_all_slots_fallback']['fraction'], .5)
        self.assertEqual(result['per_plane']['Coronal']['treatment_joint_slots']['total'], 4)
        self.assertIsNone(result['per_plane']['Axial']['series']['fraction'])
        self.assertFalse(result['advancement_allowed'])
        self.assertNotIn('private-a', json.dumps(result))

    def test_rejects_terminal_integrity_failures(self):
        original = copy.deepcopy(self.report)
        variants = [('status', 'RUNNING'), ('completed_studies', 1),
                    ('localizer_state_after', 'changed'), ('advancement_allowed', True),
                    ('elapsed_seconds', float('nan')), ('fallback_series', {})]
        for key, value in variants:
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.report = copy.deepcopy(original)
                self.report[key] = value
                self.run_summary()

    def test_rejects_build_mismatch(self):
        self.report['binding'] = {'protocol': {'study_count': 3}}
        with self.assertRaises(ValueError):
            self.run_summary()

    def test_rejects_wrong_slot_and_crop_receipts(self):
        for key, value in [('fallback_treatment_triplets', 0), ('series', [])]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                old = self.report['studies'][0][key]
                self.report['studies'][0][key] = value
                try:
                    self.run_summary()
                finally:
                    self.report['studies'][0][key] = old


if __name__ == '__main__':
    unittest.main()
