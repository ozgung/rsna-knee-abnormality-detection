"""Attachment resolution must not walk competition MRI trees."""
import json
import os
from pathlib import Path
import signal
import tempfile
import time
import unittest
from unittest.mock import patch

from knee.fresh_reader import fingerprint
from knee.fresh_reader_setup import (cache_root, dataset_root, notebook_root,
                                      resume_checkpoint, time_limit)


class FreshReaderSetupTests(unittest.TestCase):
    def test_explicit_mounts_resolve_with_all_tree_walks_forbidden(self):
        for modern in (False, True):
            with self.subTest(modern=modern), tempfile.TemporaryDirectory() as d:
                inputs = Path(d)
                # Distractors reproduce the problematic global recursive search.
                images = inputs/'competitions/rsna-knee-abnormality-detection/train_series/a/b'
                images.mkdir(parents=True)
                for name in ('llm_labels_full.csv', 'plan-private.json', 'last.pt'):
                    (images/name).write_text('must never be read')
                ds = inputs/'datasets/stevenleehans/rsna-knee-llm-report-labels' if modern else inputs/'rsna-knee-llm-report-labels'
                ds.mkdir(parents=True); (ds/'llm_labels_full.csv').write_text('fixture')
                nb = inputs/'notebooks/ozgung/cache' if modern else inputs/'cache'
                cache = nb/'fresh_reader_cache'; cache.mkdir(parents=True)
                plan = {'config': {'windows': 4}, 'records': []}
                plan['fingerprint'] = fingerprint(plan)
                (cache/'plan-private.json').write_text(json.dumps(plan))
                for name in ('cache-index-private.json', 'cache-pilot.json', 'cache-complete.json'):
                    (cache/name).write_text('{}')
                train = nb/'fresh_reader_training'; train.mkdir()
                for name in ('last.pt', 'best.pt'): (train/name).write_bytes(b'owned fixture')
                (train/'run-contract.json').write_text(json.dumps({'plan_fingerprint': plan['fingerprint']}))
                forbidden = AssertionError('Input directory enumeration is forbidden')
                with patch.object(Path, 'rglob', side_effect=forbidden), \
                     patch.object(Path, 'glob', side_effect=forbidden), \
                     patch.object(Path, 'iterdir', side_effect=forbidden), \
                     patch.object(os, 'walk', side_effect=forbidden), \
                     patch.object(os, 'scandir', side_effect=forbidden):
                    self.assertEqual(dataset_root(inputs, 'stevenleehans/rsna-knee-llm-report-labels', ['llm_labels_full.csv']), ds)
                    self.assertEqual(cache_root(inputs, 'ozgung/cache', plan['fingerprint'], complete=True), cache)
                    self.assertEqual(resume_checkpoint(inputs, 'ozgung/cache', plan['fingerprint']), train/'last.pt')
                    with self.assertRaisesRegex(ValueError, 'found 0'):
                        dataset_root(inputs, 'other/missing', ['llm_labels_full.csv'])

    def test_ambiguous_or_mutated_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            inputs = Path(d)
            roots = [inputs/'notebooks/ozgung/cache/fresh_reader_cache', inputs/'cache/fresh_reader_cache']
            plan = {'config': {'windows': 4}}
            plan['fingerprint'] = fingerprint(plan)
            for root in roots:
                root.mkdir(parents=True)
                (root/'plan-private.json').write_text(json.dumps(plan))
                (root/'cache-index-private.json').write_text('{}')
                (root/'cache-pilot.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'found 2'):
                cache_root(inputs, 'ozgung/cache', plan['fingerprint'])
            (roots[1]/'plan-private.json').unlink()
            plan['config']['windows'] = 8
            (roots[0]/'plan-private.json').write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                cache_root(inputs, 'ozgung/cache', plan['fingerprint'])

    def test_resume_requires_owned_complete_matching_pair(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/'train/fresh_reader_training'; root.mkdir(parents=True)
            (root/'last.pt').write_bytes(b'fixture')
            (root/'run-contract.json').write_text(json.dumps({'plan_fingerprint': 'expected'}))
            with self.assertRaisesRegex(ValueError, 'found 0'):
                resume_checkpoint(d, 'ozgung/train', 'expected')
            (root/'best.pt').write_bytes(b'fixture')
            with self.assertRaisesRegex(ValueError, 'configured owner'):
                resume_checkpoint(d, 'other/train', 'expected')
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                resume_checkpoint(d, 'ozgung/train', 'changed')
            for ref in ('ozgung/../train', 'ozgung/..', '/train', 'train'):
                with self.assertRaises(ValueError):
                    notebook_root(d, ref, 'fresh_reader_training', ['last.pt'])

    @unittest.skipUnless(hasattr(signal, 'setitimer'), 'Requires POSIX timer')
    def test_setup_deadline_interrupts_blocking_wait_and_cleans_timer(self):
        before = signal.getsignal(signal.SIGALRM)
        with self.assertRaisesRegex(TimeoutError, 'setup deadline'):
            with time_limit(.02, 'setup'):
                time.sleep(.2)
        self.assertEqual(signal.getsignal(signal.SIGALRM), before)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL)[0], 0.)

    @unittest.skipUnless(hasattr(signal, 'setitimer'), 'Requires POSIX timer')
    def test_nested_setup_does_not_extend_stage_deadline(self):
        started = time.monotonic()
        with self.assertRaises(TimeoutError):
            with time_limit(.04, 'stage'):
                with time_limit(5, 'setup'):
                    time.sleep(.2)
        self.assertLess(time.monotonic() - started, .18)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL)[0], 0.)


if __name__ == '__main__':
    unittest.main()
