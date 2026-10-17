"""A-prime-w changes only the final per-finding Raptor share of the sealed A-prime entry."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from knee import public_ws2w_entry as W
from knee import public_ws2_entry as WS2
from knee import public_ws2_recovery as REC
from knee.public_s6_qualification import LABELS, file_sha
from scripts import build_public_ws2w_projection_probe as B


def numeric():
    try:
        import numpy, pandas  # noqa: F401
    except ImportError:
        raise unittest.SkipTest('optional numpy/pandas unavailable')


def synthetic_arms(n=39, seed=7):
    import numpy as np
    rng = np.random.default_rng(seed)
    return {name: (rng.random((n, 12)), np.zeros(n, dtype=bool)) for name in REC.ARMS}


class ProfileTests(unittest.TestCase):
    def test_exact_weights_map_and_version(self):
        self.assertEqual(W.VERSION, 'ws2-a-prime-r224-v2w')
        self.assertNotEqual(W.VERSION, WS2.VERSION)
        self.assertEqual(W.RAPTOR_DEFAULT, .5)
        self.assertEqual(W.RAPTOR_PER_FINDING, {'Lateral OA': .2, 'PF OA': .6, 'Synovitis': .8, "Baker's": .35})
        shares = dict(zip(LABELS, W.raptor_shares()))
        self.assertEqual(shares['Medial OA'], .5)
        self.assertEqual({c for c, w in shares.items() if w != .5}, {'Lateral OA', 'PF OA', 'Synovitis', "Baker's"})
        self.assertEqual(W.profile()['raptor_share_by_label'], [shares[c] for c in LABELS])
        self.assertEqual(B.EXPECTED_SHARES, W.RAPTOR_PER_FINDING)

    def test_profile_survives_payload_json_roundtrip(self):
        self.assertEqual(json.loads(json.dumps(W.profile())), W.profile())


class BlendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        numeric()

    def test_only_four_findings_change_and_arithmetic_is_raptor_share(self):
        import numpy as np
        arms = synthetic_arms()
        base = REC.recover(*(arms[n][0] for n in REC.ARMS), *(arms[n][1] for n in REC.ARMS))
        new = W.recover(*(arms[n][0] for n in REC.ARMS), *(arms[n][1] for n in REC.ARMS))
        for j, c in enumerate(LABELS):
            w = W.RAPTOR_PER_FINDING.get(c, .5)
            expected = w * base['raptor_slot'][:, j] + (1 - w) * base['ranks']['r7'][:, j]
            self.assertTrue(np.array_equal(new['blend'][:, j], expected), c)
            self.assertEqual(np.array_equal(new['blend'][:, j], base['blend'][:, j]), c not in W.RAPTOR_PER_FINDING, c)
        self.assertTrue(np.array_equal(new['raptor_slot'], base['raptor_slot']))
        self.assertTrue(all(np.array_equal(new['ranks'][n], base['ranks'][n]) for n in REC.ARMS))
        self.assertEqual(new['weights']['Synovitis'], {'old320': .65 * .8, 'crop224': .35 * .8, 'r7': 1 - .8})

    def test_all_default_profile_is_bit_identical_to_aprime(self):
        import numpy as np
        arms = synthetic_arms(seed=3)
        base = REC.recover(*(arms[n][0] for n in REC.ARMS), *(arms[n][1] for n in REC.ARMS))
        with mock.patch.object(W, 'RAPTOR_PER_FINDING', {}):
            same = W.reblend(base)
        self.assertTrue(np.array_equal(same['blend'], base['blend']))

    def test_recovery_cap_and_masks_still_enforced(self):
        import numpy as np
        arms = synthetic_arms()
        masks = [np.zeros(39, dtype=bool) for _ in REC.ARMS]
        masks[2][:2] = True
        with self.assertRaisesRegex(ValueError, 'UNION_RECOVERY_CAP'):
            W.recover(*(arms[n][0] for n in REC.ARMS), *masks)


class RunWrapperTests(unittest.TestCase):
    def cfg(self, **change):
        return {'mode': 'entry', 'ws2w_blend': W.profile(), **change}

    def test_swaps_recovery_only_during_frozen_run_and_restores(self):
        seen = {}
        def fake(root, cfg, out, *, started):
            seen['rec'] = WS2.REC
            return {'status': 'PASS'}
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(WS2, 'run', fake):
            r = W.run('root', self.cfg(), folder, started=0.)
            saved = json.loads((Path(folder) / 'receipt.json').read_text())
        self.assertIs(seen['rec'], W.RECOVERY)
        self.assertIs(WS2.REC, REC)
        self.assertEqual(r['ws2w_blend'], W.profile())
        self.assertEqual(saved['ws2w_blend']['version'], W.VERSION)

    def test_restores_on_failure(self):
        def fake(*a, **k):
            raise TimeoutError('synthetic')
        with mock.patch.object(WS2, 'run', fake), self.assertRaises(TimeoutError):
            W.run('root', self.cfg(), 'unused', started=0.)
        self.assertIs(WS2.REC, REC)

    def test_rejects_rehearsal_or_unbound_profile(self):
        for cfg in (self.cfg(mode='rehearsal'), self.cfg(ws2w_blend=None),
                    self.cfg(ws2w_blend={**W.profile(), 'raptor_default': .4})):
            with self.subTest(cfg=cfg.get('mode')), self.assertRaisesRegex(ValueError, 'EXACT_WS2W_ENTRY_PROFILE'):
                W.run('root', cfg, 'unused', started=0.)


class BuilderBoundaryTests(unittest.TestCase):
    def test_named_waiver_and_slug_required(self):
        args = dict(rehearsal_build='x', rehearsal_result='y', destination='z', slug=B.SLUG,
                    failed_receipt_sha256=B.APRIME['failed_receipt_sha256'])
        for waiver in (None, 'projection-8.3h-20261015', 'a3prime-projection-20261016'):
            with self.subTest(waiver=waiver), self.assertRaisesRegex(ValueError, 'APRIME_W_WAIVER'):
                B.build_probe(user_waiver=waiver, **args)
        with self.assertRaisesRegex(ValueError, 'SEPARATELY_NAMED_ENTRY_SLUG'):
            B.build_probe(user_waiver=B.WAIVER, **{**args, 'slug': 'ws2-r224-probe-waiver-20261015'})
        self.assertEqual(B.APRIME['failed_receipt_sha256'], B.PROBE.FAILED_RECEIPT_SHA)


PACKET = B.KAGGLE / B.SLUG


@unittest.skipUnless((PACKET / 'build-receipt.json').is_file() and (B.APRIME['root'] / 'notebook.ipynb').is_file(),
                     'sealed A-prime and A-prime-w packets are local ignored outputs')
class PacketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.aprime, cls.aprime_code = B.unpack_notebook(B.APRIME['root'] / 'notebook.ipynb')
        cls.entry, cls.code = B.unpack_notebook(PACKET / 'notebook.ipynb')
        cls.receipt = json.loads((PACKET / 'build-receipt.json').read_bytes())

    def test_sealed_files_and_manifest(self):
        self.assertEqual(self.receipt['files'], {n: file_sha(PACKET / n) for n in ('notebook.ipynb', 'kernel-metadata.json')})
        manifest = json.loads((PACKET / 'MANIFEST.json').read_bytes())
        self.assertEqual(manifest['payload_sha256'], B.digest(self.entry))
        for name, pin in manifest['files'].items():
            self.assertEqual(pin['sha256'], file_sha(PACKET / name), name)
        self.assertEqual((self.receipt['version'], self.receipt['qualification_pass']), (W.VERSION, False))
        self.assertEqual(self.receipt['waiver']['id'], 'aprime-w-projection-20261017')
        self.assertEqual(self.receipt['waiver']['failed_receipt_sha256'], B.PROBE.FAILED_RECEIPT_SHA)
        self.assertEqual(self.receipt['waiver']['authorization']['reported_by'], 'Claude')

    def test_no_other_change_against_aprime(self):
        self.assertEqual(B.digest(self.aprime), B.APRIME['payload'])
        self.assertTrue(B.require_no_other_change(self.aprime, self.entry))
        self.assertEqual(set(self.entry['sources']) - set(self.aprime['sources']), {B.NEW_SOURCE})
        self.assertEqual(self.entry['sources'][B.NEW_SOURCE], (B.PROJECT / B.NEW_SOURCE).read_text())
        self.assertEqual(self.entry['cfg']['ws2w_blend'], W.profile())
        self.assertEqual(self.entry['cfg']['ws2_binding'], self.aprime['cfg']['ws2_binding'])
        self.assertEqual(self.entry['cfg']['r224_state_pin'], self.aprime['cfg']['r224_state_pin'])
        self.assertTrue(B.require_bootstrap_preservation(self.code, self.aprime_code))
        self.assertEqual(self.code, B.bootstrap(self.entry))

    def test_metadata_private_offline_t4_no_timeout(self):
        meta = json.loads((PACKET / 'kernel-metadata.json').read_bytes())
        old = json.loads((B.APRIME['root'] / 'kernel-metadata.json').read_bytes())
        self.assertEqual((meta['id'], meta['title']), ('ozgung/' + B.SLUG, B.SLUG))
        self.assertEqual((meta['is_private'], meta['enable_internet'], meta['machine_shape']), (True, False, 'NvidiaTeslaT4'))
        self.assertNotIn('timeout', meta)
        self.assertEqual({k: v for k, v in meta.items() if k not in ('id', 'title')},
                         {k: v for k, v in old.items() if k not in ('id', 'title')})

    def test_fresh_isolated_unpacked_import_runs_embedded_blend(self):
        numeric()
        import numpy as np
        arms = synthetic_arms(seed=11)
        report, blends = B.isolated_replay(PACKET / 'notebook.ipynb', {'synthetic': arms})
        self.assertEqual((report['status'], report['isolated'], report['all_from_unpacked_project']), ('PASS', True, True))
        local = W.recover(*(arms[n][0] for n in REC.ARMS), *(arms[n][1] for n in REC.ARMS))
        self.assertTrue(np.array_equal(blends['synthetic'], local['blend']))

    def test_recorded_real_replay_only_changed_findings_differ(self):
        for key, n in (('qual39', 39), ('visible3', 3)):
            r = self.receipt['blend_replay'][key]
            self.assertEqual(r['studies'], n)
            self.assertTrue(set(r['differing_findings']) <= set(W.RAPTOR_PER_FINDING))
            self.assertTrue(r['unchanged_findings_bit_identical'] and r['embedded_equals_independent_exactly'])
            self.assertTrue(r['aprime_csv_recomputed'])


if __name__ == '__main__':
    unittest.main()
