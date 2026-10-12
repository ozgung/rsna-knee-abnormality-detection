import unittest
try:
    import numpy as np
    import pandas as pd
except ImportError:
    np = None
from knee.public_s6_recovery import recover, merge_shards, freeze_protocol, digest, quantize

@unittest.skipIf(np is None, 'optional numpy/pandas unavailable')
class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.c = np.random.default_rng(7).uniform(size=(100, 12)).astype(np.float32)
        self.s = np.random.default_rng(9).uniform(size=(100, 12))
        self.z = np.zeros(100, dtype=bool)

    def test_cap_boundary_union(self):
        a, b = self.z.copy(), self.z.copy()
        a[:2] = True; b[1:3] = True
        self.assertEqual(recover(self.c, self.s, a, b)['union_recovered'], 3)
        b[3] = True
        with self.assertRaisesRegex(ValueError, 'UNION'): recover(self.c, self.s, a, b)

    def test_floor_actual_population(self):
        m = np.zeros(33, bool); m[0] = True
        with self.assertRaisesRegex(ValueError, 'UNION'): recover(self.c[:33], self.s[:33], m, m)
        m = np.zeros(34, bool); m[0] = True
        self.assertEqual(recover(self.c[:34], self.s[:34], m, m)['cap'], 1)

    def test_global_means_both_arms_rounding_status(self):
        a, b = self.z.copy(), self.z.copy(); a[0] = True; b[1:3] = True
        c, s = self.c.copy(), self.s.copy(); c[a] = np.nan; s[b] = np.nan
        r = recover(c, s, a, b)
        np.testing.assert_array_equal(r['control'][0], self.c[~a].astype(float).mean(0))
        np.testing.assert_array_equal(r['s6'][1], quantize(quantize(self.s[~b]).mean(0)[None])[0])
        self.assertEqual(r['statuses']['old320'][0], 'recovered')
        self.assertEqual(r['statuses']['s6'][0], 'genuine')
        self.assertEqual(r['union_recovered'], 3)

    def test_zero_mask_exact_original_ranks(self):
        r = recover(self.c, self.s, self.z, self.z)
        np.testing.assert_array_equal(r['control_rank'], pd.DataFrame(self.c).rank(method='average', pct=True))
        np.testing.assert_array_equal(r['s6_rank'], pd.DataFrame(quantize(self.s)).rank(method='average', pct=True))

    def test_ties_and_permutation(self):
        c = np.full((100, 12), .25); s = np.full_like(c, .12345649)
        a = self.z.copy(); a[:3] = True
        r = recover(c, s, a, a)
        np.testing.assert_array_equal(r['blend'], np.full_like(c, .505))
        perm = np.random.default_rng(3).permutation(100)
        base = recover(self.c, self.s, a, a)
        changed = recover(self.c[perm], self.s[perm], a[perm], a[perm])
        np.testing.assert_allclose(changed['blend'][np.argsort(perm)], base['blend'], atol=0, rtol=0)

    def test_shards_canonical_order_global_fill(self):
        a = self.z.copy(); a[:3] = True
        shards = []
        for ix in (np.arange(1, 100, 2)[::-1], np.arange(0, 100, 2)):
            shards.append(dict(ordinals=ix, control=self.c[ix], s6=self.s[ix], control_missing=a[ix], s6_missing=self.z[ix]))
        r = recover(*merge_shards(shards, 100)); direct = recover(self.c, self.s, a, self.z)
        np.testing.assert_array_equal(r['blend'], direct['blend'])
        with self.assertRaisesRegex(ValueError, 'COVERAGE'): merge_shards(shards[:1], 100)
        with self.assertRaisesRegex(ValueError, 'DUPLICATE'): merge_shards(shards + shards[:1], 100)

    def test_nonfinite_out_of_range_and_fatal_reject(self):
        for value in (np.nan, np.inf, -.01, 1.01):
            c = self.c.copy(); c[0, 0] = value
            with self.assertRaisesRegex(ValueError, 'GENUINE'): recover(c, self.s, self.z, self.z)
        with self.assertRaisesRegex(ValueError, 'FATAL'): recover(self.c, self.s, self.z, self.z, fatal_errors=['source_state'])
        with self.assertRaisesRegex(ValueError, 'CAP|SUCCESS'): recover(self.c, self.s, ~self.z, self.z)
        with self.assertRaisesRegex(ValueError, 'MASK'): recover(self.c, self.s, self.z.astype(int), self.z)

    def test_protocol_13_frozen_nested_masks_tail_tie(self):
        p = freeze_protocol(np.ones(870))
        self.assertEqual(len(p['scenarios']), 13)
        self.assertEqual(p['workload_order'][:26], list(range(26)))
        self.assertEqual(digest(p), digest(freeze_protocol(np.ones(870))))
        self.assertEqual(p['scenarios'][1]['old_missing'], p['permutation'][:8])
        for x in p['scenarios'][1:]:
            self.assertIn(len(set(x['old_missing']) | set(x['s6_missing'])), (8, 26))
            if x['name'].endswith('disjoint'):
                self.assertFalse(set(x['old_missing']) & set(x['s6_missing']))

if __name__ == '__main__': unittest.main()
