import unittest

try:
    import numpy as np
    import sklearn  # noqa: F401
except ImportError:  # dependency-free discovery
    np = None


def _rows(uid, site='S', sex='F', planes=('Sagittal', 'Coronal', 'Axial'), contrast=''):
    return [{'StudyInstanceUID': uid, 'plane': p, 'fluid': '1', 'fatsat': '1', 'PatientSex': sex,
             'Manufacturer': site, 'ManufacturerModelName': 'M', 'MagneticFieldStrength': '1.5',
             'SoftwareVersions': 'v', 'ReceiveCoilName': 'c', 'ContrastBolusAgent': contrast,
             'SliceThickness': '3', 'PixelSpacing': '0.3|0.3'} for p in planes]


@unittest.skipIf(np is None, 'numpy/sklearn not installed')
class SitePriorTests(unittest.TestCase):
    def setUp(self):
        from knee import site_prior
        self.sp = site_prior

    def test_features_count_planes_and_contrast(self):
        f = self.sp.study_features(_rows('a', planes=('Sagittal', 'Sagittal', 'Axial'), contrast='GADO'))['a']
        self.assertEqual(f['n_series'], 3.0)
        self.assertEqual(f['n_Sagittal_11'], 2.0)
        self.assertEqual(f['n_Coronal_11'], 0.0)
        self.assertEqual(f['contrast'], 1.0)
        self.assertTrue(np.isnan(f['thick_Coronal']))
        self.assertEqual(f['pix_Axial'], 0.3)

    def test_rare_site_bucketed_and_unseen_study_handled(self):
        rows = []
        for i in range(25):
            rows += _rows(f'common{i}', site='A')
        rows += _rows('rare', site='B')
        feats = self.sp.study_features(rows)
        ids = sorted(feats)
        enc = self.sp.Encoder().fit(feats, ids)
        self.assertEqual(enc.sites, ['A×M×1.5×v×c'])
        X = enc.transform(feats, ['rare', 'common0', 'missing'])
        self.assertEqual(X.shape[0], 3)
        self.assertTrue(np.isfinite(X).all())
        other = X[:, -len(enc.sexes) - 1]
        self.assertEqual(other.tolist(), [1.0, 0.0, 1.0])

    def test_combine_is_identity_when_meta_equals_prevalence(self):
        p = np.array([[0.2, 0.7]])
        prev = np.array([0.3, 0.4])
        np.testing.assert_allclose(self.sp.combine(p, np.array([[0.3, 0.4]]), prev), p, atol=1e-9)
        self.assertGreater(self.sp.combine(p, np.array([[0.6, 0.4]]), prev)[0, 0], 0.2)

    def test_masked_auc_ignores_unobserved(self):
        Y = np.array([[0.0], [1.0], [1.0], [0.0]])
        M = np.array([[1], [1], [0], [1]])
        pred = np.array([[0.1], [0.9], [0.0], [0.2]])
        self.assertEqual(self.sp.masked_aucs(pred, Y, M).tolist(), [1.0])

    def test_verdict_rules(self):
        ok = {'macro_delta': 0.003, 'macro_ci': [0.001, 0.005], 'target_delta': [0.0, -0.004]}
        self.assertTrue(self.sp.verdict(ok)['pass'])
        bad = dict(ok, target_delta=[0.01, -0.006])
        self.assertEqual(self.sp.verdict(bad)['reasons'], ['target_drop_over_0.005'])
        self.assertIn('ci_lower_not_positive', self.sp.verdict(dict(ok, macro_ci=[0.0, 0.01]))['reasons'])

    def test_fit_meta_recovers_site_signal(self):
        rng = np.random.default_rng(0)
        X = np.c_[rng.integers(0, 2, 400)].astype(float)
        Y = (rng.random((400, 1)) < np.where(X == 1, 0.7, 0.1)).astype(float)
        M = np.ones_like(Y)
        models, chosen, prev = self.sp.fit_meta(X, Y, M)
        p = models[0].predict_proba(np.array([[0.0], [1.0]]))[:, 1]
        self.assertLess(p[0], 0.25)
        self.assertGreater(p[1], 0.55)
        self.assertAlmostEqual(prev[0], Y.mean(), places=9)


if __name__ == '__main__':
    unittest.main()
