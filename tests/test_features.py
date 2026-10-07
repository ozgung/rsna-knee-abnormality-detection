import unittest
try:
    import numpy as np
    import torch
    from knee.features import StudyHead, fit_feature_stats, masked_loss
except ImportError:
    np = None


@unittest.skipIf(np is None, "Optional ML dependencies unavailable")
class FeatureTests(unittest.TestCase):
    def test_stats_exclude_validation_and_padding(self):
        x = np.array([[[[1., 2.], [999., 999.]]], [[[50., 60.], [80., 90.]]]])
        mask = np.array([[[1, 0]], [[1, 1]]], bool)
        mean, std = fit_feature_stats(x, mask, [0])
        np.testing.assert_array_equal(mean, [1., 2.])
        self.assertTrue((std > 0).all())

    def test_padding_and_missing_study_invariance(self):
        torch.manual_seed(42)
        for kind in ("attention", "meanmax"):
            model = StudyHead(np.zeros(8), np.ones(8), [.2, .7], hidden=4, kind=kind).eval()
            x = torch.randn(2, 3, 4, 8)
            mask = torch.zeros(2, 3, 4, dtype=torch.bool)
            mask[0, 0, :2] = True
            with torch.no_grad():
                baseline = model(x, mask)
                x[~mask] = 10000
                changed = model(x, mask)
            torch.testing.assert_close(baseline, changed)
            torch.testing.assert_close(changed[1].sigmoid(), torch.tensor([.2, .7]))

    def test_unknown_labels_have_no_gradient(self):
        logits = torch.zeros(2, 2, requires_grad=True)
        targets = torch.tensor([[1., .5], [0., .5]])
        confidence = (targets - .5).abs() * 2
        masked_loss(logits, targets, confidence).backward()
        self.assertTrue((logits.grad[:, 1] == 0).all())
        self.assertTrue((logits.grad[:, 0] != 0).all())

    def test_transitive_patient_report_groups_excluded(self):
        from knee.experiment import exclude_related
        actual = exclude_related(["gold", "train", "train", "train", "weak_validation", "train"],
                                 ["a", "b", "b", "c", "d", "e"],
                                 [["p"], ["p"], ["q"], ["q"], ["p"], ["safe"]])
        self.assertEqual(actual.tolist(), ["gold", "excluded", "excluded", "excluded", "excluded", "train"])

    def test_indivisible_group_limits_do_not_starve_training(self):
        from knee.experiment import choose_studies
        rows = [{"StudyInstanceUID":str(i), "Report":str(i//2)} for i in range(8)]
        _, splits = choose_studies(rows, {str(i):"train" for i in range(8)}, train_limit=4, weak_limit=3)
        self.assertEqual(int((splits == "weak_validation").sum()), 2)
        self.assertEqual(int((splits == "train").sum()), 4)
