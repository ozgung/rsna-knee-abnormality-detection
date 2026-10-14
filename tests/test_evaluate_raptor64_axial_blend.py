"""It4 axial blend evaluator: rank blend and in-distil fallback for studies without axial content (synthetic)."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
HAVE = importlib.util.find_spec('numpy') is not None


def load():
    spec = importlib.util.spec_from_file_location('eval_axial', ROOT / 'scripts/evaluate_raptor64_axial_blend.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


@unittest.skipUnless(HAVE, 'NumPy required')
class AxialBlendTests(unittest.TestCase):
    def test_ranks_and_fixed_weight_blend_with_fallback(self):
        import numpy as np
        ev = load()
        self.assertEqual(ev.WEIGHT, 0.3); self.assertEqual(ev.PRIMARY, 'PF OA'); self.assertEqual(ev.SECONDARY, ('Effusion', 'Synovitis'))
        self.assertTrue(np.allclose(ev.ranks([0.1, 0.5, 0.5, 0.9]), [0.25, 0.625, 0.625, 1.0]))
        rng = np.random.default_rng(0)
        base = rng.uniform(0, 1, (20, 12)); arm = rng.uniform(0, 1, (20, 12)); present = np.ones(20, bool); present[[3, 7]] = False
        b = ev.blend(base, arm, present)
        for j in (0, 6):
            rb = ev.ranks(base[:, j]); ra = ev.ranks(arm[present, j])
            self.assertTrue(np.allclose(b[~present, j], rb[~present]), 'absent studies fall back to the in-distil rank')
            self.assertTrue(np.allclose(b[present, j], 0.7 * rb[present] + 0.3 * ra))
        self.assertTrue(np.allclose(ev.blend(base, base, np.ones(20, bool)), np.stack([ev.ranks(base[:, j]) for j in range(12)], 1)))
        arm_garbage = arm.copy(); arm_garbage[~present] = 123.0   # absent-study arm values never matter
        self.assertTrue(np.allclose(ev.blend(base, arm_garbage, present), b))


if __name__ == '__main__':
    unittest.main()
