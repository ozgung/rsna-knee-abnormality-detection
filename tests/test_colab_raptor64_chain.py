"""Chained-VM driver helpers (dependency-free): session naming, VM-side sources compile, loss detection."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location('r64chain', ROOT / 'scripts/colab_raptor64_chain.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class ChainTests(unittest.TestCase):
    def test_session_names_use_rn_for_restarts(self):
        c = load()
        self.assertEqual(c.session_name('raptor64-sagcrop-l2', '20261014', 1), 'raptor64-sagcrop-l2-20261014')
        self.assertEqual(c.session_name('raptor64-sagcrop-l2', '20261014', 3), 'raptor64-sagcrop-l2-r3-20261014')
        self.assertNotIn('axial', c.session_name('raptor64-sagcrop-l2', '20261014', 2), 'never collides with the lane-1 session family')

    def test_vm_sources_compile_and_loss_detection(self):
        c = load()
        compile(c.LAUNCHER, 'launcher', 'exec')
        compile("CONFIG = 'x'\nNAME = 'n'\nENDPOINT = 'e'\nACCEL = 'A100'\n" + c.RECONNECT, 'reconnect', 'exec')
        for name in ('vm_raptor64_chain_boot.py', 'vm_raptor64_chain_status.py'):
            compile((ROOT / 'scripts' / name).read_text(), name, 'exec')
        self.assertTrue(c.lost('Session raptor appears to be lost')); self.assertFalse(c.lost('STATUS {}'))
        self.assertEqual(c.TERMINAL, ('exported', 'export_failed', 'failed', 'stopped_health'))


if __name__ == '__main__':
    unittest.main()
