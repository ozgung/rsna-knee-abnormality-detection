import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('train04_sequence',Path(__file__).resolve().parents[1]/'scripts/run_train04_sequence.py')
seq=importlib.util.module_from_spec(spec);spec.loader.exec_module(seq)


class SequenceTests(unittest.TestCase):
    def test_completed_fit_above_old_cap_is_accepted(self):
        report={'status':'PASS','completed_epochs':12,'elapsed_seconds':20000}
        seq.require_training_complete(report)
        for changes in ({'status':'PLATFORM_SESSION_STOP'},{'completed_epochs':11},
                        {'elapsed_seconds':float('nan')},{'elapsed_seconds':0}):
            with self.assertRaises(ValueError):seq.require_training_complete(report|changes)

    def test_status_and_quota_are_explicit(self):
        self.assertEqual(seq.kernel_state('has status "KernelWorkerStatus.COMPLETE"'),'COMPLETE')
        self.assertEqual(seq.gpu_remaining([{'resource':'GPU','remaining':'27.77h'}]),27.77)
        for text in ('timeout','404','completed?'):
            with self.assertRaises(ValueError):seq.kernel_state(text)
        with self.assertRaises(ValueError):seq.gpu_remaining([{'resource':'GPU','remaining':'unknown'}])

    def test_source_drift_stops_sequence(self):
        with patch.object(seq,'source_pins',return_value={'sampler':'new'}):
            with self.assertRaises(ValueError):seq.require_pins({'sampler':'old'})
            seq.require_pins({'sampler':'new'})


if __name__=='__main__':unittest.main()
