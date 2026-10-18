import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = Path(os.environ.get('SOURCE_SPAN_EVIDENCE_ROOT', '/nonexistent'))
PERSISTER = EVIDENCE / 'outputs/analysis/colab-fold3-recovery-20261011/fold4-uploaded-persist.py'
READY = PERSISTER.is_file() and (EVIDENCE / 'outputs/kaggle/trackA-coatnet224-fold01-20261009/train_knee_patched.py').is_file() and importlib.util.find_spec('numpy')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


vm = load('raptor64_vm', ROOT / 'scripts/colab_raptor64_vm.py')


class LayoutTests(unittest.TestCase):
    def test_every_arm_path_is_isolated_and_slugs_fit(self):
        for arm in vm.ARMS:
            p = vm.paths(arm)
            for key, value in p.items():
                if key != 'init':
                    self.assertIn(f'raptor64-{arm}', str(value), key)
            self.assertLessEqual(len(p['dataset'].split('/')[1]), 50)
            self.assertLessEqual(len(f'knee-raptor64-{arm}-fold0-final-{vm.DATE}'), 50)
        with self.assertRaises(SystemExit):
            vm.arm_check('imagenet')


@unittest.skipUnless(READY, 'Pinned evidence (historical trainer, persister) required')
class ConfigTests(unittest.TestCase):
    def test_generated_config_passes_runner_validation(self):
        builder = load('build_raptor64_training', ROOT / 'scripts/build_raptor64_training.py')
        runner = load('raptor64_runner', ROOT / 'scripts/colab_raptor64_run.py')
        with tempfile.TemporaryDirectory() as tmp:
            content = Path(tmp) / 'content'; content.mkdir()
            for arm in ('in-plane', 'dr-orig'):
                builder.build(content / f'raptor64-{arm}-pkg', evidence_root=EVIDENCE, arm=arm)
            (content / 'raptor64_cache').mkdir()
            (content / 'raptor64_cache/staging-receipt.json').write_text(json.dumps({'status': 'PASS', 'studies': 4345, 'manifest_sha256': 'a' * 64, 'content_sha256': 'b' * 64}))
            for init in ('imagenet', 'dread_v4'):
                d = content / f'raptor64-init-{init}'; d.mkdir(); w = d / 'weights.pt'; w.write_bytes(init.encode())
                (d / 'initializer.json').write_text(json.dumps({'init': init, 'weights_file': str(w), 'weights_sha256': vm.sha(w)}))
            shutil.copyfile(PERSISTER, content / 'colab_tracka_persist.py')
            configs = {arm: vm.make_config(arm, content, vm_session_id='raptor64-fold0-20261013') for arm in ('in-plane', 'dr-orig')}
            for arm, c in configs.items():
                runner.validate(c)
                self.assertEqual(c['arm'], arm); self.assertEqual(c['persistence']['dataset'], f'ozgung/knee-raptor64-{arm}-fold0-20261013')
            crossed = dict(configs['in-plane'], initializer_descriptor=configs['dr-orig']['initializer_descriptor'],
                           initializer_descriptor_sha256=configs['dr-orig']['initializer_descriptor_sha256'])
            with self.assertRaises(ValueError):
                runner.validate(crossed)
            for arm in ('in-distil', 'in-distil-w1'):
                builder.build(content / f'raptor64-{arm}-pkg', evidence_root=EVIDENCE, arm=arm, teacher_sha256='a' * 64)
            d0 = vm.make_config('in-distil', content, vm_session_id='raptor64-orig-it3-20261014')
            d1 = vm.make_config('in-distil-w1', content, vm_session_id='raptor64-orig-it3-20261014')
            runner.validate(d0); runner.validate(d1)   # boundary-aware: in-distil tag is a prefix of in-distil-w1
            with self.assertRaises(ValueError):
                runner.validate(dict(d1, workdir=d0['workdir']))
            with self.assertRaises(ValueError):
                runner.validate(dict(d0, controller_lock=d1['controller_lock']))
            for arm in ('ax-crop', 'ax-full'):
                builder.build(content / f'raptor64-{arm}-pkg', evidence_root=EVIDENCE, arm=arm, teacher_sha256='a' * 64)
                runner.validate(vm.make_config(arm, content, vm_session_id='raptor64-axial-it4-20261014'))
            with self.assertRaises(ValueError):   # an axial package cannot run under a non-axial arm config
                runner.validate(dict(d0, package_root=str(content / 'raptor64-ax-crop-pkg')))
            mixed = dict(configs['in-plane'], package_root=configs['dr-orig']['package_root'],
                         package_manifest_sha256=configs['dr-orig']['package_manifest_sha256'])
            with self.assertRaises(ValueError):
                runner.validate(mixed)


if __name__ == '__main__':
    unittest.main()
