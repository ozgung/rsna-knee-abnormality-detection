"""Raptor64 single-arm Colab controller tests (synthetic artifacts; no GPU, Kaggle or Colab)."""
import fcntl
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('run64', ROOT / 'scripts/colab_raptor64_run.py')
runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
EVIDENCE = Path(os.environ.get('SOURCE_SPAN_EVIDENCE_ROOT', '/nonexistent'))
HISTORICAL = EVIDENCE / 'outputs/kaggle/trackA-coatnet224-fold01-20261009/train_knee_patched.py'
PERSISTER = EVIDENCE / 'outputs/analysis/colab-fold3-recovery-20261011/fold4-uploaded-persist.py'
ARM = 'in-plane'
TAG = 'raptor64-' + ARM


def load_builder():
    s = importlib.util.spec_from_file_location('build_raptor64_training', ROOT / 'scripts/build_raptor64_training.py')
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


class PublishedManifestTests(unittest.TestCase):
    def test_download_is_exact_version_and_temporary_staging_is_removed(self):
        seen = []

        def download(command, **kwargs):
            self.assertEqual(command[:4], ['kaggle', 'datasets', 'download', 'fixture/private/7'])
            self.assertEqual(command[command.index('-f') + 1], 'payload_manifest.json')
            path = Path(command[command.index('-p') + 1]); seen.append(path)
            (path / 'payload_manifest.json').write_text('{"epoch": 11}')
            return subprocess.CompletedProcess(command, 0)
        with patch.object(runner.subprocess, 'run', side_effect=download):
            self.assertEqual(runner.published_manifest('fixture/private', 7), {'epoch': 11})
        self.assertFalse(seen[0].exists())

    def test_download_failure_missing_or_invalid_manifest_fails(self):
        for mode in ('failed', 'missing', 'invalid', 'large'):
            def download(command, **kwargs):
                p = Path(command[command.index('-p') + 1]) / 'payload_manifest.json'
                if mode == 'invalid':
                    p.write_text('invalid')
                if mode == 'large':
                    p.write_bytes(b'x' * (1024 * 1024 + 1))
                return subprocess.CompletedProcess(command, 1 if mode == 'failed' else 0)
            with self.subTest(mode=mode), patch.object(runner.subprocess, 'run', side_effect=download), self.assertRaises(ValueError):
                runner.published_manifest('fixture/private', 7)


@unittest.skipUnless(HISTORICAL.is_file() and PERSISTER.is_file(), 'Pinned historical trainer and authoritative persister required')
class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pkgtmp = tempfile.TemporaryDirectory()
        cls.package = Path(cls.pkgtmp.name) / f'{TAG}-pkg'
        load_builder().build(cls.package, evidence_root=EVIDENCE, arm=ARM)

    @classmethod
    def tearDownClass(cls):
        cls.pkgtmp.cleanup()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        stage = self.root / 'cache'; stage.mkdir(); init = self.root / 'initializer.json'; weights = self.root / 'weights.pt'; weights.write_bytes(b'synthetic')
        init.write_text(json.dumps({'init': 'imagenet', 'weights_file': str(weights), 'weights_sha256': runner.sha(weights)}))
        rp = stage / 'staging-receipt.json'; rp.write_text(json.dumps({'status': 'PASS', 'manifest_sha256': 'a' * 64, 'content_sha256': 'b' * 64}))
        self.policy = json.loads((self.package / 'raptor64-recipe.json').read_text())
        self.c = {'version': runner.VERSION, 'name': f'{TAG}-fold0-v1', 'arm': ARM, 'vm_session_id': 'fixtureVM', 'package_root': str(self.package),
                  'package_manifest_sha256': runner.sha(self.package / 'manifest.json'), 'workdir': str(self.root / f'{TAG}-fold0'),
                  'input_root': str(self.root / 'input'), 'stage_root': str(stage), 'manifest_sha256': 'a' * 64, 'content_sha256': 'b' * 64,
                  'stage_receipt_sha256': runner.sha(rp), 'initializer_descriptor': str(init), 'initializer_descriptor_sha256': runner.sha(init),
                  'persister': str(PERSISTER), 'persister_sha256': runner.PERSISTER_SHA,
                  'persistence': {'dataset': f'fixture/knee-{TAG}-fold0-20261013', 'private': True, 'staging': str(self.root / f'{TAG}-backup'),
                                  'status_file': str(self.root / f'{TAG}-persist-status.json')},
                  'controller_lock': str(self.root / f'{TAG}-controller.lock'), 'workers': 2, 'resume': False, 'fold': 0}
        self.procs = []; self.calls = []; self.privacy_calls = []

    def tearDown(self):
        self.tmp.cleanup()

    def artifacts(self):
        work = Path(self.c['workdir']); work.mkdir(exist_ok=True)
        (work / 'raptor_resume_fold0.pt').write_bytes(b'private-resume-epoch11'); (work / 'raptor_ft_fold0_swa.pt').write_bytes(b'fixed-selected-swa')
        (work / 'oof_fold0.csv').write_bytes(b'synthetic schema tested separately')
        meta = {'partial': False, 'hist': [{'ep': i} for i in range(12)]}
        (work / 'raptor_ft_fold0.json').write_text(json.dumps(meta))
        record = {'fold': 0, 'epochs_requested': 12, 'epochs_completed': 12, 'stopped_by_time_budget': False, 'checkpoint_kind': 'swa_topk',
                  'comparison_endpoint': 'K42 primary (selection K16)', 'history': meta['hist'], 'export': {'k_eval': 42, 'n_val': 870, 'n_gold': 0},
                  'selected_epochs': [8, 10, 11], 'checkpoint_file': 'raptor_ft_fold0_swa.pt', 'checkpoint_sha256': runner.sha(work / 'raptor_ft_fold0_swa.pt'),
                  'raptor64_policy': {'arm': ARM}, 'raptor64_experiment_split': self.policy['experiment_split'], 'raptor64_cache_content_sha256': 'b' * 64}
        (work / 'run_fold0.json').write_text(json.dumps(record))
        Path(self.c['persistence']['staging']).mkdir(exist_ok=True)
        self.published_payload = {'dataset': self.c['persistence']['dataset'], 'fold': 0, 'epoch': 11, 'resume_sha256': runner.sha(work / 'raptor_resume_fold0.pt')}
        st = (work / 'raptor_resume_fold0.pt').stat()
        return {'dataset': self.c['persistence']['dataset'], 'fold': 0, 'verified_epoch': 11, 'verified_version': 2,
                'verified_signature': [st.st_size, st.st_mtime_ns], 'final': {'state': 'verified', 'epoch': 11, 'version': 2}}

    def factory(self, train_rc=0, persist_rc=0, build=True):
        outer = self

        class Process:
            pid = 987654321

            def __init__(self, is_persist):
                self.is_persist = is_persist; self.returncode = None if is_persist else train_rc
                self.stdout = io.StringIO('[train:fold0] ep0 loss 0.4 | OOF masked macro-AUC 0.8 IDs SECRET\nsecret token must not stream\n'); self.signals = []
                if not is_persist and build:
                    outer.status = outer.artifacts()

            def poll(self):
                return self.returncode

            def send_signal(self, s):
                self.signals.append(s)

            def wait(self, timeout=None):
                self.returncode = persist_rc if self.is_persist else train_rc; return self.returncode

        def popen(cmd, **kw):
            self.calls.append((cmd, kw)); p = Process(len(self.procs) == 0); self.procs.append(p); return p
        return popen

    def invoke(self, **kw):
        defaults = {'popen': self.factory(), 'probe': lambda c, e: {'device': 'cuda', 'threads': 1, 'studies': 4345, 'timm': '1.0.26'},
                    'privacy': lambda d: self.privacy_calls.append(d), 'read_status': lambda _: self.status, 'install_signals': False, 'sleep': lambda _: None}
        defaults.update(kw)
        with patch('builtins.print'), patch.object(runner, 'published_manifest', side_effect=lambda dataset, version: self.published_payload):
            return runner.run(self.c, **defaults)

    def test_success_final_epoch11_keeps_resume_and_old_versions(self):
        r = self.invoke(); self.assertEqual(r['status'], 'PASS'); self.assertTrue(r['resume_preserved'])
        self.assertEqual(r['backup']['epoch'], 11); self.assertEqual(r['arm'], ARM); self.assertEqual((r['init'], r['head']), ('imagenet', 'plane'))
        self.assertEqual(self.procs[0].signals, [signal.SIGUSR1]); self.assertIn('--keep-old-versions', self.calls[0][0])
        self.assertNotIn('--delete-old-versions', self.calls[0][0])
        env = self.calls[1][1]['env']
        self.assertEqual(env['WORKERS'], '2'); self.assertEqual(env['RAPTOR_PRELOAD'], '0'); self.assertEqual(env['EPOCHS'], '12')
        self.assertEqual(env['TIME_BUDGET_H'], '0'); self.assertEqual(env['RES'], '384'); self.assertEqual(env['K_EVAL'], '16')
        self.assertEqual(env['RAPTOR64_CONTENT_SHA256'], 'b' * 64); self.assertEqual(env['WORKDIR'], self.c['workdir'])
        self.assertFalse({k for k in env if k.startswith('SOURCE96_')} - set(os.environ))
        self.assertEqual(len(self.privacy_calls), 2)
        work = Path(self.c['workdir'])
        self.assertTrue((work / 'raptor64-run-fingerprint.json').is_file()); self.assertTrue((work / 'raptor64-run-receipt.json').is_file())

    def test_training_failure_still_requests_final_backup(self):
        r = self.invoke(popen=self.factory(train_rc=1)); self.assertEqual(r['status'], 'FAIL'); self.assertTrue(r['resume_preserved'])
        self.assertEqual(self.procs[0].signals, [signal.SIGUSR1])

    def test_final_uses_published_version_after_staging_cleanup(self):
        status = self.artifacts()
        with patch.object(runner, 'published_manifest', return_value=self.published_payload) as download:
            result = runner.final_verified(self.c, status)
        download.assert_called_once_with(self.c['persistence']['dataset'], 2)
        self.assertEqual(result['epoch'], 11)
        for key, value in [('dataset', 'fixture/wrong'), ('fold', 1), ('epoch', 10), ('resume_sha256', 'c' * 64)]:
            with self.subTest(key=key), patch.object(runner, 'published_manifest', return_value={**self.published_payload, key: value}):
                with self.assertRaises(ValueError):
                    runner.final_verified(self.c, status)
        for change in ({'verified_epoch': 9}, {'final': {'state': 'verified', 'epoch': 9, 'version': 2}}):
            with self.subTest(change=change), patch.object(runner, 'published_manifest', return_value=self.published_payload):
                with self.assertRaises(ValueError):
                    runner.final_verified(self.c, {**status, **change})

    def test_final_rejects_missing_remote_manifest_even_with_local_copy(self):
        status = self.artifacts()
        (Path(self.c['persistence']['staging']) / 'payload_manifest.json').write_text(json.dumps(self.published_payload))
        with patch.object(runner, 'published_manifest', side_effect=ValueError('unavailable')):
            with self.assertRaises(ValueError):
                runner.final_verified(self.c, status)

    def test_backup_failure_and_stale_final_signature_fail(self):
        r = self.invoke(popen=self.factory(persist_rc=3)); self.assertEqual(r['status'], 'FAIL')
        self.procs = []; self.calls = []; self.c['resume'] = True; self.c['resume_sha256'] = runner.sha(Path(self.c['workdir']) / 'raptor_resume_fold0.pt')
        r = self.invoke(read_status=lambda _: {**self.status, 'verified_signature': [1, 2]}); self.assertEqual(r['status'], 'FAIL'); self.assertTrue(r['resume_preserved'])

    def test_artifact_worker_privacy_and_staging_pins_fail_before_process(self):
        for key, value in [('package_manifest_sha256', 'c' * 64), ('persister_sha256', 'c' * 64), ('stage_receipt_sha256', 'c' * 64),
                           ('initializer_descriptor_sha256', 'c' * 64), ('workers', 10), ('fold', 1), ('version', 'colab-source96-run-v1')]:
            old = self.c[key]; self.c[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.invoke()
            self.c[key] = old
        self.c['persistence']['private'] = False
        with self.assertRaises(ValueError):
            self.invoke()
        self.assertEqual(self.calls, [])

    def test_arm_isolation_of_paths_and_datasets(self):
        cases = [('workdir', str(self.root / 'raptor64-fold0')),                                   # lacks the arm tag
                 ('workdir', str(self.root / f'{TAG}-raptor64-dr-orig-fold0')),                    # also names another arm
                 ('controller_lock', str(self.root / 'raptor64-in-orig-controller.lock')),
                 ('name', 'raptor64-fold0-v1')]
        for key, value in cases:
            old = self.c[key]; self.c[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                runner.validate(self.c)
            self.c[key] = old
        for sub, value in (('dataset', 'fixture/knee-raptor64-dr-plane-fold0-20261013'), ('staging', str(self.root / 'backup')),
                           ('status_file', str(self.root / 'status.json'))):
            old = self.c['persistence'][sub]; self.c['persistence'][sub] = value
            with self.subTest(sub=sub), self.assertRaises(ValueError):
                runner.validate(self.c)
            self.c['persistence'][sub] = old
        runner.validate(self.c)

    def test_config_arm_must_match_package_and_initializer(self):
        self.c['arm'] = 'in-orig'
        for key in ('name', 'workdir', 'controller_lock'):
            self.c[key] = self.c[key].replace(TAG, 'raptor64-in-orig')
        for sub in ('staging', 'status_file', 'dataset'):
            self.c['persistence'][sub] = self.c['persistence'][sub].replace(TAG, 'raptor64-in-orig')
        moved = self.root / 'raptor64-in-orig-pkg'; shutil.copytree(self.package, moved); self.c['package_root'] = str(moved)
        with self.assertRaisesRegex(ValueError, 'Package arm differs'):
            runner.validate(self.c)
        self.c['arm'] = 'bogus'
        with self.assertRaises(ValueError):
            runner.validate(self.c)

    def test_initializer_descriptor_must_name_the_arm_init(self):
        init = Path(self.c['initializer_descriptor']); d = json.loads(init.read_text()); d['init'] = 'dread_v4'; init.write_text(json.dumps(d))
        self.c['initializer_descriptor_sha256'] = runner.sha(init)
        with self.assertRaisesRegex(ValueError, 'Initializer'):
            runner.validate(self.c)

    def test_existing_work_explicit_resume_required(self):
        self.artifacts()
        with self.assertRaises(ValueError):
            runner.validate(self.c)
        self.c.update(resume=True, resume_sha256=runner.sha(Path(self.c['workdir']) / 'raptor_resume_fold0.pt'))
        runner.validate(self.c); self.assertEqual(runner.environment(self.c)['RAPTOR64_RESUME_SHA256'], self.c['resume_sha256'])
        self.assertEqual(runner.environment(self.c)['RESUME'], '1')
        self.c['resume_sha256'] = 'c' * 64
        with self.assertRaises(ValueError):
            runner.validate(self.c)

    def test_cross_fold_work_artifacts_rejected(self):
        work = Path(self.c['workdir']); work.mkdir(); (work / 'raptor_resume_fold1.pt').write_bytes(b'other fold')
        with self.assertRaisesRegex(ValueError, 'Cross-fold'):
            runner.validate(self.c)

    def test_lock_prevents_duplicate_controller(self):
        with open(self.c['controller_lock'], 'w') as f:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(RuntimeError):
                self.invoke()
        self.assertEqual(self.calls, [])

    def test_primary_incomplete_or_wrong_endpoint_cannot_pass(self):
        self.artifacts(); p = Path(self.c['workdir']) / 'run_fold0.json'; r = json.loads(p.read_text())
        self.assertEqual(runner.primary_complete(self.c['workdir'], 0, 870, self.policy['experiment_split'], 'b' * 64, ARM)['selected_epochs'], [8, 10, 11])
        for change in ({'export': {**r['export'], 'k_eval': 16}}, {'epochs_completed': 11}, {'comparison_endpoint': 'matched K16 primary'},
                       {'selected_epochs': [10, 10, 11]}, {'raptor64_policy': {'arm': 'dr-plane'}}, {'raptor64_cache_content_sha256': 'c' * 64},
                       {'export': {**r['export'], 'n_val': 866}}):
            p.write_text(json.dumps({**r, **change}))
            with self.subTest(change=change), self.assertRaises(ValueError):
                runner.primary_complete(self.c['workdir'], 0, 870, self.policy['experiment_split'], 'b' * 64, ARM)
        p.write_text(json.dumps(r))
        with self.assertRaises(ValueError):
            runner.primary_complete(self.c['workdir'], 1)
        (Path(self.c['workdir']) / 'raptor_ft_fold0_swa.pt').write_bytes(b'mutated')
        with self.assertRaises(ValueError):
            runner.primary_complete(self.c['workdir'])

    def test_shutdown_signal_preserves_checkpoint_and_finalizes_backup(self):
        handlers = {}; base = self.factory()

        def popen(cmd, **kw):
            p = base(cmd, **kw)
            if not p.is_persist:
                p.returncode = None
            return p

        def register(sig, handler):
            handlers[sig] = handler; return signal.SIG_DFL

        def killpg(pid, sig):
            self.procs[1].returncode = 143

        def tick(_):
            handlers[signal.SIGTERM](signal.SIGTERM, None)
        with patch.object(runner.signal, 'signal', side_effect=register), patch.object(runner.os, 'killpg', side_effect=killpg):
            r = self.invoke(popen=popen, sleep=tick, install_signals=True)
        self.assertEqual(r['status'], 'FAIL'); self.assertEqual(r['interrupted_signal'], signal.SIGTERM)
        self.assertTrue(r['resume_preserved']); self.assertEqual(self.procs[0].signals, [signal.SIGUSR1])

    def test_private_tracebacks_retained_but_not_printed(self):
        class Output:
            stdout = io.StringIO('Traceback: private diagnostic\nSECRET report or credential\n')
        target = self.root / 'private-diagnostic.log'
        with patch('builtins.print') as print_:
            runner.stream(Output(), 'training', target)
        self.assertEqual(target.read_text(), 'Traceback: private diagnostic\nSECRET report or credential\n')
        self.assertEqual(target.stat().st_mode & 0o777, 0o600); print_.assert_not_called()
        completed = subprocess.CompletedProcess([], 1, 'runtime private stdout\n', 'Traceback: runtime private stderr\n')
        with patch.object(runner.subprocess, 'run', return_value=completed), self.assertRaises(RuntimeError):
            runner.runtime_probe(self.c, {})
        work = Path(self.c['workdir'])
        self.assertEqual((work / 'raptor64-runtime-stdout.log').read_text(), completed.stdout)
        self.assertEqual((work / 'raptor64-runtime-stderr.log').read_text(), completed.stderr)

    def test_runtime_probe_result_checks(self):
        good = {'device': 'cuda', 'threads': 1, 'studies': 4345, 'timm': '1.0.26',
                'initializer': {'missing': 0, 'unexpected': 0, 'init': 'imagenet'},
                'probe': {'finite': True, 'eval_windows': 42}}
        def run_with(value):
            completed = subprocess.CompletedProcess([], 0, 'noise\n' + json.dumps(value) + '\n', '')
            with patch.object(runner.subprocess, 'run', return_value=completed):
                return runner.runtime_probe(self.c, {})
        self.assertEqual(run_with(good), good)
        bad = [{**good, 'device': 'cpu'}, {**good, 'threads': 4}, {**good, 'studies': 4344}, {**good, 'timm': '1.0.25'},
               {**good, 'initializer': {**good['initializer'], 'missing': 1}}, {**good, 'initializer': {**good['initializer'], 'unexpected': 2}},
               {**good, 'initializer': {**good['initializer'], 'init': 'dread_v4'}}, {**good, 'probe': {'finite': False, 'eval_windows': 42}},
               {**good, 'probe': {'finite': True, 'eval_windows': 16}}]
        for value in bad:
            with self.subTest(value=value), self.assertRaises(ValueError):
                run_with(value)

    def test_dead_persister_ignored_term_escalates_after60(self):
        clock = [0.]; base = self.factory(persist_rc=3)

        def popen(cmd, **kw):
            p = base(cmd, **kw)
            if p.is_persist:
                p.returncode = 3
            else:
                p.returncode = None

                def wait(timeout=None):
                    if p.returncode is None:
                        raise subprocess.TimeoutExpired('injected', timeout)
                    return p.returncode
                p.wait = wait
            return p
        signals = []

        def killpg(pid, sig):
            signals.append(sig)
            if sig == signal.SIGKILL:
                self.procs[1].returncode = -9

        def tick(_):
            clock[0] += 30
        with patch.object(runner.os, 'killpg', side_effect=killpg):
            r = self.invoke(popen=popen, sleep=tick, monotonic=lambda: clock[0])
        self.assertEqual(clock[0], 60); self.assertEqual(signals, [signal.SIGTERM, signal.SIGKILL])
        self.assertEqual(r['training_returncode'], -9); self.assertEqual(r['status'], 'FAIL'); self.assertTrue(r['resume_preserved'])

    def test_probe_failure_never_launches(self):
        def bad(*_):
            raise RuntimeError('CUDA unavailable')
        with self.assertRaises(RuntimeError):
            self.invoke(probe=bad)
        self.assertEqual(self.calls, [])


class SanitizerTests(unittest.TestCase):
    def test_private_metadata_fail_closed(self):
        from types import SimpleNamespace

        class API:
            def authenticate(self):
                pass

            def dataset_metadata(self, dataset, t):
                p = Path(t) / 'dataset-metadata.json'; p.write_text(json.dumps({'id': dataset, 'isPrivate': True})); return str(p)
        with patch.dict('sys.modules', {'kaggle.api.kaggle_api_extended': SimpleNamespace(KaggleApi=API)}):
            self.assertTrue(runner.private_dataset('fixture/private')['private'])

        class Public(API):
            def dataset_metadata(self, dataset, t):
                p = Path(t) / 'dataset-metadata.json'; p.write_text(json.dumps({'isPrivate': False})); return str(p)
        with patch.dict('sys.modules', {'kaggle.api.kaggle_api_extended': SimpleNamespace(KaggleApi=Public)}), self.assertRaises(ValueError):
            runner.private_dataset('fixture/public')

    def test_current_nested_metadata_requires_exact_identity_and_boolean_privacy(self):
        from types import SimpleNamespace
        holder = {'value': {'info': {'ownerUser': 'fixture', 'datasetSlug': 'private', 'isPrivate': True}}}

        class API:
            def authenticate(self):
                pass

            def dataset_metadata(self, dataset, t):
                p = Path(t) / 'dataset-metadata.json'; p.write_text(json.dumps(holder['value'])); return str(p)
        with patch.dict('sys.modules', {'kaggle.api.kaggle_api_extended': SimpleNamespace(KaggleApi=API)}):
            self.assertTrue(runner.private_dataset('fixture/private')['private'])
            for bad in [{'info': {'ownerUser': 'other', 'datasetSlug': 'private', 'isPrivate': True}},
                        {'info': {'ownerUser': 'fixture', 'datasetSlug': 'private', 'isPrivate': False}},
                        {'info': {'ownerUser': 'fixture', 'datasetSlug': 'private', 'isPrivate': 'true'}}, {'isPrivate': True}]:
                holder['value'] = bad
                with self.assertRaises(ValueError):
                    runner.private_dataset('fixture/private')

    def test_logs_whitelist_only_numeric_progress(self):
        self.assertEqual(runner.sanitized('[train:fold0] ep1 loss 0.2 | OOF masked macro-AUC 0.8 ID private'), 'ep1 loss 0.2 | OOF masked macro-AUC 0.8')
        self.assertIsNone(runner.sanitized('Authorization: private-token'))
        self.assertIsNone(runner.sanitized('https://signed.example/secret?id=patient'))
        self.assertEqual(runner.sanitized('[persist 12:12:12] FINAL PERSIST VERIFIED epoch 11 version 2'), 'FINAL PERSIST VERIFIED epoch 11 version 2')
        self.assertEqual(runner.sanitized('[raptor64] init dread_v4 loaded 478 tensors, missing 0, unexpected 0'),
                         '[raptor64] init dread_v4 loaded 478 tensors, missing 0, unexpected 0')
        self.assertIsNone(runner.sanitized('[raptor64] init dread_v4 loaded 470 tensors, missing 8, unexpected 0'))


if __name__ == '__main__':
    unittest.main()
