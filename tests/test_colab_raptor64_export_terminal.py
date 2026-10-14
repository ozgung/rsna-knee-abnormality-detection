"""Synthetic tests for scripts/colab_raptor64_export_terminal.py (port of the source96 exporter tests).

No Colab, Kaggle, credential, network or real training artifact is used: a fake transport stands in for Kaggle and a
synthetic arm workdir stands in for a completed (or failed) Raptor64 fit. RAPTOR64_EXPORTER_UNDER_TEST can point at a
mutated copy of the exporter for mutation checks.
"""
import contextlib
import fcntl
import hashlib
import importlib.util
import io
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
EXPORTER = Path(os.environ.get('RAPTOR64_EXPORTER_UNDER_TEST', ROOT / 'scripts/colab_raptor64_export_terminal.py'))
RUNNER = ROOT / 'scripts/colab_raptor64_run.py'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ex = load('export64', EXPORTER)
runner = load('run64_for_export', RUNNER)
RUNNER_SHA = hashlib.sha256(RUNNER.read_bytes()).hexdigest()
PKG, CONTENT = 'a' * 64, 'c' * 64
ARM = 'in-plane'
BACKUP = f'ozgung/knee-raptor64-{ARM}-fold0-20261013'
FINAL = f'ozgung/knee-raptor64-{ARM}-fold0-final-20261013'
FAILED = f'ozgung/knee-raptor64-{ARM}-fold0-failed-20261013'
RESUME_BACKUPS = [f'ozgung/knee-raptor64-{a}-fold0-20261013' for a in ('in-orig', 'in-plane', 'dr-orig', 'dr-plane')]
SPLIT = {'fold': 0, 'eligible': 4345, 'training': 3475, 'development': 870,
         'ordered_eligible_sha256': '1' * 64, 'ordered_training_sha256': '2' * 64, 'ordered_development_sha256': '3' * 64}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def canonical(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def tree_state(root):
    """Content hash + mtime of every entry; any workdir write, delete or touch changes it."""
    out = {}
    for base, dirs, files in os.walk(root):
        for n in dirs + files:
            p = Path(base) / n
            st = p.lstat()
            out[str(p.relative_to(root))] = (sha(p.read_bytes()) if p.is_file() and not p.is_symlink() else None, st.st_mtime_ns, st.st_size)
    return out


class Fit:
    """A synthetic arm workdir plus the config/receipts that a finished (or failed) controller leaves behind."""

    def __init__(self, root, fold=0, kind='pass', backup=None, epochs=12, arm=ARM):
        self.root, self.fold, self.kind, self.arm = Path(root), 0, kind, arm
        self.counter = itertools.count()
        tag = f'raptor64-{arm}'
        self.work = self.root / f'{tag}-fold0'
        self.work.mkdir()
        k = 0
        hist = [{'ep': i, 'auc': 0.5 + i / 100} for i in range(epochs)]
        self.swa = self.work / f'raptor_ft_fold{k}_swa.pt'
        self.swa.write_bytes(b'synthetic-swa-%d-' % k * 5000)
        (self.work / f'oof_fold{k}.csv').write_text('StudyInstanceUID,a\nx,0.5\n')
        (self.work / f'raptor_resume_fold{k}.pt').write_bytes(b'private-resume-epoch11' * 100)
        (self.work / f'raptor_ft_fold{k}.json').write_text(json.dumps({'partial': False, 'hist': hist}))
        (self.work / f'train_meta_fold{k}.json').write_text('{"rc": 0}')
        (self.work / 'initial-state-record.json').write_text('{"initial": true}')
        (self.work / 'raptor_oof_fold0_swa_fold0.npz').write_bytes(b'synthetic k16 swa oof')
        for log in (f'log_fold{k}.txt', 'raptor64-training.log'):
            (self.work / log).write_text('synthetic private log\n')
        self.split = dict(SPLIT)
        policy = {'arm': arm, 'experiment_split': self.split, 'experiment_split_sha256': canonical(self.split)}
        record = {'fold': k, 'epochs_requested': 12, 'epochs_completed': epochs, 'stopped_by_time_budget': False,
                  'checkpoint_kind': 'swa_topk', 'comparison_endpoint': 'K42 primary (selection K16)',
                  'export': {'k_eval': 42, 'n_val': 870, 'n_gold': 0}, 'selected_epochs': [11, 10, 9], 'history': hist,
                  'checkpoint_file': self.swa.name, 'checkpoint_sha256': sha(self.swa.read_bytes()),
                  'raptor64_policy': policy, 'raptor64_experiment_split': self.split, 'raptor64_cache_content_sha256': CONTENT}
        self.write_json(f'run_fold{k}.json', record)
        # external pins
        stage = self.root / 'cache'
        stage.mkdir()
        self.stage_receipt = stage / 'staging-receipt.json'
        self.stage_receipt.write_text(json.dumps({'status': 'PASS', 'manifest_sha256': 'b' * 64, 'content_sha256': CONTENT}))
        self.init = self.root / 'initializer.json'
        self.init.write_text('{"init": "imagenet", "weights_file": "x"}')
        self.lock = self.root / f'{tag}-controller.lock'
        self.lock.write_text('')
        self.status_file = self.root / f'{tag}-persist-status.json'
        self.status_file.write_text(json.dumps({'fold': k, 'verified_epoch': 11}))
        self.config = {
            'version': 'colab-raptor64-run-v1', 'name': f'{tag}-fold0-v1', 'arm': arm, 'vm_session_id': 'fixtureVM',
            'package_root': str(self.root / f'{tag}-pkg'), 'package_manifest_sha256': PKG, 'workdir': str(self.work),
            'input_root': str(self.root / 'input'), 'stage_root': str(stage), 'manifest_sha256': 'b' * 64,
            'content_sha256': CONTENT, 'stage_receipt_sha256': sha(self.stage_receipt.read_bytes()),
            'initializer_descriptor': str(self.init), 'initializer_descriptor_sha256': sha(self.init.read_bytes()),
            'persister': '/x/persist.py', 'persister_sha256': runner.PERSISTER_SHA,
            'persistence': {'dataset': backup or f'ozgung/knee-{tag}-fold0-20261013', 'private': True,
                            'staging': str(self.root / f'{tag}-backup-staging'), 'status_file': str(self.status_file)},
            'controller_lock': str(self.lock), 'workers': 2, 'resume': False, 'fold': k}
        self.config_path = self.root / f'{tag}-run-config.json'
        self.config_path.write_text(json.dumps(self.config))
        self.config_sha = sha(self.config_path.read_bytes())
        self.fingerprint = {'controller_sha256': 'e' * 64, 'config_sha256_canonical': canonical(self.config),
                            'version': 'colab-raptor64-run-v1', 'name': self.config['name'], 'arm': arm,
                            'package_manifest_sha256': PKG, 'cache_content_sha256': CONTENT, 'fold': k,
                            'experiment_split': self.split, 'policy': policy}
        self.write_json('raptor64-run-fingerprint.json', self.fingerprint)
        receipt = dict(self.fingerprint, training_returncode=0, persister_returncode=0, interrupted_signal=None,
                       resume_preserved=True)
        if kind == 'pass':
            receipt.update(status='PASS', checkpoint_sha256=record['checkpoint_sha256'],
                           backup={'epoch': 11, 'version': 3, 'resume_sha256': 'f' * 64})
        else:
            receipt.update(status='FAIL', failure='FileNotFoundError')
        self.write_json('raptor64-run-receipt.json', receipt)

    def write_json(self, name, value):
        (self.work / name).write_text(json.dumps(value))

    def edit_json(self, name, fn):
        p = self.work / name
        v = json.loads(p.read_text())
        fn(v)
        p.write_text(json.dumps(v))

    def request(self, **kw):
        out = self.root / 'out'
        out.mkdir(exist_ok=True)
        tag = f'-{next(self.counter)}'
        d = dict(fold=0, arm=self.arm, workdir=self.work, config=self.config_path, config_sha256=self.config_sha,
                 package_manifest_sha256=PKG, cache_content_sha256=CONTENT, runner=RUNNER, runner_sha256=RUNNER_SHA,
                 dataset=f'ozgung/knee-raptor64-{self.arm}-fold0-final-20261013', staging=out / ('staging' + tag),
                 verify_root=out / ('verify' + tag), receipt=out / ('receipt' + tag + '.json'),
                 source_revision='abcdef1234567', exporter_path=EXPORTER, ready_timeout=900, poll=15)
        d.update(kw)
        return ex.Request(**d)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 0.001
        return self.t

    def sleep(self, seconds):
        self.t += seconds


class FakeTransport:
    """In-memory Kaggle stand-in; records every call."""

    def __init__(self, exists=False, private=True, identity=None, create_private=True, tamper=None, hook=None, lag=0,
                 serve=None, forbidden_until_created=False, forbidden_always=False, create_error=None):
        self.exists, self.private, self.identity, self.create_private = exists, private, identity, create_private
        self.tamper, self.hook, self.lag, self.serve = tamper, hook, lag, serve
        self.forbidden_until_created, self.forbidden_always, self.create_error = (
            forbidden_until_created, forbidden_always, create_error)
        self.base = 1 if exists else 0
        self.versions, self.calls, self.status_calls = [], [], 0
        self.store = Path(tempfile.mkdtemp())

    def writes(self):
        return [c for c in self.calls if c[0] in ('create', 'new_version')]

    def dataset_state(self, slug):
        self.calls.append(('state', slug))
        if self.forbidden_always or (self.forbidden_until_created and not self.versions):
            raise ex.MetadataInaccessible('metadata lookup failed: HTTPError: 403 Client Error: Forbidden')
        if not self.exists:
            return None
        return {'identity': self.identity or slug, 'private': self.private}

    def latest_version(self, slug):
        self.calls.append(('latest', slug))
        return self.base + len(self.versions)

    def _store(self, folder):
        dest = self.store / f'v{len(self.versions)}'
        shutil.copytree(folder, dest)
        self.versions.append(dest)
        if self.hook:
            self.hook()

    def create(self, slug, folder):
        self.calls.append(('create', slug))
        if self.create_error:
            raise self.create_error
        self._store(folder)
        self.exists, self.private = True, self.create_private

    def new_version(self, slug, folder, note):
        self.calls.append(('new_version', slug))
        self._store(folder)

    def status(self, slug):
        self.status_calls += 1
        if self.status_calls <= self.lag:
            return 'ready', self.base
        return 'ready', self.base + len(self.versions)

    def download(self, slug, version, dest):
        self.calls.append(('download', slug, version))
        src = self.serve or self.versions[version - self.base - 1]
        shutil.copytree(src, dest, dirs_exist_ok=True)
        if self.tamper:
            self.tamper(Path(dest))


class Base(unittest.TestCase):
    def fit(self, fold=0, kind='pass', **kw):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        return Fit(d, fold, kind, **kw)

    def go(self, fit, transport=None, req=None, **kw):
        req = req or fit.request(**kw)
        clock = Clock()
        t = transport or FakeTransport()
        receipt = ex.run(req, transport=t, runner=runner, sleep=clock.sleep, clock=clock, free_bytes=lambda p: 10 ** 12)
        return req, t, receipt

    def refused(self, fit, transport=None, contains='', phase=None, **kw):
        before = tree_state(fit.work)
        req, t, r = self.go(fit, transport, **kw)
        self.assertEqual(r['state'], 'failed', r)
        self.assertFalse(r['upload_attempted'])
        self.assertEqual(t.writes(), [], 'refused export must not upload')
        self.assertIn(contains, ' | '.join(r['problems']))
        if phase:
            self.assertEqual(r['phase'], phase)
        self.assertEqual(tree_state(fit.work), before, 'refusal must leave the workdir untouched')
        self.assertTrue(req.receipt.is_file())
        return req, t, r


class CompleteExportTests(Base):
    def test_verified_export_and_manifest(self):
        f = self.fit()
        before = tree_state(f.work)
        req, t, r = self.go(f)
        self.assertEqual((r['state'], r['phase'], r['dataset_version'], r['action']), ('verified', 'done', 1, 'create'), r)
        self.assertEqual(t.calls[0], ('state', req.dataset))
        self.assertEqual([c[0] for c in t.writes()], ['create'])
        expected = {f'raptor_ft_fold0_swa.pt', 'oof_fold0.csv', 'run_fold0.json', 'raptor_ft_fold0.json',
                    'raptor64-run-fingerprint.json', 'raptor64-run-receipt.json', 'train_meta_fold0.json',
                    'raptor_oof_fold0_swa_fold0.npz',
                    'initial-state-record.json', 'run-config.json', 'staging-receipt.json', 'initializer-descriptor.json',
                    'persist-status-fold0.json'}
        m = json.loads((req.staging / 'MANIFEST.json').read_text())
        self.assertEqual(set(m['files']), expected)
        self.assertEqual({p.name for p in req.staging.iterdir()}, expected | {'MANIFEST.json', 'dataset-metadata.json'})
        for name, meta in m['files'].items():
            data = (req.staging / name).read_bytes()
            self.assertEqual((meta['size'], meta['sha256']), (len(data), sha(data)))
            if meta['origin'] == 'workdir':
                self.assertEqual(data, (f.work / name).read_bytes())
        self.assertNotIn('MANIFEST.json', m['files'])
        self.assertEqual(m['checkpoint_sha256'], sha(f.swa.read_bytes()))
        self.assertEqual(set(m['checkpoint_sha256_sources'].values()), {sha(f.swa.read_bytes())})
        self.assertEqual(m['selected_epochs'], [11, 10, 9])
        self.assertEqual((m['fold'], m['arm'], m['label'], m['complete'], m['k_eval'], m['n_val'], m['epochs_completed']),
                         (0, ARM, 'FINAL', True, 42, 870, 12))
        self.assertEqual(m['comparison_endpoint'], 'K42 primary (selection K16)')
        self.assertEqual((m['source_revision'], m['runner_sha256'], m['receipt_mode']), ('abcdef1234567', RUNNER_SHA, 'controller_pass'))
        self.assertEqual(m['pins']['config_sha256'], f.config_sha)
        self.assertEqual(m['exporter_sha256'], sha(EXPORTER.read_bytes()))
        self.assertEqual(m['resume_backup_dataset'], BACKUP)
        self.assertNotIn('raptor_resume_fold0.pt', m['files'])
        self.assertEqual(r['manifest_sha256'], sha((req.staging / 'MANIFEST.json').read_bytes()))
        self.assertEqual(json.loads((req.staging / 'dataset-metadata.json').read_text())['id'], req.dataset)
        saved = json.loads(req.receipt.read_text())
        self.assertEqual(saved['state'], 'verified')
        self.assertTrue(saved['workdir_unchanged'])
        self.assertEqual(tree_state(f.work), before, 'workdir must be unchanged after a verified export')
        self.assertEqual(list(req.verify_root.iterdir()), [], 'verification download removed on success')

    def test_logs_only_with_flag_and_controller_log(self):
        f = self.fit()
        req, _, r = self.go(f)
        self.assertNotIn('raptor64-training.log', json.loads((req.staging / 'MANIFEST.json').read_text())['files'])
        clog = f.root / 'controller.log'
        clog.write_text('controller')
        req, _, r = self.go(f, include_logs=True, controller_log=clog)
        self.assertEqual(r['state'], 'verified', r)
        m = json.loads((req.staging / 'MANIFEST.json').read_text())
        self.assertTrue({'log_fold0.txt', 'raptor64-training.log', 'controller-fold0.log'} <= set(m['files']))
        self.assertTrue(m['includes_private_logs'])
        self.assertIn('raptor64-backup.log', m['absent_optional'])

    def test_arm_split_and_reconciliation_rules(self):
        f = self.fit()
        self.refused(f, contains='--accept-reconciled-receipt is a source96-only path', accept_reconciled_receipt=True)
        self.refused(self.fit(), contains='Raptor64 exports are fold 0 only', fold=1)
        self.refused(self.fit(), contains='--arm must be one of', arm='imagenet-original')
        # Request arm differs from config arm (destination renamed accordingly so only the arm check can fire).
        self.refused(self.fit(), contains='Run config arm differs', arm='dr-orig',
                     dataset='ozgung/knee-raptor64-dr-orig-fold0-final-20261013')
        g = self.fit()
        g.edit_json('raptor64-run-fingerprint.json', lambda v: v.update(arm='dr-plane'))
        self.refused(g, contains='Run fingerprint arm differs')
        g = self.fit()
        g.edit_json('raptor64-run-fingerprint.json', lambda v: v['experiment_split'].update(development=866))
        self.refused(g, contains='Experiment split / arm policy identity differs')
        g = self.fit()
        g.edit_json('run_fold0.json', lambda v: v['raptor64_policy'].update(arm='dr-orig'))
        self.refused(g, contains='primary_complete rejected the workdir')
        g = self.fit()
        g.edit_json('raptor64-run-receipt.json', lambda v: v.update(status='FAIL', failure='FileNotFoundError'))
        self.refused(g, contains='Controller receipt is not PASS')
        g = self.fit()
        g.edit_json('raptor64-run-receipt.json', lambda v: v.update(backup={'epoch': 9, 'version': 3}))
        self.refused(g, contains='epoch11 backup')
        g = self.fit()
        g.edit_json('raptor64-run-receipt.json', lambda v: v.update(arm='in-orig'))
        self.refused(g, contains='Controller receipt arm differs')

    def full_fit(self):
        f = self.fit(arm='dr-full')
        split = {'fold': None, 'eligible': 4345, 'training': 4345, 'development': 0,
                 'ordered_eligible_sha256': '1' * 64, 'ordered_training_sha256': '1' * 64, 'ordered_development_sha256': '4' * 64}
        policy = {'arm': 'dr-full', 'mode': 'full', 'swa_epochs': [9, 10, 11], 'experiment_split': split, 'experiment_split_sha256': canonical(split)}

        def record(v):
            v.update(mode='full', checkpoint_kind='swa_fixed_epochs', export=None, comparison_endpoint='none (full-data fit for an LB probe)',
                     raptor64_policy=policy, raptor64_experiment_split=split)
        f.edit_json('run_fold0.json', record)
        for name in ('raptor64-run-fingerprint.json', 'raptor64-run-receipt.json'):
            f.edit_json(name, lambda v: v.update(experiment_split=split, policy=policy))
        (f.work / 'oof_fold0.csv').unlink(); (f.work / 'raptor_oof_fold0_swa_fold0.npz').unlink()
        return f

    def test_full_data_arm_exports_without_development_predictions(self):
        f = self.full_fit()
        req, _, r = self.go(f)
        self.assertEqual(r['state'], 'verified', r)
        m = json.loads((req.staging / 'MANIFEST.json').read_text())
        self.assertEqual((m['arm'], m['dataset'], m['checkpoint_kind'], m['k_eval'], m['n_val']),
                         ('dr-full', 'ozgung/knee-raptor64-dr-full-fold0-final-20261013', 'swa_fixed_epochs', None, None))
        self.assertNotIn('oof_fold0.csv', m['files']); self.assertIn('raptor_ft_fold0_swa.pt', m['files'])
        g = self.full_fit()
        g.edit_json('run_fold0.json', lambda v: v.update(selected_epochs=[8, 10, 11]))
        self.refused(g, contains='primary_complete')
        h = self.full_fit()
        h.edit_json('raptor64-run-fingerprint.json', lambda v: v['policy'].update(mode='fold0'))
        self.refused(h, contains='policy identity')

    def test_other_arm_exports_and_optional_k16_npz(self):
        f = self.fit(arm='dr-orig')
        req, _, r = self.go(f)
        self.assertEqual(r['state'], 'verified', r)
        m = json.loads((req.staging / 'MANIFEST.json').read_text())
        self.assertEqual((m['arm'], m['dataset']), ('dr-orig', 'ozgung/knee-raptor64-dr-orig-fold0-final-20261013'))
        self.assertEqual(m['files']['raptor_oof_fold0_swa_fold0.npz']['role'], 'k16_in_training_swa_oof')
        g = self.fit()
        (g.work / 'raptor_oof_fold0_swa_fold0.npz').unlink()
        req, _, r = self.go(g)
        self.assertEqual(r['state'], 'verified', r)
        self.assertIn('raptor_oof_fold0_swa_fold0.npz', json.loads((req.staging / 'MANIFEST.json').read_text())['absent_optional'])

    def test_staged_checkpoint_must_match_the_recorded_hash(self):
        f = self.fit()
        real_copy = ex.copy_hashed

        def drifting(src, dst, limit=None):
            digest, size = real_copy(src, dst, limit)
            return ('0' * 64, size) if Path(src).name.endswith('_swa.pt') else (digest, size)
        with patch.object(ex, 'copy_hashed', drifting):
            req, t, r = self.go(f)
        self.assertEqual((r['state'], r['phase'], r['upload_attempted']), ('failed', 'staging', False), r)
        self.assertIn('checkpoint_sha256 differs between sources', ' '.join(r['problems']))
        self.assertEqual(t.writes(), [])

    def test_pass_receipt_must_bind_checkpoint(self):
        f = self.fit()
        f.edit_json('raptor64-run-receipt.json', lambda v: v.update(checkpoint_sha256='0' * 64))
        self.refused(f, contains='PASS controller receipt does not match')
        f = self.fit()
        f.edit_json('raptor64-run-receipt.json', lambda v: v.update(cache_content_sha256='0' * 64))
        self.refused(f, contains='Controller receipt cache_content_sha256 differs')


class RefusalTests(Base):
    def test_incomplete_or_corrupt_fits_are_refused(self):
        def nine(g):
            g.edit_json('run_fold0.json', lambda v: (v.update(epochs_completed=11), v.update(history=v['history'][:11])))
            g.edit_json('raptor_ft_fold0.json', lambda v: v.update(hist=v['hist'][:11]))

        def symlink(path):
            def make(g):
                target = g.root / 'elsewhere'
                shutil.copy(g.work / path, target)
                (g.work / path).unlink()
                (g.work / path).symlink_to(target)
            return make

        cases = {
            'eleven epochs': nine,
            'missing oof': lambda g: (g.work / 'oof_fold0.csv').unlink(),
            'checkpoint hash mismatch': lambda g: g.swa.write_bytes(g.swa.read_bytes() + b'x'),
            'missing checkpoint': lambda g: g.swa.unlink(),
            'symlinked checkpoint': symlink('raptor_ft_fold0_swa.pt'),
            'symlinked oof': symlink('oof_fold0.csv'),
            'partial meta': lambda g: g.edit_json('raptor_ft_fold0.json', lambda v: v.update(partial=True)),
            'time budget stop': lambda g: g.edit_json('run_fold0.json', lambda v: v.update(stopped_by_time_budget=True)),
            'k16 export': lambda g: g.edit_json('run_fold0.json', lambda v: v['export'].update(k_eval=16)),
            'source96 endpoint': lambda g: g.edit_json('run_fold0.json', lambda v: v.update(comparison_endpoint='matched K16 primary')),
            'cache identity': lambda g: g.edit_json('run_fold0.json', lambda v: v.update(raptor64_cache_content_sha256='0' * 64)),
            'repeated selected epoch': lambda g: g.edit_json('run_fold0.json', lambda v: v.update(selected_epochs=[11, 11, 10])),
            'wrong n_val': lambda g: g.edit_json('run_fold0.json', lambda v: v['export'].update(n_val=866)),
            'two selected epochs': lambda g: g.edit_json('run_fold0.json', lambda v: v.update(selected_epochs=[11, 10])),
        }
        for label, mutate in cases.items():
            with self.subTest(label):
                g = self.fit()
                mutate(g)
                _, _, r = self.refused(g, contains='primary_complete rejected the workdir', phase='preflight')
                self.assertFalse(r['problems'] == [])

    def test_parity_with_runner_primary_complete(self):
        """The exporter's completeness verdict is the runner's own primary_complete()."""
        good = self.fit()
        runner.primary_complete(str(good.work), 0, 870, SPLIT, CONTENT, ARM)
        self.assertEqual(self.go(good)[2]['state'], 'verified')
        for label, mutate in {
            'oof': lambda g: (g.work / 'oof_fold0.csv').unlink(),
            'hash': lambda g: g.swa.write_bytes(b'tampered'),
            'epochs': lambda g: g.edit_json('run_fold0.json', lambda v: v.update(epochs_completed=11)),
        }.items():
            with self.subTest(label):
                g = self.fit()
                mutate(g)
                with self.assertRaises(Exception):
                    runner.primary_complete(str(g.work), 0, 870, SPLIT, CONTENT, ARM)
                self.refused(g, contains='primary_complete rejected the workdir')

    def test_missing_or_mismatched_receipts_and_pins(self):
        cases = {
            'missing fingerprint': (lambda g: (g.work / 'raptor64-run-fingerprint.json').unlink(), {}, 'raptor64-run-fingerprint.json'),
            'missing controller receipt': (lambda g: (g.work / 'raptor64-run-receipt.json').unlink(), {}, 'raptor64-run-receipt.json'),
            'symlinked controller receipt': (lambda g: (shutil.copy(g.work / 'raptor64-run-receipt.json', g.root / 'r'), (g.work / 'raptor64-run-receipt.json').unlink(), (g.work / 'raptor64-run-receipt.json').symlink_to(g.root / 'r')), {}, 'not a readable regular file'),
            'fingerprint fold': (lambda g: g.edit_json('raptor64-run-fingerprint.json', lambda v: v.update(fold=1)), {}, 'fingerprint fold differs'),
            'fingerprint config hash': (lambda g: g.edit_json('raptor64-run-fingerprint.json', lambda v: v.update(config_sha256_canonical='0' * 64)), {}, 'config_sha256_canonical'),
            'config pin': (lambda g: None, {'config_sha256': '0' * 64}, 'sha256 differs from the pin'),
            'package pin': (lambda g: None, {'package_manifest_sha256': 'd' * 64}, 'package/cache hashes differ'),
            'cache pin': (lambda g: None, {'cache_content_sha256': 'd' * 64}, 'package/cache hashes differ'),
            'controller pin': (lambda g: None, {'controller_sha256': 'd' * 64}, 'controller_sha256 differs'),
            'runner pin': (lambda g: None, {}, 'Pinned runner source differs'),
        }
        for label, (mutate, kw, text) in cases.items():
            with self.subTest(label):
                g = self.fit()
                mutate(g)
                if label == 'runner pin':
                    req = g.request(runner_sha256='0' * 64)
                    clock = Clock()
                    t = FakeTransport()
                    r = ex.run(req, transport=t, runner=None, sleep=clock.sleep, clock=clock, free_bytes=lambda p: 10 ** 12)
                    self.assertEqual(r['state'], 'failed')
                    self.assertIn(text, ' '.join(r['problems']))
                    self.assertEqual(t.writes(), [])
                else:
                    self.refused(g, contains=text, **kw)

    def test_stage_receipt_and_initializer_pins_are_rechecked(self):
        g = self.fit()
        g.stage_receipt.write_text(json.dumps({'status': 'FAIL'}))
        self.refused(g, contains='staging-receipt.json')
        g = self.fit()
        g.init.write_text('tampered')
        self.refused(g, contains='initializer.json')

    def test_running_controller_lock_blocks_export(self):
        g = self.fit()
        fd = os.open(g.lock, os.O_RDWR)
        self.addCleanup(os.close, fd)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.refused(g, contains='controller has not ended')
        fcntl.flock(fd, fcntl.LOCK_UN)
        self.assertEqual(self.go(g)[2]['state'], 'verified')

    def test_missing_lock_file_blocks_export(self):
        g = self.fit()
        g.lock.unlink()
        self.refused(g, contains='Controller lock file absent')

    def test_exporter_holds_the_controller_lock_while_running(self):
        g = self.fit()
        seen = []

        def hook():
            fd = os.open(g.lock, os.O_RDONLY)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                seen.append('free')
            except BlockingIOError:
                seen.append('held')
            finally:
                os.close(fd)
        self.assertEqual(self.go(g, FakeTransport(hook=hook))[2]['state'], 'verified')
        self.assertEqual(seen, ['held'])

    def test_path_guards(self):
        g = self.fit()
        out = g.root / 'out'
        out.mkdir()
        (out / 'taken').mkdir()
        self.refused(g, contains='already exists', staging=out / 'taken')
        self.refused(g, contains='outside the workdir', staging=g.work / 'staging')
        self.refused(g, contains='outside the workdir', verify_root=g.work / 'verify')
        self.refused(g, contains='separate from the persister staging', staging=Path(g.config['persistence']['staging']))
        self.refused(g, contains='separate', staging=out / 'same', verify_root=out / 'same' / 'inner')
        self.refused(g, contains='absolute path', staging=Path('relative-staging'))
        self.refused(g, contains='Workdir must be', workdir=g.root / 'missing')
        self.refused(g, contains='source-revision', source_revision='not a revision')
        self.refused(g, contains='must be a 64-hex pin', package_manifest_sha256='short')

    def test_insufficient_disk_refuses_before_staging(self):
        g = self.fit()
        req = g.request()
        clock = Clock()
        t = FakeTransport()
        r = ex.run(req, transport=t, runner=runner, sleep=clock.sleep, clock=clock, free_bytes=lambda p: 1000)
        self.assertEqual(r['state'], 'failed')
        self.assertIn('Insufficient free disk', ' '.join(r['problems']))
        self.assertFalse(req.staging.exists())
        self.assertEqual(t.writes(), [])

    def test_receipt_path_guards(self):
        g = self.fit()
        existing = g.root / 'existing-receipt.json'
        existing.write_text('keep')
        with self.assertRaises(SystemExit):
            self.go(g, receipt=existing)
        self.assertEqual(existing.read_text(), 'keep')
        before = tree_state(g.work)
        with self.assertRaises(SystemExit):
            self.go(g, receipt=g.work / 'receipt.json')
        self.assertEqual(tree_state(g.work), before)


class DestinationTests(Base):
    def test_resume_backup_datasets_are_never_targeted(self):
        for name in RESUME_BACKUPS + [BACKUP + '-r2', 'ozgung/knee-raptor64-in-plane-fold1-20261013']:
            with self.subTest(dataset=name):
                g = self.fit()
                t = FakeTransport(exists=True)
                self.refused(g, t, contains='resume-backup dataset', dataset=name, new_version=True)
                self.assertEqual(t.calls, [], 'no remote call may be made for a forbidden destination')
        g = self.fit(backup='ozgung/some-other-backup-20261012')
        t = FakeTransport(exists=True)
        self.refused(g, t, contains='resume-backup dataset', dataset='ozgung/some-other-backup-20261012', new_version=True)
        self.assertEqual(t.calls, [])

    def test_name_pattern_owner_and_mode(self):
        for name, text in {
            'ozgung/knee-raptor64-in-plane-fold1-final-20261013': 'must match',
            'ozgung/knee-raptor64-dr-plane-fold0-final-20261013': 'must match',
            'ozgung/knee-source96-fold0-final-20261012': 'must match',
            FAILED: 'must match',
            'ozgung/knee-raptor64-in-plane-fold0-final': 'must match',
            'ozgung/knee-raptor64-in-plane-fold0-final-20261013-extra': 'must match',
            'someone/knee-raptor64-in-plane-fold0-final-20261013': 'owner',
            'Ozgung/Knee Source': 'owner/lowercase-slug',
        }.items():
            with self.subTest(name):
                self.refused(self.fit(), FakeTransport(), contains=text, dataset=name)
        req, _, r = self.go(self.fit(), dataset=FINAL + '-r2')
        self.assertEqual(r['state'], 'verified', r)

    def test_slug_length_guard(self):
        self.assertLessEqual(max(len(f'knee-raptor64-{a}-fold0-final-20261014') for a in ex.ARMS), ex.SLUG_MAX)
        # Long it3 arm names leave no room for a two-digit retry suffix; the runtime guard refuses those slugs.
        self.assertLessEqual(max(len(f'knee-raptor64-{a}-fold0-failed-20261013-r99') for a in ex.ARMS if len(a) <= 9), ex.SLUG_MAX)
        with patch.object(ex, 'SLUG_MAX', 40):
            self.refused(self.fit(), FakeTransport(), contains='at most 40')

    def test_existing_destination_needs_new_version_flag(self):
        t = FakeTransport(exists=True)
        self.refused(self.fit(), t, contains='--new-version', phase='destination')
        self.assertEqual([c[0] for c in t.calls], ['state'])

    def test_public_or_misidentified_destination_is_refused(self):
        for label, t in {'public': FakeTransport(exists=True, private=False),
                         'unknown privacy': FakeTransport(exists=True, private=None),
                         'string true': FakeTransport(exists=True, private='true'),
                         'misidentified': FakeTransport(exists=True, identity=BACKUP)}.items():
            with self.subTest(label):
                self.refused(self.fit(), t, contains='identity/privacy is unverified or public', new_version=True)
                self.assertEqual(t.writes(), [])

    def test_new_version_on_existing_private_destination(self):
        t = FakeTransport(exists=True)
        req, t, r = self.go(self.fit(), t, new_version=True)
        self.assertEqual((r['state'], r['action'], r['prior_version'], r['dataset_version']), ('verified', 'version', 1, 2), r)
        self.assertEqual([c[0] for c in t.writes()], ['new_version'])
        # an unchanged old version is never what gets verified
        self.assertIn(('download', req.dataset, 2), t.calls)

    def test_created_public_dataset_is_reported_failed(self):
        req, t, r = self.go(self.fit(), FakeTransport(create_private=False))
        self.assertEqual((r['state'], r['phase']), ('failed', 'verify'), r)
        self.assertTrue(r['upload_attempted'])
        self.assertIn('identity/privacy', ' '.join(r['problems']))
        self.assertNotIn('download', [c[0] for c in t.calls], 'a public dataset must be flagged before any download')


class CreateFreshTests(Base):
    """Real metadata for a not-yet-created slug returned HTTP 403, which is UNKNOWN, not absence."""

    def test_403_without_the_flag_is_unknown_and_refused(self):
        t = FakeTransport(forbidden_until_created=True)
        req, t, r = self.refused(self.fit(), t, contains='existence is UNKNOWN', phase='destination')
        self.assertNotEqual(r.get('existence'), 'absent')

    def test_create_fresh_creates_privately_and_verifies(self):
        f = self.fit()
        req = f.request()
        req.create_fresh = req.dataset
        req, t, r = self.go(f, FakeTransport(forbidden_until_created=True), req=req)
        self.assertEqual((r['state'], r['action'], r['dataset_version'], r['existence']),
                         ('verified', 'create', 1, 'unknown_metadata_inaccessible'), r)
        self.assertEqual([c[0] for c in t.writes()], ['create'])
        self.assertTrue(t.private)

    def test_create_fresh_must_name_the_exact_slug_and_not_combine_with_new_version(self):
        for kw in ({'create_fresh': FINAL + '-r2'},
                   {'create_fresh': FINAL, 'new_version': True}):
            with self.subTest(kw):
                t = FakeTransport(forbidden_until_created=True)
                self.refused(self.fit(), t, contains='--create-fresh must name exactly', **kw)
                self.assertEqual(t.calls, [])

    def test_create_fresh_refuses_a_visible_existing_destination(self):
        t = FakeTransport(exists=True)
        self.refused(self.fit(), t, contains='--create-fresh given but the destination already exists', phase='destination',
                     create_fresh=FINAL)

    def test_create_conflict_fails_without_verification_or_versioning(self):
        t = FakeTransport(exists=True, forbidden_until_created=True,
                          create_error=ex.TransportError('create failed: already exists'))
        req, t, r = self.go(self.fit(), t, create_fresh=FINAL)
        self.assertEqual((r['state'], r['phase']), ('failed', 'upload'), r)
        self.assertIn('already exists', ' '.join(r['problems']))
        self.assertEqual([c[0] for c in t.calls if c[0] in ('new_version', 'download')], [])

    def test_silent_new_version_of_an_existing_dataset_is_not_accepted(self):
        t = FakeTransport(exists=True, forbidden_until_created=True)
        req, t, r = self.go(self.fit(), t, create_fresh=FINAL)
        self.assertEqual((r['state'], r['phase'], r['dataset_version']), ('failed', 'verify', 2), r)
        self.assertIn('not a fresh dataset', ' '.join(r['problems']))
        self.assertNotIn('download', [c[0] for c in t.calls])

    def test_identity_and_privacy_are_mandatory_after_create(self):
        for label, t in {'public': FakeTransport(forbidden_until_created=True, create_private=False),
                         'still inaccessible': FakeTransport(forbidden_always=True),
                         'misidentified': FakeTransport(forbidden_until_created=True, identity='ozgung/other')}.items():
            with self.subTest(label):
                req, t, r = self.go(self.fit(), t, create_fresh=FINAL, ready_timeout=60)
                self.assertEqual((r['state'], r['phase']), ('failed', 'verify'), r)
                self.assertIn('identity/privacy', ' '.join(r['problems']))
                self.assertNotIn('download', [c[0] for c in t.calls])
                self.assertEqual(len(t.writes()), 1)

    def test_resume_backup_names_still_refused_with_create_fresh(self):
        t = FakeTransport(forbidden_until_created=True)
        self.refused(self.fit(), t, contains='resume-backup dataset', dataset=RESUME_BACKUPS[0],
                     create_fresh=RESUME_BACKUPS[0])
        self.assertEqual(t.calls, [])


class VerificationTests(Base):
    def assertFailed(self, r, text, phase='verify'):
        self.assertEqual((r['state'], r['phase']), ('failed', phase), r)
        self.assertIn(text, ' | '.join(r['problems']))

    def test_download_hash_mismatch_fails_and_keeps_the_download(self):
        def tamper(d):
            (d / 'raptor_ft_fold0_swa.pt').write_bytes(b'different bytes')
        f = self.fit()
        req, t, r = self.go(f, FakeTransport(tamper=tamper))
        self.assertFailed(r, 'raptor_ft_fold0_swa.pt: sha256 differs')
        self.assertTrue(Path(r['kept_download']).is_dir())
        self.assertEqual(Path(r['kept_download']).parent, req.verify_root)
        self.assertEqual(json.loads(req.receipt.read_text())['state'], 'failed')

    def test_same_size_single_byte_flip_is_detected(self):
        def tamper(d):
            p = d / 'oof_fold0.csv'
            data = bytearray(p.read_bytes())
            data[-2] ^= 1
            p.write_bytes(bytes(data))
        req, t, r = self.go(self.fit(), FakeTransport(tamper=tamper))
        self.assertFailed(r, 'oof_fold0.csv: sha256 differs')

    def test_missing_extra_and_resized_files_fail(self):
        def missing(d):
            (d / 'run_fold0.json').unlink()

        def extra(d):
            (d / 'surprise.bin').write_bytes(b'x')

        def resized(d):
            p = d / 'raptor_ft_fold0.json'
            p.write_bytes(p.read_bytes() + b' ')
        for label, tamper, text in (('missing', missing, 'run_fold0.json: missing'), ('extra', extra, 'surprise.bin: unexpected file'),
                                    ('resized', resized, 'raptor_ft_fold0.json: size')):
            with self.subTest(label):
                self.assertFailed(self.go(self.fit(), FakeTransport(tamper=tamper))[2], text)

    def test_lagging_server_version_is_retried_then_verified(self):
        req, t, r = self.go(self.fit(), FakeTransport(lag=3))
        self.assertEqual(r['state'], 'verified', r)
        self.assertGreaterEqual(t.status_calls, 4)

    def test_never_ready_times_out_as_failed(self):
        req, t, r = self.go(self.fit(), FakeTransport(lag=10 ** 6), ready_timeout=60, poll=15)
        self.assertFailed(r, 'not verified within 60s')

    def test_foreign_manifest_is_not_accepted(self):
        other = self.fit()
        _, other_t, other_r = self.go(other)
        self.assertEqual(other_r['state'], 'verified')
        req, t, r = self.go(self.fit(), FakeTransport(serve=other_t.versions[0]), ready_timeout=60)
        self.assertFailed(r, 'not verified within 60s')
        self.assertIn('MANIFEST.json', ' '.join(r['problems']) + json.dumps(r))

    def test_transient_transport_errors_and_late_visibility_are_retried(self):
        class Flaky(FakeTransport):
            def __init__(self):
                super().__init__()
                self.state_none, self.status_errors, self.download_errors = 2, 2, 1

            def dataset_state(self, slug):
                s = super().dataset_state(slug)
                if self.versions and self.state_none > 0:
                    self.state_none -= 1
                    return None
                return s

            def status(self, slug):
                if self.versions and self.status_errors > 0:
                    self.status_errors -= 1
                    raise ex.TransportError('status failed: 404')
                return super().status(slug)

            def download(self, slug, version, dest):
                if self.download_errors > 0:
                    self.download_errors -= 1
                    raise ex.TransportError('download failed: 502')
                return super().download(slug, version, dest)
        req, t, r = self.go(self.fit(), Flaky())
        self.assertEqual(r['state'], 'verified', r)
        self.assertEqual(len(t.writes()), 1, 'retries must never re-upload')

    def test_dataset_never_visible_after_upload_fails(self):
        class Invisible(FakeTransport):
            def dataset_state(self, slug):
                return super().dataset_state(slug) if not self.versions else None
        req, t, r = self.go(self.fit(), Invisible(), ready_timeout=60)
        self.assertFailed(r, 'identity/privacy')
        self.assertEqual(len(t.writes()), 1)

    def test_operator_interrupt_still_writes_a_failed_receipt(self):
        class Interrupted(FakeTransport):
            def create(self, slug, folder):
                raise KeyboardInterrupt
        f = self.fit()
        req = f.request()
        with self.assertRaises(KeyboardInterrupt):
            self.go(f, Interrupted(), req=req)
        saved = json.loads(req.receipt.read_text())
        self.assertEqual((saved['state'], saved['upload_attempted']), ('failed', True))
        self.assertIn('interrupted', ' '.join(saved['problems']))

    def test_workdir_change_during_export_fails(self):
        f = self.fit()

        def hook():
            (f.work / 'late-write.txt').write_text('trainer still writing')
        req, t, r = self.go(f, FakeTransport(hook=hook))
        self.assertFailed(r, 'workdir changed during the export')

    def test_transport_errors_become_failed_receipts(self):
        class Boom(FakeTransport):
            def create(self, slug, folder):
                raise ex.TransportError('create failed: 500')
        req, t, r = self.go(self.fit(), Boom())
        self.assertEqual((r['state'], r['phase'], r['upload_attempted']), ('failed', 'upload', True))
        self.assertIn('create failed', ' '.join(r['problems']))

        class Unknown(FakeTransport):
            def dataset_state(self, slug):
                raise ex.TransportError('metadata lookup failed: 403')
        req, t, r = self.go(self.fit(), Unknown())
        self.assertEqual((r['state'], r['upload_attempted']), ('failed', False))


class FailedModeTests(Base):
    def failed_fit(self):
        f = self.fit(kind='fail')
        f.edit_json('raptor64-run-receipt.json', lambda v: v.update(failure='RuntimeError', training_returncode=1))
        f.edit_json('run_fold0.json', lambda v: (v.update(epochs_completed=4), v.update(history=v['history'][:4])))
        return f

    def test_receipts_only_and_labelled_failed(self):
        f = self.failed_fit()
        before = tree_state(f.work)
        req, t, r = self.go(f, dataset=FAILED, mode='failed')
        self.assertEqual(r['state'], 'verified', r)
        names = {p.name for p in req.staging.iterdir()}
        self.assertFalse({n for n in names if n.endswith(('.pt', '.csv'))}, names)
        self.assertNotIn('raptor_resume_fold0.pt', names)
        self.assertIn('FAILED.txt', names)
        m = json.loads((req.staging / 'MANIFEST.json').read_text())
        self.assertEqual((m['label'], m['complete']), ('FAILED', False))
        self.assertEqual(m['failure_summary']['failure'], 'RuntimeError')
        self.assertEqual(m['failure_summary']['training_returncode'], 1)
        self.assertIn('FAILED.txt', m['files'])
        self.assertIn('raptor64-run-receipt.json', m['files'])
        self.assertNotIn('checkpoint_sha256', m)
        self.assertTrue((req.staging / 'FAILED.txt').read_text().startswith('FAILED'))
        self.assertEqual(r['dataset'], FAILED)
        self.assertEqual(tree_state(f.work), before)

    def test_failed_mode_refuses_a_complete_fit(self):
        f = self.fit()
        self.refused(f, contains='fit is complete', dataset=FAILED, mode='failed')

    def test_failed_mode_destination_and_flags(self):
        f = self.failed_fit()
        self.refused(f, contains='must match', mode='failed')   # default (final) name
        self.refused(f, contains='--accept-reconciled-receipt is a source96-only path', dataset=FAILED,
                     mode='failed', accept_reconciled_receipt=True)
        self.refused(f, contains='resume-backup', dataset=BACKUP, mode='failed', new_version=True)

    def test_failed_mode_works_with_only_the_config_present(self):
        f = self.failed_fit()
        for p in list(f.work.iterdir()):
            if p.name.startswith(('raptor64-run', 'raptor_ft', 'run_fold', 'train_meta', 'initial')):
                p.unlink()
        req, t, r = self.go(f, dataset=FAILED, mode='failed')
        self.assertEqual(r['state'], 'verified', r)


class RealTransportTests(unittest.TestCase):
    def setUp(self):
        # never let a test read a real Kaggle credential file while exercising redact()
        empty = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, empty, True)
        patcher = patch.dict(os.environ, {'KAGGLE_CONFIG_DIR': empty, 'KAGGLE_KEY': '', 'KAGGLE_API_TOKEN': ''})
        patcher.start()
        self.addCleanup(patcher.stop)

    def cli(self, rc=0, out=''):
        calls = []

        def run(cmd, **kw):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, rc, out, '')
        return calls, ex.KaggleTransport(run=run, metadata_reader=lambda s: {})

    def test_create_is_private_and_version_never_deletes(self):
        calls, t = self.cli(out='Your private Dataset is being created. Starting upload for file FAILED.txt')
        t.create('o/s', '/stage')
        t.new_version('o/s', '/stage', 'note')
        for cmd in calls:
            self.assertIn('--keep-tabular', cmd)
            self.assertNotIn('--public', cmd)
            self.assertNotIn('-u', cmd)
            self.assertNotIn('-d', cmd)
            self.assertNotIn('--delete-old-versions', cmd)
        self.assertEqual(calls[0][1:3], ['datasets', 'create'])
        self.assertEqual(calls[1][1:3], ['datasets', 'version'])

    def test_cli_failures_raise(self):
        for rc, out in ((1, 'boom'), (0, 'Error: dataset already exists'), (0, 'The requested title is already in use')):
            with self.subTest(rc=rc, out=out):
                _, t = self.cli(rc, out)
                with self.assertRaises(ex.TransportError):
                    t.create('o/s', '/stage')

    def test_status_and_download_commands(self):
        calls, t = self.cli(out='noise\n{"status": "ready", "current_version_number": 4}')
        self.assertEqual(t.status('o/s'), ('ready', 4))
        calls.clear()
        with tempfile.TemporaryDirectory() as d:
            t.download('o/s', 4, d)
        self.assertEqual(calls[0][1:4], ['datasets', 'download', 'o/s/4'])
        self.assertIn('--unzip', calls[0])
        _, bad = self.cli(out='not json')
        with self.assertRaises(ex.TransportError):
            bad.status('o/s')
        _, failing = self.cli(rc=1, out='nope')
        with self.assertRaises(ex.TransportError):
            failing.status('o/s')

    def test_dataset_state_mapping_matches_runner_rule(self):
        nested = {'info': {'ownerUser': 'o', 'datasetSlug': 's', 'isPrivate': True}}
        legacy = {'id': 'o/s', 'isPrivate': True}
        self.assertEqual(ex.parse_metadata(nested, 'o/s'), {'identity': 'o/s', 'private': True})
        self.assertEqual(ex.parse_metadata(legacy, 'o/s'), {'identity': 'o/s', 'private': True})
        self.assertEqual(ex.parse_metadata({'info': {'ownerUser': 'o', 'datasetSlug': 's'}}, 'o/s')['private'], None)
        with self.assertRaises(ex.TransportError):
            ex.parse_metadata([], 'o/s')
        for reader, expect in ((lambda s: nested, {'identity': 'o/s', 'private': True}),
                               (lambda s: (_ for _ in ()).throw(Exception('404 Client Error: Not Found')), None)):
            t = ex.KaggleTransport(metadata_reader=reader)
            self.assertEqual(t.dataset_state('o/s'), expect)
        forbidden = 'HTTPError: 403 Client Error: Forbidden for url: https://api.kaggle.com/v1/x'
        t = ex.KaggleTransport(metadata_reader=lambda s: (_ for _ in ()).throw(Exception(forbidden)))
        with self.assertRaises(ex.MetadataInaccessible):   # unknown, never None (absent)
            t.dataset_state('o/s')
        self.assertTrue(issubclass(ex.MetadataInaccessible, ex.TransportError))
        t = ex.KaggleTransport(metadata_reader=lambda s: (_ for _ in ()).throw(Exception('500 Internal Server Error')))
        with self.assertRaises(ex.TransportError) as caught:
            t.dataset_state('o/s')
        self.assertNotIsInstance(caught.exception, ex.MetadataInaccessible)

    def test_redact_scrubs_token_from_cli_output(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / 'kaggle.json').write_text(json.dumps({'username': 'u', 'key': 'SYNTHETICSECRET123'}))
            with patch.dict(os.environ, {'KAGGLE_CONFIG_DIR': d}):
                self.assertEqual(ex.redact('x SYNTHETICSECRET123 y'), 'x *** y')


class HelperTests(unittest.TestCase):
    def test_load_runner_is_pinned(self):
        self.assertTrue(callable(ex.load_runner(RUNNER, RUNNER_SHA).primary_complete))
        with self.assertRaises(ex.Refused):
            ex.load_runner(RUNNER, '0' * 64)
        with tempfile.TemporaryDirectory() as d:
            link = Path(d) / 'runner.py'
            link.symlink_to(RUNNER)
            with self.assertRaises(ex.Refused):
                ex.load_runner(link, RUNNER_SHA)

    def test_copy_hashed_refuses_symlink_and_oversize(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / 'a').write_bytes(b'abc')
            (d / 'link').symlink_to(d / 'a')
            with self.assertRaises(ex.Refused):
                ex.copy_hashed(d / 'link', d / 'out1')
            with self.assertRaises(ex.Refused):
                ex.copy_hashed(d / 'a', d / 'out2', limit=2)
            self.assertEqual(ex.copy_hashed(d / 'a', d / 'out3'), (sha(b'abc'), 3))

    def test_cli_requires_every_pin(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            ex.main(['--arm', ARM])
        self.assertEqual(caught.exception.code, 2)

    def test_cli_end_to_end_with_fake_transport(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        f = Fit(d)
        out = Path(d) / 'cli'
        out.mkdir()
        argv = ['--arm', ARM, '--workdir', str(f.work), '--config', str(f.config_path), '--config-sha256', f.config_sha,
                '--package-manifest-sha256', PKG, '--cache-content-sha256', CONTENT, '--runner', str(RUNNER),
                '--runner-sha256', RUNNER_SHA, '--dataset', FINAL,
                '--staging', str(out / 'staging'), '--verify-root', str(out / 'verify'), '--receipt', str(out / 'receipt.json'),
                '--source-revision', 'unversioned']
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = ex.main(argv, transport=FakeTransport(), runner=runner)
        self.assertEqual(code, 0)
        summary = json.loads(buffer.getvalue())
        self.assertEqual((summary['state'], summary['dataset_version']), ('verified', 1))
        self.assertEqual(json.loads((out / 'receipt.json').read_text())['state'], 'verified')


if __name__ == '__main__':
    unittest.main()
