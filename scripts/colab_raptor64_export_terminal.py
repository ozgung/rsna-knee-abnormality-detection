#!/usr/bin/env python3
"""Terminal artifact exporter for completed Raptor64 fold-0 Colab arms (runs ON the Colab VM after a fit has ended).

Port of scripts/colab_source96_export_terminal.py (unchanged) for the four Raptor64 arms (in-orig, in-plane, dr-orig,
dr-plane). The persister (colab_tracka_persist.py) only backs up resume checkpoints. The final top-3 SWA checkpoint, the
K42 OOF predictions, run/meta JSON and controller receipts live only in the arm's VM workdir. This script copies them
to a fresh staging directory, writes a MANIFEST.json (sha256 + size per file), publishes the directory as a NEW private
Kaggle dataset, downloads the exact published version into a separate temp directory, compares every sha256 and writes
a local receipt with state `verified` or `failed`. It never writes to, renames or deletes anything in the workdir and
never touches the resume-backup datasets.

Modes
  complete (default): refuses unless the pinned Raptor64 runner's own `primary_complete()` accepts the workdir (twelve
      epochs, top-3 SWA, K42 export of the 870 fold-0 development studies, matching arm/split/cache), the pinned
      config/package/cache hashes match, the arm agrees across request/config/fingerprint, the controller receipt is
      PASS with a verified epoch-11 backup, the controller lock is free and no input is a symlink.
      Destination: OWNER/knee-raptor64-{ARM}-fold0-final-YYYYMMDD[-rN]. There is no reconciled-receipt path.
  --failed-receipts-only: for a fit that is NOT complete. Refuses when `primary_complete()` accepts (a complete fit must
      use the normal mode). Copies receipts only (no checkpoint, OOF or resume), writes a FAILED.txt sentinel and a
      manifest labelled FAILED. Destination: OWNER/knee-raptor64-{ARM}-fold0-failed-YYYYMMDD[-rN].

Destination rules: slugs are at most 50 characters. The dataset must be absent (created private: the Kaggle CLI
default, `--public` is never passed; `--keep-tabular` preserves CSV bytes) or, only with --new-version, an existing
dataset whose metadata identity equals the slug and isPrivate is exactly true. Metadata HTTP 403 is UNKNOWN, never
absence: it is refused unless --create-fresh names exactly the destination, which allows a create-only run (no version,
no overwrite; the published version must be 1 and an exact identity/privacy check is mandatory after the create). The
config's persistence dataset, any resume-backup-shaped slug knee-raptor64-{ARM}-fold0-YYYYMMDD and any slug not
matching the arm/mode pattern are refused.

Transport is injectable (see `KaggleTransport` for the real Kaggle CLI/SDK implementation; tests use a fake).
Secrets: the Kaggle token is read only by the Kaggle tools themselves (and by `redact`, to scrub their output).
"""
import argparse
import contextlib
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

VERSION = 'colab-raptor64-export-terminal-v1'
SCHEMA = 'raptor64-terminal-export-v1'
CONFIG_VERSION = 'colab-raptor64-run-v1'
PREFIX = 'raptor64'
ARMS = ('in-orig', 'in-plane', 'dr-orig', 'dr-plane', 'dr-full', 'in-mirror', 'in-distil', 'in-distil-w1', 'in-distil-flip', 'ax-crop', 'ax-full')
FULL_ARMS = frozenset({'dr-full'})  # all-eligible fit: no development predictions, fixed pre-registered SWA epochs
FULL_SPLIT_COUNTS = (4345, 4345, 0)
DEVELOPMENT = 870
EPOCHS = 12
FINAL_EPOCH = EPOCHS - 1
SPLIT_COUNTS = (4345, 3475, 870)
SLUG_MAX = 50
FINGERPRINT = PREFIX + '-run-fingerprint.json'
RUN_RECEIPT = PREFIX + '-run-receipt.json'
SHA_RE = re.compile(r'[0-9a-f]{64}')
DATASET_RE = re.compile(r'[a-z0-9_-]+/[a-z0-9-]{3,50}')
REVISION_RE = re.compile(r'(?:[0-9a-f]{7,40}|unversioned)')
CHUNK = 8 * 1024 * 1024
MAX_RECEIPT_BYTES = 8 * 1024 * 1024
MAX_LOG_BYTES = 256 * 1024 * 1024
DISK_MARGIN = 1 << 30
RESERVED_NAMES = frozenset({'MANIFEST.json', 'dataset-metadata.json'})


class Refused(Exception):
    """A pre-upload guard failed; nothing was uploaded."""


class TransportError(Exception):
    """The Kaggle transport failed or returned something unusable."""


class MetadataInaccessible(TransportError):
    """Dataset metadata could not be read (HTTP 403/forbidden). This is UNKNOWN, never evidence of absence."""


class VerificationFailed(Exception):
    def __init__(self, version, problems, kept=None):
        super().__init__('; '.join(problems[:5]))
        self.version, self.problems, self.kept = version, problems, kept


def utc_now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


# ----------------------------------------------------------------------------- file helpers
def _open_regular(path):
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as error:
        raise Refused(f'{Path(path).name}: not a readable regular file ({error.strerror})') from None
    st = os.fstat(fd)
    if not stat.S_ISREG(st.st_mode):
        os.close(fd)
        raise Refused(f'{Path(path).name}: not a regular file')
    return fd, st


def sha256_file(path):
    fd, _ = _open_regular(path)
    h = hashlib.sha256()
    with os.fdopen(fd, 'rb') as f:
        for block in iter(lambda: f.read(CHUNK), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path, limit=MAX_RECEIPT_BYTES):
    fd, st = _open_regular(path)
    with os.fdopen(fd, 'rb') as f:
        if st.st_size > limit:
            raise Refused(f'{Path(path).name}: larger than {limit} bytes')
        data = f.read()
    try:
        return json.loads(data)
    except ValueError:
        raise Refused(f'{Path(path).name}: not valid JSON') from None


def read_json_pinned(path, digest):
    if not SHA_RE.fullmatch(str(digest)):
        raise Refused('A 64-hex sha256 pin is required')
    if sha256_file(path) != digest:
        raise Refused(f'{Path(path).name}: sha256 differs from the pin')
    return read_json(path)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def copy_hashed(src, dst, limit=None):
    """Copy src to the NEW file dst; return (sha256, size). Refuses a symlink or a source that changes while copying."""
    fd, st = _open_regular(src)
    if limit is not None and st.st_size > limit:
        os.close(fd)
        raise Refused(f'{Path(src).name}: {st.st_size} bytes exceeds the {limit} byte limit')
    h, n = hashlib.sha256(), 0
    with os.fdopen(fd, 'rb') as f, open(dst, 'xb') as out:
        for block in iter(lambda: f.read(CHUNK), b''):
            h.update(block)
            out.write(block)
            n += len(block)
        out.flush()
        os.fsync(out.fileno())
    after = os.stat(src)
    if n != st.st_size or (after.st_size, after.st_mtime_ns) != (st.st_size, st.st_mtime_ns):
        raise Refused(f'{Path(src).name}: changed while being copied')
    digest = h.hexdigest()
    if sha256_file(dst) != digest:
        raise Refused(f'{Path(src).name}: staged copy differs from the source bytes')
    return digest, n


def snapshot(root):
    """Metadata of every entry under root (no file reads, symlinks not followed)."""
    state = {}
    for base, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            p = os.path.join(base, name)
            st = os.lstat(p)
            state[os.path.relpath(p, root)] = (st.st_mode, st.st_size, st.st_mtime_ns, st.st_ino)
    return state


def real(path):
    return Path(os.path.realpath(path))


def inside(child, parent):
    c, p = real(child), real(parent)
    return c == p or p in c.parents


def atomic_json(path, value):
    p = Path(path)
    tmp = p.with_name(p.name + '.tmp')
    if p.is_symlink() or tmp.is_symlink():
        raise ValueError('Unsafe receipt path')
    with open(tmp, 'w') as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def load_runner(path, digest):
    """Import the pinned controller source so that its own primary_complete() defines completeness."""
    p = Path(path)
    if not SHA_RE.fullmatch(str(digest)) or p.is_symlink() or sha256_file(p) != digest:
        raise Refused('Pinned runner source differs')
    spec = importlib.util.spec_from_file_location('pinned_raptor64_runner', p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, 'primary_complete', None)):
        raise Refused('Pinned runner has no primary_complete()')
    return module


@contextlib.contextmanager
def controller_quiesced(lock_path):
    """Hold the fold's controller lock for the whole export; refuse when the controller still owns it."""
    p = Path(lock_path)
    if p.is_symlink() or not p.is_file():
        raise Refused('Controller lock file absent or unsafe; cannot confirm the controller ended')
    fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise Refused('Controller lock is held: the training controller has not ended') from None
        yield
    finally:
        os.close(fd)  # closing releases the flock


# ----------------------------------------------------------------------------- request / names
@dataclass
class Request:
    fold: int
    workdir: Path
    config: Path
    config_sha256: str
    package_manifest_sha256: str
    cache_content_sha256: str
    runner: Path
    runner_sha256: str
    dataset: str
    staging: Path
    verify_root: Path
    receipt: Path
    source_revision: str
    arm: str = None
    mode: str = 'complete'
    new_version: bool = False
    create_fresh: str = None
    accept_reconciled_receipt: bool = False
    include_logs: bool = False
    controller_log: Path = None
    controller_sha256: str = None
    ready_timeout: float = 900.0
    poll: float = 15.0
    exporter_path: Path = Path(__file__)

    def __post_init__(self):
        for name in ('workdir', 'config', 'runner', 'staging', 'verify_root', 'receipt', 'exporter_path'):
            setattr(self, name, Path(getattr(self, name)))
        if self.controller_log is not None:
            self.controller_log = Path(self.controller_log)


def check_request_shape(req):
    if type(req.fold) is not int or req.fold != 0:
        raise Refused('Raptor64 exports are fold 0 only')
    if req.arm not in ARMS:
        raise Refused(f'--arm must be one of {", ".join(ARMS)}')
    if req.accept_reconciled_receipt:
        raise Refused('--accept-reconciled-receipt is a source96-only path; Raptor64 requires a PASS controller receipt')
    if req.mode not in ('complete', 'failed'):
        raise Refused('Unknown mode')
    if not REVISION_RE.fullmatch(req.source_revision or ''):
        raise Refused('--source-revision must be a 7-40 hex git revision or the word unversioned')
    for name in ('config_sha256', 'package_manifest_sha256', 'cache_content_sha256', 'runner_sha256'):
        if not SHA_RE.fullmatch(str(getattr(req, name))):
            raise Refused(f'{name} must be a 64-hex pin')
    if req.controller_sha256 is not None and not SHA_RE.fullmatch(req.controller_sha256):
        raise Refused('controller_sha256 must be a 64-hex pin')
    if req.create_fresh is not None and (req.new_version or req.create_fresh != req.dataset):
        raise Refused('--create-fresh must name exactly the destination dataset and cannot be combined with --new-version')
    for name in ('workdir', 'config', 'runner', 'staging', 'verify_root', 'receipt'):
        if not getattr(req, name).is_absolute():
            raise Refused(f'{name} must be an absolute path')


def check_paths(req):
    work = req.workdir
    if work.is_symlink() or not work.is_dir():
        raise Refused('Workdir must be an existing real directory')
    if inside(req.staging, req.verify_root) or inside(req.verify_root, req.staging):
        raise Refused('staging and verify_root must be separate')
    for name in ('staging', 'verify_root', 'receipt'):
        p = getattr(req, name)
        if inside(p, work):
            raise Refused(f'{name} must be outside the workdir')
        if os.path.lexists(p):
            raise Refused(f'{name} already exists; a fresh path is required')
        if not p.parent.is_dir():
            raise Refused(f'{name} parent directory must exist')
    if inside(req.receipt, req.staging) or inside(req.receipt, req.verify_root):
        raise Refused('receipt must be outside staging and verify_root')


def check_dataset_name(req, config):
    dataset = req.dataset
    if not DATASET_RE.fullmatch(dataset or ''):
        raise Refused('Dataset must look like owner/lowercase-slug')
    backup = config['persistence']['dataset']
    owner_, slug_ = dataset.split('/', 1)
    if dataset == backup or re.fullmatch(rf'knee-{PREFIX}-(?:{"|".join(ARMS)})-fold\d-\d{{8}}(?:-r\d{{1,2}})?', slug_):
        raise Refused('Refusing to write to a resume-backup dataset')
    owner, slug = dataset.split('/', 1)
    if owner != backup.split('/', 1)[0]:
        raise Refused('Destination owner must equal the backup dataset owner')
    word = 'final' if req.mode == 'complete' else 'failed'
    if len(slug) > SLUG_MAX or not re.fullmatch(rf'knee-{PREFIX}-{re.escape(req.arm)}-fold0-{word}-\d{{8}}(?:-r\d{{1,2}})?', slug):
        raise Refused(f'Destination slug must match knee-{PREFIX}-{req.arm}-fold0-{word}-YYYYMMDD[-rN] (at most {SLUG_MAX} chars)')


def check_privacy(state, dataset):
    """Same rule as the runner's private_dataset(): exact identity and isPrivate true."""
    if not isinstance(state, dict) or state.get('identity') != dataset or state.get('private') is not True:
        raise Refused('Destination dataset identity/privacy is unverified or public')


# ----------------------------------------------------------------------------- preflight
def load_config(req):
    c = read_json_pinned(req.config, req.config_sha256)
    if not isinstance(c, dict) or c.get('version') != CONFIG_VERSION:
        raise Refused('Run config version differs')
    if c.get('fold', 0) != req.fold or Path(str(c.get('workdir'))) != req.workdir:
        raise Refused('Run config fold/workdir differ from the request')
    if c.get('arm') != req.arm:
        raise Refused('Run config arm differs from the request')
    if c.get('package_manifest_sha256') != req.package_manifest_sha256 or c.get('content_sha256') != req.cache_content_sha256:
        raise Refused('Run config package/cache hashes differ from the expected pins')
    for key in ('stage_root', 'stage_receipt_sha256', 'manifest_sha256', 'initializer_descriptor',
                'initializer_descriptor_sha256'):
        if not isinstance(c.get(key), str):
            raise Refused(f'Run config lacks {key}')
    persistence = c.get('persistence') or {}
    if persistence.get('private') is not True or not DATASET_RE.fullmatch(str(persistence.get('dataset'))):
        raise Refused('Run config has no private persistence dataset')
    if not all(isinstance(v, str) for v in (c.get('controller_lock'), persistence.get('staging'), persistence.get('status_file'))):
        raise Refused('Run config lacks controller_lock/persistence staging/status_file')
    if inside(req.staging, persistence['staging']) or inside(persistence['staging'], req.staging):
        raise Refused('Export staging must be separate from the persister staging')
    return c


def check_fingerprint(req, config):
    fp = read_json(req.workdir / FINGERPRINT)
    expected = {'fold': req.fold, 'arm': req.arm, 'package_manifest_sha256': req.package_manifest_sha256,
                'cache_content_sha256': req.cache_content_sha256, 'config_sha256_canonical': canonical_sha(config),
                'version': CONFIG_VERSION}
    for key, value in expected.items():
        if fp.get(key) != value:
            raise Refused(f'Run fingerprint {key} differs from the pinned config')
    if req.controller_sha256 is not None and fp.get('controller_sha256') != req.controller_sha256:
        raise Refused('Run fingerprint controller_sha256 differs from the pin')
    split = fp.get('experiment_split')
    policy = fp.get('policy') or {}
    full = req.arm in FULL_ARMS
    if (not isinstance(split, dict) or split.get('fold') != (None if full else 0) or policy.get('experiment_split_sha256') != canonical_sha(split)
            or (split.get('eligible'), split.get('training'), split.get('development')) != (FULL_SPLIT_COUNTS if full else SPLIT_COUNTS)
            or policy.get('arm') != req.arm or (policy.get('mode') == 'full') != full):
        raise Refused('Experiment split / arm policy identity differs')
    return fp


def check_complete(req, runner, fp):
    try:
        record = runner.primary_complete(str(req.workdir), 0, DEVELOPMENT, fp.get('experiment_split'),
                                         req.cache_content_sha256, req.arm)
    except Exception as error:
        raise Refused(f'primary_complete rejected the workdir: {type(error).__name__}: {error}') from None
    sel = record.get('selected_epochs')
    if not (isinstance(sel, list) and len(sel) == 3 and len(set(sel)) == 3
            and all(type(e) is int and 0 <= e <= FINAL_EPOCH for e in sel)):
        raise Refused(f'selected_epochs must be three distinct epochs in 0..{FINAL_EPOCH}')
    return record


def check_terminal_receipts(req, runner_sha, fp, record):
    """Return (receipt_mode, None); only a PASS controller receipt with a verified epoch-11 backup is accepted."""
    r = read_json(req.workdir / RUN_RECEIPT)
    for key in ('fold', 'arm', 'package_manifest_sha256', 'cache_content_sha256', 'config_sha256_canonical', 'controller_sha256'):
        if r.get(key) != fp.get(key):
            raise Refused(f'Controller receipt {key} differs from the run fingerprint')
    if r.get('status') != 'PASS':
        raise Refused('Controller receipt is not PASS; use --failed-receipts-only for an incomplete fit')
    if r.get('checkpoint_sha256') != record['checkpoint_sha256'] or (r.get('backup') or {}).get('epoch') != FINAL_EPOCH:
        raise Refused(f'PASS controller receipt does not match the checkpoint/epoch{FINAL_EPOCH} backup')
    return 'controller_pass', None


@dataclass(frozen=True)
class Item:
    name: str
    src: Path
    role: str
    origin: str
    required: bool
    limit: int = MAX_RECEIPT_BYTES


def plan_items(req, config, reconciliation):
    k, w = req.fold, req.workdir
    items = []

    def add(name, src, role, origin, required, limit=MAX_RECEIPT_BYTES):
        items.append(Item(name, Path(src), role, origin, required, limit))

    if req.mode == 'complete':
        big = 1 << 40
        add(f'raptor_ft_fold{k}_swa.pt', w / f'raptor_ft_fold{k}_swa.pt', 'final_swa_checkpoint', 'workdir', True, big)
        if req.arm not in FULL_ARMS:
            add(f'oof_fold{k}.csv', w / f'oof_fold{k}.csv', 'oof_predictions', 'workdir', True, big)
        add(f'run_fold{k}.json', w / f'run_fold{k}.json', 'run_record', 'workdir', True)
        add(f'raptor_ft_fold{k}.json', w / f'raptor_ft_fold{k}.json', 'training_meta', 'workdir', True)
        add(FINGERPRINT, w / FINGERPRINT, 'controller_fingerprint', 'workdir', True)
        add(RUN_RECEIPT, w / RUN_RECEIPT, 'controller_receipt', 'workdir', True)
        add(f'oof_fold{k}_mirror_tta.csv', w / f'oof_fold{k}_mirror_tta.csv', 'mirror_tta_oof_predictions', 'workdir', False, big)
        add(f'raptor_oof_fold{k}_swa_fold{k}.npz', w / f'raptor_oof_fold{k}_swa_fold{k}.npz', 'k16_in_training_swa_oof', 'workdir',
            False, big)
    else:
        add(FINGERPRINT, w / FINGERPRINT, 'controller_fingerprint', 'workdir', False)
        add(RUN_RECEIPT, w / RUN_RECEIPT, 'controller_receipt', 'workdir', False)
        add(f'raptor_ft_fold{k}.json', w / f'raptor_ft_fold{k}.json', 'training_meta', 'workdir', False)
        add(f'run_fold{k}.json', w / f'run_fold{k}.json', 'run_record', 'workdir', False)
    add(f'train_meta_fold{k}.json', w / f'train_meta_fold{k}.json', 'stage_train_meta', 'workdir', False)
    add('initial-state-record.json', w / 'initial-state-record.json', 'initial_state_record', 'workdir', False)
    add('run-config.json', req.config, 'run_config', 'external', True)
    add('staging-receipt.json', Path(config['stage_root']) / 'staging-receipt.json', 'cache_stage_receipt', 'external',
        req.mode == 'complete')
    add('initializer-descriptor.json', config['initializer_descriptor'], 'initializer_descriptor', 'external',
        req.mode == 'complete')
    add(f'persist-status-fold{k}.json', config['persistence']['status_file'], 'persister_status', 'external', False)
    if req.include_logs:
        for name in (f'log_fold{k}.txt', f'log_export_fold{k}.txt', f'{PREFIX}-training.log', f'{PREFIX}-backup.log',
                     f'{PREFIX}-runtime-stdout.log', f'{PREFIX}-runtime-stderr.log'):
            add(name, w / name, 'private_log', 'workdir', False, MAX_LOG_BYTES)
        if req.controller_log is not None:
            add(f'controller-fold{k}.log', req.controller_log, 'private_log', 'external', False, MAX_LOG_BYTES)
    seen = set()
    for item in items:
        if item.name in seen or '/' in item.name or item.name in RESERVED_NAMES:
            raise Refused(f'Invalid or duplicate staged name {item.name}')
        seen.add(item.name)
    return items


def check_external_pins(req, config):
    """Cache stage receipt and initializer descriptor are pinned by the run config; re-check them when present."""
    stage = Path(config['stage_root']) / 'staging-receipt.json'
    if stage.exists() or req.mode == 'complete':
        r = read_json_pinned(stage, config.get('stage_receipt_sha256'))
        if r.get('status') != 'PASS' or r.get('manifest_sha256') != config.get('manifest_sha256') \
                or r.get('content_sha256') != config.get('content_sha256'):
            raise Refused('Cache staging receipt does not match the run config')
    init = Path(str(config.get('initializer_descriptor')))
    if init.exists() or req.mode == 'complete':
        read_json_pinned(init, config.get('initializer_descriptor_sha256'))


def preflight(req, runner, runner_sha):
    check_request_shape(req)
    check_paths(req)
    config = load_config(req)
    check_dataset_name(req, config)
    check_external_pins(req, config)
    plan = {'config': config, 'record': None, 'receipt_mode': None, 'reconciliation': None, 'fingerprint': None}
    if req.mode == 'complete':
        fp = check_fingerprint(req, config)
        record = check_complete(req, runner, fp)
        receipt_mode, reconciliation = check_terminal_receipts(req, runner_sha, fp, record)
        plan.update(fingerprint=fp, record=record, receipt_mode=receipt_mode, reconciliation=reconciliation)
    else:
        try:
            fp = read_json(req.workdir / FINGERPRINT)
        except Refused:
            fp = {}
        try:
            record = runner.primary_complete(str(req.workdir), 0, DEVELOPMENT, fp.get('experiment_split'),
                                             req.cache_content_sha256, req.arm)
        except Exception:
            record = None
        if record is not None:
            raise Refused('The fit is complete (primary_complete accepts it): use the normal export, not FAILED mode')
        plan.update(fingerprint=fp, receipt_mode='failed')
    plan['items'] = plan_items(req, config, plan['reconciliation'])
    return plan


# ----------------------------------------------------------------------------- staging
def copy_items(req, plan, free_bytes):
    present, absent = [], []
    for item in plan['items']:
        if item.src.is_symlink():
            raise Refused(f'{item.name}: source is a symlink')
        if item.src.exists():
            present.append(item)
        elif item.required:
            raise Refused(f'{item.name}: required source is missing')
        else:
            absent.append(item.name)
    need = 2 * sum(os.stat(i.src).st_size for i in present) + DISK_MARGIN
    free = free_bytes(req.staging.parent)
    if free < need:
        raise Refused(f'Insufficient free disk: {free} bytes free, {need} needed for staging plus verification download')
    req.staging.mkdir(mode=0o700)
    files = {}
    for item in present:
        digest, size = copy_hashed(item.src, req.staging / item.name, item.limit)
        files[item.name] = {'size': size, 'sha256': digest, 'role': item.role, 'origin': item.origin,
                            'source': str(item.src)}
    return files, sorted(absent)


def build_manifest(req, plan, files, absent, runner_sha, now):
    complete = req.mode == 'complete'
    manifest = {
        'schema': SCHEMA, 'exporter_version': VERSION, 'label': 'FINAL' if complete else 'FAILED', 'complete': complete,
        'created_utc': now(), 'fold': req.fold, 'arm': req.arm, 'dataset': req.dataset, 'workdir': str(req.workdir),
        'source_revision': req.source_revision, 'exporter_sha256': sha256_file(req.exporter_path),
        'runner_sha256': runner_sha, 'receipt_mode': plan['receipt_mode'],
        'pins': {'config_sha256': req.config_sha256, 'package_manifest_sha256': req.package_manifest_sha256,
                 'cache_content_sha256': req.cache_content_sha256, 'controller_sha256': req.controller_sha256,
                 'stage_receipt_sha256': plan['config'].get('stage_receipt_sha256'),
                 'initializer_descriptor_sha256': plan['config'].get('initializer_descriptor_sha256'),
                 'persister_sha256': plan['config'].get('persister_sha256')},
        'resume_backup_dataset': plan['config']['persistence']['dataset'],
        'absent_optional': absent, 'includes_private_logs': bool(req.include_logs),
        'files': dict(sorted(files.items())),
    }
    if complete:
        record = plan['record']
        ckpt = record['checkpoint_file']
        staged = files[ckpt]['sha256']
        sources = {'run_record': record['checkpoint_sha256'], 'staged_copy': staged}
        receipt = read_json(req.staging / RUN_RECEIPT)
        sources['controller_receipt'] = receipt.get('checkpoint_sha256')
        if len(set(sources.values())) != 1:
            raise Refused('checkpoint_sha256 differs between sources')
        manifest.update(
            checkpoint_file=ckpt, checkpoint_sha256=staged, checkpoint_sha256_sources=sources, selected_epochs=record['selected_epochs'],
            epochs_completed=record['epochs_completed'], k_eval=(record.get('export') or {}).get('k_eval'), n_val=(record.get('export') or {}).get('n_val'),
            checkpoint_kind=record['checkpoint_kind'], comparison_endpoint=record['comparison_endpoint'])
    else:
        receipt = {}
        try:
            receipt = read_json(req.staging / RUN_RECEIPT)
        except Refused:
            pass
        manifest['failure_summary'] = {k: receipt.get(k) for k in (
            'status', 'failure', 'training_returncode', 'persister_returncode', 'interrupted_signal', 'resume_preserved')}
        manifest['workdir_listing'] = sorted(
            (p.name, p.stat().st_size) for p in req.workdir.iterdir() if p.is_file() and not p.is_symlink())
        manifest['not_exported'] = 'checkpoints, OOF predictions and resume state (receipts only)'
    return manifest


def write_staging_metadata(req, manifest):
    if req.mode == 'failed':
        text = (f'FAILED: raptor64 arm {req.arm} fold {req.fold} did not complete a verified primary export. Receipts only; no checkpoint '
                f'or predictions are included. See MANIFEST.json failure_summary. Resume state is in '
                f'{manifest["resume_backup_dataset"]}.\n')
        (req.staging / 'FAILED.txt').write_text(text)
        data = text.encode()
        manifest['files']['FAILED.txt'] = {'size': len(data), 'sha256': hashlib.sha256(data).hexdigest(), 'role': 'label',
                                           'origin': 'generated', 'source': None}
        manifest['files'] = dict(sorted(manifest['files'].items()))
    (req.staging / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    slug = req.dataset.split('/', 1)[1]
    title = slug.replace('-', ' ').replace('_', ' ')[:50]
    (req.staging / 'dataset-metadata.json').write_text(
        json.dumps({'title': title, 'id': req.dataset, 'licenses': [{'name': 'CC0-1.0'}]}, indent=1) + '\n')
    return sha256_file(req.staging / 'MANIFEST.json')


# ----------------------------------------------------------------------------- upload / verify
def compare_download(directory, expected, manifest_sha):
    """Return (problems, bound). bound: the downloaded MANIFEST.json is byte-identical to ours (so it is OUR version)."""
    found, problems = {}, []
    for p in sorted(Path(directory).rglob('*')):
        rel = p.relative_to(directory).as_posix()
        if p.is_symlink():
            problems.append(f'{rel}: symlink in download')
        elif p.is_file():
            found[rel] = p
    manifest = found.get('MANIFEST.json')
    bound = manifest is not None and sha256_file(manifest) == manifest_sha
    if manifest is None:
        problems.append('MANIFEST.json: missing')
    elif not bound:
        problems.append('MANIFEST.json: sha256 differs')
    for name, meta in expected.items():
        p = found.get(name)
        if p is None:
            problems.append(f'{name}: missing')
            continue
        if p.stat().st_size != meta['size']:
            problems.append(f'{name}: size {p.stat().st_size} != {meta["size"]}')
        if sha256_file(p) != meta['sha256']:
            problems.append(f'{name}: sha256 differs')
    for rel in sorted(set(found) - set(expected) - RESERVED_NAMES):
        problems.append(f'{rel}: unexpected file')
    return problems, bound


def await_private(req, transport, sleep, clock):
    """Identity/privacy check after an upload; a just-created dataset may be invisible for a short while."""
    deadline = clock() + req.ready_timeout
    while True:
        try:
            state = transport.dataset_state(req.dataset)
        except MetadataInaccessible:
            state = None
        if state is not None or clock() + req.poll > deadline:
            check_privacy(state, req.dataset)
            return
        sleep(req.poll)


def verify_published(req, transport, files, manifest_sha, min_version, sleep, clock, exact_version=None):
    deadline = clock() + req.ready_timeout
    last = 'dataset not ready'
    while True:
        try:
            status, version = transport.status(req.dataset)
            if status == 'ready' and type(version) is int and version >= min_version:
                if exact_version is not None and version != exact_version:
                    raise VerificationFailed(version, [f'published version {version} != expected {exact_version}: '
                                                       'the destination was not a fresh dataset'])
                dest = Path(tempfile.mkdtemp(prefix=f'v{version}-', dir=req.verify_root))
                transport.download(req.dataset, version, dest)
                problems, bound = compare_download(dest, files, manifest_sha)
                if not problems:
                    shutil.rmtree(dest)
                    return version
                if bound:
                    raise VerificationFailed(version, problems, kept=str(dest))
                last = '; '.join(problems[:3])
                shutil.rmtree(dest)  # possibly a lagging/other version: retry within the timeout
            else:
                last = f'status={status} version={version}'
        except TransportError as error:  # status/download can fail transiently right after an upload
            last = f'transport: {error}'
        if clock() + req.poll > deadline:
            raise VerificationFailed(None, [f'not verified within {req.ready_timeout:.0f}s ({last})'])
        sleep(req.poll)


# ----------------------------------------------------------------------------- main flow
def run(req, *, transport, runner=None, sleep=time.sleep, clock=time.monotonic, now=utc_now, free_bytes=None):
    """Return the receipt dict (also written to req.receipt). state is `verified` or `failed`."""
    free_bytes = free_bytes or (lambda p: shutil.disk_usage(p).free)
    interrupt = None
    receipt = {'schema': SCHEMA + '-receipt', 'exporter_version': VERSION, 'state': 'failed', 'phase': 'preflight',
               'upload_attempted': False, 'mode': req.mode, 'fold': req.fold, 'arm': req.arm, 'dataset': req.dataset,
               'source_revision': req.source_revision, 'started_utc': now(), 'workdir': str(req.workdir),
               'staging': str(req.staging), 'problems': []}
    try:
        check_request_shape(req)
        if os.path.lexists(req.receipt):
            raise Refused('receipt path already exists')
        if req.receipt.parent.is_dir() and inside(req.receipt, req.workdir):
            raise Refused('receipt must be outside the workdir')
        runner_sha = req.runner_sha256
        if runner is None:
            runner = load_runner(req.runner, req.runner_sha256)
        t0 = clock()
        plan = preflight(req, runner, runner_sha)
        lock = controller_quiesced(plan['config']['controller_lock'])
        with lock:
            before = snapshot(req.workdir)
            receipt['phase'] = 'destination'
            try:
                state = transport.dataset_state(req.dataset)
                receipt['existence'] = 'absent' if state is None else 'exists'
            except MetadataInaccessible as error:
                if req.create_fresh != req.dataset:
                    raise TransportError(f'{error}; existence is UNKNOWN (not treated as absent). A create-only run needs '
                                         '--create-fresh with the exact slug') from None
                state = None
                receipt['existence'] = 'unknown_metadata_inaccessible'
            if req.create_fresh is not None and state is not None:
                raise Refused('--create-fresh given but the destination already exists')
            if state is None:
                action, prior = 'create', 0
            else:
                check_privacy(state, req.dataset)
                if not req.new_version:
                    raise Refused('Destination already exists; pass --new-version to publish a new version')
                prior = transport.latest_version(req.dataset)
                if type(prior) is not int or prior < 1:
                    raise Refused('Existing destination has no positive latest version')
                action = 'version'
            receipt.update(action=action, prior_version=prior or None)
            receipt['phase'] = 'staging'
            files, absent = copy_items(req, plan, free_bytes)
            manifest = build_manifest(req, plan, files, absent, runner_sha, now)
            manifest_sha = write_staging_metadata(req, manifest)
            files = manifest['files']
            receipt.update(manifest_sha256=manifest_sha, files=files, absent_optional=absent,
                           checkpoint_sha256=manifest.get('checkpoint_sha256'), selected_epochs=manifest.get('selected_epochs'),
                           payload_bytes=sum(m['size'] for m in files.values()), staging_seconds=round(clock() - t0, 3))
            receipt['phase'] = 'upload'
            receipt['upload_attempted'] = True
            t1 = clock()
            note = f'raptor64 {req.arm} fold{req.fold} {manifest["label"]} export {manifest_sha[:12]}'
            if action == 'create':
                transport.create(req.dataset, req.staging)
            else:
                transport.new_version(req.dataset, req.staging, note)
            receipt['upload_seconds'] = round(clock() - t1, 3)
            receipt['phase'] = 'verify'
            await_private(req, transport, sleep, clock)
            req.verify_root.mkdir(mode=0o700)
            t2 = clock()
            receipt['dataset_version'] = verify_published(req, transport, files, manifest_sha, prior + 1, sleep, clock,
                                                          exact_version=1 if action == 'create' else None)
            receipt['verify_seconds'] = round(clock() - t2, 3)
            await_private(req, transport, sleep, clock)
            if snapshot(req.workdir) != before:
                raise VerificationFailed(receipt['dataset_version'], ['workdir changed during the export'])
            receipt['workdir_unchanged'] = True
            receipt.update(state='verified', phase='done')
    except Refused as error:
        receipt['problems'].append(str(error))
    except VerificationFailed as error:
        receipt['problems'].extend(error.problems)
        receipt['dataset_version'] = error.version
        if error.kept:
            receipt['kept_download'] = error.kept
    except TransportError as error:
        receipt['problems'].append(f'transport: {error}')
    except Exception as error:  # a receipt must exist whatever happens; never claim verified
        receipt['problems'].append(f'unexpected {type(error).__name__}: {error}')
    except KeyboardInterrupt as error:  # record the interruption, then re-raise after the receipt is written
        receipt['problems'].append('interrupted by operator (KeyboardInterrupt); the published state is unverified')
        interrupt = error
    receipt['finished_utc'] = now()
    if receipt['state'] != 'verified':
        receipt.setdefault('workdir_unchanged', None)
    if os.path.lexists(req.receipt) or not req.receipt.parent.is_dir() or inside(req.receipt, req.workdir):
        raise SystemExit('Cannot write the local receipt safely: ' + json.dumps(receipt, sort_keys=True))
    atomic_json(req.receipt, receipt)
    if interrupt is not None:
        raise interrupt
    return receipt


# ----------------------------------------------------------------------------- real transport
def redact(text):
    secrets = set()
    try:
        with open(os.path.join(os.environ.get('KAGGLE_CONFIG_DIR', os.path.expanduser('~/.kaggle')), 'kaggle.json')) as f:
            j = json.load(f)
        secrets.update(str(v) for k, v in j.items() if k in ('key', 'token') and v)
    except Exception:
        pass
    secrets.update(v for k, v in os.environ.items() if k in ('KAGGLE_KEY', 'KAGGLE_API_TOKEN') and v)
    for s in secrets:
        text = text.replace(s, '***')
    return text


def parse_metadata(m, slug):
    """Runner private_dataset() rule: nested info.ownerUser/datasetSlug or legacy id, and info.isPrivate."""
    if not isinstance(m, dict):
        raise TransportError('Dataset metadata is not an object')
    info = m.get('info', m)
    if not isinstance(info, dict):
        raise TransportError('Dataset metadata info is not an object')
    identity = (str(info.get('ownerUser')) + '/' + str(info.get('datasetSlug'))) if 'info' in m else info.get('id')
    return {'identity': identity, 'private': info.get('isPrivate')}


def sdk_metadata(slug):
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    with tempfile.TemporaryDirectory() as t:
        return json.loads(Path(api.dataset_metadata(slug, t)).read_bytes())


class KaggleTransport:
    """Kaggle CLI (create/version/status/download) plus the SDK metadata call used by the runner's private_dataset()."""
    NOT_FOUND = re.compile(r'\b404\b|not\s*found|does not exist', re.I)
    FORBIDDEN = re.compile(r'\b403\b|forbidden', re.I)
    BAD_OUTPUT = re.compile(r'^\W*(?:error|invalid|forbidden|failed|cannot|could not)\b|already (?:in use|exists)', re.I | re.M)

    def __init__(self, kaggle='kaggle', env=None, run=subprocess.run, metadata_reader=sdk_metadata):
        self.kaggle, self.env, self._run, self._metadata = kaggle, env, run, metadata_reader

    def _cli(self, args, timeout):
        try:
            p = self._run([self.kaggle] + args, capture_output=True, text=True, timeout=timeout, env=self.env)
        except subprocess.TimeoutExpired:
            raise TransportError(f'kaggle {args[:2]} timed out after {timeout}s') from None
        except FileNotFoundError:
            raise TransportError('kaggle CLI not found on PATH') from None
        return p.returncode, redact(((p.stdout or '') + (p.stderr or '')).strip())

    def dataset_state(self, slug):
        try:
            meta = self._metadata(slug)
        except TransportError:
            raise
        except Exception as error:
            text = redact(f'{type(error).__name__}: {error}')
            if self.NOT_FOUND.search(text):
                return None
            if self.FORBIDDEN.search(text):
                raise MetadataInaccessible('metadata lookup failed: ' + text[:200]) from None
            raise TransportError('metadata lookup failed: ' + text[:200]) from None
        return parse_metadata(meta, slug)

    def status(self, slug):
        rc, out = self._cli(['datasets', 'status', slug, '--format', 'json(status,current_version_number)'], 120)
        if rc != 0:
            raise TransportError('status failed: ' + out[:200])
        for line in reversed(out.splitlines()):
            if line.strip().startswith('{'):
                try:
                    j = json.loads(line)
                except ValueError:
                    break
                version = j.get('current_version_number')
                if isinstance(version, str) and version.isdigit():
                    version = int(version)
                return str(j.get('status')).lower(), version
        raise TransportError('status output unparseable')

    def latest_version(self, slug):
        return self.status(slug)[1]

    def create(self, slug, folder):
        # No --public/-u: the Kaggle CLI creates private datasets by default. --keep-tabular preserves CSV bytes.
        rc, out = self._cli(['datasets', 'create', '-p', str(folder), '-q', '--keep-tabular'], 3600)
        if rc != 0 or self.BAD_OUTPUT.search(out):
            raise TransportError('create failed: ' + out[:300])

    def new_version(self, slug, folder, note):
        # No -d: older versions are never deleted.
        rc, out = self._cli(['datasets', 'version', '-p', str(folder), '-m', note, '-q', '--keep-tabular'], 3600)
        if rc != 0 or self.BAD_OUTPUT.search(out):
            raise TransportError('version failed: ' + out[:300])

    def download(self, slug, version, dest):
        rc, out = self._cli(['datasets', 'download', f'{slug}/{version}', '-p', str(dest), '--unzip', '-o', '-q'], 3600)
        if rc != 0:
            raise TransportError('download failed: ' + out[:300])
        import zipfile
        for z in Path(dest).glob('*.zip'):
            zipfile.ZipFile(z).extractall(dest)
            z.unlink()


# ----------------------------------------------------------------------------- CLI
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fold', type=int, default=0, choices=(0,))
    ap.add_argument('--arm', required=True, choices=ARMS)
    ap.add_argument('--workdir', type=Path, required=True)
    ap.add_argument('--config', type=Path, required=True, help='the exact run config JSON the controller was launched with')
    ap.add_argument('--config-sha256', required=True)
    ap.add_argument('--package-manifest-sha256', required=True)
    ap.add_argument('--cache-content-sha256', required=True)
    ap.add_argument('--runner', type=Path, required=True, help='controller source providing primary_complete()')
    ap.add_argument('--runner-sha256', required=True)
    ap.add_argument('--controller-sha256', default=None, help='optional pin of the controller that launched the fit')
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--staging', type=Path, required=True, help='fresh directory (must not exist)')
    ap.add_argument('--verify-root', type=Path, required=True, help='fresh directory for the verification download')
    ap.add_argument('--receipt', type=Path, required=True, help='local receipt JSON to create (must not exist)')
    ap.add_argument('--source-revision', required=True, help='git revision of this exporter, or unversioned')
    ap.add_argument('--new-version', action='store_true')
    ap.add_argument('--create-fresh', metavar='SLUG', default=None,
                    help='create-only path for a fresh exact slug whose metadata is inaccessible (HTTP 403); must equal --dataset')
    ap.add_argument('--failed-receipts-only', action='store_true')
    ap.add_argument('--accept-reconciled-receipt', action='store_true', help='source96-only; always refused here')
    ap.add_argument('--include-logs', action='store_true')
    ap.add_argument('--controller-log', type=Path, default=None)
    ap.add_argument('--ready-timeout', type=float, default=900.0)
    ap.add_argument('--poll', type=float, default=15.0)
    return ap


def main(argv=None, *, transport=None, runner=None):
    a = build_parser().parse_args(argv)
    req = Request(fold=a.fold, workdir=a.workdir, config=a.config, config_sha256=a.config_sha256,
                  package_manifest_sha256=a.package_manifest_sha256, cache_content_sha256=a.cache_content_sha256,
                  runner=a.runner, runner_sha256=a.runner_sha256, dataset=a.dataset, staging=a.staging,
                  verify_root=a.verify_root, receipt=a.receipt, source_revision=a.source_revision, arm=a.arm,
                  mode='failed' if a.failed_receipts_only else 'complete', new_version=a.new_version,
                  create_fresh=a.create_fresh,
                  accept_reconciled_receipt=a.accept_reconciled_receipt, include_logs=a.include_logs,
                  controller_log=a.controller_log, controller_sha256=a.controller_sha256,
                  ready_timeout=a.ready_timeout, poll=a.poll, exporter_path=Path(__file__).resolve())
    result = run(req, transport=transport or KaggleTransport(), runner=runner)
    print(json.dumps({'state': result['state'], 'phase': result['phase'], 'dataset': result['dataset'],
                      'dataset_version': result.get('dataset_version'), 'problems': result['problems'],
                      'receipt': str(req.receipt)}))
    return 0 if result['state'] == 'verified' else 1


if __name__ == '__main__':
    raise SystemExit(main())
