"""Private sharded Raptor64 cache: Dread "widedense v4" reader contract, fail-closed resume.

The volume of each study is produced by the exact pinned Raptor public reader that the
`raptor` profile of `knee.reference` audits (source SHA 859aae...): slots 18/14/12/8/12,
6-94% slice span, 336 px / 140 mm storage. `knee.raptor_input.prepare_study` is called
unchanged, so slot selection, ordering, intensity windowing, missing-slot masks and the
resource/fatal-error policy are the qualified production ones. Model input is the 336-px
volume resized to 384 by the trainer/inference window code; nothing here resizes to 384.

This module is a parallel variant; the qualified source96 cache is not modified. Generic,
depth-independent helpers (identity checks, partitions, membership, filenames, atomic JSON)
are imported from `knee.source96_cache` unchanged.
"""
from collections import Counter
import csv
import importlib.metadata
import json
from pathlib import Path
import resource
import shutil
import sys
import time
import zipfile

from knee import public_training_smoke as smoke
from knee import source96_cache as base
from knee.reference import RAPTOR_RECIPE, RAPTOR_SOURCE_SHA256

VERSION = 'raptor64-v4-336-sharded-v1'
DEPTH = 64
STORAGE = 336
SLOTS = [('Sagittal', 1, 18), ('Sagittal', 0, 14), ('Coronal', 1, 12), ('Coronal', 0, 8), ('Axial', -1, 12)]
SPAN = [0.06, 0.94]
V4_CHECKPOINT_SHA256 = '89606f05849838529e1b4658d28fb049623205d9853371548403bca460361ded'
RAW_BYTES = DEPTH * STORAGE * STORAGE + DEPTH
MAX_FILE_BYTES = RAW_BYTES + 65536  # NPY/ZIP/deflate overhead, including incompressible input.
DISK_RESERVE = 512 * 1024**2
CHECK_WINDOWS = 42
CHECK_RES = 384
SLUG_MAX = 50
CONFIG = {'reader': 'raptor-public-v4-unmodified', 'reader_source_sha256': RAPTOR_SOURCE_SHA256,
          'depth': DEPTH, 'storage': STORAGE, 'crop_mm': 140., 'slots': [list(s) for s in SLOTS], 'span': SPAN,
          'model_input_resolution': CHECK_RES, 'window_checks': {'k': CHECK_WINDOWS, 'res': CHECK_RES, 'function': 'public.eval_windows'},
          'slot_metadata': 'private index: per-slot plane, selected, actual Fluid_Sensitive, series frames, count',
          'cpu_threads': 1, 'default_shards': 4, 'raw_bytes_per_study': RAW_BYTES, 'max_npz_bytes_per_study': MAX_FILE_BYTES,
          'disk_reserve_bytes': DISK_RESERVE, 'training': False, 'models_loaded': 0}

# Re-exported depth-independent helpers (unchanged qualified source96 code).
safe_id, make_shards, membership, filename, safe_file, atomic_json = (
    base.safe_id, base.make_shards, base.membership, base.filename, base.safe_file, base.atomic_json)


def check_slug(slug):
    """Kaggle kernel/dataset slugs must be at most 50 characters (owner excluded)."""
    import re
    if not isinstance(slug, str) or not re.fullmatch('[a-z0-9]+(?:-[a-z0-9]+)*', slug) or not 5 <= len(slug) <= SLUG_MAX:
        raise ValueError(f'Slug must be lowercase-hyphenated and 5..{SLUG_MAX} characters')
    return slug


def require_public(public):
    """The loaded reader module must carry the v4 geometry verbatim."""
    if ([tuple(s) for s in public.SLOTS] != SLOTS or public.MAXS != DEPTH or public.IMG != STORAGE or
            float(public.CROP_MM) != 140. or public.K_EVAL != CHECK_WINDOWS):
        raise ValueError('Loaded Raptor reader geometry differs from the v4 contract')
    return public


def load_public(module_source, manifest):
    """Exact `knee.reference` raptor-profile loading; no substitutions."""
    from knee.reference import load_definitions
    if (manifest.get('profile') != 'raptor' or manifest.get('source_sha256') != RAPTOR_SOURCE_SHA256 or
            manifest.get('checkpoint_sha256') != V4_CHECKPOINT_SHA256 or manifest.get('recipe') != RAPTOR_RECIPE):
        raise ValueError('Raptor64 reader manifest differs from the pinned v4 K42 recipe')
    public, decoder = load_definitions(module_source, '', manifest)
    if decoder is not None:
        raise ValueError('Raptor profile must not carry a decoder')
    return require_public(public)


def validate_arrays(volume, mask):
    import numpy as np
    if volume.shape != (DEPTH, STORAGE, STORAGE) or volume.dtype != np.uint8:
        raise ValueError('Volume geometry changed')
    if mask.shape != (DEPTH,) or mask.dtype != np.uint8 or not np.isin(mask, [0, 1]).all():
        raise ValueError('Mask geometry changed')
    if not np.array_equal(mask.astype(bool), volume.reshape(DEPTH, -1).any(axis=1)):
        raise ValueError('Pixel/mask consistency failed')
    if not mask.any():
        raise ValueError('Whole-empty study cannot qualify')


PLANES = {'Sagittal': 0, 'Coronal': 1, 'Axial': 2}


def slot_records(public, sid, records, frames):
    """Per-slot plane/actual Fluid_Sensitive/series length; no identifiers."""
    from knee.reference import slot_plan
    plan, _ = slot_plan(public, sid, records)
    result = []
    for (plane, _fluid, count), (row, planned) in zip(public.SLOTS, plan):
        if planned != count:
            raise ValueError('Slot plan differs')
        if row is None:
            result.append({'plane': plane, 'selected': False, 'fluid_sensitive': None, 'frames': 0, 'count': count})
            continue
        fs = int(row.get('Fluid_Sensitive', 0) or 0)
        if fs not in (0, 1):
            raise ValueError('Fluid_Sensitive must be 0/1')
        result.append({'plane': plane, 'selected': True, 'fluid_sensitive': fs,
                       'frames': int(frames.get(str(row['SeriesInstanceUID']), 0)), 'count': count})
    return result


def tracked_reader(public, frames):
    """Wrap the public reader factory to observe each ordered series length (no extra I/O)."""
    original = public._make_reader

    def factory():
        order, pixel, resize = original()

        def tracked_order(path):
            files, spacing = order(path)
            frames[Path(path).name] = len(files)
            return files, spacing
        return tracked_order, pixel, resize
    return original, factory


def slice_metadata(slots):
    """Per-slice (plane*2+FS code, normalized depth within its series), mirroring public build_study picks."""
    import numpy as np
    codes, depth = [], []
    for s in slots:
        if s['plane'] not in PLANES or type(s['count']) is not int or s['count'] <= 0:
            raise ValueError('Invalid slot metadata')
        fs = s['fluid_sensitive'] if s['fluid_sensitive'] in (0, 1) else 0
        n, k = int(s['frames']), s['count']
        if n >= 1:
            lo, hi = int(n * SPAN[0]), int(n * SPAN[1]) - 1; hi = max(hi, lo)
            picks = np.linspace(lo, hi, k).round().astype(int) if n > 1 else np.zeros(k, int)
            d = np.minimum(picks, n - 1) / max(n - 1, 1)
        else:
            d = np.zeros(k)
        codes.extend([PLANES[s['plane']] * 2 + fs] * k); depth.extend(d.tolist())
    if len(codes) != DEPTH:
        raise ValueError('Slot metadata does not cover the volume depth')
    return np.asarray(codes, np.int64), np.asarray(depth, np.float32)


def window_metadata(codes, depth, centers):
    """(K,2) float32 [code, depth] of each window's (clamped) centre slice; train and eval share this."""
    import numpy as np
    D = len(codes)
    c = np.asarray([max(1, min(int(x), D - 2)) for x in centers])
    return np.stack([codes[c].astype(np.float32), depth[c].astype(np.float32)], 1)


def decode(public, sid, records, root):
    """Call the qualified production reader; return arrays plus aggregate counts and slot metadata."""
    from knee.raptor_input import prepare_study
    frames = {}
    original, factory = tracked_reader(public, frames)
    public._make_reader = factory
    try:
        volume, mask, counts = prepare_study(public, sid, records, root)
    finally:
        public._make_reader = original
    if counts.get('headers', 0) <= 0 or counts.get('decoded_unique_slices', 0) <= 0:
        raise ValueError('No successful raw DICOM decoding evidence')
    validate_arrays(volume, mask)
    slots = slot_records(public, sid, records, frames)
    slice_metadata(slots)
    return volume, mask, {'reader_counts': counts, 'slots': slots}


def window_checks(public, volume, mask):
    """Exact public evaluation windows at K42/384 (the endpoint geometry)."""
    import numpy as np
    import torch
    validate_arrays(volume, mask)
    slot = np.repeat(np.arange(len(SLOTS)), [s[2] for s in SLOTS])
    centers = np.asarray(public._eval_centers(mask, DEPTH, CHECK_WINDOWS))
    x = public.eval_windows(volume, mask, CHECK_WINDOWS, CHECK_RES)
    if tuple(x.shape) != (CHECK_WINDOWS, 3, CHECK_RES, CHECK_RES) or not torch.isfinite(x).all():
        raise ValueError('Raptor64 window shape/finiteness failed')
    triplets = np.stack([centers - 1, centers, centers + 1], 1)
    return {'k': CHECK_WINDOWS, 'res': CHECK_RES, 'shape': list(x.shape), 'finite': True,
            'tensor_sha256': smoke.digest(x.contiguous().numpy().tobytes()),
            'unique_centers': int(len(set(centers.tolist()))),
            'cross_slot_windows': int(np.count_nonzero(np.ptp(slot[triplets], axis=1))),
            'masked_slice_windows': int(np.count_nonzero(~mask[triplets].astype(bool).all(axis=1)))}


def array_hashes(volume, mask):
    validate_arrays(volume, mask)
    return {'volume_sha256': smoke.digest(volume.tobytes()), 'mask_sha256': smoke.digest(mask.tobytes())}


def read_bounded(path):
    """Bounded NPZ schema check before allocation; returns (volume, mask)."""
    import numpy as np
    with zipfile.ZipFile(path) as zipped:
        infos = zipped.infolist()
        limits = {'volume.npy': DEPTH * STORAGE * STORAGE + 4096, 'mask.npy': DEPTH + 4096}
        if len(infos) != 2 or {i.filename for i in infos} != set(limits) or any(i.file_size > limits[i.filename] for i in infos):
            raise ValueError('Cache archive members exceed bounded schema')
        for info in infos:
            with zipped.open(info) as stream:
                if np.lib.format.read_magic(stream) != (1, 0):
                    raise ValueError('Unexpected NPY format')
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
                expected = (DEPTH, STORAGE, STORAGE) if info.filename == 'volume.npy' else (DEPTH,)
                if shape != expected or fortran or dtype != np.dtype('uint8') or info.file_size != stream.tell() + int(np.prod(shape)):
                    raise ValueError('Cache NPY header differs from bounded schema')
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {'volume', 'mask'}:
            raise ValueError('Cache archive schema differs')
        return archive['volume'], archive['mask']


def verify_file(path, expected=None):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= MAX_FILE_BYTES:
        raise ValueError('Invalid cache file')
    volume, mask = read_bounded(path)
    result = {**array_hashes(volume, mask), 'file_sha256': smoke.file_sha(path), 'bytes': path.stat().st_size}
    if expected is not None and any(result[k] != expected.get(k) for k in result):
        raise ValueError('Cache hashes/bytes differ')
    return result


def atomic_npz(path, volume, mask):
    import numpy as np
    path = Path(path); tmp = path.with_name(path.name + '.tmp'); hashes = array_hashes(volume, mask)
    if path.is_symlink() or tmp.is_symlink():
        raise ValueError('Cache symlinks forbidden')
    if path.exists():  # Never overwrite an unindexed completed file: require raw equality.
        result = verify_file(path)
        if any(result[k] != v for k, v in hashes.items()):
            raise ValueError('Unindexed final differs from current decode')
        return result
    try:
        with tmp.open('wb') as stream:
            np.savez_compressed(stream, volume=volume, mask=mask)
        result = verify_file(tmp)
        if any(result[k] != v for k, v in hashes.items()):
            raise ValueError('Save/reload bytes differ')
        tmp.replace(path)
        return result
    except BaseException:
        if tmp.exists() and not tmp.is_symlink():
            tmp.unlink()
        raise


def index_rows(root, document, expected_contract, members, ids):
    if document.get('version') != VERSION or document.get('contract_sha256') != expected_contract:
        raise ValueError('Resume contract differs')
    entries = document.get('entries')
    if not isinstance(entries, list):
        raise ValueError('Private index missing')
    seen = {}
    for entry in entries:
        n = entry.get('ordinal')
        if type(n) is not int or n not in members or n in seen or entry.get('StudyInstanceUID') != ids[n] or entry.get('file') != filename(n):
            raise ValueError('Resume membership/order differs')
        verify_file(safe_file(root, entry['file']), entry); seen[n] = entry
    return seen


def disk_required(remaining, import_bytes=0):
    if type(remaining) is not int or remaining < 0 or type(import_bytes) is not int or import_bytes < 0:
        raise ValueError('Invalid disk projection')
    return remaining * MAX_FILE_BYTES + import_bytes + 2 * MAX_FILE_BYTES + DISK_RESERVE


def require_disk(root, remaining, import_bytes=0):
    required = disk_required(remaining, import_bytes); free = shutil.disk_usage(root).free
    if free < required:
        raise OSError('Insufficient disk for worst-case outputs and temporary space')
    return {'free_bytes': free, 'required_bytes': required, 'remaining_studies': remaining, 'import_bytes': import_bytes}


def environment():
    import cv2
    import torch
    return {'python': sys.version.split()[0], 'device': 'cpu', 'torch_threads': torch.get_num_threads(),
            'opencv_threads': cv2.getNumThreads(),
            'packages': {p: importlib.metadata.version(p) for p in
                         ('torch', 'numpy', 'pydicom', 'opencv-python', 'pylibjpeg', 'pylibjpeg-libjpeg', 'pylibjpeg-openjpeg')}}


def stream_cache(out, contract, ids, records, raw, public, *, resume_root=None, started=None):
    """Stream one study at a time; private aligned index, public aggregate receipt without IDs."""
    import torch, cv2
    from knee.fresh_reader_setup import time_limit
    started = time.monotonic() if started is None else started
    torch.set_num_threads(1); cv2.setNumThreads(1)
    out = Path(out)
    if out.is_symlink():
        raise ValueError('Output symlink forbidden')
    out.mkdir(parents=True, exist_ok=True)
    fingerprint = smoke.digest(smoke.canonical(contract))
    runfile = out / 'run-contract.json'; indexfile = out / 'index-private.json'
    if runfile.exists():
        if json.loads(runfile.read_bytes()) != contract:
            raise ValueError('Output contract differs')
    elif any(out.iterdir()):
        raise ValueError('Unbound existing output directory')
    else:
        atomic_json(runfile, contract)
    previous = out / 'receipt.json'
    if previous.exists():
        if previous.is_symlink():
            raise ValueError('Receipt symlink forbidden')
        archive = out / 'receipt-history'; archive.mkdir(exist_ok=True)
        if archive.is_symlink():
            raise ValueError('Receipt history symlink forbidden')
        atomic_json(archive / (smoke.file_sha(previous) + '.json'), json.loads(previous.read_bytes()))
    members = make_shards(ids, contract['shard_count'])[contract['shard_index']]
    report = {'status': 'RUNNING', 'phase': 'resume', 'version': VERSION, 'contract_sha256': fingerprint,
              'config': CONFIG, 'shard_index': contract['shard_index'], 'shard_count': contract['shard_count'],
              'membership': contract['membership'], 'partition': contract['partitions'][contract['shard_index']],
              'code_revision': contract['code_revision'], 'source_hashes': contract['sources'],
              'input_pins': contract['input_pins'], 'private_outputs': True, 'aggregate_contains_ids': False,
              'environment': environment(), 'completed_studies': 0, 'newly_decoded_studies': 0, 'resumed_studies': 0,
              'bytes': 0, 'decode_seconds': 0., 'window_seconds': 0., 'write_seconds': 0., 'counters': {}}
    entries = {}; counters = Counter()

    def write():
        report.update(completed_studies=len(entries), bytes=sum(e['bytes'] for e in entries.values()),
                      elapsed_seconds=time.monotonic() - started, counters=dict(counters),
                      peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024))
        free = shutil.disk_usage(out).free
        report['minimum_free_disk_bytes'] = min(report.get('minimum_free_disk_bytes', free), free)
        atomic_json(out / 'receipt.json', report)

    def save_index():
        atomic_json(indexfile, {'version': VERSION, 'contract_sha256': fingerprint,
                                'entries': [entries[n] for n in members if n in entries]})
        report['private_index_sha256'] = smoke.file_sha(indexfile)
    write()
    try:
        with time_limit(max(1, 43200 - (time.monotonic() - started)), 'actual CPU platform12h limit'):
            if report['environment'] != contract['qualified_environment']:
                raise ValueError('Qualified CPU environment differs')
            require_public(public)
            if indexfile.exists():
                entries = index_rows(out, json.loads(indexfile.read_bytes()), fingerprint, members, ids)
            if resume_root is not None:
                if entries:
                    raise ValueError('Cannot combine local and external resume')
                resume_root = Path(resume_root)
                if json.loads((resume_root / 'run-contract.json').read_bytes()) != contract:
                    raise ValueError('External resume contract differs')
                imported = index_rows(resume_root, json.loads((resume_root / 'index-private.json').read_bytes()), fingerprint, members, ids)
                report['disk_preflight'] = require_disk(out, len(members) - len(imported), sum(e['bytes'] for e in imported.values()))
                for n, entry in imported.items():
                    src = safe_file(resume_root, entry['file']); dest = safe_file(out, entry['file']); tmp = dest.with_name(dest.name + '.tmp')
                    if tmp.is_symlink():
                        raise ValueError('Resume temporary symlink forbidden')
                    shutil.copyfile(src, tmp); verify_file(tmp, entry); tmp.replace(dest); entries[n] = entry; save_index()
            if not indexfile.exists():
                save_index()
            report['resumed_studies'] = len(entries)
            report['disk_preflight_after_resume'] = require_disk(out, len(members) - len(entries))
            allowed = {filename(n) for n in members}
            if any(p.name.endswith('.npz') and p.name not in allowed for p in out.iterdir()):
                raise ValueError('Unexpected final cache file')
            report['phase'] = 'decode'; write()
            for n in members:
                if n in entries:
                    continue
                report['active_ordinal'] = n; write(); require_disk(out, 1)
                begin = time.monotonic(); volume, mask, audit = decode(public, ids[n], records, raw)
                report['decode_seconds'] += time.monotonic() - begin; counters.update(audit['reader_counts'])
                begin = time.monotonic(); windows = window_checks(public, volume, mask)
                report['window_seconds'] += time.monotonic() - begin
                begin = time.monotonic(); name = filename(n); saved = atomic_npz(safe_file(out, name), volume, mask)
                report['write_seconds'] += time.monotonic() - begin
                entries[n] = {'ordinal': n, 'StudyInstanceUID': ids[n], 'file': name, **saved, 'windows': windows,
                              'slots': audit['slots']}
                save_index(); report['newly_decoded_studies'] += 1; del volume, mask; write()
            if len(entries) != len(members):
                raise ValueError('Incomplete partition')
            report.pop('active_ordinal', None)
            report.update(status='PASS', phase='complete', private_index_sha256=smoke.file_sha(indexfile),
                          final_free_disk_bytes=shutil.disk_usage(out).free)
            write()
    except BaseException as error:
        report.update(status='ERROR', failure={'phase': report['phase'], 'type': type(error).__name__}); write()
        raise RuntimeError('Raptor64 cache failed; completed private files and sanitized aggregate receipt preserved') from None
    return report


def _run(inputs, out, contract, module_source, manifest, folds_sha256, *, started, resume_kernel=None, resume_dataset=None):
    import numpy as np
    from knee.fresh_reader_setup import dataset_root, notebook_root, exact_root
    from knee.corpus44_parity import require_image_tree
    if contract['version'] != VERSION or contract['config'] != CONFIG or folds_sha256 != contract['input_pins']['folds_sha256']:
        raise ValueError('Cache configuration differs')
    if type(contract['shard_index']) is not int or not 0 <= contract['shard_index'] < contract['shard_count']:
        raise ValueError('Invalid shard index')
    if smoke.digest(module_source.encode()) != contract['reader']['module_sha256'] or smoke.digest(smoke.canonical(manifest)) != contract['reader']['manifest_sha256']:
        raise ValueError('Qualified reader differs')
    public = load_public(module_source, manifest)
    from knee.raptor_input import policy_contract
    if policy_contract() != contract['reader']['input_policy']:
        raise ValueError('Raptor input policy differs')
    identity_root = notebook_root(inputs, smoke.IDENTITY_KERNEL, 'sealed_identity', ['sealed-manifest-private.json'])
    identity_path = identity_root / 'sealed-manifest-private.json'; smoke.require_file(identity_path, smoke.MANIFEST_SHA)
    identity = json.loads(identity_path.read_bytes())
    competition = exact_root([Path(inputs) / 'competitions/rsna-knee-abnormality-detection', Path(inputs) / 'rsna-knee-abnormality-detection'], ['train.csv', 'train_series.csv'])
    for name, sha in contract['input_pins']['competition'].items():
        smoke.require_file(competition / name, sha)
    historical = []
    for ref, prefix in zip(smoke.CORPORA, ['all', 'extra']):
        root = dataset_root(inputs, ref, [prefix + '_ids.npy']); name = prefix + '_ids.npy'
        smoke.require_file(root / name, smoke.PINS[name]); historical.extend(np.load(root / name, allow_pickle=False).astype(str).tolist())
    hpo = dataset_root(inputs, smoke.ASSETS, ['folds_v0.json']); smoke.require_file(hpo / 'folds_v0.json', folds_sha256)
    ids, summary = membership(historical, competition / 'train.csv', json.loads((hpo / 'folds_v0.json').read_bytes())['folds'], identity)
    if summary != contract['membership']:
        raise ValueError('Eligible population differs')
    partitions = make_shards(ids, contract['shard_count'])
    for i, members in enumerate(partitions):
        expected = {'studies': len(members), 'ordinals_sha256': smoke.digest(smoke.canonical(members)),
                    'ordered_ids_sha256': smoke.digest(smoke.canonical([ids[n] for n in members]))}
        if expected != contract['partitions'][i]:
            raise ValueError('Partition differs')
    selected = [ids[n] for n in partitions[contract['shard_index']]]; records = {u: [] for u in selected}
    with (competition / 'train_series.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['StudyInstanceUID'] in records:
                records[row['StudyInstanceUID']].append(row)
    raw = require_image_tree(competition, selected, records)
    resume = None
    if resume_kernel:
        resume = notebook_root(inputs, resume_kernel, 'raptor64_cache', ['run-contract.json', 'index-private.json'])
    if resume_dataset:
        resume = dataset_root(inputs, resume_dataset, ['run-contract.json', 'index-private.json'])
    return stream_cache(out, contract, ids, records, raw, public, resume_root=resume, started=started)


def run(inputs, out, contract, module_source, manifest, folds_sha256, *, started, resume_kernel=None, resume_dataset=None):
    try:
        return _run(inputs, out, contract, module_source, manifest, folds_sha256, started=started,
                    resume_kernel=resume_kernel, resume_dataset=resume_dataset)
    except BaseException as error:
        path = Path(out)
        if not path.is_symlink():
            path.mkdir(parents=True, exist_ok=True)
            if not (path / 'receipt.json').exists():
                atomic_json(path / 'receipt.json', {'version': VERSION, 'status': 'ERROR', 'phase': 'setup',
                    'contract_sha256': smoke.digest(smoke.canonical(contract)),
                    'failure': {'type': type(error).__name__}, 'elapsed_seconds': time.monotonic() - started,
                    'aggregate_contains_ids': False, 'private_outputs': True})
        raise RuntimeError('Raptor64 cache failed; completed private files and sanitized aggregate receipt preserved') from None
