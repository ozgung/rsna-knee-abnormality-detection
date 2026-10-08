"""TRAIN-04 variable-series cache; no fitting and no heavy import at module load."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import time

from knee.fresh_reader import (CONFIG, SOURCE_HASHES, environment, fingerprint, load_source,
                               make_plan, validate_plan, write_json)
from knee.fresh_reader_setup import progress, time_limit
from knee.validation import file_sha256

VERSION = 'coverage256-full-series-v1'
ORIGINAL_PLAN = '1bd346c05c0aac20f54b82b790e356d3d6c1b95606accaa4a5d1da7e3e421942'
IDENTITY_FP = '911f1605497f5c181402e3fbf70c26d588ffb9c73d6cb2d67e45f22eec52245d'
IDENTITY_SHA = '21c71745703d742585684298494a6269ae4217fadc5fc4c26f3bf0d10a289467'
CACHE_CONFIG = dict(image_size=256, max_slices=64, pilot_studies=64, pilot_seconds=1800,
                    full_seconds=21600, bytes_limit=16 * 1024**3, disk_reserve=5 * 1024**3)


def bind_cohort(candidate, original, identity):
    """Membership is checked before any image decode or alternate labels."""
    validate_plan(original)
    if original['fingerprint'] != ORIGINAL_PLAN:
        raise ValueError('Original development plan differs')
    validate_plan(identity)
    if identity['fingerprint'] != IDENTITY_FP:
        raise ValueError('Frozen identity manifest differs')
    if candidate['targets'] != original['targets'] or candidate['files'] != original['files']:
        raise ValueError('Competition target/schema/file identity differs')
    old_dev = [r for r in original['records'] if r['split'] == 'weak_validation']
    new_dev = [r for r in candidate['records'] if r['split'] == 'weak_validation']
    if new_dev != old_dev or len(new_dev) != 200:
        raise ValueError('Development membership/order/labels changed')
    rows = candidate['records']
    if Counter(r['split'] for r in rows) != {'train': 2000, 'weak_validation': 200}:
        raise ValueError('Expected original2000 training and unchanged200 development')
    known = {r['uid']: r for r in identity['records']}
    if len(known) != len(identity['records']) or len({r['uid'] for r in rows}) != len(rows):
        raise ValueError('Duplicate identity/cohort rows')
    role_components = {'train': set(), 'weak_validation': set()}
    retained = []
    for row in rows:
        item = known[row['uid']]
        expected = 'training_candidate' if row['split'] == 'train' else 'development'
        if (item['future_role'] != expected or item['original_split'] != row['split'] or
                item['report_group'] != row['group'] or item['quarantine_reasons']):
            raise ValueError('Original membership conflicts with frozen identity exclusion')
        role_components[row['split']].add(item['component'])
        retained.append(row | {'component': item['component']})
    all_dev = {r['component'] for r in identity['records'] if r['future_role'] == 'development'}
    excluded = {r['component'] for r in identity['records']
                if r['future_role'] != 'training_candidate'}
    if role_components['train'] & (all_dev | excluded):
        raise ValueError('Training component overlaps held-out/excluded identities')
    if not {r['uid'] for r in original['records'] if r['split'] == 'train'} <= {
            r['uid'] for r in retained if r['split'] == 'train'}:
        raise ValueError('Original512 training cohort is not retained')
    plan = {'version': VERSION, 'records': retained, 'targets': candidate['targets'],
            'files': candidate['files'], 'weak_labels_sha256': candidate['weak_labels_sha256'],
            'original_plan_fingerprint': ORIGINAL_PLAN, 'identity_fingerprint': IDENTITY_FP,
            'source_hashes': SOURCE_HASHES, 'config': CACHE_CONFIG,
            'selection': 'original2000train/unchanged200dev; largest usable series per slot; stable ID ties'}
    return plan | {'fingerprint': fingerprint(plan)}


def make_coverage_plan(data, weak, original_path, identity_path):
    if file_sha256(identity_path) != IDENTITY_SHA:
        raise ValueError('Identity file hash differs')
    candidate = make_plan(data, weak, CONFIG | {'train_limit': 2000})
    return bind_cohort(candidate, json.loads(Path(original_path).read_text()),
                       json.loads(Path(identity_path).read_text()))


def pilot_rows(plan):
    return sorted(plan['records'], key=lambda r: hashlib.sha256(
        ('coverage256-pilot-v1:' + r['uid']).encode()).hexdigest())[:CACHE_CONFIG['pilot_studies']]


def projection(sample, total, overhead, *, bytes_limit=CACHE_CONFIG['bytes_limit']):
    import math
    if (not sample or len(sample) > total or not math.isfinite(overhead) or overhead < 0 or
            any(not math.isfinite(r['seconds']) or r['seconds'] <= 0 or r['bytes'] <= 0 for r in sample)):
        raise ValueError('Invalid cache pilot measurements')
    if type(bytes_limit) is not int or not CACHE_CONFIG['bytes_limit']<=bytes_limit<=17*1024**3:
        raise ValueError('Remote byte allowance must be16..17GiB')
    seconds = overhead + 1.5 * sum(r['seconds'] for r in sample) * total / len(sample)
    size = 1.25 * sum(r['bytes'] for r in sample) * total / len(sample)
    return {'passed': seconds <= CACHE_CONFIG['full_seconds'] and size <= bytes_limit,
            'estimated_seconds_with_margin': seconds, 'estimated_bytes_with_margin': size,
            'time_margin': 1.5, 'size_margin': 1.25, 'measured_studies': len(sample),
            'target_studies': total, 'overhead_seconds': overhead,'bytes_limit':bytes_limit}


def validate_volumes(volumes, *, image_size=256):
    import numpy as np
    if len(volumes) != 6:
        raise ValueError('Six variable series slots required')
    for v in volumes:
        if (v.dtype != np.uint8 or v.ndim != 3 or v.shape[1:] != (image_size, image_size) or
                (len(v) and not 3 <= len(v) <= 64) or (len(v) and not np.any(v))):
            raise ValueError('Invalid variable series array')
    if not any(len(v) for v in volumes):
        raise ValueError('Study has no usable series')


def sample_windows(volumes, k, *, train, rng, image_size=256):
    """Draw within each series only; RNG is explicit and evaluation consumes none."""
    import numpy as np
    validate_volumes(volumes, image_size=image_size)
    if type(k) is not int or k not in (8, 12):
        raise ValueError('TRAIN04 uses eight train or twelve eval windows')
    x = np.zeros((6, k, 3, image_size, image_size), np.uint8)
    mask, pos = np.zeros(6, bool), np.zeros((6, k), np.float32)
    for s, v in enumerate(volumes):
        n = len(v)
        if not n:
            continue
        a, b = max(1., .04 * (n-1)), min(n-2., .96 * (n-1))
        if train:
            edges = np.linspace(a, b, k+1)
            centers = edges[:-1] + rng.random(k) * np.diff(edges)
        else:
            centers = np.linspace(a, b, k)
        centers = np.clip(np.round(centers).astype(int), 1, n-2)
        x[s] = v[centers[:, None] + np.array([-1, 0, 1])[None, :]]
        mask[s], pos[s] = True, centers / (n-1)
    return x, mask, pos


def read_volumes(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != {f'slot{i}' for i in range(6)}:
            raise ValueError('Cache slot schema differs')
        result = [z[f'slot{i}'] for i in range(6)]
    validate_volumes(result)
    return result


def verify_index(root, plan, index, *, complete):
    expected = {r['uid'] for r in plan['records']}
    if not set(index) <= expected or (complete and set(index) != expected):
        raise ValueError('Cache membership differs')
    for uid, row in index.items():
        filename = hashlib.sha256(uid.encode()).hexdigest() + '.npz'
        if row['file'] != filename:
            raise ValueError('Cache filename differs')
        path = Path(root) / 'arrays' / filename
        if path.stat().st_size != row['bytes'] or file_sha256(path) != row['sha256']:
            raise ValueError('Cache file identity differs')


def build_cache(data, sources, plan, out, *, previous=None, expansion_bytes=None):
    import cv2
    import numpy as np
    import pandas as pd
    validate_plan(plan)
    if plan['version'] != VERSION or plan['config'] != CACHE_CONFIG:
        raise ValueError('Cache recipe differs')
    pilot = previous is None
    if expansion_bytes is not None and (pilot or type(expansion_bytes) is not int or
            not CACHE_CONFIG['bytes_limit']<=expansion_bytes<=17*1024**3):
        raise ValueError('Explicit byte revision applies to full cache only; max17GiB')
    byte_limit=expansion_bytes or CACHE_CONFIG['bytes_limit']
    started = time.monotonic()
    limit = CACHE_CONFIG['pilot_seconds' if pilot else 'full_seconds']
    data, out = Path(data), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out/'arrays').mkdir()
    write_json(out/'plan-private.json', plan)
    audit = {'status': 'RUNNING', 'stage': 'pilot' if pilot else 'complete',
             'plan_fingerprint': plan['fingerprint'], 'environment': environment(),
             'selected_series': 0, 'missing_slots': 0, 'failures': 0, 'cached_studies': 0,
             'expansion_bytes_override':expansion_bytes,'effective_bytes_limit':byte_limit}
    index, total_bytes, new_seconds = {}, 0, 0.
    try:
        with time_limit(limit, 'TRAIN04 CPU cache'):
            if previous:
                previous = Path(previous)
                old_plan = json.loads((previous/'plan-private.json').read_text())
                old_audit = json.loads((previous/'cache-report.json').read_text())
                if old_plan != plan or old_audit['status'] != 'PASS' or (not old_audit['gate']['passed'] and expansion_bytes is None):
                    raise ValueError('Matching successful pilot required')
                index = json.loads((previous/'index-private.json').read_text())
                if set(index) != {r['uid'] for r in pilot_rows(plan)}:
                    raise ValueError('Exact64 pilot coverage required')
                verify_index(previous, plan, index, complete=False)
                gate = projection(list(index.values()), len(plan['records']), old_audit['overhead_seconds'],bytes_limit=byte_limit)
                audit['original_pilot_gate']=old_audit['gate']
                audit['expansion_gate']=gate
                if not gate['passed'] or shutil.disk_usage(out).free < gate['estimated_bytes_with_margin'] + CACHE_CONFIG['disk_reserve']:
                    raise ValueError('CPU expansion time/disk gate failed')
                for row in index.values():
                    shutil.copy2(previous/'arrays'/row['file'], out/'arrays'/row['file'])
                total_bytes = sum(r['bytes'] for r in index.values())
                audit['reused_studies'] = len(index)
            prep, model = load_source(sources, 'preprocess'), load_source(sources, 'knee')
            if prep.MAX_SLICES != 64 or model.LABELS != plan['targets']:
                raise ValueError('External preprocessing/schema differs')
            selected = pilot_rows(plan) if pilot else plan['records']
            ids = {r['uid'] for r in selected}
            series = pd.read_csv(data/'train_series.csv', dtype=str)
            series = series[series.StudyInstanceUID.isin(ids)].sort_values('SeriesInstanceUID')
            series['Fat_Suppression'] = series['Fat_Suppression'].astype(int)
            counts = {}
            for row in series.itertuples():
                folder = data/'train_series'/row.StudyInstanceUID/row.SeriesInstanceUID
                counts[row.SeriesInstanceUID] = min(sum(p.is_file() for p in folder.iterdir()), 64)
            table = model.build_study_table(series, counts)
            for row in selected:
                uid = row['uid']
                if uid in index:
                    continue
                if shutil.disk_usage(out).free < CACHE_CONFIG['disk_reserve'] + 64*6*256**2:
                    raise RuntimeError('Remote disk reserve reached')
                tick = time.monotonic()
                volumes, selected_series = [], []
                for slot, candidates in enumerate(table.get(uid, [[] for _ in range(6)])):
                    if not candidates:
                        volumes.append(np.empty((0,256,256), np.uint8)); selected_series.append(None)
                        audit['missing_slots'] += 1
                        continue
                    volume, info = prep.load_series(str(data/'train_series'/uid/candidates[0]), model.PLANES[slot%3])
                    if info['n_bad'] or len(volume) < 3 or not np.any(volume):
                        raise ValueError('Selected series strict decoding failed')
                    volumes.append(np.stack([cv2.resize(v, (256,256), interpolation=cv2.INTER_AREA) for v in volume]))
                    selected_series.append(candidates[0]); audit['selected_series'] += 1
                validate_volumes(volumes)
                filename = hashlib.sha256(uid.encode()).hexdigest()+'.npz'
                path = out/'arrays'/filename
                np.savez_compressed(path, **{f'slot{i}': v for i,v in enumerate(volumes)})
                size = path.stat().st_size
                if total_bytes + size > byte_limit:
                    path.unlink(); raise RuntimeError('Remote cache size limit exceeded')
                duration = time.monotonic()-tick
                index[uid] = dict(file=filename, bytes=size, sha256=file_sha256(path), seconds=duration,
                                  series=selected_series, lengths=[len(v) for v in volumes])
                total_bytes += size; new_seconds += duration
                write_json(out/'index-private.json', index)
                if len(index)%8 == 0:
                    progress('coverage_cache', studies=len(index), bytes=total_bytes, seconds=time.monotonic()-started)
            sample = [index[r['uid']] for r in pilot_rows(plan)]
            overhead = time.monotonic()-started-new_seconds
            audit.update(status='PASS', gate=projection(sample,len(plan['records']), overhead,bytes_limit=byte_limit),
                         overhead_seconds=overhead, new_decode_seconds=new_seconds,
                         cached_studies=len(index), cache_bytes=total_bytes)
            verify_index(out, plan, index, complete=not pilot)
    except BaseException as exc:
        audit.update(status='FAIL', failures=1, failure_type=type(exc).__name__)
        raise
    finally:
        audit['elapsed_seconds'] = time.monotonic()-started
        write_json(out/'cache-report.json', audit)
    return audit
