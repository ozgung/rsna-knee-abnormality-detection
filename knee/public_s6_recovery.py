"""Versioned capped arm-mean recovery. Array-only: no labels or image readers.

This policy does not qualify an entry and cannot consume strict-v2 PASS receipts.
Only explicitly declared missing whole rows may be filled. Callers must propagate
fatal errors; this module never catches prediction, source or resource errors.
"""
import hashlib
import json
import math

RECIPE = 's6-fixed70-30-capped-arm-mean-v1'
RECEIPT_VERSION = 's6-recovery-cpu-sensitivity-v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def quantize(values):
    import numpy as np
    return np.asarray([[float('%.6f' % float(v)) for v in row] for row in values], dtype=np.float64)


def recover(control, s6, control_missing, s6_missing, *, fatal_errors=()):
    """Fill globally across both shards, then rank the entire actual population.

    Genuine S6 values and its recovered mean are source-six-decimal values.
    Old320 genuine values and means retain their original precision.
    Explicit missing rows may contain NaN; genuine nonfinite rows are fatal.
    """
    import numpy as np
    import pandas as pd
    if fatal_errors:
        raise ValueError('FATAL_ERRORS_REQUIRE_ABORT')
    c, s = np.asarray(control), np.asarray(s6)
    cm, sm = np.asarray(control_missing), np.asarray(s6_missing)
    n = len(c)
    if n == 0 or c.shape != (n, 12) or s.shape != c.shape:
        raise ValueError('ARM_SCHEMA')
    if cm.shape != (n,) or sm.shape != (n,) or cm.dtype != bool or sm.dtype != bool:
        raise ValueError('EXPLICIT_ROW_MASK_REQUIRED')
    cap = math.floor(3 * n / 100)
    count = int((cm | sm).sum())
    if count > cap:
        raise ValueError('UNION_RECOVERY_CAP')
    filled = []
    means = []
    for name, a, missing in [('old320', c, cm), ('s6', s, sm)]:
        genuine = np.asarray(a[~missing], dtype=np.float64)
        if not len(genuine):
            raise ValueError('GENUINE_SUCCESS_REQUIRED_' + name)
        if not np.isfinite(genuine).all() or not ((genuine >= 0) & (genuine <= 1)).all():
            raise ValueError('GENUINE_PROBABILITIES_' + name)
        if name == 's6':
            genuine = quantize(genuine)
        mean = genuine.mean(axis=0, dtype=np.float64)
        if name == 's6':
            mean = quantize(mean[None])[0]
        a = np.empty((n, 12), dtype=np.float64)
        a[~missing] = genuine
        a[missing] = mean
        filled.append(a)
        means.append(mean.tolist())
    cr, sr = [pd.DataFrame(a).rank(method='average', pct=True).to_numpy() for a in filled]
    statuses = {'old320': np.where(cm, 'recovered', 'genuine').tolist(),
                's6': np.where(sm, 'recovered', 'genuine').tolist()}
    return {'recipe': RECIPE, 'receipt_version': RECEIPT_VERSION, 'population': n,
            'cap': cap, 'union_recovered': count, 'means': dict(zip(('old320', 's6'), means)),
            'statuses': statuses, 'control': filled[0], 's6': filled[1],
            'control_rank': cr, 's6_rank': sr, 'blend': .7 * cr + .3 * sr}


def merge_shards(shards, population):
    """Collect exact canonical ordinals; never fill or rank within a shard."""
    import numpy as np
    c = np.empty((population, 12), dtype=np.float64)
    s = np.empty_like(c)
    cm = np.empty(population, dtype=bool)
    sm = np.empty(population, dtype=bool)
    seen = set()
    for shard in shards:
        indices = np.asarray(shard['ordinals'])
        if indices.ndim != 1 or indices.dtype.kind not in 'iu':
            raise ValueError('SHARD_ORDINAL_SCHEMA')
        if any(int(i) in seen or i < 0 or i >= population for i in indices) or len(set(indices.tolist())) != len(indices):
            raise ValueError('SHARD_DUPLICATE_OR_RANGE')
        for key, target in [('control', c), ('s6', s), ('control_missing', cm), ('s6_missing', sm)]:
            a = np.asarray(shard[key])
            if a.shape != target[indices].shape or (target.dtype == bool and a.dtype != bool):
                raise ValueError('SHARD_ARRAY_SCHEMA')
            target[indices] = a
        seen.update(indices.tolist())
    if len(seen) != population:
        raise ValueError('SHARD_COVERAGE')
    return c, s, cm, sm


def freeze_protocol(windows, *, seed=20261012):
    """Label-blind ordinal masks, including the fixed workload tail."""
    import numpy as np
    w = np.asarray(windows)
    if w.shape != (870,) or not np.isfinite(w).all() or not (w > 0).all():
        raise ValueError('FIXED870_WORKLOAD_REQUIRED')
    permutation = np.random.default_rng(seed).permutation(870).tolist()
    tail = sorted(range(870), key=lambda i: (-int(w[i]), i))
    scenarios = [{'name': 'zero', 'old_missing': [], 's6_missing': []}]
    for family, order, sizes in [('ordinal', permutation, (8, 26)), ('workload_tail', tail, (26,))]:
        for size in sizes:
            indices = order[:size]
            for pattern in ('old_only', 's6_only', 'coincident', 'disjoint'):
                old = indices if pattern in ('old_only', 'coincident') else indices[:size // 2] if pattern == 'disjoint' else []
                s6 = indices if pattern in ('s6_only', 'coincident') else indices[size // 2:] if pattern == 'disjoint' else []
                scenarios.append({'name': f'{family}_{size}_{pattern}', 'old_missing': old, 's6_missing': s6})
    protocol = {'recipe': RECIPE, 'receipt_version': RECEIPT_VERSION, 'seed': seed,
                'population': 870, 'permutation': permutation, 'workload_order': tail,
                'scenarios': scenarios, 'bootstrap': {'draws': 2000, 'seed': 42},
                'gate': 'each scenario: delta vs intact original320 >0; lower paired component CI >0; fewer than3 supported target declines >=.01',
                'means': 'per target, each arm genuine successes across entire population before average pct ranks; S6 genuine/mean six decimals',
                'interpretation': 'prespecified synthetic sensitivity; not natural-failure validation or runtime qualification'}
    return protocol
