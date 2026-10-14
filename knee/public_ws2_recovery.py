"""Fixed A-prime rank composition with three-arm capped global-mean recovery.

Array-only and entry-independent. Masks mean classified data-local whole-arm
failures, never absent/unreached work. The caller must prove that classification,
bind IDs/source/environment, and abort unknown/resource/platform failures.
"""
import math

RECIPE = 'ws2-fixed35-raptor-slot-three-arm-capped-v1'
ARMS = ('old320', 'r7', 'crop224')
A224 = .35


def quantize9(values):
    import numpy as np
    return np.asarray([[float('%.9g' % float(v)) for v in row] for row in values], dtype=np.float64)


def source224_csv_values(probabilities):
    """Round-trip the public raw224 float32 pandas CSV before ranking.

    Recovery happens before this conversion, over the complete canonical population.
    The current public recipe directly percentile-ranks this CSV once; the older
    WS2 argsort/second-rank proposal is intentionally absent.
    """
    from io import StringIO
    import numpy as np
    import pandas as pd
    p = np.asarray(probabilities)
    if p.ndim != 2 or p.shape[1] != 12 or len(p) == 0 or not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError('WS2_224_PROBABILITIES')
    handle = StringIO()
    pd.DataFrame(p.astype(np.float32)).to_csv(handle, index=False)
    handle.seek(0)
    result = pd.read_csv(handle).to_numpy(dtype=np.float64)
    if result.shape != p.shape or not np.isfinite(result).all() or not ((result >= 0) & (result <= 1)).all():
        raise ValueError('WS2_224_CSV_ROUNDTRIP')
    return result


def recover(old320, r7, crop224, old_missing, r7_missing, crop224_missing, *, fatal_errors=()):
    import numpy as np
    import pandas as pd
    if fatal_errors:
        raise ValueError('FATAL_ERRORS_REQUIRE_ABORT')
    arrays = [np.asarray(a) for a in (old320, r7, crop224)]
    masks = [np.asarray(m) for m in (old_missing, r7_missing, crop224_missing)]
    if arrays[0].ndim != 2 or arrays[0].shape[1] != 12 or len(arrays[0]) == 0:
        raise ValueError('ARM_SCHEMA')
    n = len(arrays[0])
    if any(a.shape != (n, 12) for a in arrays):
        raise ValueError('ARM_SCHEMA')
    if any(m.shape != (n,) or m.dtype != bool for m in masks):
        raise ValueError('EXPLICIT_ROW_MASK_REQUIRED')
    cap = math.floor(3 * n / 100)
    recovered = int(np.logical_or.reduce(masks).sum())
    if recovered > cap:
        raise ValueError('UNION_RECOVERY_CAP')
    filled, means, statuses = {}, {}, {}
    for name, source, missing in zip(ARMS, arrays, masks):
        genuine = np.asarray(source[~missing], dtype=np.float64)
        if not len(genuine):
            raise ValueError('GENUINE_SUCCESS_REQUIRED_' + name)
        if not np.isfinite(genuine).all() or not ((genuine >= 0) & (genuine <= 1)).all():
            raise ValueError('GENUINE_PROBABILITIES_' + name)
        if name == 'r7':
            genuine = quantize9(genuine)
        mean = genuine.mean(axis=0, dtype=np.float64)
        if name == 'r7':
            mean = quantize9(mean[None])[0]
        a = np.empty((n, 12), dtype=np.float64)
        a[~missing], a[missing] = genuine, mean
        filled[name] = a
        means[name] = mean.tolist()
        statuses[name] = np.where(missing, 'recovered', 'genuine').tolist()
    raw224 = source224_csv_values(filled['crop224'])
    ranks = {name: pd.DataFrame(raw224 if name == 'crop224' else filled[name]).rank(
        method='average', pct=True).to_numpy() for name in ARMS}
    # Preserve published arithmetic order: the35% mix precedes the50:50 blend.
    raptor_slot = (1.0 - A224) * ranks['old320'] + A224 * ranks['crop224']
    blend = .5 * raptor_slot + .5 * ranks['r7']
    return {'recipe': RECIPE, 'population': n, 'cap': cap, 'union_recovered': recovered,
            'means': means, 'statuses': statuses, 'filled': filled, 'ranks': ranks,
            'source224_csv_values': raw224, 'raptor_slot': raptor_slot, 'blend': blend,
            'weights': {'old320': .325, 'r7': .5, 'crop224': .175},
            'precision': {'old320': 'original float64 conversion; no quantization',
                          'r7': 'Python %.9g genuine values and mean',
                          'crop224': 'global mean then pandas float32 raw CSV roundtrip; one average percentile rank'}}


def merge_shards(shards, population):
    """Require complete three-arm ledgers and canonical unique ordinals.

    Missing ledger rows/fields are fatal, not data-local prediction failures.
    Output order: old320/r7/crop224, then the three explicit whole-row masks.
    """
    import numpy as np
    if type(population) is not int or population <= 0:
        raise ValueError('LEDGER_POPULATION')
    arrays = {name: np.empty((population, 12), dtype=np.float64) for name in ARMS}
    masks = {name + '_missing': np.empty(population, dtype=bool) for name in ARMS}
    seen = set()
    for shard in shards:
        if not isinstance(shard, dict) or not {'ordinals', *arrays, *masks} <= shard.keys():
            raise ValueError('MISSING_LEDGER_FIELDS')
        ix = np.asarray(shard['ordinals'])
        if ix.ndim != 1 or ix.dtype.kind not in 'iu':
            raise ValueError('SHARD_ORDINAL_SCHEMA')
        if len(set(ix.tolist())) != len(ix) or any(int(i) in seen or i < 0 or i >= population for i in ix):
            raise ValueError('SHARD_DUPLICATE_OR_RANGE')
        for key, target in {**arrays, **masks}.items():
            a = np.asarray(shard[key])
            if a.shape != target[ix].shape or (target.dtype == bool and a.dtype != bool):
                raise ValueError('SHARD_ARRAY_SCHEMA')
            target[ix] = a
        seen.update(ix.tolist())
    if len(seen) != population:
        raise ValueError('SHARD_COVERAGE')
    return (*(arrays[name] for name in ARMS), *(masks[name + '_missing'] for name in ARMS))
