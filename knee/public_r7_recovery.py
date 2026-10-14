"""R7 capped arm-mean recovery: the frozen S6 policy with R7 blend weights and precision.

knee/public_s6_recovery.py is frozen (its bytes are pinned by the CPU sensitivity
gate) and hard-codes the 70:30 blend and the S6 worker's '%.6f' output precision.
`recover_with` is that function's algorithm with exactly two parameters exposed;
tests prove that `recover_with(..., weights=(.7, .3), quantizer=frozen.quantize)`
reproduces the frozen function bit for bit. The R7 entry uses heliosli's fixed
0.5/0.5 per-finding weights (including the three OA findings) and the R7 worker's
'%.9g' predictions.csv precision. Cap, global genuine means, whole-population
average percentile ranks and the explicit missing-row masks are unchanged.
"""
import math

from knee.public_r7_stackers import quantize9

RECIPE = 'r7-fixed50-50-capped-arm-mean-v1'
RECEIPT_VERSION = 's6-recovery-cpu-sensitivity-v1'
WEIGHTS = (.5, .5)


def recover_with(control, s6, control_missing, s6_missing, *, weights, quantizer, recipe, fatal_errors=()):
    """Frozen recover() algorithm; slot names stay 'old320'/'s6' (the compact slot)."""
    import numpy as np
    import pandas as pd
    if fatal_errors:
        raise ValueError('FATAL_ERRORS_REQUIRE_ABORT')
    if len(weights) != 2 or not all(0 <= w <= 1 for w in weights) or abs(sum(weights) - 1) > 1e-12:
        raise ValueError('BLEND_WEIGHTS')
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
            genuine = quantizer(genuine)
        mean = genuine.mean(axis=0, dtype=np.float64)
        if name == 's6':
            mean = quantizer(mean[None])[0]
        a = np.empty((n, 12), dtype=np.float64)
        a[~missing] = genuine
        a[missing] = mean
        filled.append(a)
        means.append(mean.tolist())
    cr, sr = [pd.DataFrame(a).rank(method='average', pct=True).to_numpy() for a in filled]
    statuses = {'old320': np.where(cm, 'recovered', 'genuine').tolist(),
                's6': np.where(sm, 'recovered', 'genuine').tolist()}
    return {'recipe': recipe, 'receipt_version': RECEIPT_VERSION, 'population': n,
            'cap': cap, 'union_recovered': count, 'means': dict(zip(('old320', 's6'), means)),
            'statuses': statuses, 'control': filled[0], 's6': filled[1],
            'control_rank': cr, 's6_rank': sr, 'blend': weights[0] * cr + weights[1] * sr}


def recover(control, r7, control_missing, r7_missing, *, fatal_errors=()):
    """Raptor original320 0.5 + R7 0.5 per finding; R7 values at '%.9g'."""
    return recover_with(control, r7, control_missing, r7_missing, weights=WEIGHTS, quantizer=quantize9,
                        recipe=RECIPE, fatal_errors=fatal_errors)
