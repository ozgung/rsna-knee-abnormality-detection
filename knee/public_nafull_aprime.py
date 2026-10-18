"""Fixed A‴ profile and full-population recovery contract.

This module contains the new K62 checkpoint binding and arithmetic only. The
archived WS2 runtime is intentionally left untouched; a packet cannot become an
entry until its K62 worker is explicitly wired into the dedicated profile.
"""
from pathlib import Path
import hashlib
import json
import math

from knee.public_s6_qualification import LABELS, file_sha, require

VERSION = 'nafull-aprime-raptor62-r224-r7-v1'
METHOD_OLD = 'na-full SWA9-11 Raptor64 K62 one-view; 336 px to 384; CUDA fp16'
CPU_RECEIPT_SHA256 = 'b80f59edf27880773db067931c34e6eaf804afa614f245b356645555aafad1d6'
CONSTRUCTOR_GRID_SHA256 = '67ae62b6e4ba164abf4757e0948b4fa9cc410b587538dd19ce7be58186d465bd'
CHECKPOINT = {
    'dataset': 'ozgung/knee-raptor64-na-full-fold0-final-20261016',
    'version': 1,
    'file': 'raptor_ft_fold0_swa.pt',
    'sha256': 'c5cda6abf3234867d21baf067270b2f45540356802fb556ddc7cf23daa7740c8',
    'bytes': 292826627,
    'state_sha256': 'b77ad89d6299a2464bf5b452e1132678d3af21a75cf48211357fd6320d5d03e5',
}
RAPTOR_PROFILE = {
    'profile': 'raptor', 'windows': 62, 'input_res': 384, 'image_size': 336,
    'slots': 64, 'slot_sizes': [18, 14, 12, 8, 12], 'slice_span': [.06, .94],
    'tta': 'none', 'precision': 'CUDA fp16; no fp32 fallback',
    'decoder': 'memoized pydicom (raptor_input.prepare_study)',
}
WEIGHTS = {'na_full': .325, 'r224': .175, 'r7': .5}
ARMS = ('na_full', 'r224', 'r7')


def checkpoint_manifest(cpu_receipt):
    """Bind verified export bytes to an actual strict CPU K62 receipt."""
    import numpy as np
    require(file_sha(cpu_receipt) == CPU_RECEIPT_SHA256, 'NAFULL_EXACT_CPU_RECEIPT')
    r = json.loads(Path(cpu_receipt).read_bytes())
    require(r.get('checkpoint_sha256') == CHECKPOINT['sha256']
            and r.get('checkpoint_bytes') == CHECKPOINT['bytes'], 'NAFULL_CHECKPOINT_IDENTITY')
    require(r.get('state_sha256') == CHECKPOINT['state_sha256']
            and r.get('strict_state_load') is True, 'NAFULL_STRICT_CPU_STATE')
    require(r.get('device') == 'cpu' and r.get('windows_shape') == [62, 3, 384, 384]
            and r.get('forward_windows') == 62 and r.get('uses_raw_data') is False
            and r.get('uses_labels') is False, 'NAFULL_CPU_K62_PROFILE')
    require(r.get('dread_pin_method_check') is True and r.get('labels') == LABELS
            and r.get('mirror') is False and r.get('checkpoint_meta') == {'epoch': [9, 10, 11], 'swa_over': [9, 10, 11]}
            and 0 <= r.get('cpu_repeat_max_abs', math.inf) <= 1e-6,
            'NAFULL_DREAD_PIN_METHOD_AND_REPEAT')
    grid = r.get('constructor_grid_sha256')
    require(grid == CONSTRUCTOR_GRID_SHA256, 'NAFULL_CONSTRUCTOR_GRID_RECEIPT')
    probs = np.asarray(r.get('cpu_probabilities'), dtype=np.float64)
    require(probs.shape == (12,) and np.isfinite(probs).all() and bool(((probs >= 0) & (probs <= 1)).all()),
            'NAFULL_CPU_PROBABILITIES')
    m = r.get('manifest', {})
    require(m.get('source_sha256') == '859aae12ecea593b45ab0a5f6463fa573cd3083f340c161e920f10fbba29e81b'
            and m.get('profile') == 'raptor' and m.get('checkpoint_sha256') == CHECKPOINT['sha256']
            and m.get('checkpoint_file') == CHECKPOINT['file'] and m.get('weight_dataset') == CHECKPOINT['dataset'],
            'NAFULL_EXACT_AUDITED_SOURCE_MANIFEST')
    return {'checkpoint': dict(CHECKPOINT), 'constructor_grid_sha256': grid,
            'cpu_receipt_sha256': CPU_RECEIPT_SHA256, 'cpu_device': 'cpu',
            'windows': 62, 'strict_state_load': True}


def require_profile(manifest, cpu_binding):
    """Reject old320 and stale/partial K62 profiles without mutating them."""
    require(manifest.get('checkpoint') == cpu_binding['checkpoint'], 'NAFULL_PROFILE_CHECKPOINT')
    require(manifest.get('constructor_grid_sha256') == cpu_binding['constructor_grid_sha256'],
            'NAFULL_PROFILE_GRID')
    require(manifest.get('profile') == RAPTOR_PROFILE, 'NAFULL_PROFILE_RECIPE')
    require(manifest.get('labels') == LABELS, 'NAFULL_PROFILE_LABEL_ORDER')
    return True


def _quantize9(values):
    import numpy as np
    return np.asarray([[float('%.9g' % float(x)) for x in row] for row in values], dtype=np.float64)


def _r224_csv_roundtrip(values):
    import numpy as np
    import pandas as pd
    p = np.asarray(values, dtype=np.float64)
    require(p.ndim == 2 and p.shape[1] == 12 and len(p) > 0 and np.isfinite(p).all()
            and bool(((p >= 0) & (p <= 1)).all()), 'APRIME_R224_PROBABILITIES')
    from io import StringIO
    f = StringIO()
    pd.DataFrame(p.astype(np.float32)).to_csv(f, index=False)
    f.seek(0)
    roundtrip = pd.read_csv(f).to_numpy(dtype=np.float64)
    require(roundtrip.shape == p.shape and np.isfinite(roundtrip).all()
            and bool(((roundtrip >= 0) & (roundtrip <= 1)).all()), 'APRIME_R224_CSV_ROUNDTRIP')
    return roundtrip


def recover(na_full, r224, r7, na_full_missing, r224_missing, r7_missing, *, fatal_errors=()):
    """Mean-fill each arm on the complete population, then rank and blend once."""
    import numpy as np
    import pandas as pd
    if fatal_errors:
        raise ValueError('FATAL_ERRORS_REQUIRE_ABORT')
    arrays = [np.asarray(a) for a in (na_full, r224, r7)]
    masks = [np.asarray(m) for m in (na_full_missing, r224_missing, r7_missing)]
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
            genuine = _quantize9(genuine)
        mean = genuine.mean(axis=0, dtype=np.float64)
        if name == 'r7':
            mean = _quantize9(mean[None])[0]
        a = np.empty((n, 12), dtype=np.float64)
        a[~missing], a[missing] = genuine, mean
        filled[name] = a
        means[name] = mean.tolist()
        statuses[name] = np.where(missing, 'recovered', 'genuine').tolist()
    raw_r224 = _r224_csv_roundtrip(filled['r224'])
    ranks = {
        'na_full': pd.DataFrame(filled['na_full']).rank(method='average', pct=True).to_numpy(),
        'r224': pd.DataFrame(raw_r224).rank(method='average', pct=True).to_numpy(),
        'r7': pd.DataFrame(filled['r7']).rank(method='average', pct=True).to_numpy(),
    }
    # Keep A′ arithmetic order: 65/35 within the Raptor slot, then 50/50 with R7.
    raptor_slot = .65 * ranks['na_full'] + .35 * ranks['r224']
    blend = .5 * raptor_slot + .5 * ranks['r7']
    return {'recipe': VERSION, 'population': n, 'cap': cap, 'union_recovered': recovered,
            'means': means, 'statuses': statuses, 'filled': filled, 'ranks': ranks,
            'r224_csv_values': raw_r224, 'raptor_slot': raptor_slot, 'blend': blend,
            'weights': dict(WEIGHTS),
            'precision': {'na_full': 'original float64 conversion; no quantization',
                'r224': 'global mean then pandas float32 raw CSV roundtrip; one average percentile rank',
                'r7': 'Python %.9g genuine values and mean'}}


def recover_ws2_order(old320, r7, crop224, old_missing, r7_missing, crop224_missing, *, fatal_errors=()):
    """WS2 worker-order adapter; the old320 slot now means the bound na-full arm."""
    result = recover(old320, crop224, r7, old_missing, crop224_missing, r7_missing,
                     fatal_errors=fatal_errors)
    result['weights'] = {'old320': WEIGHTS['na_full'], 'crop224': WEIGHTS['r224'], 'r7': WEIGHTS['r7']}
    result['ranks'] = {'old320': result['ranks']['na_full'], 'crop224': result['ranks']['r224'],
                       'r7': result['ranks']['r7']}
    result['filled'] = {'old320': result['filled']['na_full'], 'crop224': result['filled']['r224'],
                        'r7': result['filled']['r7']}
    result['means'] = {'old320': result['means']['na_full'], 'crop224': result['means']['r224'],
                       'r7': result['means']['r7']}
    result['statuses'] = {'old320': result['statuses']['na_full'], 'crop224': result['statuses']['r224'],
                         'r7': result['statuses']['r7']}
    return result
