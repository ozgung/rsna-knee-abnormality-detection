"""Frozen R6-flip frame: A-prime Raptor half, heliosli R6 + flip TTA compact half, per-finding weights.

Dependency-free at import. Array helpers import NumPy/pandas/torch inside calls.

Frame (frozen 2026-10-17, before any R6 run of ours):
* Raptor half = 0.65 rank(R384 two-view mirror, A-prime old320 worker) + 0.35 rank(R224 one view,
  A-prime crop224 worker), both unchanged from A-prime (57229227, public 0.956, the baseline).
* Compact half = rank(heliosli R6 bundle fe055406fad6a12f, compact_infer v4c --flip-tta).
* Final, per finding f: w_f * Raptor half + (1 - w_f) * rank(R6), w = RAPTOR_WEIGHTS, 0.5 otherwise.
* Recovery = A-prime's: whole-arm missing rows get the arm's global genuine mean, union cap
  floor(3% N), complete-population average percentile ranks; R6 values at '%.9g'.
Excluded: na-full, V2, extra readers and the R224 mirror.
"""
import hashlib
import json
import math

from knee.public_r6_profile import (BUNDLE_ID, DEFAULT_RAPTOR_WEIGHT, FLIP_TTA, R224_WEIGHT, R384_WEIGHT,
                                    RAPTOR_WEIGHTS)

VERSION = 'r6flip-raptor-r6-perfinding-v1'
FRAME_ID = 'r6flip-aprime-raptor-r6fe055406-flip-perfinding-20261017'
LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
          'Synovitis', "Baker's", 'Contusion', 'Fracture']
ARMS = ('old320', 'r6', 'crop224')
R6_DATASET = 'heliosli/rsna-knee-r6-frozen-fe055406-w1'
R6_DATASET_VERSION = 1
R6_MOUNT = 'rsna-knee-r6-frozen-fe055406-w1'
R6_ARM = 'E_lrprior'
MIRROR_PAIRS = (('Medial Meniscus', 'Lateral Meniscus'), ('Medial OA', 'Lateral OA'))
FRAME = {
    'frame_id': FRAME_ID, 'version': VERSION, 'baseline': 'A-prime 57229227 (public 0.956)',
    'raptor_half': {'old320': R384_WEIGHT, 'crop224': R224_WEIGHT,
                    'source': 'A-prime R384 two-view mirror and R224 one-view workers, unchanged'},
    'compact_half': {'bundle_id': BUNDLE_ID, 'dataset': '%s/%d' % (R6_DATASET, R6_DATASET_VERSION),
                     'flip_tta': FLIP_TTA, 'worker': 'compact_infer.py v4c (captured, unchanged), 2 loader workers'},
    'raptor_weight_per_finding': {k: RAPTOR_WEIGHTS[k] for k in RAPTOR_WEIGHTS},
    'raptor_weight_default': DEFAULT_RAPTOR_WEIGHT,
    'recovery': 'A-prime whole-arm global genuine means; union floor(3% N) cap; complete-population average pct ranks',
    'excluded': ['na-full', 'V2', 'extra readers', 'R224 mirror'],
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def raptor_weight_vector(labels=LABELS):
    """Per-finding Raptor-half weight in official target order (R6 gets 1 - w)."""
    require(list(labels) == LABELS, 'OFFICIAL_TARGET_ORDER')
    require(set(RAPTOR_WEIGHTS) <= set(LABELS), 'WEIGHT_TARGETS')
    w = [float(RAPTOR_WEIGHTS.get(name, DEFAULT_RAPTOR_WEIGHT)) for name in labels]
    require(all(0.0 <= x <= 1.0 for x in w), 'WEIGHT_RANGE')
    return w


def recover(old320, r6, crop224, old_missing, r6_missing, crop224_missing, *, fatal_errors=()):
    """A-prime's frozen three-arm fill/rank (knee.public_ws2_recovery), then the per-finding blend.

    The frozen function fills, quantizes the compact slot at '%.9g', round-trips R224 through the
    public float32 CSV and ranks; its 0.5/0.5 blend is replaced column by column with
    w_f * raptor_slot + (1 - w_f) * rank(R6) in the published arithmetic order.
    """
    import numpy as np
    from knee import public_ws2_recovery as REC
    rec = REC.recover(old320, r6, crop224, old_missing, r6_missing, crop224_missing, fatal_errors=fatal_errors)
    w = np.asarray(raptor_weight_vector(), dtype=np.float64)
    raptor = rec['raptor_slot']
    r6_rank = rec['ranks']['r7']
    blend = np.empty_like(raptor)
    for j in range(len(LABELS)):
        blend[:, j] = w[j] * raptor[:, j] + (1.0 - w[j]) * r6_rank[:, j]
    require(np.isfinite(blend).all() and ((blend >= 0) & (blend <= 1)).all(), 'BLEND_UNIT_INTERVAL')
    ranks = {'old320': rec['ranks']['old320'], 'r6': r6_rank, 'crop224': rec['ranks']['crop224']}
    statuses = {'old320': rec['statuses']['old320'], 'r6': rec['statuses']['r7'], 'crop224': rec['statuses']['crop224']}
    means = {'old320': rec['means']['old320'], 'r6': rec['means']['r7'], 'crop224': rec['means']['crop224']}
    return {'recipe': VERSION, 'frozen_fill_rank_recipe': rec['recipe'], 'population': rec['population'], 'cap': rec['cap'],
            'union_recovered': rec['union_recovered'], 'means': means, 'statuses': statuses, 'ranks': ranks,
            'raptor_slot': raptor, 'blend': blend, 'raptor_weights': dict(zip(LABELS, w.tolist())),
            'precision': {'old320': rec['precision']['old320'], 'r6': rec['precision']['r7'],
                          'crop224': rec['precision']['crop224']}}


def recovery_receipt(rec):
    """JSON-safe summary (no arrays, no study identifiers)."""
    import numpy as np
    keep = {k: rec[k] for k in ('recipe', 'frozen_fill_rank_recipe', 'population', 'cap', 'union_recovered', 'means',
                                'raptor_weights', 'precision')}
    keep['recovered_by_arm'] = {a: int(sum(s == 'recovered' for s in rec['statuses'][a])) for a in ARMS}
    keep['blend_sha256'] = hashlib.sha256(np.ascontiguousarray(rec['blend']).tobytes()).hexdigest()
    keep['rank_sha256'] = {a: hashlib.sha256(np.ascontiguousarray(rec['ranks'][a]).tobytes()).hexdigest() for a in ARMS}
    return keep


# ---- flip TTA, independent of compact_infer (same published definition) -------------------------------------------------
def mirror_perm(labels=LABELS, spec=None, bin_names=None):
    """Class permutation P (its own inverse) and, with a state spec and Net.bin_names, the binary permutation."""
    swap = {}
    for a, b in MIRROR_PAIRS:
        swap[a], swap[b] = b, a
    perm = [list(labels).index(swap.get(x, x)) for x in labels]
    bin_perm = None
    if spec is not None:
        for a, b in MIRROR_PAIRS:
            require(spec['categories'][a] == spec['categories'][b], 'FLIP_STATE_CATEGORIES_' + a)
        if bin_names is not None:
            names = [tuple(x) for x in bin_names]
            bin_perm = []
            for binary, label in names:
                partner = (binary, swap.get(label, label))
                require(partner in names, 'FLIP_BINARY_PARTNER')
                bin_perm.append(names.index(partner))
    require(sorted(perm) == list(range(len(labels))) and all(perm[perm[i]] == i for i in range(len(labels))), 'FLIP_INVOLUTION')
    return perm, bin_perm


def mirror_inputs(img, meta):
    """Expected mirrored windows: coronal/axial width flip, sagittal slice reversal with rel_pos negated,
    coronal bilateral views 2 <-> 3. img [B, W, K, S, S], meta [B, W, >= 8] with the plane in column 7."""
    import torch
    require(meta.shape[-1] >= 8 and img.dim() == 5, 'FLIP_INPUT_SCHEMA')
    plane = meta[..., 7].round()
    sag = plane == 0
    out = img.flip(-1).clone()
    if bool(sag.any()):
        out[sag] = img[sag].flip(1)
    m = meta.clone()
    m[..., 2] = torch.where(sag, -meta[..., 2], meta[..., 2])
    view = meta[..., 1]
    m[..., 1] = torch.where(view == 2, torch.full_like(view, 3.), torch.where(view == 3, torch.full_like(view, 2.), view))
    return out, m


def state_readout(state_k, sl, bl):
    """Per class the softmax over its K_c valid slots, then every binary's sigmoid (float32 [B, D])."""
    import torch
    p = sl.float().softmax(-1)
    return torch.cat([p[:, c, :k] for c, k in enumerate(state_k)] + [torch.sigmoid(bl.float())], 1).float()


def deployed_logits(plain, mirror, perm, mcl=LABELS.index('MCL')):
    """Per fold model 0.5 (z + P z_mirror) in logit space, MCL from the plain pass."""
    import torch
    out = []
    for z, zm in zip(plain, mirror):
        t = 0.5 * (z + zm[perm])
        t[mcl] = z[mcl]
        out.append(t)
    return torch.stack(out)


def deployed_state(state_k, plain_aux, mirror_aux, perm, bin_perm, mcl_columns):
    """Fold-mean state read-out with flip TTA (mirror re-indexed by P / bin_perm; MCL block from the plain pass)."""
    import torch
    rows = []
    for (sl, bl), (slm, blm) in zip(plain_aux, mirror_aux):
        st = state_readout(state_k, sl, bl)
        stm = state_readout(state_k, slm[:, perm], blm[:, bin_perm])
        t = 0.5 * (st[0] + stm[0])
        t[mcl_columns] = st[0][mcl_columns]
        rows.append(t)
    return torch.stack(rows).double().mean(0)


def mcl_state_columns(layout):
    return [i for c in layout['columns'] if c['class'] == 'MCL'
            for i in (range(c['start'], c['stop']) if 'categories' in c else [c['column']])]


def reconstruct(observation, fluid_record, fluid_doc, state_doc, spec, labels=LABELS, stack_fallback=()):
    """Deployed study probabilities from the observed raw passes (independent of compact_infer's own arithmetic).

    observation: {'plain': [z [12]] * 4, 'mirror': [...], 'plain_aux': [(sl, bl)] * 4, 'mirror_aux': [...],
    'state_k': [...], 'bin_names': [...]} as float32 CPU tensors.
    """
    import numpy as np
    import torch
    from knee import public_r7_stackers as stackers
    perm, bin_perm = mirror_perm(labels, spec, observation['bin_names'])
    zs = deployed_logits(observation['plain'], observation['mirror'], perm, labels.index('MCL'))
    zmean = zs.mean(0)
    p_model = torch.sigmoid(zmean).double().numpy()
    zf = zs.mean(0).double().numpy()
    layout = stackers.state_layout(spec, labels)
    stv = deployed_state(observation['state_k'], observation['plain_aux'], observation['mirror_aux'], perm, bin_perm,
                         mcl_state_columns(layout)).numpy()
    p, receipt = stackers.apply(p_model, zf, stv, fluid_record, fluid_doc, state_doc, labels)
    for name in stack_fallback:
        j = labels.index(name)
        p[j] = p_model[j]
        receipt[name]['source_fallback'] = True
    return np.asarray(p, dtype=np.float64), {'perm': perm, 'bin_perm': bin_perm, 'logits': zs.numpy().tolist(),
                                             'model_probabilities': p_model.tolist(), 'stackers': receipt}


# ---- state fingerprints that survive compact_infer's in-place `m.enc = torch.compile(enc)` --------------------------------
def normalized_name(name):
    return name.replace('._orig_mod.', '.').replace('_orig_mod.', '')


def module_digest(module):
    """SHA256 over every named parameter and buffer (compile wrapper prefixes removed), name-sorted."""
    h = hashlib.sha256()
    tensors = {'parameter/' + normalized_name(n): t for n, t in module.named_parameters()}
    tensors.update({'buffer/' + normalized_name(n): t for n, t in module.named_buffers()})
    for name, t in sorted(tensors.items()):
        a = t.detach().cpu().contiguous()
        h.update(canonical([name, str(a.dtype), list(a.shape)]))
        h.update(a.reshape(-1).view(__import__('torch').uint8).numpy().tobytes() if a.numel() else b'')
    return h.hexdigest()


def buffers_digest(module):
    h = hashlib.sha256()
    for name, t in sorted((normalized_name(n), b) for n, b in module.named_buffers()):
        a = t.detach().cpu().contiguous()
        h.update(canonical([name, str(a.dtype), list(a.shape)]))
        h.update(a.reshape(-1).view(__import__('torch').uint8).numpy().tobytes() if a.numel() else b'')
    return h.hexdigest()


def constructor_digest(module):
    """Architecture identity: class, parameter/buffer names, dtypes, shapes, trainability."""
    cls = type(module)
    spec = {'class': cls.__module__ + '.' + cls.__qualname__,
            'parameters': [[normalized_name(n), str(p.dtype), list(p.shape), bool(p.requires_grad)]
                           for n, p in module.named_parameters()],
            'buffers': [[normalized_name(n), str(b.dtype), list(b.shape)] for n, b in module.named_buffers()]}
    return digest(spec)


def checksum(module):
    """Cheap exact per-study mutation check (float64 sums on the module's device)."""
    import torch
    s = torch.zeros((), dtype=torch.float64, device=next(module.parameters()).device)
    q = torch.zeros_like(s)
    with torch.no_grad():
        for t in list(module.parameters()) + list(module.buffers()):
            if t.is_floating_point():
                x = t.detach().double()
                s += x.sum()
                q += (x * x).sum()
            else:
                s += t.detach().double().sum()
    return [float(s.item()), float(q.item())]


def finite_unit(p):
    return all(isinstance(x, float) and math.isfinite(x) and 0.0 <= x <= 1.0 for x in p)


# ---- deadline guard (strategy revision 2026-10-17): never abort, always a complete submission ---------------------------
DEADLINE_SECONDS = 8.3 * 3600          # from kernel start T0
RAPTOR_ARMS = (('old320', R384_WEIGHT), ('crop224', R224_WEIGHT))   # frozen; a na-full arm may be added here by config later
STATUSES = ('genuine', 'data_local_failure', 'unavailable')


def recover_guarded(arrays, statuses, raptor_arms=RAPTOR_ARMS):
    """A-prime recovery when every arm is complete (genuine or data-local within the cap); otherwise the public
    fallback: each arm ranked over its own available rows, a Raptor arm missing takes the other Raptor arms' rank
    (weights renormalized), the compact arm missing takes the Raptor half and vice versa, nothing at all gives 0.5.

    arrays: {arm: [n, 12] probabilities}; statuses: {arm: [n] of STATUSES}. Never raises on missing work.
    """
    import numpy as np
    import pandas as pd
    from knee import public_ws2_recovery as REC
    names = [a for a, _ in raptor_arms] + ['r6']
    n = len(statuses['r6'])
    st = {a: np.asarray(statuses[a]) for a in names}
    require(all(len(st[a]) == n and set(st[a].tolist()) <= set(STATUSES) for a in names) and n > 0, 'GUARDED_STATUS_SCHEMA')
    unavailable = {a: st[a] == 'unavailable' for a in names}
    data_local = {a: st[a] == 'data_local_failure' for a in names}
    cap = math.floor(3 * n / 100)
    union = np.logical_or.reduce([data_local[a] for a in names])
    frozen = tuple(raptor_arms) == RAPTOR_ARMS
    if frozen and not any(u.any() for u in unavailable.values()) and int(union.sum()) <= cap:
        rec = recover(arrays['old320'], arrays['r6'], arrays['crop224'], data_local['old320'], data_local['r6'], data_local['crop224'])
        rec.update(mode='aprime_complete_population', fallback_studies=0, fallback={'raptor_only': 0, 'r6_only': 0, 'constant': 0})
        return rec
    from knee.public_ws2_recovery import quantize9, source224_csv_values
    w = np.asarray(raptor_weight_vector(), dtype=np.float64)
    ok = {a: st[a] == 'genuine' for a in names}
    ranks = {}
    for a in names:
        r = np.full((n, 12), np.nan)
        if ok[a].any():
            v = np.asarray(arrays[a], dtype=np.float64)[ok[a]]
            v = quantize9(v) if a == 'r6' else source224_csv_values(v) if a == 'crop224' else v
            r[ok[a]] = pd.DataFrame(v).rank(method='average', pct=True).to_numpy()
        ranks[a] = r
    num = np.zeros((n, 12))
    den = np.zeros((n, 1))
    for a, weight in raptor_arms:
        m = ok[a][:, None]
        num += np.where(m, weight * np.nan_to_num(ranks[a]), 0.)
        den += np.where(m, weight, 0.)
    have_raptor = den[:, 0] > 0
    rr = np.where(have_raptor[:, None], num / np.where(den > 0, den, 1.), np.nan)
    ro = ranks['r6']
    have_r6 = ok['r6']
    blend = np.where(have_raptor[:, None] & have_r6[:, None], w * rr + (1. - w) * ro,
                     np.where(have_raptor[:, None], rr, np.where(have_r6[:, None], ro, .5)))
    require(np.isfinite(blend).all() and ((blend >= 0) & (blend <= 1)).all(), 'BLEND_UNIT_INTERVAL')
    complete = np.logical_and.reduce([ok[a] for a in names])
    return {'recipe': VERSION, 'mode': 'deadline_fallback_public_semantics', 'population': n, 'cap': cap,
            'union_recovered': int((~complete).sum()), 'fallback_studies': int((~complete).sum()),
            'fallback': {'raptor_only': int((have_raptor & ~have_r6).sum()), 'r6_only': int((~have_raptor & have_r6).sum()),
                         'constant': int((~have_raptor & ~have_r6).sum()),
                         'partial_raptor': int((have_raptor & ~np.logical_and.reduce([ok[a] for a, _ in raptor_arms])).sum())},
            'unavailable_by_arm': {a: int(unavailable[a].sum()) for a in names},
            'data_local_by_arm': {a: int(data_local[a].sum()) for a in names},
            'means': {}, 'statuses': {a: st[a].tolist() for a in names}, 'ranks': ranks, 'raptor_slot': rr, 'blend': blend,
            'raptor_weights': dict(zip(LABELS, w.tolist())), 'precision': {'r6': '%.9g', 'crop224': 'float32 CSV', 'old320': 'float64'},
            'frozen_fill_rank_recipe': 'public per-model available-row ranks'}


def guarded_receipt(rec):
    import numpy as np
    keep = {k: rec.get(k) for k in ('recipe', 'mode', 'population', 'cap', 'union_recovered', 'fallback_studies', 'fallback',
                                    'unavailable_by_arm', 'data_local_by_arm', 'raptor_weights')}
    keep['blend_sha256'] = hashlib.sha256(np.ascontiguousarray(rec['blend']).tobytes()).hexdigest()
    return keep
