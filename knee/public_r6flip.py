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


def raptor_weight_vector(labels=LABELS, blend=None):
    """Per-finding Raptor-half weight in official target order (R6 gets 1 - w). blend=None is the frozen v1 table;
    otherwise the sealed cfg['blend'] (see resolve_blend)."""
    require(list(labels) == LABELS, 'OFFICIAL_TARGET_ORDER')
    if blend is not None and (blend.get('own_only') or blend.get('no_raptor')):   # P11 / P12: no Raptor half, weight 0 everywhere
        resolve_blend(blend)
        return [0.0] * len(labels)
    _, weights, default = resolve_blend(blend)
    require(set(weights) <= set(LABELS), 'WEIGHT_TARGETS')
    w = [float(weights.get(name, default)) for name in labels]
    require(all(0.0 <= x <= 1.0 for x in w), 'WEIGHT_RANGE')
    return w


def recover(old320, r6, crop224, old_missing, r6_missing, crop224_missing, *, fatal_errors=(), blend=None):
    """A-prime's frozen three-arm fill/rank (knee.public_ws2_recovery), then the per-finding blend.

    The frozen function fills, quantizes the compact slot at '%.9g', round-trips R224 through the
    public float32 CSV and ranks; its 0.5/0.5 blend is replaced column by column with
    w_f * raptor_slot + (1 - w_f) * rank(R6) in the published arithmetic order.
    """
    import numpy as np
    from knee import public_ws2_recovery as REC
    require(resolve_blend(blend)[0] == RAPTOR_ARMS, 'RECOVER_FROZEN_RAPTOR_ARMS_ONLY')   # other arm sets: recover_arms
    require(resolve_compact(blend)[0] == DEFAULT_COMPACT, 'RECOVER_FROZEN_COMPACT_ONLY')     # P6 dual compact: recover_arms
    rec = REC.recover(old320, r6, crop224, old_missing, r6_missing, crop224_missing, fatal_errors=fatal_errors)
    w = np.asarray(raptor_weight_vector(blend=blend), dtype=np.float64)
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
    keep['recovered_by_arm'] = {a: int(sum(s == 'recovered' for s in rec['statuses'][a])) for a in rec['ranks']}
    keep['blend_sha256'] = hashlib.sha256(np.ascontiguousarray(rec['blend']).tobytes()).hexdigest()
    keep['rank_sha256'] = {a: hashlib.sha256(np.ascontiguousarray(rec['ranks'][a]).tobytes()).hexdigest() for a in rec['ranks']}
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


def plain_state(state_k, plain_aux):
    """Fold-mean state read-out of the plain pass only (compact_infer v4c without --flip-tta)."""
    import torch
    return torch.stack([state_readout(state_k, sl, bl)[0] for sl, bl in plain_aux]).double().mean(0)


def reconstruct(observation, fluid_record, fluid_doc, state_doc, spec, labels=LABELS, stack_fallback=(), flip=True):
    """Deployed study probabilities from the observed raw passes (independent of compact_infer's own arithmetic).

    observation: {'plain': [z [12]] * 4, 'mirror': [...], 'plain_aux': [(sl, bl)] * 4, 'mirror_aux': [...],
    'state_k': [...], 'bin_names': [...]} as float32 CPU tensors. flip=False (P6 compact arms): plain pass only,
    per-fold logits as observed and the fold-mean plain state read-out; no mirror entries are used.
    """
    import numpy as np
    import torch
    from knee import public_r7_stackers as stackers
    layout = stackers.state_layout(spec, labels)
    if flip:
        perm, bin_perm = mirror_perm(labels, spec, observation['bin_names'])
        zs = deployed_logits(observation['plain'], observation['mirror'], perm, labels.index('MCL'))
        stv = deployed_state(observation['state_k'], observation['plain_aux'], observation['mirror_aux'], perm, bin_perm,
                             mcl_state_columns(layout)).numpy()
    else:
        perm = bin_perm = None
        zs = torch.stack(list(observation['plain']))
        stv = plain_state(observation['state_k'], observation['plain_aux']).numpy()
    zmean = zs.mean(0)
    p_model = torch.sigmoid(zmean).double().numpy()
    zf = zs.mean(0).double().numpy()
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
RAPTOR_ARMS = (('old320', R384_WEIGHT), ('crop224', R224_WEIGHT))   # frozen default; variants seal their own in cfg['blend']
STATUSES = ('genuine', 'data_local_failure', 'unavailable')
BLEND_ARM_NAMES = ('old320', 'crop224', 'nafull')    # stage order: R384 two-view, R224 one-view, na-full K62 one-view
ALL_ARMS = ('old320', 'r6', 'crop224', 'nafull', 'r7', 's6c')
COMPACT_ARM_NAMES = ('r6', 'r7', 's6c')        # stage order: R6, then R7 (P6) or our own s6c members (P9) last, under the guard
DEFAULT_COMPACT = (('r6', 1.0),)               # P2 and P3-P5: rank(R6 flip) is the whole compact slot
DEFAULT_COMPACT_FLIP = True
RAPTOR_WEIGHTS_V2 = {'Lateral OA': .15, 'PF OA': .65, 'Synovitis': .85, "Baker's": .30}   # heliosli V2 table, others 0.5
# Entry variants. None of these is the default: no variant means no cfg['blend'] key and the frozen v1 arithmetic.
VARIANTS = {
    'p3': {'raptor_arms': [['old320', .52], ['crop224', .28], ['nafull', .20]], 'weights': dict(RAPTOR_WEIGHTS),
           'default_weight': DEFAULT_RAPTOR_WEIGHT},
    'p4': {'raptor_arms': [['old320', .52], ['crop224', .28], ['nafull', .20]], 'weights': dict(RAPTOR_WEIGHTS_V2),
           'default_weight': DEFAULT_RAPTOR_WEIGHT},
    'p5': {'raptor_arms': [['old320', .65], ['nafull', .35]], 'weights': dict(RAPTOR_WEIGHTS),
           'default_weight': DEFAULT_RAPTOR_WEIGHT},
    # P6 dual-compact (final-push plan 2026-10-19, Lane A): P2's Raptor half and per-finding weights; the compact slot is
    # 1/2 rank(R6 plain) + 1/2 rank(R7 plain), no flip TTA, weights fixed before any P6 output exists. R7 runs last.
    'p6': {'raptor_arms': [['old320', R384_WEIGHT], ['crop224', R224_WEIGHT]], 'weights': dict(RAPTOR_WEIGHTS),
           'default_weight': DEFAULT_RAPTOR_WEIGHT, 'compact_arms': [['r6', .5], ['r7', .5]], 'compact_flip': False},
    # P6-prime: P6 without the R224 stage (only if the P6 rehearsal point exceeds 26,000 s).
    'p6p': {'raptor_arms': [['old320', 1.0]], 'weights': dict(RAPTOR_WEIGHTS),
            'default_weight': DEFAULT_RAPTOR_WEIGHT, 'compact_arms': [['r6', .5], ['r7', .5]], 'compact_flip': False},
    # P6s (2026-10-20, after Lane B's out-of-fold win): P6 with one decode per study -- R7 scored in the R6 plain worker on
    # R6's rendered windows (compact_shared). Same arithmetic as P6; p6sp is P6s without R224.
    'p6s': {'raptor_arms': [['old320', R384_WEIGHT], ['crop224', R224_WEIGHT]], 'weights': dict(RAPTOR_WEIGHTS),
            'default_weight': DEFAULT_RAPTOR_WEIGHT, 'compact_arms': [['r6', .5], ['r7', .5]], 'compact_flip': False,
            'compact_shared': True},
    'p6sp': {'raptor_arms': [['old320', 1.0]], 'weights': dict(RAPTOR_WEIGHTS),
             'default_weight': DEFAULT_RAPTOR_WEIGHT, 'compact_arms': [['r6', .5], ['r7', .5]], 'compact_flip': False,
             'compact_shared': True},
    # P7-prime (Lane C): P2's frozen Raptor half and v1 weights; the compact slot is our S6 seed ensemble (knee.public_s6c)
    'p7': {'raptor_arms': [['old320', R384_WEIGHT], ['crop224', R224_WEIGHT]], 'weights': dict(RAPTOR_WEIGHTS),
           'default_weight': DEFAULT_RAPTOR_WEIGHT},
    # P9 (gap assessment 2026-10-20 §3.1, Lane P9): P2's Raptor arm R384 only (R224 dropped for runtime, as P6s-prime) and v1
    # weights; the compact slot is 1/2 rank(R6 plain) + 1/2 rank(own S6 members' mean rank, knee.public_s6c), weights fixed
    # before any P9 output exists. The own stage runs last under the guard; trimmed studies keep rank(R6 plain). p9r adds R224.
    'p9': {'raptor_arms': [['old320', 1.0]], 'weights': dict(RAPTOR_WEIGHTS), 'default_weight': DEFAULT_RAPTOR_WEIGHT,
           'compact_arms': [['r6', .5], ['s6c', .5]], 'compact_flip': False},
    'p9r': {'raptor_arms': [['old320', R384_WEIGHT], ['crop224', R224_WEIGHT]], 'weights': dict(RAPTOR_WEIGHTS),
            'default_weight': DEFAULT_RAPTOR_WEIGHT, 'compact_arms': [['r6', .5], ['s6c', .5]], 'compact_flip': False},
    # P9s: P9's arithmetic with one decode per study -- the members rendered at 320 px from R6's native cache and scored inside the
    # R6 plain worker (knee.public_s6c_shared); no separate own stage. p9sr adds R224.
    'p9s': {'raptor_arms': [['old320', 1.0]], 'weights': dict(RAPTOR_WEIGHTS), 'default_weight': DEFAULT_RAPTOR_WEIGHT,
            'compact_arms': [['r6', .5], ['s6c', .5]], 'compact_flip': False, 'own_shared': True},
    'p9sr': {'raptor_arms': [['old320', R384_WEIGHT], ['crop224', R224_WEIGHT]], 'weights': dict(RAPTOR_WEIGHTS),
             'default_weight': DEFAULT_RAPTOR_WEIGHT, 'compact_arms': [['r6', .5], ['s6c', .5]], 'compact_flip': False, 'own_shared': True},
}


def own_dual_blend(w, r224=False, stride=1):
    """P9s family (parent, 2026-10-20 22:55): compact slot = w rank(own members) + (1 - w) rank(R6 plain), shared decode, the
    members at eval stride 1 or 2 (stride 2 halves their cost; the stride-1 parity and reference checks stay in the rehearsal)."""
    b = {'raptor_arms': [['old320', R384_WEIGHT], ['crop224', R224_WEIGHT]] if r224 else [['old320', 1.0]], 'weights': dict(RAPTOR_WEIGHTS),
         'default_weight': DEFAULT_RAPTOR_WEIGHT, 'compact_arms': [['r6', round(1.0 - w, 2)], ['s6c', round(w, 2)]], 'compact_flip': False,
         'own_shared': True}
    if stride != 1:
        b['own_stride'] = int(stride)
    return b


def own_compact_blend(raptor='both', aggregation='mean_logit'):
    """P10 / P11 family (parent, 2026-10-20 22:55): P7's machinery (the own members replace the R6 bundle in the compact slot)
    with a sealed aggregation and, for P11, no Raptor half at all (the final column is the ranked own compact alone)."""
    arms = {'both': [['old320', R384_WEIGHT], ['crop224', R224_WEIGHT]], 'r384': [['old320', 1.0]], 'none': []}[raptor]
    b = {'raptor_arms': arms, 'weights': dict(RAPTOR_WEIGHTS), 'default_weight': DEFAULT_RAPTOR_WEIGHT, 'own_aggregation': aggregation}
    if raptor == 'none':
        b['own_only'] = True
    return b


OWN_WEIGHTS = (50, 60, 70)        # the parent pre-registers w from the four-fold grid; 0.5 to 0.7 expected
# 'p9s2...' = the second member set (sealed separately from P9s v1's two members): p9s2[r][w60|w70][s2]
for _w in OWN_WEIGHTS:
    for _r224 in (False, True):
        for _stride in (1, 2):
            VARIANTS['p9s2' + ('r' if _r224 else '') + ('' if _w == 50 else 'w%d' % _w) + ('' if _stride == 1 else 's2')] = \
                own_dual_blend(_w / 100.0, _r224, _stride)
# P10: Raptor half as P2 (p10p: R384 only) + own-only compact (mean logit; *mr = mean rank); P11: own-only compact alone.
VARIANTS.update({'p10': own_compact_blend('both', 'mean_logit'), 'p10mr': own_compact_blend('both', 'mean_rank'),
                 'p10p': own_compact_blend('r384', 'mean_logit'), 'p10pmr': own_compact_blend('r384', 'mean_rank'),
                 'p11': own_compact_blend('none', 'mean_logit'), 'p11mr': own_compact_blend('none', 'mean_rank')})
COMPACT_VARIANTS = {'p7': 's6c', **{k: 's6c' for k in VARIANTS if k.startswith('p10') or k.startswith('p11')}}   # compact slot not the R6 bundle


def no_raptor_blend(flip, stride=2, w=0.5):
    """P12 (parent, 2026-10-21 12:55 UTC): no Raptor stage; compact = (1 - w) rank(R6 [flip|plain]) + w rank(own mean logit), both from
    the shared decode, members at eval stride 2, cut first at the guard."""
    b = {'raptor_arms': [], 'weights': dict(RAPTOR_WEIGHTS), 'default_weight': DEFAULT_RAPTOR_WEIGHT,
         'compact_arms': [['r6', round(1.0 - w, 2)], ['s6c', round(w, 2)]], 'compact_flip': bool(flip), 'own_shared': True,
         'own_aggregation': 'mean_logit', 'no_raptor': True}
    if stride != 1:
        b['own_stride'] = int(stride)
    return b


# p12 = R6 plain + the P10 eight; p12f = R6 flip + the P10 eight; p12fn = R6 flip + a sealed subset (as many as fit the gate);
# p12v5 / p12fv5 = the Oct 22 member swap (v5 members), sealed when the exports exist
VARIANTS.update({'p12': no_raptor_blend(False), 'p12f': no_raptor_blend(True), 'p12fn': no_raptor_blend(True),
                 'p12v5': no_raptor_blend(False), 'p12fv5': no_raptor_blend(True),
                 'p11v5': own_compact_blend('none', 'mean_logit'), 'p9s2s2b': own_dual_blend(0.5, False, 2)})
COMPACT_VARIANTS['p11v5'] = 's6c'
# Oct 22 wave 1 E1 (parent 21:45): P10 with R224 on the E1 list; p10pv5 = the R384-only fallback
VARIANTS.update({'p10v5': own_compact_blend('both', 'mean_logit'), 'p10pv5': own_compact_blend('r384', 'mean_logit')})
COMPACT_VARIANTS.update({'p10v5': 's6c', 'p10pv5': 's6c'})


def p2_own_blend(w, stride):
    """P13 (parent, pre-registered 2026-10-21 22:00 UTC): P2 exactly -- R384 0.65 + R224 0.35, v1 per-finding weights, R6 FLIP -- with
    the compact slot (1 - w) rank(R6 flip) + w rank(own mean logit); members from the shared decode inside the R6 worker."""
    return {'raptor_arms': [['old320', R384_WEIGHT], ['crop224', R224_WEIGHT]], 'weights': dict(RAPTOR_WEIGHTS),
            'default_weight': DEFAULT_RAPTOR_WEIGHT, 'compact_arms': [['r6', round(1.0 - w, 2)], ['s6c', round(w, 2)]], 'compact_flip': True,
            'own_shared': True, 'own_stride': int(stride), 'own_aggregation': 'mean_logit'}


# S4 (slot 5): P10 with R224 on the S4 list, members in the separate worker at stride 2 (p10s4) or 3 (p10s4s)
VARIANTS.update({'p10s4': {**own_compact_blend('both', 'mean_logit'), 'own_stride': 2},
                 'p10s4s': {**own_compact_blend('both', 'mean_logit'), 'own_stride': 3}})
COMPACT_VARIANTS.update({'p10s4': 's6c', 'p10s4s': 's6c'})
# p13a: k=3 at stride 2; p13b: k=3 at stride 3; p13c: k=2 at stride 3; *w65: slot 4 (0.65 own / 0.35 R6 flip), same members/stride
VARIANTS.update({'p13a': p2_own_blend(.5, 2), 'p13b': p2_own_blend(.5, 3), 'p13c': p2_own_blend(.5, 3),
                 'p13aw65': p2_own_blend(.65, 2), 'p13bw65': p2_own_blend(.65, 3), 'p13cw65': p2_own_blend(.65, 3)})
PRECISION = {'old320': 'float64', 'r6': '%.9g', 'crop224': 'float32 CSV', 'nafull': 'float64', 'r7': '%.9g', 's6c': '%.9g'}
BLEND_KEYS = {'raptor_arms', 'weights', 'default_weight'}
COMPACT_KEYS = {'compact_arms', 'compact_flip'}


def resolve_blend(blend=None):
    """(raptor arms, per-finding weights, default weight) from the sealed cfg['blend']; None is the frozen default."""
    if blend is None:
        return RAPTOR_ARMS, dict(RAPTOR_WEIGHTS), DEFAULT_RAPTOR_WEIGHT
    require(set(blend) in (BLEND_KEYS, BLEND_KEYS | COMPACT_KEYS, BLEND_KEYS | COMPACT_KEYS | {'compact_shared'},
                           BLEND_KEYS | COMPACT_KEYS | {'own_shared'}, BLEND_KEYS | COMPACT_KEYS | {'own_shared', 'own_stride'},
                           BLEND_KEYS | {'own_aggregation'}, BLEND_KEYS | {'own_aggregation', 'own_only'},
                           BLEND_KEYS | {'own_aggregation', 'own_stride'},
                           BLEND_KEYS | COMPACT_KEYS | {'own_shared', 'own_stride', 'own_aggregation', 'no_raptor'},
                           BLEND_KEYS | COMPACT_KEYS | {'own_shared', 'own_stride', 'own_aggregation'}), 'BLEND_SCHEMA')
    arms = tuple((str(a), float(w)) for a, w in blend['raptor_arms'])
    names = [a for a, _ in arms]
    if blend.get('own_only'):
        require(blend['own_only'] is True and not arms and 'own_aggregation' in blend, 'OWN_ONLY_NO_RAPTOR_ARMS')
    elif blend.get('no_raptor'):                    # P12: no Raptor stage; the compact slot (R6 + own) is the whole column
        require(blend['no_raptor'] is True and not arms and 'compact_arms' in blend and blend.get('own_shared') is True, 'NO_RAPTOR_NEEDS_SHARED_COMPACT')
    else:
        require(names and 'old320' in names and names == [x for x in BLEND_ARM_NAMES if x in names], 'BLEND_ARM_ORDER')
        require(all(0.0 < w <= 1.0 for _, w in arms) and abs(sum(w for _, w in arms) - 1.0) <= 1e-12, 'BLEND_ARM_WEIGHTS_SUM_TO_ONE')
    if 'own_aggregation' in blend:
        require(blend['own_aggregation'] in ('mean_logit', 'mean_rank'), 'OWN_AGGREGATION')
    if 'own_stride' in blend:
        require((blend.get('own_shared') is True or 'own_aggregation' in blend) and blend['own_stride'] in (1, 2, 3), 'OWN_STRIDE')
    weights = {str(k): float(v) for k, v in blend['weights'].items()}
    default = float(blend['default_weight'])
    require(set(weights) <= set(LABELS) and all(0.0 <= v <= 1.0 for v in weights.values()) and 0.0 <= default <= 1.0, 'BLEND_WEIGHTS')
    return arms, weights, default


def blend_arm_names(blend=None):
    """Raptor arm names enabled by the sealed blend, in stage order."""
    return [a for a, _ in resolve_blend(blend)[0]]


def resolve_compact(blend=None):
    """(compact arms with weights, flip TTA) from the sealed cfg['blend']; absent keys are P2's rank(R6 flip)."""
    resolve_blend(blend)
    if blend is None or 'compact_arms' not in blend:
        return DEFAULT_COMPACT, DEFAULT_COMPACT_FLIP
    arms = tuple((str(a), float(w)) for a, w in blend['compact_arms'])
    names = [a for a, _ in arms]
    require(names and names[0] == 'r6' and names == [x for x in COMPACT_ARM_NAMES if x in names], 'COMPACT_ARM_ORDER')
    require(all(0.0 < w <= 1.0 for _, w in arms) and abs(sum(w for _, w in arms) - 1.0) <= 1e-12, 'COMPACT_ARM_WEIGHTS_SUM_TO_ONE')
    require(isinstance(blend['compact_flip'], bool), 'COMPACT_FLIP_FLAG')
    return arms, blend['compact_flip']


def compact_shared(blend=None):
    """P6s: R7 scored inside the R6 plain worker on the same rendered windows (one decode); requires the r6+r7 plain pair."""
    if blend is None or not blend.get('compact_shared'):
        return False
    require(blend['compact_shared'] is True and compact_arm_names(blend) == ['r6', 'r7'] and resolve_compact(blend)[1] is False,
            'COMPACT_SHARED_NEEDS_R6_R7_PLAIN')
    return True


def compact_arm_names(blend=None):
    """Compact arm names enabled by the sealed blend, in stage order (['r6'] unless a variant seals more)."""
    return [a for a, _ in resolve_compact(blend)[0]]


def own_shared(blend=None):
    """P9s: the own members scored inside the R6 plain worker from one decode (no separate s6c stage)."""
    if blend is None or not blend.get('own_shared'):
        return False
    require(blend['own_shared'] is True and own_compact_arm(blend), 'OWN_SHARED_NEEDS_OWN_COMPACT_ARM')
    return True


def own_stride(blend=None):
    """Eval stride of the own members in the shared decode (1 unless the variant seals 2)."""
    resolve_blend(blend)
    return int((blend or {}).get('own_stride', 1))


def own_aggregation(blend=None):
    """P10/P11: how the own members are combined in the compact slot ('mean_rank' is P7-prime's)."""
    resolve_blend(blend)
    return (blend or {}).get('own_aggregation', 'mean_rank')


def own_only(blend=None):
    """P11: no Raptor half; the final column is the ranked own compact alone."""
    resolve_blend(blend)
    return bool((blend or {}).get('own_only'))


def own_compact_arm(blend=None):
    """P9: our S6 members (knee.public_s6c) run as a second compact arm next to the heliosli R6 bundle (plain), last under the
    guard. Distinct from P7, where the s6c block replaces the R6 bundle in the 'r6' slot."""
    names = compact_arm_names(blend)
    if 's6c' not in names:
        return False
    # R6 flip TTA is allowed only with the shared decode (P12): the members get their own 320 px windows, R6 its plain + mirror passes
    require(names == ['r6', 's6c'] and (resolve_compact(blend)[1] is False or blend.get('own_shared') is True) and not compact_shared(blend),
            'OWN_COMPACT_NEEDS_R6_PLAIN_PAIR')
    return True


def recover_arms(arrays, missing, blend=None, *, fatal_errors=()):
    """Complete-population recovery for sealed variants (A-prime's fill/rank rules on any subset of the Raptor arms).

    Per arm: genuine rows only, global genuine mean for classified data-local rows (R6 values and mean at '%.9g'), union cap
    floor(3% N), complete-population average percentile ranks (R224 through the public float32 CSV round trip), raptor slot =
    sum of arm weights * ranks in stage order, then w_f * slot + (1 - w_f) * compact slot per finding, where the compact slot
    is rank(R6) by default or, for a sealed compact_arms list (P6), the weighted sum of the compact arms' ranks (each '%.9g').
    """
    import numpy as np
    import pandas as pd
    from knee.public_ws2_recovery import quantize9, source224_csv_values
    require(not fatal_errors, 'FATAL_ERRORS_REQUIRE_ABORT')
    arms = resolve_blend(blend)[0]
    compact = resolve_compact(blend)[0]
    compact_names = [a for a, _ in compact]
    names = [a for a, _ in arms] + compact_names
    src = {a: np.asarray(arrays[a]) for a in names}
    msk = {a: np.asarray(missing[a]) for a in names}
    n = len(src['r6'])
    require(n > 0 and all(src[a].shape == (n, 12) for a in names), 'ARM_SCHEMA')
    require(all(msk[a].shape == (n,) and msk[a].dtype == bool for a in names), 'EXPLICIT_ROW_MASK_REQUIRED')
    cap = math.floor(3 * n / 100)
    union = int(np.logical_or.reduce([msk[a] for a in names]).sum())
    require(union <= cap, 'UNION_RECOVERY_CAP')
    means, statuses, ranks = {}, {}, {}
    for a in names:
        genuine = np.asarray(src[a][~msk[a]], dtype=np.float64)
        require(len(genuine) > 0, 'GENUINE_SUCCESS_REQUIRED_' + a)
        require(np.isfinite(genuine).all() and ((genuine >= 0) & (genuine <= 1)).all(), 'GENUINE_PROBABILITIES_' + a)
        if a in compact_names:
            genuine = quantize9(genuine)
        mean = genuine.mean(axis=0, dtype=np.float64)
        if a in compact_names:
            mean = quantize9(mean[None])[0]
        filled = np.empty((n, 12), dtype=np.float64)
        filled[~msk[a]], filled[msk[a]] = genuine, mean
        means[a] = mean.tolist()
        statuses[a] = np.where(msk[a], 'recovered', 'genuine').tolist()
        ranks[a] = pd.DataFrame(source224_csv_values(filled) if a == 'crop224' else filled).rank(method='average', pct=True).to_numpy()
    slot = None
    for a, weight in arms:
        term = weight * ranks[a]
        slot = term if slot is None else slot + term
    if compact == DEFAULT_COMPACT:
        cslot = ranks['r6']
    else:
        cslot = None
        for a, weight in compact:
            term = weight * ranks[a]
            cslot = term if cslot is None else cslot + term
    w = np.asarray(raptor_weight_vector(blend=blend), dtype=np.float64)
    if slot is None:                          # P11: no Raptor half (w is all zeros); the compact slot is the whole column
        slot = np.zeros_like(cslot)
    out = np.empty_like(slot)
    for j in range(len(LABELS)):
        out[:, j] = w[j] * slot[:, j] + (1.0 - w[j]) * cslot[:, j]
    require(np.isfinite(out).all() and ((out >= 0) & (out <= 1)).all(), 'BLEND_UNIT_INTERVAL')
    extra = {} if compact == DEFAULT_COMPACT else {'compact_slot': cslot, 'compact_weights': dict(compact)}
    return {'recipe': VERSION, 'frozen_fill_rank_recipe': 'A-prime fill/rank rules on the sealed arm set', 'population': n, 'cap': cap,
            'union_recovered': union, 'means': means, 'statuses': statuses, 'ranks': ranks, 'raptor_slot': slot, 'blend': out,
            'raptor_weights': dict(zip(LABELS, w.tolist())), 'precision': {a: PRECISION[a] for a in names}, **extra}


def recover_guarded(arrays, statuses, raptor_arms=None, blend=None):
    """A-prime recovery when every arm is complete (genuine or data-local within the cap); otherwise the public
    fallback: each arm ranked over its own available rows, a Raptor arm missing takes the other Raptor arms' rank
    (weights renormalized), the compact arm missing takes the Raptor half and vice versa, nothing at all gives 0.5.
    With sealed compact_arms (P6) a compact arm missing takes the other compact arms' rank the same way (a study the guard
    cut from R7 gets rank(R6) as its compact slot).

    arrays: {arm: [n, 12] probabilities}; statuses: {arm: [n] of STATUSES}. Never raises on missing work.
    The Raptor arms and per-finding weights come from the sealed cfg['blend'] (None: the frozen default); an explicit
    raptor_arms overrides only the arm list and always takes the fallback arithmetic unless it is the frozen one.
    """
    import numpy as np
    import pandas as pd
    arms = tuple(tuple(x) for x in raptor_arms) if raptor_arms is not None else resolve_blend(blend)[0]
    compact = resolve_compact(blend)[0]
    compact_names = [a for a, _ in compact]
    names = [a for a, _ in arms] + compact_names
    n = len(statuses['r6'])
    st = {a: np.asarray(statuses[a]) for a in names}
    require(all(len(st[a]) == n and set(st[a].tolist()) <= set(STATUSES) for a in names) and n > 0, 'GUARDED_STATUS_SCHEMA')
    unavailable = {a: st[a] == 'unavailable' for a in names}
    data_local = {a: st[a] == 'data_local_failure' for a in names}
    cap = math.floor(3 * n / 100)
    union = np.logical_or.reduce([data_local[a] for a in names])
    frozen = arms == RAPTOR_ARMS and compact == DEFAULT_COMPACT
    complete = not any(u.any() for u in unavailable.values()) and int(union.sum()) <= cap
    if complete and (frozen or raptor_arms is None):
        if frozen:
            rec = recover(arrays['old320'], arrays['r6'], arrays['crop224'], data_local['old320'], data_local['r6'], data_local['crop224'],
                          blend=blend)
            rec.update(mode='aprime_complete_population')
        else:
            rec = recover_arms(arrays, data_local, blend)
            rec.update(mode='variant_complete_population')
        rec.update(fallback_studies=0, fallback={'raptor_only': 0, 'r6_only': 0, 'constant': 0})
        return rec
    from knee.public_ws2_recovery import quantize9, source224_csv_values
    w = np.asarray(raptor_weight_vector(blend=blend), dtype=np.float64)
    ok = {a: st[a] == 'genuine' for a in names}
    ranks = {}
    for a in names:
        r = np.full((n, 12), np.nan)
        if ok[a].any():
            v = np.asarray(arrays[a], dtype=np.float64)[ok[a]]
            v = quantize9(v) if a in compact_names else source224_csv_values(v) if a == 'crop224' else v
            r[ok[a]] = pd.DataFrame(v).rank(method='average', pct=True).to_numpy()
        ranks[a] = r
    num = np.zeros((n, 12))
    den = np.zeros((n, 1))
    for a, weight in arms:
        m = ok[a][:, None]
        num += np.where(m, weight * np.nan_to_num(ranks[a]), 0.)
        den += np.where(m, weight, 0.)
    have_raptor = den[:, 0] > 0
    rr = np.where(have_raptor[:, None], num / np.where(den > 0, den, 1.), np.nan)
    if compact == DEFAULT_COMPACT:
        ro = ranks['r6']
        have_r6 = ok['r6']
        extra = {}
    else:
        cnum = np.zeros((n, 12))
        cden = np.zeros((n, 1))
        for a, weight in compact:
            m = ok[a][:, None]
            cnum += np.where(m, weight * np.nan_to_num(ranks[a]), 0.)
            cden += np.where(m, weight, 0.)
        have_r6 = cden[:, 0] > 0                  # "r6" here means any compact arm (fallback count keys kept)
        ro = np.where(have_r6[:, None], cnum / np.where(cden > 0, cden, 1.), np.nan)
        partial = have_r6 & ~np.logical_and.reduce([ok[a] for a in compact_names])
        extra = {'compact_slot': ro, 'compact_weights': dict(compact),
                 'compact_fallback': {'partial_compact': int(partial.sum()),
                                      **{'%s_only' % a: int((ok[a] & ~np.logical_or.reduce([ok[b] for b in compact_names if b != a]))
                                                            .sum()) for a in compact_names}}}
    out = np.where(have_raptor[:, None] & have_r6[:, None], w * rr + (1. - w) * ro,
                   np.where(have_raptor[:, None], rr, np.where(have_r6[:, None], ro, .5)))
    require(np.isfinite(out).all() and ((out >= 0) & (out <= 1)).all(), 'BLEND_UNIT_INTERVAL')
    done = np.logical_and.reduce([ok[a] for a in names])
    return {'recipe': VERSION, 'mode': 'deadline_fallback_public_semantics', 'population': n, 'cap': cap,
            'union_recovered': int((~done).sum()), 'fallback_studies': int((~done).sum()),
            'fallback': {'raptor_only': int((have_raptor & ~have_r6).sum()), 'r6_only': int((~have_raptor & have_r6).sum()),
                         'constant': int((~have_raptor & ~have_r6).sum()),
                         'partial_raptor': int((have_raptor & ~np.logical_and.reduce([ok[a] for a, _ in arms])).sum())},
            'unavailable_by_arm': {a: int(unavailable[a].sum()) for a in names},
            'data_local_by_arm': {a: int(data_local[a].sum()) for a in names},
            'means': {}, 'statuses': {a: st[a].tolist() for a in names}, 'ranks': ranks, 'raptor_slot': rr, 'blend': out,
            'raptor_weights': dict(zip(LABELS, w.tolist())), 'precision': {a: PRECISION[a] for a in names},
            'frozen_fill_rank_recipe': 'public per-model available-row ranks', **extra}


def guarded_receipt(rec):
    import numpy as np
    keep = {k: rec.get(k) for k in ('recipe', 'mode', 'population', 'cap', 'union_recovered', 'fallback_studies', 'fallback',
                                    'unavailable_by_arm', 'data_local_by_arm', 'raptor_weights')}
    for k in ('compact_weights', 'compact_fallback'):         # P6 only; absent keys keep the P2-P5 receipts unchanged
        if k in rec:
            keep[k] = rec[k]
    keep['blend_sha256'] = hashlib.sha256(np.ascontiguousarray(rec['blend']).tobytes()).hexdigest()
    return keep
