"""A-prime-w: the sealed A-prime entry with only the final per-finding Raptor share changed.

Final blend per finding f: w_f * raptor_slot + (1 - w_f) * rank(R7), where
raptor_slot = 0.65 rank(R384) + 0.35 rank(R224) exactly as in knee.public_ws2_recovery.
w_f is the Raptor-half share (heliosli blend-mine W_RAPTOR_C semantics). Members,
checkpoints, recovery, ranks, workers, runtime and every guard are the unchanged
A-prime modules: run() delegates to knee.public_ws2_entry.run with only its
recovery reference swapped for the duration of the call.
"""
from pathlib import Path
import types

from knee import public_ws2_entry as WS2
from knee import public_ws2_recovery as REC
from knee.public_s6_qualification import LABELS, require, save

VERSION = 'ws2-a-prime-r224-v2w'
RECIPE = 'ws2w-fixed35-raptor-slot-per-finding-raptor-share-v1'
RAPTOR_DEFAULT = .5
RAPTOR_PER_FINDING = {'Lateral OA': .2, 'PF OA': .6, 'Synovitis': .8, "Baker's": .35}


def raptor_shares():
    require(set(RAPTOR_PER_FINDING) <= set(LABELS), 'WS2W_KNOWN_FINDINGS')
    shares = [float(RAPTOR_PER_FINDING.get(c, RAPTOR_DEFAULT)) for c in LABELS]
    require(len(shares) == 12 and all(0. <= w <= 1. for w in shares), 'WS2W_SHARE_RANGE')
    return shares


def profile():
    shares = raptor_shares()
    return {'version': VERSION, 'recipe': RECIPE, 'base_version': WS2.VERSION, 'base_rank_recipe': REC.RECIPE,
            'raptor_slot': '(1 - %.2f) rank(old320) + %.2f rank(crop224)' % (REC.A224, REC.A224),
            'raptor_default': RAPTOR_DEFAULT, 'raptor_per_finding': dict(RAPTOR_PER_FINDING),
            'labels': list(LABELS), 'raptor_share_by_label': shares}


def reblend(rec):
    """Replace only the final blend of a complete A-prime recovery result."""
    import numpy as np
    w = np.asarray(raptor_shares(), dtype=np.float64)
    slot, r7 = rec['raptor_slot'], rec['ranks']['r7']
    require(slot.shape == r7.shape == (rec['population'], 12), 'WS2W_RANK_SCHEMA')
    blend = w * slot + (1.0 - w) * r7
    require(np.isfinite(blend).all() and ((blend >= 0) & (blend <= 1)).all(), 'WS2W_BLEND_RANGE')
    weights = {c: {'old320': (1.0 - REC.A224) * s, 'crop224': REC.A224 * s, 'r7': 1.0 - s}
               for c, s in zip(LABELS, w.tolist())}
    return {**rec, 'recipe': RECIPE, 'base_recipe': rec['recipe'], 'blend': blend,
            'weights': weights, 'raptor_share_by_label': w.tolist()}


def recover(*args, **kwargs):
    return reblend(REC.recover(*args, **kwargs))


RECOVERY = types.SimpleNamespace(ARMS=REC.ARMS, RECIPE=RECIPE, recover=recover)


def run(root, cfg, out, *, started):
    """Bound image-only A-prime-w entry; rehearsal stays on the unchanged A-prime path."""
    require(cfg.get('mode') == 'entry' and cfg.get('ws2w_blend') == profile(), 'EXACT_WS2W_ENTRY_PROFILE')
    require(WS2.REC is REC, 'WS2W_UNPATCHED_BASE')
    WS2.REC = RECOVERY
    try:
        r = WS2.run(root, cfg, out, started=started)
    finally:
        WS2.REC = REC
    r['ws2w_blend'] = profile()
    save(Path(out) / 'receipt.json', r)
    return r
