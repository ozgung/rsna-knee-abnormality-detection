"""Lane P9 diversity report: per-finding Spearman of the P9 compact and blend against P2's on the frozen-39 rehearsal studies.

Usage: python scripts/lanep9/spearman_p9.py <p2 rehearsal r6flip dir> <p9 rehearsal r6flip dir> <variant> <out json>
Reads only rehearsal ledgers (no labels). Aggregate numbers only are written; study ids are not.
"""
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
import numpy as np
import pandas as pd

from knee import public_r6flip as R6F
from knee import public_s6c as S6C
from knee.public_ws2_recovery import quantize9


def ledger(path):
    with np.load(path, allow_pickle=False) as a:
        return dict(zip(a['study_ids'].tolist(), a['probabilities']))


def arm(root, ids, folders, sub):
    rows = {}
    for f in folders:
        rows.update(ledger(Path(root) / f / sub / 'predictions.npz'))
    return np.array([rows[s] for s in ids])


def own(root, ids):
    seed_p, status = [], []
    for board in range(2):
        with np.load(Path(root) / ('board%d-s6c-production' % board) / 's6c' / 'predictions.npz', allow_pickle=False) as a:
            for sid, z, st in zip(a['study_ids'].tolist(), a['seed_logits'], a['statuses'].tolist()):
                seed_p.append((sid, S6C.sigmoid(z), st))
    by = {sid: (p, st) for sid, p, st in seed_p}
    p = np.array([by[s][0] for s in ids])
    good = np.array([by[s][1] == 'genuine' for s in ids])
    return S6C.mean_rank(p, good), p


def pct(v):
    return pd.DataFrame(np.asarray(v, dtype=np.float64)).rank(method='average', pct=True).to_numpy()


def spearman(a, b):
    return {R6F.LABELS[j]: float(pd.Series(a[:, j]).corr(pd.Series(b[:, j]), method='spearman')) for j in range(12)}


def main(p2_root, p9_root, variant, out_path):
    ids = json.loads((Path(p2_root) / 'assignment.json').read_bytes())['ids']
    ids9 = json.loads((Path(p9_root) / 'assignment.json').read_bytes())['ids']
    assert ids == ids9 and len(ids) == 39, 'frozen-39 order differs'
    old = arm(p2_root, ids, ['board0-old', 'board1-old'], 'old')
    crop = arm(p2_root, ids, ['board0-crop224', 'board1-crop224'], 'crop224')
    r6flip = arm(p2_root, ids, ['board0-r6-production', 'board1-r6-production'], 'r6')
    old9 = arm(p9_root, ids, ['board0-old', 'board1-old'], 'old')
    r6plain = arm(p9_root, ids, ['board0-r6-production', 'board1-r6-production'], 'r6')
    own_rank, member_p = own(p9_root, ids)
    n = len(ids)
    genuine = {k: ['genuine'] * n for k in ('old320', 'crop224', 'r6', 's6c')}
    p2 = R6F.recover_guarded({'old320': old, 'crop224': crop, 'r6': r6flip}, {k: genuine[k] for k in ('old320', 'crop224', 'r6')})
    blend = R6F.VARIANTS[variant]
    names = R6F.blend_arm_names(blend) + ['r6', 's6c']
    arrays = {'old320': old9, 'crop224': crop, 'r6': r6plain, 's6c': own_rank}
    p9 = R6F.recover_guarded({k: arrays[k] for k in names}, {k: genuine[k] for k in names}, blend=blend)
    compact_p2 = pct(quantize9(r6flip))
    compact_p9 = p9['compact_slot']
    members = {S6C.LABELS[j]: None for j in range(12)}
    rep = {'studies': n, 'variant': variant, 'labels': R6F.LABELS,
           'raptor_old320_identical_across_rehearsals': bool(np.allclose(old, old9, atol=1e-6)),
           'p9_compact_vs_p2_r6flip_compact': spearman(compact_p9, compact_p2),
           'r6plain_vs_r6flip': spearman(r6plain, r6flip),
           'own_meanrank_vs_r6flip': spearman(own_rank, r6flip),
           'own_meanrank_vs_r6plain': spearman(own_rank, r6plain),
           'p9_vs_p2_blend': spearman(p9['blend'], p2['blend']),
           'members_vs_each_other': spearman(member_p[:, 0, :], member_p[:, 1, :]) if member_p.shape[1] == 2 else None}
    for k in ('p9_compact_vs_p2_r6flip_compact', 'r6plain_vs_r6flip', 'own_meanrank_vs_r6flip', 'own_meanrank_vs_r6plain', 'p9_vs_p2_blend'):
        vals = list(rep[k].values())
        rep[k + '_mean'], rep[k + '_min'] = float(np.mean(vals)), float(np.min(vals))
    Path(out_path).write_text(json.dumps(rep, indent=1) + '\n')
    for k in ('p9_compact_vs_p2_r6flip_compact', 'own_meanrank_vs_r6flip', 'r6plain_vs_r6flip', 'p9_vs_p2_blend'):
        print('%-36s mean %.4f min %.4f' % (k, rep[k + '_mean'], rep[k + '_min']))


if __name__ == '__main__':
    main(*sys.argv[1:5])
