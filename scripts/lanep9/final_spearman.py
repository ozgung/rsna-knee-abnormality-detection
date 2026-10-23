"""Final-selection input: pairwise per-finding Spearman of the official blended columns on the frozen rehearsal studies.

Usage: python scripts/lanep9/final_spearman.py <out json>
Every entry is replayed from its own rehearsal ledgers with its sealed blend (knee.public_r6flip_entry.arm_table +
knee.public_r6flip.recover_guarded, the rehearsal's own replay path); C1 w0.65 is replayed from p13n's ledgers with the w 0.65
blend (its entry reused p13n's rehearsal). No labels, no gold-58. Aggregate numbers only; no study ids are written.
"""
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
import numpy as np
import pandas as pd

from knee import public_r6flip as R6F
from knee import public_r6flip_entry as ENTRY

K = Path('/Users/ozgun/Dev/rsna-knee-abnormality-detection/outputs/kaggle')
ENTRIES = [  # (label, submission, rehearsal result dir, variant whose sealed blend is replayed)
    ('P2', 57340072, 'r6flip-rehearsal-20261017-v4-out', None),
    ('P10p', 57472040, 'r6flip-p10p-rehearsal-20261021-out', 'p10p'),
    ('P9 v2', 57464435, 'r6flip-p9s2s2-rehearsal-20261021-out', 'p9s2s2'),
    ('P11', 57471109, 'r6flip-p11-rehearsal-20261021-out', 'p11'),
    ('E1', 57500952, 'r6flip-p10v5-rehearsal-20261022-out', 'p10v5'),
    ('S4', 57504011, 'r6flip-p10s4-rehearsal-20261022-out', 'p10s4'),
    ('C1', 57506666, 'r6flip-p13n-rehearsal-20261022-out', 'p13n'),
    ('C1 w0.65', 57506593, 'r6flip-p13n-rehearsal-20261022-out', 'p13nw65'),
    ('p13nk2', None, 'r6flip-p13nk2-rehearsal-20261022-out', 'p13nk2'),
]


def replay(result_dir, variant):
    root = K / result_dir / 'r6flip'
    r = json.loads((root / 'receipt.json').read_bytes())
    cfg = json.loads((K / result_dir / 'r6flip-project' / 'runtime-config.json').read_bytes())
    ids = json.loads((root / 'assignment.json').read_bytes())['ids']
    blend = cfg.get('blend')
    if variant is not None and blend != R6F.VARIANTS[variant]:
        R6F.require(variant == 'p13nw65' and blend == R6F.VARIANTS['p13n'], 'UNEXPECTED_BLEND_' + variant)
    arrays, statuses, _ = ENTRY.arm_table(ids, r['workers'], root, entry=False, blend=blend, compact=cfg.get('s6c'))
    use = R6F.VARIANTS[variant] if variant is not None else blend
    rec = R6F.recover_guarded(arrays, statuses, blend=use)
    if use == blend:                                       # the saved receipt must replay exactly (same check as the rehearsal gate)
        R6F.require(R6F.guarded_receipt(rec) == r['recovery'], 'REPLAY_RECOVERY_' + result_dir)
    return ids, rec['blend']


def main(out):
    cols, ids0 = {}, None
    for label, sub, d, v in ENTRIES:
        ids, b = replay(d, v)
        if ids0 is None:
            ids0 = ids
        R6F.require(ids == ids0, 'SAME_FROZEN_STUDIES')
        cols[label] = np.asarray(b)
    labels = [e[0] for e in ENTRIES]
    per, macro = {}, {}
    for a in labels:
        for b in labels:
            v = [float(pd.Series(cols[a][:, j]).corr(pd.Series(cols[b][:, j]), method='spearman')) for j in range(12)]
            per['%s | %s' % (a, b)] = dict(zip(R6F.LABELS, v))
            macro.setdefault(a, {})[b] = float(np.mean(v))
    doc = {'n_studies': len(ids0), 'study_set': 'frozen rehearsal set (36 training probes + 3 visible test studies), identical across all rehearsals',
           'entries': [{'label': l, 'submission': s, 'rehearsal_result': d, 'replayed_blend': v or 'P2 (frozen default)',
                        'replay': 'from p13n rehearsal ledgers with the w 0.65 blend' if v == 'p13nw65' else 'own rehearsal ledgers; saved recovery receipt reproduced'}
                       for l, s, d, v in ENTRIES],
           'macro_mean_spearman': macro, 'per_finding_spearman': per, 'labels': R6F.LABELS, 'gold58_read': False}
    Path(out).write_text(json.dumps(doc, indent=1) + '\n')
    print('n =', len(ids0))
    print('%-9s' % '' + ''.join('%-9s' % l[:8] for l in labels))
    for a in labels:
        print('%-9s' % a[:8] + ''.join('%-9.3f' % macro[a][b] for b in labels))


if __name__ == '__main__':
    main(sys.argv[1])
