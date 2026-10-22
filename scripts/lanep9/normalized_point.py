"""Speed-normalized rehearsal point for the P13 family (parent amendment 2026-10-22 01:15 UTC).

Usage: python scripts/lanep9/normalized_point.py <p2 rehearsal receipt.json> <rehearsal dir (…-out/r6flip)> [--out json]

Public arms at P2's rehearsal rate (R384 production and fixed cost, R224 if present, R6 flip per window and its non-sentinel fixed
cost), the member cost at its measured value divided by this rehearsal's R6-flip slowdown ratio, and the rehearsal-only overheads
(sentinel passes and the stride-1 check) as measured. Point = multiplier 1, imbalance 1 (knee.public_r6flip_entry.projection's
point scenario). Prints the per-stage rates, both slowdown ratios and the result.
"""
import json
from pathlib import Path
import sys


def member_board_seconds(root, board):
    x = json.loads((Path(root) / ('board%d-s6c-production' % board) / 's6c' / 'receipt.json').read_bytes())
    return sum(sum(float(t) for t in row['member_seconds']) for row in x['records'] if row['status'] == 'genuine')


def main(p2_path, root, out=None):
    p2 = json.loads(Path(p2_path).read_bytes())
    r = json.loads((Path(root) / 'receipt.json').read_bytes())
    P, R = p2['projection'], r['projection']
    pop, mean = R['population'], R['window_distribution']['mean']
    assert P['population'] == pop and abs(P['window_distribution']['mean'] - mean) < 1e-9
    W2, W = P['measured_r6_windows'], R['measured_r6_windows']
    # R6 flip alone per board in the rehearsal: R6-stage production minus members minus the counted-out stride-1 check
    check = R.get('s6c_stride1_check_seconds') or [0.0, 0.0]
    boards = []
    for b, w in enumerate(r['workers']['r6']):
        prod = sum(float(x['production_seconds']) for x in w['records'] if x['status'] == 'genuine')
        mem = member_board_seconds(root, b)
        boards.append({'board': b, 'r6_stage_production': prod, 'stride1_check': check[b], 'members': mem, 'r6_flip_only': prod - check[b] - mem})
    r6_only = max(x['r6_flip_only'] for x in boards)
    members = max(x['members'] for x in boards)
    rate_r6_p2, rate_r6 = P['measured_production_wall_seconds']['r6'] / W2, r6_only / W
    ratio_r6 = rate_r6 / rate_r6_p2
    arms = [a for a in ('old', 'crop224') if a in R['measured_production_wall_seconds']]
    ratio_old = (R['measured_production_wall_seconds']['old'] / R['measured_studies']['old']) / \
                (P['measured_production_wall_seconds']['old'] / P['measured_studies']['old'])
    raptor = {a: P['measured_production_wall_seconds'][a] * pop / P['measured_studies'][a] + P['measured_fixed_phase_overhead_seconds'][a] for a in arms}
    scale = pop * mean / W2
    r6_public = P['measured_production_wall_seconds']['r6'] * scale + (P['measured_fixed_phase_overhead_seconds']['r6'] - P['r6_sentinel_pass_wall_seconds'])
    member_norm = members / ratio_r6
    member_stage = member_norm * pop * mean / W
    sentinels = sum(v['wall_seconds'] for k, v in r['phases'].items() if k.startswith('r6_sentinel'))
    point = R['setup_seconds'] + R['merge_seconds'] + sum(raptor.values()) + r6_public + member_stage + sentinels + sum(check)
    doc = {'rule': 'parent amendment 2026-10-22 01:15 UTC: public arms at P2 rehearsal rates; members / R6 slowdown; rehearsal-only overhead as measured',
           'p2_receipt': str(p2_path), 'rehearsal': str(root),
           'rates': {'r384_s_per_study': {'p2': P['measured_production_wall_seconds']['old'] / P['measured_studies']['old'],
                                          'rehearsal': R['measured_production_wall_seconds']['old'] / R['measured_studies']['old']},
                     'r6_flip_s_per_window': {'p2': rate_r6_p2, 'rehearsal': rate_r6}},
           'ratios': {'r384_slowdown': ratio_old, 'r6_flip_slowdown': ratio_r6},
           'boards': boards,
           'components_seconds': {'setup_merge_measured': R['setup_seconds'] + R['merge_seconds'], **{'%s_at_p2' % a: v for a, v in raptor.items()},
                                  'r6_flip_at_p2_incl_fixed': r6_public, 'members_normalized': member_stage,
                                  'members_measured_board_max_seconds': members, 'members_normalized_board_max_seconds': member_norm,
                                  'sentinel_passes_measured': sentinels, 'stride1_check_measured': sum(check)},
           'saved_point_seconds': R['point_seconds'], 'normalized_point_seconds': point}
    if out:
        Path(out).write_text(json.dumps(doc, indent=1) + '\n')
    print(json.dumps({k: doc[k] for k in ('rates', 'ratios', 'components_seconds', 'saved_point_seconds', 'normalized_point_seconds')}, indent=1))
    return doc


if __name__ == '__main__':
    a = sys.argv[1:]
    main(a[0], a[1], a[3] if len(a) > 3 and a[2] == '--out' else None)
