"""Local Raptor64 endpoint: masked report macro AUC on the fixed 870 fold-0 studies at K42.

Truth, masks and 843 report-component groups are the pinned own224 same-run endpoint; the
control is own44 (own224 Track-A) K42 = 0.905877. Each arm's exported oof_fold0.csv (val rows,
K42) must cover exactly the endpoint IDs in the same order. Comparisons use the qualified paired
whole-component bootstrap (2,000 draws, seed 42) from knee.public_adaptation_fit. Output is
aggregate only (no IDs). All values are exposed report-selected development metrics.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from knee import public_training_smoke as smoke
from knee.public_adaptation_fit import metric_report, paired_bootstrap

ENDPOINT_SHA = '1c7a0d0928e625551a47ab10082fcb9a5837cd1c3eefd4adc2fac0d921aa27c9'
CONTROL_SHA = '2e5d807b4cdb202e63f95e036c8dbc30d7f5d89eb933eae350693127d9232c42'
CONTROL_MACRO = 0.9058770197220801
ARMS = ('in-orig', 'in-plane', 'dr-orig', 'dr-plane', 'dr-full', 'in-mirror', 'in-mirror-tta', 'in-distil', 'in-distil-w1', 'in-distil-flip')
CONTRASTS = [('in-plane', 'in-orig', 'clean plane-aware head test (ImageNet init)'),
             ('dr-plane', 'dr-orig', 'plane-aware head under Dread init (exposed)'),
             ('dr-orig', 'in-orig', 'Dread v4 init vs ImageNet init (exposed; not a clean init comparison)'),
             ('in-mirror', 'in-orig', 'M1 mirror-equivariant training, plain K42 (clean single factor)'),
             ('in-mirror-tta', 'in-orig', 'M1 mirror training plus mirror TTA at export (two factors)'),
             ('in-distil', 'in-orig', 'M2 distillation toward Dread v4 (near-clean: second-order teacher exposure)'),
             ('in-distil-w1', 'in-distil', 'it3: distillation weight 0.5 -> 1.0 (single factor)'),
             ('in-distil-flip', 'in-distil', 'it3: label-preserving anatomical flip p 0.5 (single factor)'),
             ('in-distil-w1', 'in-orig', 'it3 w1 vs in-orig'), ('in-distil-flip', 'in-orig', 'it3 flip vs in-orig')]
SECONDARY = ['Lateral Meniscus', 'Lateral OA', 'PF OA', 'Medial OA', 'Medial Meniscus']


def load_oof(path, ids):
    import numpy as np
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['StudyInstanceUID', *smoke.TARGETS, 'set']:
            raise ValueError('OOF schema differs')
        rows = [r for r in reader]
    if any(r['set'] != 'val' for r in rows) or [r['StudyInstanceUID'] for r in rows] != list(ids):
        raise ValueError('OOF must contain exactly the 870 endpoint studies in endpoint order (no gold rows)')
    p = np.asarray([[float(r[t]) for t in smoke.TARGETS] for r in rows], dtype=np.float64)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('Probabilities must be finite in [0,1]')
    return p


def evaluate(endpoint_path, control_path, arms):
    import numpy as np
    smoke.require_file(endpoint_path, ENDPOINT_SHA); smoke.require_file(control_path, CONTROL_SHA)
    e = np.load(endpoint_path, allow_pickle=False); c = np.load(control_path, allow_pickle=False)
    ids = e['ids'].astype(str).tolist(); y, m, groups = e['truth'], e['masks'], e['groups'].astype(str)
    if len(ids) != 870 or c['ids'].astype(str).tolist() != ids:
        raise ValueError('Endpoint/control identity differs')
    control = c['probabilities'].astype(np.float64)
    base = metric_report(y, m, control, smoke.TARGETS)
    if abs(base['macro_auc'] - CONTROL_MACRO) > 1e-12:
        raise ValueError('Pinned own44 K42 control macro differs')
    prevalence = metric_report(y, m, np.tile(np.where(m > 0, y, 0).sum(0) / np.maximum(m.sum(0), 1), (len(ids), 1)), smoke.TARGETS)
    preds = {arm: load_oof(path, ids) for arm, path in arms.items()}
    result = {'endpoint': 'masked report macro AUC, fixed 870 fold-0 development, K42', 'endpoint_sha256': ENDPOINT_SHA,
              'control': {'name': 'own44 K42 (own224 Track-A SWA, same-run diagnostic)', 'sha256': CONTROL_SHA, **base},
              'prevalence_baseline': {'macro_auc': prevalence['macro_auc'], 'note': 'constant per-target prevalence: AUC 0.5 by construction'},
              'groups': {'clusters': int(len(set(groups.tolist()))), 'policy': 'normalized report components, pinned with the endpoint'},
              'reporting': 'Raw K42 macro and paired deltas only; no pass/fail gates are applied here (dr-orig epoch-1 K16 < 0.85 is a training health check)',
              'arms': {}, 'contrasts': {}, 'limits': 'Exposed report-selected development; patient identity unavailable; dread_v4 arms additionally exposed '
                                                    '(Dread trained on all non-gold studies). Not a public-score forecast.'}
    for arm, p in preds.items():
        report = metric_report(y, m, p, smoke.TARGETS)
        boot = paired_bootstrap(y, m, control, p, groups, draws=2000, seed=42, targets=smoke.TARGETS)
        delta = report['macro_auc'] - base['macro_auc']
        entry = {**report, 'delta_vs_own44_k42': delta, 'paired_bootstrap_vs_own44_k42': boot,
                 'per_target_delta_vs_own44_k42': {t: (report['targets'][t]['auc'] - base['targets'][t]['auc'])
                                                    if report['targets'][t]['auc'] is not None and base['targets'][t]['auc'] is not None else None
                                                    for t in smoke.TARGETS}}
        result['arms'][arm] = entry
    for a, b, label in CONTRASTS:
        if a in preds and b in preds:
            boot = paired_bootstrap(y, m, preds[b], preds[a], groups, draws=2000, seed=42, targets=smoke.TARGETS)
            cols = [smoke.TARGETS.index(t) for t in SECONDARY]
            sec = paired_bootstrap(y[:, cols], m[:, cols], preds[b][:, cols], preds[a][:, cols], groups, draws=2000, seed=42, targets=SECONDARY)
            ra, rb = metric_report(y[:, cols], m[:, cols], preds[a][:, cols], SECONDARY), metric_report(y[:, cols], m[:, cols], preds[b][:, cols], SECONDARY)
            result['contrasts'][f'{a} minus {b}'] = {'label': label, 'delta': result['arms'][a]['macro_auc'] - result['arms'][b]['macro_auc'],
                                                     'paired_bootstrap': boot,
                                                     'per_target_delta': {t: (result['arms'][a]['targets'][t]['auc'] - result['arms'][b]['targets'][t]['auc'])
                                                                          if result['arms'][a]['targets'][t]['auc'] is not None and result['arms'][b]['targets'][t]['auc'] is not None else None
                                                                          for t in smoke.TARGETS},
                                                     'secondary_five_target': {'targets': SECONDARY, 'delta': ra['macro_auc'] - rb['macro_auc'], 'paired_bootstrap': sec}}
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--endpoint', type=Path, required=True); p.add_argument('--control', type=Path, required=True)
    p.add_argument('--arm', action='append', default=[], help='ARM=path/to/oof_fold0.csv (in-orig, in-plane, dr-orig, dr-plane)')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)
    arms = {}
    for item in a.arm:
        name, _, path = item.partition('=')
        if name not in ARMS or name in arms or not path:
            raise SystemExit('Use --arm NAME=PATH once per arm')
        arms[name] = Path(path)
    if a.out.exists():
        raise SystemExit('Fresh output path required')
    result = evaluate(a.endpoint, a.control, arms)
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'control': result['control']['macro_auc'],
                      **{arm: {'macro': v['macro_auc'], 'delta': v['delta_vs_own44_k42'], 'ci95': v['paired_bootstrap_vs_own44_k42']['ci95']}
                         for arm, v in result['arms'].items()}}, indent=2))


if __name__ == '__main__':
    main()
