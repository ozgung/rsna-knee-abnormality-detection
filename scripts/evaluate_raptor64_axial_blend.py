"""Iteration-4 axial PF arm read-out (pre-registered: docs/experiments/raptor64-axial-pf-it4-protocol-20261014.md).

Primary: PF OA AUC of a fixed 0.7 x in-distil + 0.3 x arm rank blend against in-distil K42 alone, on the fixed 870
fold-0 development studies (pinned own224 endpoint truth/masks/843 report components), paired whole-component
bootstrap (2,000 draws, seed 42). Ranks are normalised average ranks over the 870 (in-distil) and over the axial-present
development studies (arm). Studies without axial content fall back to in-distil alone (their arm rank is replaced by
their in-distil rank, so the blend equals the in-distil rank). The weight is fixed; nothing is tuned on fold 0.

Kill-rule inputs computed here, for ax-crop (ax-full is the control):
  1. blended PF OA delta >= +0.010 with CI lower bound > 0;
  2. arm-alone PF OA >= 0.80 (on axial-present studies; all-870 also reported);
  3. same delta sign for the SWA export and the best single epoch;
  4. ax-crop blended PF OA > ax-full blended PF OA;
  5. secondary gate: the same 0.3 blend into the pinned Dread v4_full K42 fold-0 predictions must not lower PF OA.
Effusion and Synovitis are secondary reads only. Output is aggregate only (no IDs).
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from knee import public_training_smoke as smoke
from knee.public_adaptation_fit import metric_report, paired_bootstrap

ENDPOINT_SHA = '1c7a0d0928e625551a47ab10082fcb9a5837cd1c3eefd4adc2fac0d921aa27c9'
IN_DISTIL_SHA = '0eb638e2fed52c01ca542f410bbaa912e15e52a9f687e8f6dc68ae131e3d2d86'   # it2 in-distil K42 oof_fold0.csv
TEACHER_SHA = 'dab0621a2302dc46a867d007d8d90d014d62ddb2f2ad515b8c571d9e17e692f9'      # Dread v4_full K42, all 4,345 studies
IN_DISTIL_MACRO = 0.9148972797851097
WEIGHT = 0.3
PRIMARY = 'PF OA'
SECONDARY = ('Effusion', 'Synovitis')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_csv(path, ids):
    import numpy as np
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['StudyInstanceUID', *smoke.TARGETS, 'set']:
            raise ValueError('OOF schema differs')
        rows = list(reader)
    if any(r['set'] != 'val' for r in rows) or [r['StudyInstanceUID'] for r in rows] != list(ids):
        raise ValueError('OOF must contain exactly the 870 endpoint studies in endpoint order')
    return checked(np.asarray([[float(r[t]) for t in smoke.TARGETS] for r in rows], dtype=np.float64))


def load_npz(path, ids):
    import numpy as np
    with np.load(path, allow_pickle=False) as z:
        if z['ids'].astype(str).tolist() != list(ids):
            raise ValueError('Best-epoch OOF ids differ from the endpoint order')
        return checked(z['pred'].astype(np.float64))


def checked(p):
    import numpy as np
    if p.ndim != 2 or p.shape[1] != 12 or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('Probabilities must be finite in [0,1] with 12 targets')
    return p


def ranks(values):
    """Normalised average ranks in (0, 1]."""
    import numpy as np
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind='stable'); sorted_ = values[order]
    r = np.empty(len(values)); i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_[j + 1] == sorted_[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return r / len(values)


def blend(base, arm, present, weight=WEIGHT):
    """Per-target 0.7 x rank(base) + 0.3 x rank(arm); arm ranks over present studies; absent studies keep rank(base)."""
    import numpy as np
    out = np.empty_like(base)
    for j in range(base.shape[1]):
        rb = ranks(base[:, j]); ra = rb.copy()
        ra[present] = ranks(arm[present, j])
        out[:, j] = (1 - weight) * rb + weight * ra
    return out


def single(y, m, groups, base, new, target):
    j = smoke.TARGETS.index(target)
    rb, rn = metric_report(y[:, [j]], m[:, [j]], base[:, [j]], [target]), metric_report(y[:, [j]], m[:, [j]], new[:, [j]], [target])
    boot = paired_bootstrap(y[:, [j]], m[:, [j]], base[:, [j]], new[:, [j]], groups, draws=2000, seed=42, targets=[target])
    return {'base_auc': rb['macro_auc'], 'new_auc': rn['macro_auc'], 'delta': rn['macro_auc'] - rb['macro_auc'], 'ci95': boot['ci95'],
            'valid_draws': boot['valid_draws'], 'support': {k: rn['targets'][target][k] for k in ('positive', 'negative', 'observed')}}


def evaluate(endpoint, in_distil, teacher, presence, arms, best=None):
    import numpy as np
    if sha(endpoint) != ENDPOINT_SHA or sha(in_distil) != IN_DISTIL_SHA or sha(teacher) != TEACHER_SHA:
        raise ValueError('Pinned endpoint / in-distil / teacher bytes differ')
    e = np.load(endpoint, allow_pickle=False)
    ids = e['ids'].astype(str).tolist(); y, m, groups = e['truth'], e['masks'], e['groups'].astype(str)
    if len(ids) != 870:
        raise ValueError('Endpoint population differs')
    d = load_csv(in_distil, ids)
    if abs(metric_report(y, m, d, smoke.TARGETS)['macro_auc'] - IN_DISTIL_MACRO) > 1e-12:
        raise ValueError('Pinned in-distil K42 macro differs')
    with np.load(teacher, allow_pickle=False) as z:
        tid = {u: i for i, u in enumerate(z['ids'].astype(str).tolist())}; tp = z['probs'].astype(np.float64)
    dread = checked(tp[[tid[u] for u in ids]])
    pres_map = json.loads(Path(presence).read_bytes())
    if sorted(pres_map) != sorted(ids):
        raise ValueError('Axial presence must cover exactly the endpoint studies')
    present = np.asarray([bool(pres_map[u]) for u in ids])
    preds = {name: load_csv(path, ids) for name, path in arms.items()}
    best_preds = {name: load_npz(path, ids) for name, path in (best or {}).items()}
    j = smoke.TARGETS.index(PRIMARY)
    out = {'endpoint': 'PF OA AUC, fixed 870 fold-0 development (masked weak report labels), 0.7/0.3 rank blend',
           'pins': {'endpoint_sha256': ENDPOINT_SHA, 'in_distil_oof_sha256': IN_DISTIL_SHA, 'dread_v4_full_k42_sha256': TEACHER_SHA,
                    'presence_sha256': sha(presence), 'arms': {k: sha(v) for k, v in arms.items()}, 'best_epoch': {k: sha(v) for k, v in (best or {}).items()}},
           'axial_present': {'development': int(present.sum()), 'absent': int((~present).sum()),
                             'pf_oa_observed_absent': int(((m[:, j] > 0) & ~present).sum())},
           'blend_policy': 'per-target normalised average ranks; arm ranks over axial-present studies; absent studies keep the in-distil rank',
           'in_distil_alone': metric_report(y, m, d, smoke.TARGETS)['targets'][PRIMARY],
           'dread_alone': metric_report(y, m, dread, smoke.TARGETS)['targets'][PRIMARY], 'arms': {}}
    for name, p in preds.items():
        alone_all = metric_report(y, m, p, smoke.TARGETS)
        mp = m.copy(); mp[~present] = 0
        alone_present = metric_report(y, mp, p, smoke.TARGETS)
        b = blend(d, p, present); bd = blend(dread, p, present)
        entry = {'arm_alone_all870': {t: alone_all['targets'][t]['auc'] for t in smoke.TARGETS},
                 'arm_alone_axial_present': {t: alone_present['targets'][t]['auc'] for t in smoke.TARGETS},
                 'arm_alone_macro_all870': alone_all['macro_auc'], 'arm_alone_macro_axial_present': alone_present['macro_auc'],
                 'pf_oa_support_axial_present': {k: alone_present['targets'][PRIMARY][k] for k in ('positive', 'negative', 'observed')},
                 'primary_blend_vs_in_distil': single(y, m, groups, d, b, PRIMARY),
                 'secondary_gate_blend_vs_dread': single(y, m, groups, dread, bd, PRIMARY),
                 'secondary_reads': {t: single(y, m, groups, d, b, t) for t in SECONDARY}}
        if name in best_preds:
            entry['best_epoch_blend_vs_in_distil'] = single(y, m, groups, d, blend(d, best_preds[name], present), PRIMARY)
            entry['best_epoch_arm_alone_pf_oa_axial_present'] = metric_report(y, mp, best_preds[name], smoke.TARGETS)['targets'][PRIMARY]['auc']
        out['arms'][name] = entry
    if 'ax-crop' in preds and 'ax-full' in preds:
        bc, bf = blend(d, preds['ax-crop'], present), blend(d, preds['ax-full'], present)
        out['crop_vs_full_blended'] = single(y, m, groups, bf, bc, PRIMARY)
        c = out['arms']['ax-crop']; p = c['primary_blend_vs_in_distil']
        rules = {'1_delta_ge_0.010_and_ci_lower_gt_0': p['delta'] >= 0.010 and p['ci95'][0] > 0,
                 '2_arm_alone_pf_oa_ge_0.80_axial_present': c['arm_alone_axial_present'][PRIMARY] >= 0.80,
                 '3_sign_agrees_swa_vs_best_epoch': (None if 'best_epoch_blend_vs_in_distil' not in c else
                                                     (p['delta'] > 0) == (c['best_epoch_blend_vs_in_distil']['delta'] > 0)),
                 '4_crop_beats_full_blended': out['crop_vs_full_blended']['delta'] > 0,
                 '5_dread_gate_not_lowered': c['secondary_gate_blend_vs_dread']['delta'] >= 0}
        out['kill_rules_ax_crop'] = rules
        out['verdict'] = 'KEEP ax-crop' if all(v is True for v in rules.values()) else 'CLOSE the axial-arm idea (at least one rule fails)'
    out['limits'] = ('Exposed report-selected development (top-3 SWA chosen on the same 870); masked weak report labels; study-level folds; '
                     'second-order teacher exposure (Dread saw these studies); not a public-score forecast.')
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--endpoint', type=Path, required=True); p.add_argument('--in-distil', type=Path, required=True)
    p.add_argument('--teacher', type=Path, required=True); p.add_argument('--presence', type=Path, required=True)
    p.add_argument('--arm', action='append', default=[], help='ARM=oof_fold0.csv (SWA export)')
    p.add_argument('--best', action='append', default=[], help='ARM=raptor_oof_fold0_fold0.npz (best single epoch)')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)
    pairs = lambda items: {k: Path(v) for k, _, v in (i.partition('=') for i in items)}
    if a.out.exists():
        raise SystemExit('Fresh output path required')
    result = evaluate(a.endpoint, a.in_distil, a.teacher, a.presence, pairs(a.arm), pairs(a.best))
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('axial_present', 'kill_rules_ax_crop', 'verdict') if k in result}, indent=2))


if __name__ == '__main__':
    main()
