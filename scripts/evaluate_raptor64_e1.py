"""Local E1 evaluation: per-target AUC deltas of plane-group window ablations against the full K42 pass (in-orig)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from knee import public_training_smoke as smoke
from knee.public_adaptation_fit import metric_report, paired_bootstrap

E = '~/Dev/rsna-knee-abnormality-detection/outputs/kaggle/own224-window-samerun-20261012-result-v1/own224_window_diagnostic/endpoint-private.npz'
ENDPOINT_SHA = '1c7a0d0928e625551a47ab10082fcb9a5837cd1c3eefd4adc2fac0d921aa27c9'
WEAK = ['Lateral OA', 'PF OA', 'Lateral Meniscus', 'MCL']
npz, out = Path(sys.argv[1]), Path(sys.argv[2])
smoke.require_file(E, ENDPOINT_SHA)
e = np.load(E); z = np.load(npz)
ids = e['ids'].astype(str); assert z['ids'].astype(str).tolist() == ids.tolist()
y, m, g = e['truth'], e['masks'], e['groups'].astype(str)
full = z['full'].astype(np.float64)
result = {'method': 'drop windows centred in a plane group from the standard K42 plan (in-orig final SWA)', 'full': metric_report(y, m, full, smoke.TARGETS)}
for name in ('no_sagittal', 'no_coronal', 'no_axial'):
    p = z[name].astype(np.float64); assert np.isfinite(p).all()
    r = metric_report(y, m, p, smoke.TARGETS)
    weak_cols = [smoke.TARGETS.index(t) for t in WEAK]
    result[name] = {'macro_auc': r['macro_auc'], 'macro_delta': r['macro_auc'] - result['full']['macro_auc'],
                    'macro_ci95': paired_bootstrap(y, m, full, p, g, draws=2000, seed=42, targets=smoke.TARGETS)['ci95'],
                    'per_target_delta': {t: r['targets'][t]['auc'] - result['full']['targets'][t]['auc'] for t in smoke.TARGETS},
                    'weak_targets_mean_delta': float(np.mean([r['targets'][t]['auc'] - result['full']['targets'][t]['auc'] for t in WEAK])),
                    'weak_ci95': paired_bootstrap(y[:, weak_cols], m[:, weak_cols], full[:, weak_cols], p[:, weak_cols], g, draws=2000, seed=42, targets=WEAK)['ci95']}
out.write_text(json.dumps(result, indent=2) + '\n')
print('full macro', round(result['full']['macro_auc'], 5))
for name in ('no_sagittal', 'no_coronal', 'no_axial'):
    r = result[name]
    print(name, 'macro d', round(r['macro_delta'], 4), [round(x, 4) for x in r['macro_ci95']], '| weak mean d', round(r['weak_targets_mean_delta'], 4), [round(x, 4) for x in r['weak_ci95']])
    print('   ', {t: round(v, 4) for t, v in r['per_target_delta'].items()})
