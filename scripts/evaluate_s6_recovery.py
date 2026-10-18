"""CPU-only freeze/extract/score stages; all identity artifacts stay ignored.

Use the existing PyArrow interpreter for extract and numpy/pandas for score.
Freeze must complete before extract opens truth. No runtime qualification here.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from knee.public_s6_recovery import RECIPE, RECEIPT_VERSION, digest, freeze_protocol, recover
from knee.public_s6_qualification import LABELS
from knee.public_s6_evaluation import array_sha
from knee.public_adaptation_fit import file_sha, fingerprint, metric_report, paired_bootstrap, load_binary_truth, report_components


def save_new(path, data):
    with Path(path).open('x') as h:
        json.dump(data, h, indent=2, allow_nan=False); h.write('\n')


def paths(root):
    return (root / 'outputs/kaggle/public-s6-evaluation-fold0-20261011-result/s6-evaluation',
            root / 'outputs/kaggle/public-s6-evaluation-fold0-20261011-result/s6-evaluation-project/config-private.json')


def freeze(root, out):
    import numpy as np
    result, config = paths(root)
    r = json.loads((result / 'receipt.json').read_text())
    audit = root / 'outputs/analysis/public-s6-evaluation-20261011/independent-result-audit.json'
    if json.loads(audit.read_text())['counts'] != {'PASS': 73, 'FAIL': 0, 'UNKNOWN': 3}:
        raise ValueError('ORIGINAL_AUDIT_REQUIRED')
    p = result / 'comparison-private.npz'
    if file_sha(p) != r['prediction_artifacts_sha256']['comparison-private.npz']:
        raise ValueError('EXACT_SAVED_ARRAYS_REQUIRED')
    cfg = json.loads(config.read_text())
    with np.load(p, allow_pickle=False) as a:
        if a['study_ids'].tolist() != cfg['dev'] or fingerprint(cfg['dev']) != r['binding']['dev_order_sha256']:
            raise ValueError('CANONICAL870_ORDER')
        if array_sha(a['control']) != r['binding']['saved_control_array_sha256']:
            raise ValueError('INTACT_ORIGINAL320_REQUIRED')
        zero = np.zeros(870, bool)
        check = recover(a['control'], a['s6_raw'], zero, zero)
        if any(not np.array_equal(check[k], a[k]) for k in ('control_rank', 's6_rank')) or not np.array_equal(check['blend'], a['fixed_blend']):
            raise ValueError('ZERO_FAILURE_EXACT_PARITY')
    worker = json.loads((result / 's6/receipt.json').read_text())
    if [x['ordinal'] for x in worker['records']] != list(range(870)):
        raise ValueError('WORKLOAD_CANONICAL_ORDER')
    p = freeze_protocol([x['windows'] for x in worker['records']])
    p['inputs'] = {'comparison_sha256': file_sha(result / 'comparison-private.npz'),
                   'comparison_receipt_sha256': file_sha(result / 'receipt.json'),
                   'worker_receipt_sha256': file_sha(result / 's6/receipt.json'),
                   'config_sha256': file_sha(config), 'independent_audit_sha256': file_sha(audit),
                   'source_binding': r['binding']}
    p['implementation_sha256'] = {str(x.relative_to(Path(__file__).resolve().parents[1])): file_sha(x) for x in
                                  (Path(__file__).resolve(), Path(__file__).resolve().parents[1] / 'knee/public_s6_recovery.py')}
    p['frozen_unix_time'] = time.time()
    out.mkdir(parents=True, exist_ok=True)
    save_new(out / 'protocol-private.json', p)
    save_new(out / 'protocol-freeze.json', {'protocol_sha256': file_sha(out / 'protocol-private.json'),
             'recipe': RECIPE, 'receipt_version': RECEIPT_VERSION, 'scenarios': len(p['scenarios']), 'truth_opened': False,
             'zero_failure_exact_parity': True, 'frozen_unix_time': p['frozen_unix_time']})
    print(json.dumps({'phase': 'frozen_before_truth', 'protocol_sha256': file_sha(out / 'protocol-private.json'), 'scenarios': 13}))


def verified_protocol(root, out):
    p = json.loads((out / 'protocol-private.json').read_text())
    if file_sha(out / 'protocol-private.json') != json.loads((out / 'protocol-freeze.json').read_text())['protocol_sha256']:
        raise ValueError('FROZEN_PROTOCOL_BYTES')
    for name, sha in p['implementation_sha256'].items():
        if file_sha(Path(__file__).resolve().parents[1] / name) != sha:
            raise ValueError('FROZEN_IMPLEMENTATION_BYTES')
    result, cfg = paths(root)
    for path, key in ((result / 'comparison-private.npz', 'comparison_sha256'), (cfg, 'config_sha256'),
                      (result / 'receipt.json', 'comparison_receipt_sha256'), (result / 's6/receipt.json', 'worker_receipt_sha256')):
        if file_sha(path) != p['inputs'][key]: raise ValueError('FROZEN_INPUT_BYTES')
    return p


def extract(root, out):
    import numpy as np
    import pyarrow.parquet as pq
    p = verified_protocol(root, out)
    result, config = paths(root)
    cfg = json.loads(config.read_text())
    truth = root / 'data/weak-labels/nartaa_rsna-knee-hpo-assets/train_labels_v1.parquet'
    if file_sha(truth) != p['inputs']['source_binding']['truth_sha256']:
        raise ValueError('EXACT_TRUTH_BYTES')
    y, m = load_binary_truth(truth, cfg['dev'], LABELS)
    if fingerprint({'y': y.tolist(), 'm': m.tolist()}) != json.loads((result / 'receipt.json').read_text())['truth_sha256']:
        raise ValueError('ORIGINAL_TRUTH_FINGERPRINT')
    rows = list(csv.DictReader((root / 'data/raw/train.csv').open()))
    if file_sha(root / 'data/raw/train.csv') != p['inputs']['source_binding']['official_train_sha256']:
        raise ValueError('OFFICIAL_TRAIN_BYTES')
    labelids = set(pq.read_table(root / 'data/weak-labels/nartaa_rsna-knee-hpo-assets/labels_v1_blend.parquet', columns=['StudyInstanceUID']).column(0).to_pylist())
    gold = {r['StudyInstanceUID'] for r in rows if all(r[t].strip() for t in LABELS)}
    eligible = sorted(({r['StudyInstanceUID'] for r in rows} & labelids) - gold - set(cfg['excluded']))
    train = [s for s in eligible if cfg['folds'][s] != 0]
    dev = [s for s in eligible if cfg['folds'][s] == 0]
    if dev != cfg['dev']: raise ValueError('ORIGINAL_POPULATION')
    groups, grouping = report_components(rows, train, dev)
    if groups != cfg['groups'] or fingerprint(groups) != p['inputs']['source_binding']['groups_sha256']:
        raise ValueError('ORIGINAL_COMPONENTS')
    with (out / 'truth-components-private.npz').open('xb') as h:
        np.savez(h, y=y, masks=m, groups=np.asarray(groups))
    save_new(out / 'truth-extraction.json', {'protocol_sha256': file_sha(out / 'protocol-private.json'),
             'truth_components_sha256': file_sha(out / 'truth-components-private.npz'), 'grouping': grouping,
             'truth_sha256': file_sha(truth), 'extracted_after_freeze': True})
    print('Exact truth and original component identities extracted after mask freeze.')


def score(root, out):
    import numpy as np
    started = time.perf_counter()
    p = verified_protocol(root, out)
    result, _ = paths(root)
    extraction = json.loads((out / 'truth-extraction.json').read_text())
    if file_sha(out / 'truth-components-private.npz') != extraction['truth_components_sha256']:
        raise ValueError('EXTRACTED_TRUTH_BYTES')
    with np.load(out / 'truth-components-private.npz', allow_pickle=False) as a:
        y, m, groups = a['y'], a['masks'], a['groups']
    with np.load(result / 'comparison-private.npz', allow_pickle=False) as a:
        c, s = a['control'], a['s6_raw']
    baseline = metric_report(y, m, c, LABELS)
    results = []
    for scenario in p['scenarios']:
        cm, sm = np.zeros(870, bool), np.zeros(870, bool)
        cm[scenario['old_missing']] = True; sm[scenario['s6_missing']] = True
        r = recover(c, s, cm, sm)
        metrics = metric_report(y, m, r['blend'], LABELS)
        boot = paired_bootstrap(y, m, c, r['blend'], groups, draws=2000, seed=42, targets=LABELS)
        delta = metrics['macro_auc'] - baseline['macro_auc']
        damage = [t for t in LABELS if metrics['targets'][t]['positive'] >= 20 and metrics['targets'][t]['negative'] >= 20
                  and metrics['targets'][t]['auc'] - baseline['targets'][t]['auc'] <= -.01]
        passed = delta > 0 and boot['ci95'] is not None and boot['ci95'][0] > 0 and len(damage) < 3
        results.append({'name': scenario['name'], 'union_recovered': r['union_recovered'],
                        'arm_counts': {arm: {status: values.count(status) for status in ('genuine', 'recovered')} for arm, values in r['statuses'].items()},
                        'mask_sha256': digest(scenario), 'means': r['means'], 'metrics': metrics,
                        'paired_bootstrap': boot, 'macro_delta': delta, 'target_damage': damage, 'gate_pass': passed})
        print(json.dumps({'scenario': scenario['name'], 'macro_delta': delta, 'ci95': boot['ci95'], 'damage': damage, 'gate_pass': passed}), flush=True)
    save_new(out / 'result.json', {'recipe': RECIPE, 'receipt_version': RECEIPT_VERSION,
             'protocol_sha256': file_sha(out / 'protocol-private.json'), 'baseline': baseline, 'scenarios': results,
             'all_scenarios_pass': all(x['gate_pass'] for x in results), 'runtime_seconds': time.perf_counter() - started,
             'environment': {'python': sys.version, 'numpy': np.__version__, 'device': 'cpu'},
             'qualification': 'CPU synthetic sensitivity only; fresh image-only runtime qualification required',
             'generalization_evidence': False})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['freeze', 'extract', 'score'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    globals()[args.phase](args.root.resolve(), args.out.resolve())
