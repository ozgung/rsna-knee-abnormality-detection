"""Step-3 fold-0 test of the DICOM site/protocol prior (pre-registered in docs/research/plan-update-20261014.md).

Usage: python scripts/site_prior_fold0.py HEADERS_CSV TRUTH_CSV IMAGE_NPZ DREAD_NPZ OUT_JSON
- HEADERS_CSV: series_headers.csv from the private Kaggle kernel ozgung/knee-header-meta-20261014.
- TRUTH_CSV: train_labels_v1 (y_/m_ columns, SHA-pinned as TRUTH_SHA in the parquet form).
- IMAGE_NPZ: the image model's fold-0 predictions (ids, pred) on the fixed 870.
- DREAD_NPZ: pinned Dread v4_full K42 teacher (ids, probs) over all 4,345 eligible studies (transfer gate).
Training studies = Dread eligible ids minus the 870. Nothing is tuned on the 870.
"""
import csv
import json
import sys

import numpy as np

sys.path.insert(0, '.')
from knee import site_prior as sp  # noqa: E402


def main(headers, truth, image, dread, out):
    rows = [r for r in csv.DictReader(open(headers)) if r['split'] == 'train']
    feats = sp.study_features(rows)
    t = {r['StudyInstanceUID']: r for r in csv.DictReader(open(truth))}
    z, d = np.load(image), np.load(dread)
    dev = [str(u) for u in z['ids']]
    eligible = [str(u) for u in d['ids']]
    train = [u for u in eligible if u not in set(dev)]
    assert len(dev) == 870 and len(train) == 3475, (len(dev), len(train))
    missing = [u for u in train + dev if u not in feats]
    Y = lambda ids: np.array([[float(t[u]['y_' + k]) for k in sp.TARGETS] for u in ids])
    M = lambda ids: np.array([[int(float(t[u]['m_' + k])) for k in sp.TARGETS] for u in ids])
    enc = sp.Encoder().fit(feats, [u for u in train if u in feats])
    Xtr, Xdev = enc.transform(feats, train), enc.transform(feats, dev)
    models, chosen, prev = sp.fit_meta(Xtr, Y(train), M(train))
    p_meta = np.column_stack([m.predict_proba(Xdev)[:, 1] for m in models])
    Ydev, Mdev = Y(dev), M(dev)
    p_img = z['pred'].astype(float)
    dmap = dict(zip(eligible, d['probs'].astype(float)))
    p_dread = np.array([dmap[u] for u in dev])
    res = {'n_train': len(train), 'n_dev': len(dev), 'missing_header_studies': len(missing),
           'sites_kept': len(enc.sites), 'features': Xtr.shape[1], 'C': chosen, 'prevalence': prev.tolist(),
           'meta_alone_auc': sp.masked_aucs(p_meta, Ydev, Mdev).tolist(),
           'image_auc': sp.masked_aucs(p_img, Ydev, Mdev).tolist()}
    res['primary'] = sp.paired_bootstrap(p_img, sp.combine(p_img, p_meta, prev), Ydev, Mdev)
    res['primary_verdict'] = sp.verdict(res['primary'])
    res['transfer'] = sp.paired_bootstrap(p_dread, sp.combine(p_dread, p_meta, prev), Ydev, Mdev)
    res['transfer_gate_pass'] = res['transfer']['macro_delta'] >= 0
    json.dump(res, open(out, 'w'), indent=1)
    print(json.dumps({k: res[k] for k in ('n_train', 'missing_header_studies', 'sites_kept', 'C', 'primary_verdict', 'transfer_gate_pass')}))
    print('meta-alone', np.round(res['meta_alone_auc'], 3).tolist())
    print('primary macro Δ', round(res['primary']['macro_delta'], 5), np.round(res['primary']['macro_ci'], 5).tolist(),
          'per-target', np.round(res['primary']['target_delta'], 4).tolist())
    print('transfer macro Δ', round(res['transfer']['macro_delta'], 5), np.round(res['transfer']['macro_ci'], 5).tolist())


if __name__ == '__main__':
    main(*sys.argv[1:6])
