"""Step-3 site/protocol prior from DICOM header metadata (docs/research/plan-update-20261014.md).

Pre-registered rule: per-target L2 logistic regression on per-study header features, fitted on the training
studies only (C by inner 5-fold CV), combined with an image model's probabilities as
``logit(p) + 0.5 * (logit(p_meta) - logit(prevalence))``. No weight tuning on the development fold.
Inference uses DICOM headers only; no reports, labels or patient identifiers are read as features.
"""
from collections import Counter

import numpy as np

TARGETS = ['ACL', 'MCL', 'Medial_Meniscus', 'Lateral_Meniscus', 'Medial_OA', 'Lateral_OA', 'PF_OA', 'Effusion',
           'Synovitis', 'Bakers', 'Contusion', 'Fracture']
SITE_TAGS = ['Manufacturer', 'ManufacturerModelName', 'MagneticFieldStrength', 'SoftwareVersions', 'ReceiveCoilName']
PLANES = ['Sagittal', 'Coronal', 'Axial']
MIN_SITE = 20
WEIGHT = 0.5
EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _num(x):
    try:
        return float(str(x).split('|')[0])
    except ValueError:
        return np.nan


def study_features(series_rows):
    """Per-study raw features from per-series header rows (dicts as written by the header kernel)."""
    by_study = {}
    for r in series_rows:
        by_study.setdefault(r['StudyInstanceUID'], []).append(r)
    out = {}
    for uid, rows in by_study.items():
        first = rows[0]
        f = {'site': '×'.join(str(first.get(k, '')) for k in SITE_TAGS), 'sex': str(first.get('PatientSex', '')),
             'contrast': float(any(str(r.get('ContrastBolusAgent', '')).strip() for r in rows)),
             'n_series': float(len(rows))}
        for p in PLANES:
            pr = [r for r in rows if r['plane'] == p]
            for fl in ('0', '1'):
                for fs in ('0', '1'):
                    f[f'n_{p}_{fl}{fs}'] = float(sum(str(r['fluid']) == fl and str(r['fatsat']) == fs for r in pr))
            th = [_num(r.get('SliceThickness')) for r in pr]
            ps = [_num(r.get('PixelSpacing')) for r in pr]
            f[f'thick_{p}'] = float(np.nanmedian(th)) if pr and not np.all(np.isnan(th)) else np.nan
            f[f'pix_{p}'] = float(np.nanmedian(ps)) if pr and not np.all(np.isnan(ps)) else np.nan
        out[uid] = f
    return out


class Encoder:
    """Fitted on training studies only: site buckets (min count), sex levels, numeric medians and scales."""

    def fit(self, feats, ids):
        counts = Counter(feats[u]['site'] for u in ids)
        self.sites = sorted(s for s, c in counts.items() if c >= MIN_SITE)
        self.sexes = sorted({feats[u]['sex'] for u in ids})
        self.numeric = sorted(k for k in feats[ids[0]] if k not in ('site', 'sex'))
        x = np.array([[feats[u][k] for k in self.numeric] for u in ids], dtype=float)
        self.median = np.nanmedian(x, axis=0)
        x = np.where(np.isnan(x), self.median, x)
        self.mu, self.sd = x.mean(0), x.std(0) + 1e-9
        return self

    def transform(self, feats, ids):
        x = np.array([[feats[u][k] if u in feats else np.nan for k in self.numeric] for u in ids], dtype=float)
        miss = np.isnan(x).any(1, keepdims=True).astype(float)
        x = (np.where(np.isnan(x), self.median, x) - self.mu) / self.sd
        site = np.array([[float(feats.get(u, {}).get('site') == s) for s in self.sites] +
                         [float(feats.get(u, {}).get('site') not in self.sites)] for u in ids])
        sex = np.array([[float(feats.get(u, {}).get('sex') == s) for s in self.sexes] for u in ids])
        return np.hstack([x, miss, site, sex])


def fit_meta(X, Y, M, Cs=(0.01, 0.03, 0.1, 0.3, 1.0), seed=0):
    """Per-target L2 logistic regression; C chosen by inner 5-fold CV log loss on the training rows."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import log_loss
    models, chosen, prev = [], [], []
    for j in range(Y.shape[1]):
        m = M[:, j] == 1
        x, y = X[m], (Y[m, j] >= 0.5).astype(int)
        prev.append(y.mean())
        best = None
        for c in Cs:
            losses = []
            for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(x, y):
                lr = LogisticRegression(C=c, max_iter=2000).fit(x[tr], y[tr])
                losses.append(log_loss(y[va], lr.predict_proba(x[va])[:, 1], labels=[0, 1]))
            if best is None or np.mean(losses) < best[0]:
                best = (np.mean(losses), c)
        chosen.append(best[1])
        models.append(LogisticRegression(C=best[1], max_iter=2000).fit(x, y))
    return models, chosen, np.array(prev)


def combine(p_image, p_meta, prevalence, weight=WEIGHT):
    """The pre-registered fixed logit shift; returns probabilities."""
    z = _logit(p_image) + weight * (_logit(p_meta) - _logit(prevalence)[None, :])
    return 1 / (1 + np.exp(-z))


def masked_aucs(pred, Y, M):
    from sklearn.metrics import roc_auc_score
    return np.array([roc_auc_score((Y[M[:, j] == 1, j] >= 0.5).astype(int), pred[M[:, j] == 1, j])
                     for j in range(Y.shape[1])])


def paired_bootstrap(a, b, Y, M, n=2000, seed=0):
    """Whole-study paired bootstrap of macro and per-target Δ = AUC(b) − AUC(a); returns mean Δ and 95% CIs."""
    rng = np.random.default_rng(seed)
    base = masked_aucs(b, Y, M) - masked_aucs(a, Y, M)
    draws = []
    for _ in range(n):
        i = rng.integers(0, len(Y), len(Y))
        try:
            draws.append(masked_aucs(b[i], Y[i], M[i]) - masked_aucs(a[i], Y[i], M[i]))
        except ValueError:
            continue
    d = np.array(draws)
    macro = d.mean(1)
    return {'macro_delta': float(base.mean()), 'macro_ci': [float(np.quantile(macro, .025)), float(np.quantile(macro, .975))],
            'target_delta': base.tolist(), 'target_ci': np.quantile(d, [.025, .975], axis=0).T.tolist(), 'draws': len(d)}


def verdict(boot):
    """Pre-registered kill: macro Δ < +0.002, CI lower ≤ 0, or any target drop > 0.005."""
    reasons = []
    if boot['macro_delta'] < 0.002:
        reasons.append('macro_delta_below_0.002')
    if boot['macro_ci'][0] <= 0:
        reasons.append('ci_lower_not_positive')
    if min(boot['target_delta']) < -0.005:
        reasons.append('target_drop_over_0.005')
    return {'pass': not reasons, 'reasons': reasons}
