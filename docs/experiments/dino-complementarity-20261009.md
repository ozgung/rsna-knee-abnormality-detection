# DINO complementarity check — October 9

Decision: defer further DINO training under the current 10-epoch recipe. The predeclared equal-rank blend worsens both completed folds; prioritize ConvNeXt inference qualification and the already planned CoAtNet comparison. Existing fitted artifacts stay unchanged. Queue changes require coordination with Fable's controller.

| Validation cohort | ConvNeXt | DINO | Equal-rank mean | Difference vs ConvNeXt, 95% CI |
| --- | ---: | ---: | ---: | --- |
| Fold 0, 870 studies | 0.88095 | 0.84196 | 0.87397 | −0.00697 [−0.01148, −0.00234] |
| Fold 1, 866 studies | 0.88234 | 0.81117 | 0.86254 | −0.01980 [−0.02551, −0.01418] |
| Pooled, 1,736 studies | 0.88144 | 0.82626 | 0.86803 | −0.01341 [−0.01705, −0.00988] |

AUC is the unweighted macro over all 12 defined targets, restricted to addressed raw report-label cells. Average-tie percentile ranks are computed per target within each evaluation population and averaged equally; there was no weight search. Paired bootstrap uses 2,000 resamples of 1,685 patient/report components (zero components crossing folds); every draw retained all targets. Supports and per-target results are in the aggregate receipt. Only pooled MCL improves, with its interval crossing zero. Mean prediction Spearman correlation is 0.796, so diversity alone does not compensate for the weaker member.

Checks passed for exact 870/866 validation memberships, exclusion of gold and related studies, official target order, finite probabilities, source/truth/fold hashes, and reproduction of each recorded single-family AUC to 1e-12. Checkpoint byte validation reuses the preceding reconciliation receipts. Gold outcomes were not inspected for this analysis.

These are checkpoint-selection report labels, not an independent clinical test. Intervals are conditional on these fitted models and this cohort; they do not capture training uncertainty or guarantee a public score. This rejects this fixed blend and recipe as the next spending priority, not all possible DINO adaptations.

Evidence: `outputs/analysis/dino-complement-20261009/aggregate.json`; reproducible CPU analysis `analyze.py` alongside it. Raw labels and study-level data remain ignored. Parent reviewed the analysis and aggregate receipt; no new training or entry was launched for this comparison.
