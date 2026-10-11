# Fixed R2-224 / original-label384 blend diagnostic — October11

The single fixed 50:50 probability blend improves report-development AUC on all five folds. It does not establish a competitive improvement over the preserved public0.950 solution. No weight search, training, GPU job or submission was performed.

Before opening blend metrics, parent accepted Fable's proposal as descriptive CPU work with fixed weights. Each report study uses only predictions from its own held-out fold in each family; never average all five models on those training studies. Gold58 uses each family's five-model mean, then the same50:50 blend. Coverage and schema checks passed for4345 unique report studies and58 gold studies; fold counts870/866/870/870/869, with the four quarantined studies excluded. Every probability is finite and in[0,1], with official12-target order. Pinned original report truth uses the historical float32y_>=0.5 threshold and unchanged observation masks.

| Prediction family | Pooled report macro-AUC | Gold58 five-model macro-AUC |
| --- | ---: | ---: |
| R2 bootstrap224 | 0.910751 | 0.905867 |
| Original-label384 | 0.913156 | 0.912396 |
| Fixed50:50 blend | **0.917421** | **0.914240** |

These are recomputed prediction metrics, not means of fold AUCs. All12 targets are defined; macro policy is their unweighted arithmetic mean. Cross-fold probability scales can affect pooled AUC, so per-fold comparisons are also retained: blend0.916258/0.919394/0.915936/0.912458/0.923663, each exceeding both corresponding families.

Paired2000-draw bootstrap, seed42, resamples exact normalized-report components, preserving multiplicities:4199 report components and58 gold components; all2000 draws retained under the fixed originally-defined target policy. Blend minus384 report Δ+0.004266,95% interval[+0.003178,+0.005395]; gold Δ+0.001844, interval[-0.006343,+0.009440]. Blend minusR2 report Δ+0.006670,[+0.005280,+0.008109]; gold Δ+0.008373,[-0.000162,+0.017623]. These intervals quantify sampling uncertainty within the existing development benchmark and do not remove selection or supervision exposure.

Report harm versus384 is small: Lateral OA−0.000435 and Lateral Meniscus−0.000316;10 targets improve. VersusR2,11 improve and MCL falls0.001660. Gold changes are mixed; versus384, Baker's and Medial Meniscus each fall about0.0108. No target-specific weights or gold-selected follow-up was chosen.

Full-fold endpoint selection, R2 global-OOF teachers, prior gold-guided development and missing patient identities limit independence. Legacy384 folds0/1/2 and R2 selected weights still lack equivalent all-tensor/source qualification; a candidate needs those checks and a measured offline image-only rehearsal. The strongest active comparison remains the fixed public-checkpoint adaptation control/endpoint. Another generic384 bootstrap cycle stays lower priority until that result arrives; native ROI/localizer work remains a separate hypothesis.

Exact input/script hashes, coverage, per-target support/AUC/deltas, per-fold metrics, bootstrap diagnostics and limits: [aggregate receipt](../../outputs/analysis/fixed-family-blend-20261011/receipt.json). Executed CPU analysis: `outputs/analysis/fixed-family-blend-20261011/analyze.py`, installed miniconda Python with NumPy/PyArrow and existing project metric/bootstrap helpers. No identifiers or predictions are copied into Git. An initial analysis assertion failed because the exclusion JSON wrapper was mistaken for its study list; corrected before any metric output, then exact4345 coverage passed. Source evidence and model state were unchanged.
