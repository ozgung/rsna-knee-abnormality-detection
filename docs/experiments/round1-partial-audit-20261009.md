# Round1 fold01 audit — October 9, 2026

The first two fixed bootstrap-label folds completed. Independent artifact/metric audit passed61 checks with zero failures; the parent also passed the inference builder's exact round1 metadata validator for both folds. GlobalOOF exposure and selection on the evaluated report fold make these descriptive, biased validation results. They do not forecast the public score. No gold outcomes were evaluated in this audit.

| Fold | Studies | Round0 AUC | Round1 AUC | Change | Selected SWA epochs |
| --- | ---: | ---: | ---: | ---: | --- |
| 0 | 870 | 0.880946 | 0.888544 | +0.007598 | [9, 8, 7] |
| 1 | 866 | 0.882341 | 0.896112 | +0.013771 | [8, 9, 7] |

Pooled over1,736 distinct heldout studies: 0.881443 → 0.892478, delta+0.011035. All12 targets are defined. Macro AUC is the unweighted mean over targets with both classes; unaddressed report findings are excluded, not treated as negatives.

| Target | Pooled positive / negative / unaddressed | Fold0 AUC change | Fold1 AUC change |
| --- | --- | ---: | ---: |
| ACL | 214 / 1434 / 88 | +0.00186 | +0.01643 |
| MCL | 44 / 1582 / 110 | -0.01177 | +0.03930 |
| Medial Meniscus | 672 / 983 / 81 | +0.01486 | +0.01995 |
| Lateral Meniscus | 235 / 1404 / 97 | +0.00535 | +0.01397 |
| Medial OA | 304 / 1333 / 99 | +0.00601 | +0.01248 |
| Lateral OA | 140 / 1486 / 110 | +0.01145 | +0.00529 |
| PF OA | 335 / 1300 / 101 | +0.01121 | +0.00018 |
| Effusion | 428 / 1207 / 101 | +0.00569 | +0.00083 |
| Synovitis | 181 / 674 / 881 | +0.01138 | +0.00335 |
| Baker's | 222 / 860 / 654 | +0.01646 | +0.01740 |
| Contusion | 338 / 1237 / 161 | +0.00994 | +0.02052 |
| Fracture | 103 / 1470 / 163 | +0.00873 | +0.01554 |

Exact label SHA, raw truth hash, source/fold/exclusion bindings, ten completed epochs, top3 report-selected SWA, matching same-ID round0 coverage and identical recipe apart from labels all passed. Per-target AUC/support reproduces the run receipts within1e-12. MCL falls in fold0 despite the aggregate gain; no target-specific tuning follows.

Keep the fixed experiment: finish folds2–4, verify all checkpoint bytes, then qualify the unchanged image-only reader and fullfive ensemble. Do not substitute a two-fold entry or expand bootstrap recursively from this partial result.

Evidence: `outputs/analysis/round1-fold01-parent-audit-20261009/audit.json` and `audit.py`; transport receipt `outputs/analysis/reconcile-20261009-2030/weight-range-transport.json`. Raw identifiers and predictions remain ignored. Run receipts do not contain signed OOF checksums; the audit establishes local byte binding and reproduced runtime metrics, not a retrospectively invented signed manifest.

Checkpoint download and packaging status are maintained in [STATUS](../../STATUS.md). No new CUDA rehearsal or competition entry is claimed here.

At20:36UTC both selected checkpoints completed verified transport (112,185,311bytes each): fold0 SHA `97953dc5678c625cd2b77395ba4a0143811e8e6ddf924dae1bfccf88a1017b70`; fold1 SHA `08ae19852532f1f6dd7b738671a05ae2fb00b7f950d9e70641f9ae3191c5b168`. Parent built `outputs/kaggle/own-convnext-r1-rehearsal-fold01-20261009/` with explicit `--training-round 1`; notebookSHA `486321081a72e4db5fbd645b7c53bb240c63ab88f136d8839a9d11bf3aa2eeff`, contractSHA `b998655443a7fb7abbf8554565c99338f8aee52732ea0fa67b202cc268ccf199`. Build-only, not launched; fullfive CUDA qualification remains pending. No runtime source changed, so existing tests were not rerun; actual builder, metadata/content/hash checks passed.
