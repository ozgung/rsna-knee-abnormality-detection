# CoAtNet224 paired label comparison — October 10, 2026

The completed two-fold bootstrap arm has mixed report-validation results. Original blended labels remain the primary own-model submission candidate; this comparison does not justify another bootstrap round or default ensemble inclusion.

| Raw-report masked AUC | Original blended labels | Bootstrap-v1 | Difference |
| --- | ---: | ---: | ---: |
| Fold 0 (870 studies) | 0.901842 | 0.899828 | −0.002014 |
| Fold 1 (866 studies) | 0.902008 | 0.909633 | +0.007626 |
| Pooled (1,736 studies) | 0.901719 | 0.904833 | +0.003114 |

Pooled metrics concatenate predictions from the two separately trained folds. Macro AUC is the unweighted mean over targets with both classes present; all twelve are defined. Unknown report labels are masked, not converted to negatives.

| Target | Original AUC | Bootstrap AUC | Difference | Positive | Negative | Unaddressed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ACL | 0.943026 | 0.946343 | +0.003317 | 214 | 1434 | 88 |
| MCL | 0.892268 | 0.866122 | -0.026146 | 44 | 1582 | 110 |
| Medial Meniscus | 0.913751 | 0.914676 | +0.000925 | 672 | 983 | 81 |
| Lateral Meniscus | 0.843496 | 0.851167 | +0.007671 | 235 | 1404 | 97 |
| Medial OA | 0.905059 | 0.912041 | +0.006981 | 304 | 1333 | 99 |
| Lateral OA | 0.845943 | 0.850548 | +0.004605 | 140 | 1486 | 110 |
| PF OA | 0.853832 | 0.865146 | +0.011313 | 335 | 1300 | 101 |
| Effusion | 0.929318 | 0.931436 | +0.002118 | 428 | 1207 | 101 |
| Synovitis | 0.955424 | 0.957924 | +0.002500 | 181 | 674 | 881 |
| Baker's | 0.927216 | 0.936900 | +0.009685 | 222 | 860 | 654 |
| Contusion | 0.895244 | 0.901929 | +0.006685 | 338 | 1237 | 161 |
| Fracture | 0.916056 | 0.923770 | +0.007714 | 103 | 1470 | 163 |

The same study IDs, supports, truth/split/exclusion hashes, ten completed epochs, and report-selected top-three SWA were verified. Selected epochs are [8,9,7] for fold0 and [9,8,7] for fold1. The architecture, optimizer and window configuration match the original arm. Source changes additionally implement per-epoch persistence and disabled resume handling; model, reader, loss and export definitions are unchanged. Both runs started fresh. Thus sources are not byte-identical, and runtime comparisons include persistence overhead.

Sol produced a 69-check audit; the parent reran it with all checks passing. This verifies artifacts and metrics, not weight bytes or model forward. Evidence and reproducible script: `outputs/analysis/coatnet224-label-arm-20261010/{audit.py,audit.json}`. Source runs: `ozgung/knee-tracka-coatnet224-r1-fold01-20261010` v1 versus original `ozgung/knee-tracka-coatnet224-fold01-20261009` v3.

These are not untouched outer-fold estimates: evaluated report folds select checkpoints, and global teacher bootstrapping adds indirect validation exposure. Gold outcomes were not examined by this audit or used to select the next recipe. Pooled MCL declines by0.026146 with only44 positives; the remaining eleven targets increase. No confidence interval or claim of statistical significance is supplied.

The earlier ConvNeXt public result fell0.878→0.874 despite increased report AUC. Combined evidence supports retiring this bootstrap recipe operationally, but does not prove that teacher weakness is the sole cause or that bootstrapping always fails. No bootstrap weights were downloaded for further qualification. Next: audit original-label folds2/3, complete full-five image-only CUDA rehearsal, and submit only after exact schema/parity/runtime checks pass.
