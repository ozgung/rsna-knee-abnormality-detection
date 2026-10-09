# Round1 partial audit — October 9, 2026

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

## Folds2/3 completed; four-fold comparison

At21:34UTC the new fold23 audit passed65 checks with zero failures. Reused fold01 evidence was bound to unchanged OOF bytes; no gold outcomes evaluated. Parent inference metadata validation passed for folds2/3.

| Fold | Studies | Round0 AUC | Round1 AUC | Change | SWA epochs |
| --- | ---: | ---: | ---: | ---: | --- |
| 2 | 870 | 0.881449 | 0.893514 | +0.012064 | [8, 9, 7] |
| 3 | 870 | 0.878534 | 0.890753 | +0.012219 | [9, 8, 7] |

Four-fold pooled report AUC on3,476 distinct heldout studies: **0.880448 → 0.892044**, delta+0.011597. All12 targets improve on both new folds; prior fold0 MCL decline remains. These remain biased report-validation results, not independent clinical CV or a public-score forecast.

| Target | Four-fold positive / negative / unaddressed | Fold2 AUC change | Fold3 AUC change |
| --- | --- | ---: | ---: |
| ACL | 425 / 2886 / 165 | +0.00962 | +0.00327 |
| MCL | 89 / 3175 / 212 | +0.00930 | +0.03108 |
| Medial Meniscus | 1328 / 1995 / 153 | +0.00629 | +0.01523 |
| Lateral Meniscus | 473 / 2800 / 203 | +0.01208 | +0.01284 |
| Medial OA | 597 / 2700 / 179 | +0.00744 | +0.00402 |
| Lateral OA | 294 / 2982 / 200 | +0.02018 | +0.01220 |
| PF OA | 702 / 2590 / 184 | +0.01518 | +0.00460 |
| Effusion | 870 / 2413 / 193 | +0.00529 | +0.00801 |
| Synovitis | 368 / 1373 / 1735 | +0.01288 | +0.00273 |
| Baker's | 434 / 1755 / 1287 | +0.01358 | +0.01585 |
| Contusion | 668 / 2487 / 321 | +0.01416 | +0.01265 |
| Fracture | 214 / 2944 / 318 | +0.01879 | +0.02418 |

Evidence: `outputs/analysis/round1-fold23-parent-audit-20261009/audit.json` and `audit.py`. Per-target AUC and supports reproduce receipts within1e-12. Fullfive image-only qualification remains pending fold4. Current checkpoint transport status is in [STATUS](../../STATUS.md).

## Fullfive training artifacts accepted

Final fold4 audit:45 checks PASS. Fold4 report AUC0.889545→0.900207,869studies, selectedSWA epochs[9,8,7]. All4,345 eligible studies are covered exactly once in the pooled validation predictions. All12 targets improve on fold4 and in the fullfive pooled comparison; fold0MCL remains the sole negative fold/target change.

Pooled raw-report AUC: **0.882137 → 0.893651**, delta+0.011513. Fable's0.867392→0.879288 comparison also reproduces: it evaluates thresholded `labels_v1_blend` targets, whereas this audit uses raw `train_labels_v1 y_`; both use the same addressed-cell masks. These are different target definitions, not an unexplained discrepancy. Neither is independent clinical CV because globalOOF exposure and checkpoint selection remain. No gold-driven changes or automatic round2 follow from these numbers.

| Target | Positive / negative / unaddressed | Round0 pooled | Round1 pooled |
| --- | --- | ---: | ---: |
| ACL | 518 / 3619 / 208 | 0.924557 | 0.934517 |
| MCL | 110 / 3971 / 264 | 0.840993 | 0.857991 |
| Medial Meniscus | 1663 / 2498 / 184 | 0.867032 | 0.881485 |
| Lateral Meniscus | 607 / 3490 / 248 | 0.809857 | 0.823530 |
| Medial OA | 744 / 3376 / 225 | 0.895330 | 0.902972 |
| Lateral OA | 366 / 3731 / 248 | 0.847164 | 0.860241 |
| PF OA | 884 / 3228 / 233 | 0.846098 | 0.852816 |
| Effusion | 1086 / 3018 / 241 | 0.924584 | 0.929704 |
| Synovitis | 448 / 1710 / 2187 | 0.947710 | 0.954293 |
| Baker's | 540 / 2199 / 1606 | 0.903804 | 0.918102 |
| Contusion | 841 / 3107 / 397 | 0.877239 | 0.890511 |
| Fracture | 269 / 3682 / 394 | 0.901277 | 0.917645 |

Final checkpoint112,185,311bytes SHA `2ae47c3e8afed5a4ab3e8ffaca92c59899d84820f8ccc637871898354e3dbaeb` verified. Allfive weights and run records staged in `outputs/kaggle/own-convnext-r1-fivefold-inputs-20261009/`. Immutable fullfive rehearsal built from6b289e5 at `outputs/kaggle/own-convnext-r1-rehearsal-fivefold-20261009/`: notebookSHA `1f5be813a5b19f767cb13ed12c519c54e81adb6e69298e05553d79406fc8fdba`, contractSHA `256b8623d0820c66cca19ea687e70ed428443f4e6bcd26049d2fd8d5ba618630`. Actual launch was rejected by Kaggle's two-session GPU limit; no runtime qualification or entry yet.

Parent held the remaining long-fit queue and installed a one-shot rehearsal launcher for the next real free slot, leaving both running fits intact. The first direct push rejection and launcher startup failures are retained under `outputs/analysis/reconcile-20261009-2330/`; see STATUS for exact live state and recovery. The launcher retries only explicit capacity rejection, stops on unknown push outcome, and avoids duplicating an existing job.

Fullfive audit and metrics: `outputs/analysis/round1-fold4-parent-audit-20261009/audit.json`. No inference source changed; actual metadata/build/hash checks passed, existing runtime tests were not repeated. Fullfive GPU rehearsal and final-visible equality remain mandatory before submission.
