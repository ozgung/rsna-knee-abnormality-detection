# Source96 fold0 primary result

Collected 2026-10-12 19:33 UTC from the existing Colab session `source96-fold0-r2-20261012`. Ten epochs completed; frozen top-three SWA uses epochs 9, 8, 7. Full 870-study K16 primary masked macro AUC **0.8945295273**, all 12 targets defined. This is report-exposed, selected development evidence, not independent validation. Best single epoch was 0.8952961763; do not substitute it for the frozen primary.

| Target | AUC | Positive | Negative |
| --- | ---: | ---: | ---: |
| ACL | 0.92619548 | 109 | 716 |
| MCL | 0.87879747 | 20 | 790 |
| Medial Meniscus | 0.90796065 | 330 | 496 |
| Lateral Meniscus | 0.80806001 | 109 | 713 |
| Medial OA | 0.92680765 | 150 | 662 |
| Lateral OA | 0.83518116 | 75 | 736 |
| PF OA | 0.84332350 | 169 | 646 |
| Effusion | 0.92418805 | 222 | 595 |
| Synovitis | 0.94295756 | 103 | 337 |
| Baker's | 0.94054076 | 113 | 434 |
| Contusion | 0.88926489 | 165 | 634 |
| Fracture | 0.91107713 | 52 | 742 |

Macro policy: unweighted mean over targets with both classes. Local CSV checks pass: 870 unique studies, all predictions finite and in [0,1]; receipt macro arithmetic reproduced. Independent truth-based AUC recomputation is pending. Training wall time 9,088.80 seconds; selected export 53.02 seconds. Checkpoint SHA256 `402c5ba777af7ace00ccaedae0df53df9a122539789c1fbf958ea0531e26d658` is receipt-reported, pending final transport verification.

Original controller ended FAIL/FileNotFoundError, with training and persister return codes both zero and no interruption. This matches the previously diagnosed deleted-staging-manifest bookkeeping defect; original receipt is preserved. Epoch9 resume backup version11 is verified by the persister. Parent reconciliation and private final export wait until fold1 ends and the busy notebook supervision cell exits; no retraining.

Evidence: ignored `outputs/analysis/source96-training-20261012/fold0-primary-{run_fold0.json,source96-run-receipt.json,oof_fold0.csv}` and `fold0-primary-local-check.json`. Next: verify final transport and selected checkpoint, then execute the frozen K16 parity / K94 / header-gated comparison. Historical own224 K16 was 0.9018421014; this new primary is lower by 0.0073125740, a recipe-level comparison because historical initializer bytes remain unresolved. No submission is justified by this result alone.
