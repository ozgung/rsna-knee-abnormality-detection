# Own224 matched K16 versus K42 — October 12, 2026

The frozen 870-study, same-checkpoint comparison completed as private Kaggle v1/kernel **138336842**, `ozgung/knee-own224-window-samerun-20261012`. No optimizer updates occurred and model state remained identical. The parent independently passed **76 checks** on the collected outputs.

K16 macro AUC **0.9018421014**; K42 **0.9058770197**; delta **+0.0040349184**. A fixed-seed 2,000-draw paired bootstrap over 843 normalized-report components gave **[+0.0017689120, +0.0063442793]**, with all draws valid. Macro averages all 12 defined targets equally; unknown labels remain masked. This is an exposed report-selected development endpoint with unresolved patient identity, not independent generalization or a public-score forecast.

## Per-target evidence

| Target | K16 | K42 | Delta | Positive | Negative | Missing |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ACL | 0.946197 | 0.947965 | +0.001768 | 109 | 716 | 45 |
| MCL | 0.855633 | 0.868101 | +0.012468 | 20 | 790 | 60 |
| Medial Meniscus | 0.921786 | 0.927774 | +0.005987 | 330 | 496 | 44 |
| Lateral Meniscus | 0.855952 | 0.865782 | +0.009831 | 109 | 713 | 48 |
| Medial OA | 0.923958 | 0.928489 | +0.004532 | 150 | 662 | 58 |
| Lateral OA | 0.849312 | 0.850616 | +0.001304 | 75 | 736 | 59 |
| PF OA | 0.847253 | 0.846667 | -0.000586 | 169 | 646 | 55 |
| Effusion | 0.917102 | 0.921024 | +0.003922 | 222 | 595 | 53 |
| Synovitis | 0.946472 | 0.948287 | +0.001815 | 103 | 337 | 430 |
| Baker's | 0.943212 | 0.946597 | +0.003385 | 113 | 434 | 323 |
| Contusion | 0.899331 | 0.905994 | +0.006663 | 165 | 634 | 71 |
| Fracture | 0.915898 | 0.913228 | -0.002670 | 52 | 742 | 76 |

## Execution and audit

- Source revision `f0d48ce`; notebook SHA `76ce700d49e8cf537372ad7b20e1db68f4133762cf6a610f90e5a44ff6e691ec`; contract SHA `6c4efd08e0dd89cab34d649b5fc56b8a8b9bd8dc0243d88fea5dd9947610b2e0`.
- Total recorded runtime 680.85 seconds; inference sections K16 118.25 seconds, K42 343.63 seconds. Scheduler wait is excluded. Historical saved-probability maximum error 2.9783e-8 with corrected one-thread preprocessing.
- All 870 IDs, original label-file hash, masks, scored truth, predictions, cache pins, state hashes, target supports, AUCs and paired confidence interval independently verified.
- K16 / K42 used 13,920 / 36,540 windows. There were no duplicate centers or fallback studies. Masked triplets 536 / 2,064; series-boundary triplets 870 / 6,960. This preserves the exact legacy window policy; extra windows are not guaranteed to be anatomically valid triplets.
- Initial parent auditor v1 failed one check because it thresholded unobserved labels without applying the pinned evaluator's zero-placeholder rule. There were 22 discrepancies, all masked out; zero observed-label or mask discrepancies. Preserve that failed receipt. Auditor v2 explicitly checks the correct masked representation and zero placeholders; no output, metric or experimental policy changed.

Private local evidence: `outputs/analysis/own224-window-samerun-20261012/{terminal-collection-v1,parent-terminal-review-v1,parent-terminal-review-v2}.json`; frozen remote outputs under `outputs/kaggle/own224-window-samerun-20261012-result-v1/own224_window_diagnostic/`. The same evidence directory contains both auditor versions.

## Decision

Wider inference coverage helped this checkpoint in aggregate: ten targets increased, PF OA fell 0.000586 and fracture fell 0.002670. Keep those harms visible. The controlled source96 training hypothesis remains justified, with K16 selection/export kept fixed. Do not change the source96 primary endpoint to K42/K94 or tune on this result. No direct submission is warranted from this diagnostic alone; preserve K94/mirror as separately qualified secondary measurements.
