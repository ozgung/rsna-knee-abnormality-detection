# Fixed public adaptation result — October11

The completed fixed two-epoch adaptation is **rejected for promotion** under the prospective benchmark. Frozen public control report-development macro-AUC is **0.951330909**, versus **0.948112635** for the fixed averaged endpoint: Δ−0.003218274, paired2000-component bootstrap95% interval[−0.005741037,−0.000880128]. All12 targets are defined. Training-only prevalence predicts0.5 AUC. This is development evidence with inherited public-checkpoint exposure and report-derived truth, not an independent clinical evaluation or public-score estimate.

| Target | Frozen control | Fixed endpoint | Delta | Positive / negative support |
| --- | ---: | ---: | ---: | ---: |
| ACL | 0.982023 | 0.982446 | +0.000423 | 109 / 716 |
| MCL | 0.953038 | 0.937089 | −0.015949 | 20 / 790 |
| Medial Meniscus | 0.966532 | 0.966960 | +0.000428 | 330 / 496 |
| Lateral Meniscus | 0.963882 | 0.962402 | −0.001480 | 109 / 713 |
| Medial OA | 0.957563 | 0.958459 | +0.000896 | 150 / 662 |
| Lateral OA | 0.919022 | 0.918967 | −0.000054 | 75 / 736 |
| PF OA | 0.922289 | 0.921098 | −0.001191 | 169 / 646 |
| Effusion | 0.931365 | 0.930653 | −0.000712 | 222 / 595 |
| Synovitis | 0.957218 | 0.957420 | +0.000202 | 103 / 337 |
| Baker's | 0.958342 | 0.960462 | +0.002121 | 113 / 434 |
| Contusion | 0.942874 | 0.941679 | −0.001195 | 165 / 634 |
| Fracture | 0.961824 | 0.939716 | −0.022108 | 52 / 742 |

The bootstrap samples843 exact normalized-report components, retaining all2000 draws under the fixed originally-defined12-target policy. Two supported targets lose≥0.01; the rejection is independently triggered by the negative upper confidence bound. Most changes are small, with the largest losses in MCL and fracture. This pattern does not prove label noise, optimization or representation drift as a causal explanation. Do not rescue this experiment by selecting epochs, changing labels or fusing targets retrospectively against this development fold.

Original fit `ozgung/knee-public-adaptation-fit-fold0-20261010` v1 trained432updates/3455contributors over twoepochs, before its evaluator failed on an incorrect raw-binary-truth assumption. The documented historical float32y_>=0.5 benchmark correction preserved masks and was accepted before any adaptation metric. Recovery `ozgung/knee-public-adaptation-evaluation-fold0-20261011` v2 completed, confirmed06:23UTC. Zero optimizerupdates occurred in recovery; exact original finalresume/endpoint hashes, source/input binding, complete state/cursor/orders, reconstructed endpoint average, original BN and strict real-image endpoint reload passed runtime gates. All870 fixed development studies were evaluated with inheritedK94 anatomical-mirror inputs. Recovery artifact/launch details: [recovery record](public-adaptation-evaluation-recovery-20261011.md); [prospective protocol](public-adaptation-protocol-20261010.md).

Evaluation3865.17s; recovery total3922.65s; original training20898.04s. ActualTeslaT4, torch2.10.0+cu128/timm1.0.26/Python3.12.13; peakCUDA1,933,192,192bytes, peakRSS3,790,536,704bytes. Saved private control/endpoint arrays precede metrics, so inference need not be repeated for independent recomputation. Raw receipts/predictions remain ignored: `outputs/kaggle/public-adaptation-evaluation-fold0-20261011-result/public-adaptation-fit/`.

Parent delegates independent receipt/prediction/truth/bootstrap verification to SolLow and next-method review to AstraHigh. Bestpublic0.950 is preserved; this endpoint will not be submitted or expanded. Native ROI/localizer, coverage and supervision alternatives remain hypotheses until the next explicit controlled comparison is accepted.

## Independent result verification

SolLow independently recovered only the private prediction arrays from exact evaluatorversion2, reconstructed the sorted870-study development order from fixed folds/exclusions/gold membership, loaded the SHA-pinned report truth, retained original unknown masks and historical float32 threshold, then recomputed every target metric/support, train-only prevalence, component grouping and all2000 paired bootstrap draws. All recomputed metrics and the fixed reject decision agree exactly. Source/config/binding, complete432updates, zero recoveryupdates and actual artifact SHA bindings pass. Audit **28PASS/0FAIL/2UNKNOWN**: [aggregate review](../../outputs/analysis/public-adaptation-evaluation-recovery-20261011/completed-parent-review.json).

The two limits remain explicit: tensor/BN/image-reload qualification is verified source-and-runtime-receipt evidence, not a second independent image/tensor execution; prediction rows are bound to the reconstructed immutable order and runtime loops rather than embedded study identifiers in the arrays. No second GPU inference or endpoint selection was performed. Parent accepts the rejection and the bounded native-crop geometry prototype as the next qualification step, retaining the unchanged public global model.
