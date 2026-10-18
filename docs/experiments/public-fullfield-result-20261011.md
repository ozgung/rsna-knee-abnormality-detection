# Frozen full-field result — October 11, 2026

The fixed central320 versus full384 comparison completed with runtime mechanics PASS and a positive paired development result. Control macro AUC is **0.9513309087200222**; full384 is **0.9539771134498085**; delta **+0.0026462047297863**, with a 2000-draw component-paired 95% interval **[+0.0014776379496809, +0.0038455586416489]**. All twelve target deltas are positive; no inherited damage-rule target is present. Train-only prevalence macro AUC is 0.5. This meets the prospective gate for a fixed second-fold replication and separate image-only rehearsal.

Independent saved-artifact audit: **27 PASS, 0 FAIL, 2 UNKNOWN**. The audit reconstructs official fold membership, historical float32 thresholded truth and masks, target support, exact normalized-report components, train-only prevalence, and all per-target/macro AUCs. Its separate pairwise Mann–Whitney comparison matrices give half credit to ties and reproduce the interval exactly, using the same prospective seed42/2000 whole-component multiplicities. All draws are defined over the original twelve targets; 843 components cover 870 studies (largest14). Fresh control arrays are byte-identical to the previous recovery control.

The result binding equals the immutable build and accepted scientific constants: revision `c76388e132225bbcb268c305656f55f235b11821`, bundle `3937b4877e1a71f50f4a76a59e78b82f339d0ac9daf45884e497ed2ac274ce5f`, notebook `553760d78cc45dae5a240bfdd819f126a976da8e34efb69163381e3d297fda43`. Exact population order and prediction-file digest match. Both870×12 float32 arrays contain finite probabilities in [0,1]. Receipt/source gates report zero optimizer updates, unchanged486 persistent tensors, exactly32 constructor-qualified nonpersistent grid differences, final state/buffer immutability, and all32 nongold training-only mechanics probes. Reference/fast/window/repeat/reload and saved-control checks passed; probability errors are exactly zero. These runtime assertions were inspected, without another CUDA/image/checkpoint execution.

Runtime was7420.45s total,6548.65s evaluation and838.98s qualification, with peak CUDA2,568,470,528B and RSS2,799,906,816B. Environment: Python3.12.13, torch2.10.0+cu128, timm1.0.26, NumPy2.0.2, PyArrow24.0.0, TeslaT4. No fitting, submission, or public score was produced.

Private evidence remains ignored: `outputs/kaggle/public-fullfield-fold0-20261011-result/public-fullfield/`; reproducible independent audit script and aggregate JSON: `outputs/analysis/public-fullfield-fold0-20261011/independent_result_audit.py` and `independent-result-audit.json`. The two UNKNOWNs are an independent image/tensor rerun and competitive generalization. Public all-data checkpoint exposure and absent patient identities prevent an independent generalization claim; exact report components do not identify all repeat patients or near-duplicate reports. Positive development evidence supports replication, not candidate promotion or a leaderboard forecast. See the [implementation/protocol](public-fullfield-fold0-20261011.md).

## Per-target development evidence

| Target | Positive / negative / missing | Central320 AUC | Full384 AUC | Delta |
| --- | --- | --- | --- | --- |
| ACL | 109 / 716 / 45 | 0.982023 | 0.984701 | +0.002678 |
| MCL | 20 / 790 / 60 | 0.953038 | 0.954367 | +0.001329 |
| Medial Meniscus | 330 / 496 / 44 | 0.966532 | 0.967412 | +0.000880 |
| Lateral Meniscus | 109 / 713 / 48 | 0.963882 | 0.964281 | +0.000399 |
| Medial OA | 150 / 662 / 58 | 0.957563 | 0.957936 | +0.000373 |
| Lateral OA | 75 / 736 / 59 | 0.919022 | 0.923279 | +0.004257 |
| PF OA | 169 / 646 / 55 | 0.922289 | 0.927593 | +0.005303 |
| Effusion | 222 / 595 / 53 | 0.931365 | 0.934333 | +0.002968 |
| Synovitis | 103 / 337 / 430 | 0.957218 | 0.961079 | +0.003860 |
| Baker's | 113 / 434 / 323 | 0.958342 | 0.960524 | +0.002182 |
| Contusion | 165 / 634 / 71 | 0.942874 | 0.946382 | +0.003508 |
| Fracture | 52 / 742 / 76 | 0.961824 | 0.965841 | +0.004017 |

All12 targets have both observed classes; missing observations remain masked. The largest point gain is PF OA, followed by lateral OA and fracture. The full-field/grid change is a combined intervention; these associations do not establish which anatomical border supplies the gain.
