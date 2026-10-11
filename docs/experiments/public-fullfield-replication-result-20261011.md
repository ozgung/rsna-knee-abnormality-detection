# Full-field second-fold replication — October 11, 2026

The fixed fold1 evaluation completed with mechanics PASS. Central320 macro AUC **0.953149601496** improves to full384 **0.954966430142**: delta **+0.001816828647**, paired 2000-component 95% interval **[+0.000803860400, +0.002857967227]**. All twelve targets are defined; ten improve. Medial Meniscus and Synovitis decline by less than 0.0012; the predeclared damage set is empty. The positive second-fold interval meets the [prospective replication gate](public-fullfield-replication-protocol-20261011.md) for one separate image-only offline rehearsal, before any competition entry.

Independent audit: **51 PASS, 0 FAIL, 2 UNKNOWN**. It independently reconstructs 4345 eligible / 3479 training / 866 development studies and 842 normalized-report components, exact frozen32 training-only probes, gold/quarantine exclusions, canonical folds, official schema, masked float32 truth, all supports and training-only prevalence. Separate Mann–Whitney pairwise matrices with half credit for ties reproduce per-target/macro AUC and the seed42/2000 paired whole-component interval. All2000 draws are defined. Prevalence macro AUC is0.5.

Saved original-source central320 predictions match the fresh control exactly on all32 mechanics probes. Same decoded public384/K94/anatomical mirror inputs, original checkpoint and all486 persistent tensors remain unchanged; only the32 allowed constructor-derived nonpersistent grids differ. Receipt/source evidence reports exact reference/fast volume/mask/window equality, zero probability errors for reference/repeat/strict reload, no optimizer, no intermediate metric opening and final all-buffer immutability. This is an audit of bound saved artifacts, not another independent CUDA/image execution.

One accepted run: `ozgung/knee-public-fullfield-replication-fold1-20261011` v1/kernel138159578, COMPLETE observed14:24UTC. Source revision `c0fac03bf896589fd913f9b7929732e0a1e751cc`, bundle `a052e3945ba1742d1ae286eaaad6e900ae98bac64b06c56bc26bca9ec7ca12d9`. Receipt SHA256 `482ea13858274dce4e4e3cfffd04c3c9a9ee332eb927ed680417b7c97d2057c9`. Runtime 6753.30s total (925.76s mechanics; 5796.59s development), peak CUDA 2,572,664,832B and RSS 2,878,300,160B. Python3.12.13, torch2.10.0+cu128, timm1.0.26, NumPy2.0.2, PyArrow24.0.0, TeslaT4. Zero pixel/header/resource failures and zero empty studies; source-successful missing-plane/partial masks remain explicit.

Private receipts and arrays remain ignored under `outputs/kaggle/public-fullfield-replication-fold1-20261011-result/public-fullfield-replication/`. Reproducible independent audit: `outputs/analysis/public-fullfield-replication-fold1-20261011/independent_result_audit.py`; aggregate `independent-result-audit.json`. UNKNOWNs remain independent CUDA/image/checkpoint rerun and competitive generalization. Public all-data checkpoint exposure and unavailable patient identities persist: these are replicated development improvements, not independent validation or a public-score forecast. The [first-fold result](public-fullfield-result-20261011.md) was +0.00264620 with positive interval and all12 targets improving.

## Per-target development evidence

| Target | Positive / negative / missing | Central320 AUC | Full384 AUC | Delta |
| --- | --- | --- | --- | --- |
| ACL | 105 / 718 / 43 | 0.976986 | 0.980740 | +0.003754 |
| MCL | 24 / 792 / 50 | 0.977536 | 0.979535 | +0.001999 |
| Medial Meniscus | 342 / 487 / 37 | 0.966155 | 0.965315 | -0.000841 |
| Lateral Meniscus | 126 / 691 / 49 | 0.961363 | 0.964969 | +0.003606 |
| Medial OA | 154 / 671 / 41 | 0.929278 | 0.931629 | +0.002352 |
| Lateral OA | 65 / 750 / 51 | 0.919262 | 0.921026 | +0.001764 |
| PF OA | 166 / 654 / 46 | 0.922055 | 0.923096 | +0.001041 |
| Effusion | 206 / 612 / 48 | 0.944944 | 0.948656 | +0.003712 |
| Synovitis | 78 / 337 / 451 | 0.974815 | 0.973674 | -0.001141 |
| Baker's | 109 / 426 / 331 | 0.953138 | 0.953310 | +0.000172 |
| Contusion | 173 / 603 / 90 | 0.951102 | 0.954304 | +0.003202 |
| Fracture | 51 / 728 / 87 | 0.961161 | 0.963343 | +0.002182 |

Macro is the unweighted mean over the original twelve observed two-class targets. Missing observations stay masked; no label, prevalence, crop, checkpoint, window or target-weight search occurred.

Next action: implement and exercise the [standalone image-only rehearsal](candidate-rehearsal-readiness-20261011.md) on the same frozen32 probes plus all3 official visible test studies. Pure entry assets must exclude labels, reports, folds and private prediction fixtures. Rehearsal must bind exact qualified primitives, checkpoint/grids/state, same-image probabilities and official output schema, and measure actual resource/runtime evidence. S6 remains a separate fixed70:30 comparison; no full384 substitution.
