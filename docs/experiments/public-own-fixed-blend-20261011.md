# Fixed public/own fold0 diagnostic — 2026-10-11

The single predeclared 80% frozen public control + 10% original384 + 10% R2-224 comparison is mechanically PASS (24 checks). Fixed triage: **reject**. No rejected adaptation endpoint, gold scoring, weight search, training or downloads were used.

| Prediction | Report macro AUC |
| --- | --- |
| control | 0.951330909 |
| original384 | 0.912771053 |
| R2_224 | 0.909008784 |
| fixed_blend | 0.948295575 |

Blend minus control: **-0.003035334**; paired 2,000 normalized-report-component draws, seed42, 95% CI **[-0.004349611, -0.001943321]**. 2000 valid draws across 843 components. Damage requires delta ≤−0.01 and at least20 observed examples/class; flagged: none.

All870 exact fold0 validation IDs were independently aligned to the sorted public order; gold58/quarantine4 excluded. All12 targets defined; original float32 threshold ≥0.5 and observation masks preserved. Macro is the unweighted mean over defined targets. CSV schema, unique IDs, exact coverage, finite [0,1] probabilities and unchanged SHA256 inputs passed. Gold CSV rows were ignored.

| Target | Positive | Negative | Missing | Control AUC | Blend AUC | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ACL | 109 | 716 | 45 | 0.982023 | 0.980242 | -0.001781 |
| MCL | 20 | 790 | 60 | 0.953038 | 0.948354 | -0.004684 |
| Medial Meniscus | 330 | 496 | 44 | 0.966532 | 0.963862 | -0.002670 |
| Lateral Meniscus | 109 | 713 | 48 | 0.963882 | 0.958503 | -0.005378 |
| Medial OA | 150 | 662 | 58 | 0.957563 | 0.954814 | -0.002749 |
| Lateral OA | 75 | 736 | 59 | 0.919022 | 0.912754 | -0.006268 |
| PF OA | 169 | 646 | 55 | 0.922289 | 0.919303 | -0.002986 |
| Effusion | 222 | 595 | 53 | 0.931365 | 0.930926 | -0.000439 |
| Synovitis | 103 | 337 | 430 | 0.957218 | 0.956872 | -0.000346 |
| Baker's | 113 | 434 | 323 | 0.958342 | 0.958179 | -0.000163 |
| Contusion | 165 | 634 | 71 | 0.942874 | 0.941248 | -0.001625 |
| Fracture | 52 | 742 | 76 | 0.961824 | 0.954489 | -0.007335 |

This is exposed development evidence. Public all-data warmstart and R2 globalOOF/gold-guided development/full-fold selection limit inference. Legacy384 fold0 and R2 selected tensors still require artifact qualification; no image-only rehearsal or public-score forecast follows. Patient identity is unavailable, and normalized-report components cannot detect unknown repeat patients/near duplicates. CSV hashes bind these inputs rather than qualifying the weights. Public prediction row binding is source/preflight evidence.

Close this fixed probe without rescue or weight search.

Aggregate receipt with SHA256 inputs, checks, per-target supports and bootstrap: [receipt](../../outputs/analysis/public-own-fixed-blend-20261011/receipt.json). Authorization: [shared handoff](../claude-astra-handoff.md), 06:35UTC.
