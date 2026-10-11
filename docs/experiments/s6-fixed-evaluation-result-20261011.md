# S6 fixed comparison result — October 11

The prospective 70% original central320 / 30% S6 percentile-rank blend completed on all 870 frozen development studies. Macro AUC **0.951330909 → 0.962963989**, delta **+0.011633080**, paired 2,000-component 95% CI **[+0.009787471, +0.013714779]**. All 12 targets improve; inherited damage rule does not fire. Runtime reports PASS; independent saved-array audit **73 PASS / 0 FAIL / 3 UNKNOWN** reproduces the exact comparison and qualifies a separate image-only rehearsal. This is exposed report-derived development evidence, not a public score forecast or independent generalization.

## Fixed protocol and result

S6 uses all four unchanged released ConvNeXt readers, native 384 renders with central352 retained crops, v4 views, physical3.5mm triplets, stride1, CUDA fp16/chunk16. Original central320 control stays separate from the independently qualified full384 method. Source Python `%.6f` CSV quantization precedes average-percentile ranks over all870 rows per target; blend is fixed70:30. No target weights, intermediate metric selection, optimizer or window reduction.

| Target | Positive / negative observed | Original320 AUC | Fixed blend AUC | Delta |
| --- | --- | --- | --- | --- |
| ACL | 109 / 716 | 0.982023 | 0.988327 | +0.006304 |
| MCL | 20 / 790 | 0.953038 | 0.965063 | +0.012025 |
| Medial Meniscus | 330 / 496 | 0.966532 | 0.974465 | +0.007933 |
| Lateral Meniscus | 109 / 713 | 0.963882 | 0.971628 | +0.007746 |
| Medial OA | 150 / 662 | 0.957563 | 0.967855 | +0.010292 |
| Lateral OA | 75 / 736 | 0.919022 | 0.936531 | +0.017509 |
| PF OA | 169 / 646 | 0.922289 | 0.944327 | +0.022038 |
| Effusion | 222 / 595 | 0.931365 | 0.950511 | +0.019146 |
| Synovitis | 103 / 337 | 0.957218 | 0.965919 | +0.008700 |
| Baker's | 113 / 434 | 0.958342 | 0.962756 | +0.004415 |
| Contusion | 165 / 634 | 0.942874 | 0.957303 | +0.014430 |
| Fracture | 52 / 742 | 0.961824 | 0.970882 | +0.009058 |

Standalone S6 descriptive macro AUC0.969493847 is retained as a descriptive result. It does not replace the prospectively selected blend. Prevalence AUC0.5; all12 targets defined; 843 exact normalized-report components, all2000 bootstrap draws valid. Unknown patient identity and public all-data/checkpoint exposure limit interpretation.

## Runtime and recovery

`ozgung/knee-public-s6-evaluation-fold0-20261011` v1/kernel138162038, private/offline NvidiaTeslaT4, COMPLETE observed2026-10-11 18:35UTC. Reviewed source `dbb9d5ac3d2dd2e6c0328a7de0a9fac4c0f8ca70`; immutable826400-byte notebook SHA `b6b1b717c3bc8e8f2a8596bd9ae892f2e040cd05f68606f6b9012a6109319de5`, sourcebundle `2773d29fdc368b21764248e8aa4c42511a96405698dc5dfd5e0bcdad4f3abb7c`.

Parent runtime21731.24seconds (~6.04h); S6 worker21563.04seconds. Complete502660windows, unchanged state digest, zero failed/imputed studies and zero fp32 recoveries. PeakCUDA2.707GB, native scratch767.8MB; scratch cleaned after each study. Two TeslaT4 boards were available, but this evaluation used cuda:0 only. Scaling S6 alone to1300 studies is near9h, before the public arm, setup and tails. A combined hidden entry has not passed the actual offline runtime gate.

Small receipts and complete arrays: `outputs/kaggle/public-s6-evaluation-fold0-20261011-result/s6-evaluation/`. Comparison NPZ SHA `6fc8f0053fb28d6f77352b49b3384c089811c07e20f5041944abf2aeb7ad7d28`; four-reader NPZ SHA `655d038227054478f4ae4ef58ae22807d948d2ec0e43c094c0a65654a1aac1c1`. Kaggle CLI default pagination initially missed arrays; a bounded direct two-page SDK inventory collected only the two small NPZs (1.33MB), preserving original logs/receipts and printing no data IDs or signed URLs.

## Next gate

Independent pairwise Mann–Whitney AUC with half ties, NumPy average-rank implementation plus actual pandas oracle, float32 truth, masks/supports, training prevalence and exact report-component seed42/2000 bootstrap all pass. The three UNKNOWNs are independent CUDA rerun, competitive generalization and combined hidden runtime. Evidence: `outputs/analysis/public-s6-evaluation-20261011/independent-result-audit.json` and reproducible `independent_result_audit.py`. Parent accepts the existing positive-CI/no-harm gate. Prepare an image-only rehearsal that preserves per-study outputs while using both real T4s for deterministic study sharding; exact cross-board parity, complete-population rank order, strict state/source checks and actual runtime evidence are required. No competition submission yet.
