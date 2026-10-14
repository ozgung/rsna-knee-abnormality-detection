# Raptor64 original methods, iteration 3: results (distillation weight, label-preserving flip)

Completed 2026-10-14 at about 11:35 UTC (14:35 UTC+3). [Protocol](raptor64-original-methods-it3-protocol-20261014.md), with kill rules recorded before launch.

**Setup.** Session `raptor64-orig-it3-20261014` (A100 40 GB), bundle v6 `e1c60001…`, cache `0bdc189c…`, teacher `dab0621a…`. The endpoint is masked report macro on the fixed 870 at K42, with a paired whole-component bootstrap (843 components, 2,000 draws, seed 42). Gold-58 is unavailable. Exposure: every distil arm carries second-order exposure through the teacher.

## Summary

| Arm | K42 macro | Δ vs in-distil 0.91490 (CI) | Five-target secondary Δ vs in-distil (CI) | Δ vs in-orig (CI) | Δ vs own44 (CI) |
| --- | --- | --- | --- | --- | --- |
| in-distil-w1 (weight 1.0) | 0.91638 | +0.0015 (−0.0015, +0.0044) | +0.0023 (−0.0018, +0.0064) | +0.0096 (+0.0065, +0.0131) | +0.0105 (+0.0061, +0.0154) |
| in-distil-flip (label-preserving, p = 0.5) | 0.91412 | −0.0008 (−0.0038, +0.0025) | −0.0049 (−0.0098, −0.0001) | +0.0073 (+0.0036, +0.0113) | +0.0082 (+0.0031, +0.0136) |

The five-target secondary is Lateral Meniscus, Lateral OA, PF OA, Medial OA and Medial Meniscus.

## Pre-registered decisions

- **w1:** the CI lower bound is ≤ 0, so the weight factor is **closed**. Δ = +0.0015 < +0.003, so **distillation scaling is saturated**.
- **flip:** the CI lower bound is ≤ 0, so the flip factor is **closed**. The five-target secondary is slightly negative.
- **Full-data bar:** the best arm is 0.91638 < 0.92183, so **not met**. Iteration 3 ends as knowledge; no own arm enters an entry under the decision document.

## Per-target Δ vs in-distil

| Target | w1 | flip |
| --- | --- | --- |
| ACL | +0.009 | −0.000 |
| MCL | −0.013 | **+0.017** |
| Medial Meniscus | +0.002 | +0.001 |
| Lateral Meniscus | +0.005 | −0.008 |
| Medial OA | −0.001 | −0.005 |
| Lateral OA | +0.005 | **−0.014** |
| PF OA | +0.001 | +0.001 |
| Effusion | +0.004 | +0.002 |
| Synovitis | +0.002 | +0.001 |
| Baker's | +0.002 | −0.000 |
| Contusion | −0.001 | −0.003 |
| Fracture | +0.004 | −0.002 |

**Reading (speculative).** The flip's effect is target-dependent. MCL gains, plausibly because the flip doubles the effective MCL views. Lateral OA and Lateral Meniscus lose, which suggests the model uses absolute image laterality as a cue (the historical trainer's note: "no flips, laterality is signal"). On average the factor nets to zero. Raising the distillation weight to 1.0 is a small and uncertain gain, mostly on ACL and the lateral targets, offset by MCL.

## Runs and artifacts

| | in-distil-w1 | in-distil-flip |
| --- | --- | --- |
| Config | `53874c54…` | `56acaa25…` |
| Package | `47196f43…` | `441436db…` |
| Controller | PASS | PASS |
| Epoch-1 K16 (stop at < 0.80) | 0.8310 | 0.8210 |
| Final epoch K16 | 0.9152 | 0.9123 |
| SWA epochs | [10, 11, 9] | [11, 10, 9] |
| Checkpoint SHA | `233c8b78…` | `4a0f345e…` |
| Final backup | epoch 11, version 13 | epoch 11, version 13 |
| Private dataset | `ozgung/knee-raptor64-in-distil-w1-fold0-final-20261014` | `ozgung/knee-raptor64-in-distil-flip-fold0-final-20261014` |
| Verification | exporter timed out on the archive build at 2,700 s; per-file 13/13 match, private | per-file 13/13 match, private (session stopped before the exporter's archive wait finished) |

- **Timing:** launched 06:12 UTC; both arms concurrent at about 21 min per epoch, 8.0 GB GPU memory, 100% utilization; training ended at about 10:30 UTC.
- **Evidence:** `outputs/analysis/raptor64-20261013/it3/`, containing `endpoint-it3-v1.json`, the per-file verification receipts and the downloaded final files.
