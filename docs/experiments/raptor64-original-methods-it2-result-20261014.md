# Raptor64 original methods, iteration 2: results (E1, M1 mirror, M2 distil)

Completed 2026-10-14 at about 05:30 UTC (08:30 UTC+3). The protocol was written before launch: [protocol](raptor64-original-methods-it2-protocol-20261014.md). Session `raptor64-orig-it2-20261014`, A100 40 GB. Bundle v5 (`43f8ee6e…`); its packages are identical to v4, and only the controller probe fix differs. The cache content is `0bdc189c…`. All numbers are on the fixed 870 fold-0 development set: masked report macro AUC at K42, paired whole-component bootstrap (843 components, 2,000 draws, seed 42). Gold-58 sign check: **not available**, because the gold studies are not in the staged cache.

## Summary

| Arm | K42 macro | Δ vs in-orig (CI) | Five-target secondary Δ (CI) | Δ vs own44 K42 (CI) | Pre-registered decision |
| --- | --- | --- | --- | --- | --- |
| in-orig (comparator) | 0.90683 | – | – | +0.0010 (−0.0035, +0.0052) | – |
| **in-distil** (M2) | **0.91490** | **+0.0081 (+0.0052, +0.0111)** | +0.0117 (+0.0074, +0.0165) | +0.0090 (+0.0045, +0.0138) | **kill under the rule**: < in-orig + 0.015 (0.92183) |
| in-mirror (M1, plain K42) | 0.88863 | −0.0182 (−0.0253, −0.0111) | −0.0282 (−0.0369, −0.0196) | −0.0172 (−0.0252, −0.0093) | **kill**: CI ≤ 0 and secondary < +0.005 |
| in-mirror + mirror TTA (secondary) | 0.88895 | −0.0179 (−0.0251, −0.0107) | −0.0272 | −0.0169 | not a decision endpoint |

**M2 per-target Δ vs in-orig:** Lateral OA +0.020, PF OA +0.016, Lateral Meniscus +0.016, MCL +0.013, ACL +0.010, Medial OA +0.008, Contusion +0.008, Effusion +0.004, Baker's +0.003, Fracture +0.002, Medial Meniscus −0.001, Synovitis −0.002. The gain is broad and largest on the weak lateral/PF/MCL targets. It is **near-clean with second-order exposure**: the teacher (Dread v4_full, which scores an exposed 0.9596 on these 870) was trained on these development studies, so the gain is an upper bound on a clean distillation gain. The pre-registered kill rule (+0.015) triggers, even though the gain is significant. Whether to overrule a pre-registered rule is the parent's decision, not mine.

**M1 per-target Δ vs in-orig:** Lateral Meniscus −0.089, MCL −0.064, Medial Meniscus −0.032, Medial OA −0.017, Lateral OA −0.008, ACL −0.006, PF OA +0.005, and the rest within ±0.003. The damage is concentrated exactly on the side-swapped and masked targets. **Likely cause: the method as specified is anatomically wrong.** Mirroring a right knee left-to-right produces a valid *left* knee whose medial compartment is still the medial compartment. Mirroring changes laterality, not anatomy. The Medial↔Lateral label swap therefore mislabels every mirrored sample, and masking MCL halves its supervision. A correct mirror augmentation would keep the labels unchanged and train MCL on all samples, as a laterality-invariance augmentation. That is a different hypothesis and was not tested. The historical trainer's "no flips (laterality is signal)" choice also remains untested in its label-preserving form.

## E1 plane-group ablation (gates M3)

The in-orig final SWA (`e9e131a1…`) was run at K42 on the 870, dropping the windows whose centre lies in a plane group. On average 21 sagittal, 13 coronal and 8 axial windows remain per study. Full-pass parity with the exported K42 is a max absolute difference of 2.0e-5.

| Dropped | Macro Δ (CI) | Mean Δ on Lateral OA, PF OA, Lateral Meniscus, MCL (CI) | Notable targets |
| --- | --- | --- | --- |
| sagittal | −0.0281 (−0.0344, −0.0222) | −0.0336 (−0.046, −0.023) | ACL −0.122, Lateral Meniscus −0.103, PF OA −0.041, MM −0.026; Lateral OA +0.007, MCL +0.003 |
| coronal | −0.0067 (−0.0096, −0.0040) | −0.0101 (−0.017, −0.004) | MCL −0.037, Contusion −0.015, MM −0.010 |
| axial | −0.0027 (−0.0059, +0.0001) | −0.0034 (−0.009, +0.001) | Fracture −0.011, Lateral OA −0.009 |

**M3 gate:** removing sagittal costs the weak targets 0.034 on average (≥ 0.03), so M3 is killed as specified. Lateral Meniscus and PF OA depend on sagittal. Lateral OA and MCL do not, and MCL is carried by coronal, so a narrow coronal specialist for MCL and Lateral OA alone is not excluded.

## Runs and artifacts

| Item | in-mirror | in-distil |
| --- | --- | --- |
| Config | `d68ffd7f…` | `218830584…` |
| Package manifest | `eae3a773…` | `1b47c709…` |
| Controller | PASS | PASS |
| Epoch-1 K16 (health stop 0.80) | 0.8117 | 0.8296 |
| Final epoch K16 | 0.8860 | 0.9124 |
| SWA epochs | [10, 9, 11] | [10, 11, 9] |
| Checkpoint SHA | `42d93646…` | `df836f38…` |
| Final backup | epoch 11, version 13 | epoch 11, version 13 |
| Private final dataset | `ozgung/knee-raptor64-in-mirror-fold0-final-20261014` | `ozgung/knee-raptor64-in-distil-fold0-final-20261014` |
| Verification | per-file, 14 files match, private | exporter verified (ready-timeout 2700) and per-file, 13 files |

- **Timing:** about 21 min per epoch with both arms concurrent, 8.0 GB of GPU memory, 100% utilization; 12 epochs from 00:23 to 04:42 UTC, then about 10 min of K42 export.
- **Teacher:** Dread v4_full K42 on 4,345 studies in 608 s, npz `dab0621a…`.
- **Attempt 1** (00:18 UTC) failed in the controller's pre-launch probe: the probe applied plain BCE to the 24-wide variant targets. It was fixed with `y[:, :12]`, covered by a regression test, and relaunched as attempt 2; attempt 1 is preserved on the VM.
- **Evidence:** `outputs/analysis/raptor64-20261013/it2/`, containing:
  - `endpoint-it2-v1.json`, `e1-result-v1.json`, `e1-receipt.json`, `teacher-receipt.json`;
  - the per-file verification receipts;
  - the downloaded final files.

## Next decision (for the parent)

- M1 as specified: closed. A label-preserving mirror augmentation would be a new, separately pre-registered arm.
- M2: significant, broad, near-clean +0.008, below its own +0.015 bar. Options are to accept the kill, or to pre-register a revised decision (for example a full-data student for the B slot) with the exposure caveat stated.
- M3: closed by E1, except a possible narrow MCL/Lateral OA coronal head.
