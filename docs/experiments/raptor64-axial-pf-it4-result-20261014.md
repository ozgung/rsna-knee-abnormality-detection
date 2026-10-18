# Iteration 4 result: axial patellofemoral arm, 2026-10-14 — CLOSED (ax-crop fails rules 1, 4 and 5)

Protocol: [raptor64-axial-pf-it4-protocol-20261014.md](raptor64-axial-pf-it4-protocol-20261014.md). Branch `claude/axial-pf-it4-20261014`:
- arm code at `05ae207` and `8f0ff34`;
- derived axial cache at `5bf0fe1`.

The final run is a fresh start; the two earlier attempts are described under "Earlier attempts".

## Read-out

**Setup.**
- **Population:** fixed 870 fold-0 development studies, masked weak report labels.
- **Comparator:** in-distil K42 (`0eb638e2…`); PF OA AUC 0.8637 (169 positive, 646 negative, 55 masked).
- **Bootstrap:** paired whole-component, 2,000 draws, seed 42.
- **Blend:** fixed 0.7 × in-distil + 0.3 × arm, per-target normalised average ranks.
- **Arm predictions:** the exported SWA checkpoint at K10, which is every axial window, 870 rows. These are the final-dataset exports `oof_fold0.csv`, not the in-training selection file.
- **Axial content:** all 870 studies have it, so no fallback was used.
- **Output:** `outputs/it4/it4-axial-blend-result-v1.json` (`a1180b0f…`).

| | ax-crop | ax-full (control) |
| --- | --- | --- |
| Blended PF OA (Δ vs in-distil, 95% CI) | 0.8651 (+0.0014, −0.0052 to +0.0080) | 0.8731 (+0.0094, +0.0025 to +0.0163) |
| Best single epoch, blended Δ | +0.0015 (epoch 7) | +0.0071 (epoch 8) |
| Arm-alone PF OA: SWA / best epoch | 0.832 / 0.828 | 0.855 / 0.851 |
| Arm-alone macro (12 targets) | 0.860 | 0.884 |
| Dread v4_full K42 gate (0.9295 alone) | 0.9187 (−0.0108, −0.0183 to −0.0039) | 0.9240 (−0.0055, −0.0124 to +0.0014) |
| Effusion blended Δ (secondary only) | +0.0009 | +0.0046 |
| Synovitis blended Δ (secondary only) | −0.0012 | +0.0042 |

Blended crop minus blended full: −0.0080 (−0.0149 to −0.0013).

**Arm-alone AUC per target, all 870:**

| Target | ax-crop | ax-full |
| --- | --- | --- |
| ACL | 0.918 | 0.911 |
| MCL | 0.866 | 0.881 |
| Medial Meniscus | 0.851 | 0.868 |
| Lateral Meniscus | 0.754 | 0.806 |
| Medial OA | 0.905 | 0.913 |
| Lateral OA | 0.817 | 0.854 |
| PF OA | 0.832 | 0.855 |
| Effusion | 0.909 | 0.923 |
| Synovitis | 0.920 | 0.935 |
| Baker's | 0.816 | 0.914 |
| Contusion | 0.853 | 0.865 |
| Fracture | 0.876 | 0.877 |

**Kill rules for ax-crop:**

| Rule | Result |
| --- | --- |
| 1. Δ ≥ +0.010 with CI lower bound > 0 | **fail** |
| 2. Arm-alone PF OA ≥ 0.80 | pass (0.832) |
| 3. Δ has the same sign for SWA and best epoch | pass (both positive) |
| 4. Crop beats full | **fail**: the crop is significantly worse |
| 5. Dread gate does not lower PF OA | **fail**: −0.011 |

**Verdict: close the axial-arm idea**, as pre-registered.

ax-full is the control, not a candidate. Its +0.009 is below the +0.010 bar, and it also lowers the Dread blend, so it does not qualify either.

These are exposed, report-selected development numbers: top-3 SWA was chosen on these same 870 studies, and the teacher has second-order exposure. They are not a public-score forecast.

## Orientation (protocol step 1)

- `knee.raptor_input.prepare_study` does not canonicalise in-plane orientation. The pinned reader (`knee-mri-twelve-findings…py` lines 255–292, SHA `859aae12…`) uses IOP only to sort slices, and stores pixels in DICOM order.
- I implemented the protocol's centroid heuristic (`axial_anterior_side`) and checked it by eye before training:

  | Sample | Heuristic result |
  | --- | --- |
  | 20 random studies | correct on 19 |
  | 4,345 studies | 60 called anterior-bottom |
  | 20 of those 60, inspected | all 20 wrong: each was anterior-top |
  | 10 studies most likely to be rotated | none rotated |

- **Deviation, recorded before training:** the arms use storage order, which puts anterior at the top. The heuristic is kept for audit only, and the p = 0.5 anterior–posterior flip is applied with labels and teacher unchanged.
- All 4,345 studies have axial content.

## Runs and runtime

**Final run (`raptor64-axial-it4-r3-20261014`):**
- One A100 40 GB, allocated 13:49 UTC, terminal at 19:22, then stopped. It ran 5.5 h without a loss.
- Both arms ran concurrently: about 26 min per epoch, 7.9 GB GPU memory.
- Exported datasets (private, verified): `ozgung/knee-raptor64-{ax-crop,ax-full}-fold0-final-20261014`.

| | ax-crop | ax-full |
| --- | --- | --- |
| SWA epochs | [7, 8, 5] | [8, 11, 9] |
| Checkpoint | `aaa80b42…` | `dd689702…` |
| Final backup | epoch 11, version 15 | epoch 11, version 15 |
| Training ended (UTC) | 19:08 | 19:20 |

- **Seed and config:** seed 42. Configs `219516ff…` (ax-crop) and `02f71fea…` (ax-full); bundle `b25279c6…`; runner `60bf96b0…`.
- **Cache:**
  - The derived axial-only cache was built by the private Kaggle CPU kernel `ozgung/knee-raptor64-axial-cache-20261014` in 465 s. Every study's source hash was verified, and its axial bytes were re-read and compared.
  - It is 4.22 GB in four pinned tars; pins `ce59a6f6…`, content `021288ad…`.
  - It stages in about 4 minutes, against about 30 for the full cache.
  - Tests: `tests/test_raptor64_axial_cache.py` (3) prove the arms see byte-identical windows, targets and masks, and cover the derive→stage round trip with the unchanged loader.
  - Because the cache identity is new, the epoch-0 backups from the full-cache run could not be resumed. Their identity pins the cache content, so training restarted from epoch 0.
- **Diagnostics:** host RAM was logged every 60 s. Minimum available RAM was 72.6 of 87.5 GB. Peak process RSS was 5.05 GB; it grew stepwise at epoch ends, consistent with the retained top-3 state. **There is no OOM signature.**
- **Chain driver:** `outputs/it4/chain_driver.py` polled the VM with `colab exec` every 60 s and needed no second VM.

**Earlier attempts:**
- `raptor64-axial-it4-20261014` and `-r2-` were lost 4,578 s after allocation. In both, the keep-alive returned 404 from iteration 76, and there was no local stop or unassign. Those runs had sparse execs.
- r3 polled every 60 s and survived; the cause of the earlier losses is unresolved.

**Tests:**
- Locally with `.venv/bin/python`: the raptor64 suites ran 119 OK with 6 skipped, plus the 3 new cache tests.
- On the VM (timm 1.0.26): 118 OK with 1 skipped, including the axial end-to-end test.
