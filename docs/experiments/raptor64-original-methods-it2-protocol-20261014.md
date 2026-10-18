# Raptor64 original methods, iteration 2: protocol (E1, M1 mirror, M2 distil)

Written 2026-10-14 at about 23:30 UTC on 2026-10-13 (02:30 UTC+3), before any iteration-2 result. Plan: `docs/research/original-methods-plan-20261014.md` (M1, M2, E1). Infrastructure, cache and recipe: [raptor64 protocol](raptor64-dread-init-protocol-20261013.md).

## Fixed comparator and single factors

The baseline is **in-orig**:
- ImageNet-384 init, original head, `labels_v1_blend`;
- fixed fold-0 split (3,475 train, 870 development), seed 42, 12 epochs, OneCycle;
- K8 training windows, top-3 SWA selected per epoch by K16 report-masked macro, K42 export;
- the raptor64 Dread v4 contract cache (content `0bdc189c…`).

Its K42 macro is 0.90683, against own44 K42 at 0.90588 (`outputs/analysis/raptor64-20261013/endpoint-in-orig-v1.json`). Each new arm changes exactly one factor against in-orig. A test proves the in-orig trainer, entry and recipe built from current code are byte-identical to the frozen bundle v1 package.

| Arm | Single factor | Implementation |
| --- | --- | --- |
| `in-mirror` (M1) | mirror-equivariant training | see M1 |
| `in-distil` (M2) | distillation toward Dread v4_full | see M2 |

**M1 (`in-mirror`).** Training samples are anatomically mirrored with p = 0.5:
- slice order is reversed inside each sagittal slot (0–17 and 18–31), and image width is flipped in the coronal and axial slots (32–63);
- the mask is mirrored identically;
- labels are permuted Medial↔Lateral for Meniscus and OA (`SIDE_PERM` = [0, 1, 3, 2, 5, 4, 6, …]), and the MCL loss is masked on mirrored samples;
- the loss is the historical pos-weighted BCE with a per-element mask, identical to the historical loss when the mask is all ones.

The mirror decision draws one Python random number before the historical sampler runs. Window count (K8 train, K16/K42 eval), centres and evaluation items are otherwise unchanged. Mirror TTA at export, as the plan specifies, is written as a **secondary** file, `oof_fold0_mirror_tta.csv`: the mean of plain and mirrored-then-swapped-back probabilities, with MCL from the plain pass. The **primary** `oof_fold0.csv` is plain K42, so that the training factor is tested alone.

**M2 (`in-distil`).**
- **Loss:** the historical report-label loss (pos-weighted BCE toward `labels_v1_blend`, dense as now) plus a fixed 0.5 × plain BCE toward the teacher's probabilities on all 12 targets.
- **Teacher:** Dread v4_full CC0, full model, at K42 public eval windows on all 4,345 cached studies. The file SHA is pinned in the package recipe before launch.
- **Weight:** fixed at 0.5 and not tuned.
- **Deviation from the plan:** the plan's phrase "teacher only where the report label is masked" does not apply, because the current training loss has no mask. The coordinator's specification (report loss + 0.5 × teacher loss) is followed.
- **Exposure:** the development studies are never used for training, but the teacher saw them, so in-distil's development numbers are **near-clean with second-order exposure**.

**Code.** Both variants are subclasses and loss wrappers in `knee/raptor64_training.py` (`recipe_variant`, `loss_variant`, `mirror_tta_csv`), added to the trainer by two hook lines only for these arms. The historical sampler, model, optimizer, schedule, `lossf` assignment, training loop, top-3 and SWA AST stay identical.

**Tests:** `tests/test_raptor64_variants.py` and `tests/test_raptor64_e2e.py`. They check:
- mirror twice is the identity;
- the sampler and window count are unchanged;
- the masked loss equals the historical loss;
- the distil loss formula and teacher pinning;
- CPU end-to-end runs of both arms, including the TTA CSV.

## E1 plane-group ablation (no training; gates M3)

The input is the in-orig final SWA (`e9e131a1…`) on the 870 at K42. For each study the standard K42 window plan is computed. The windows whose centre lies in one plane group are then dropped: sagittal (slots 0–31), coronal (32–51) or axial (52–63). Window features are computed once and the attention head is rerun on the kept windows. The output is the per-target AUC delta against the full pass, with the paired component bootstrap.

Dropping windows was chosen over zeroing slices because, with the legacy centre rule, zeroed interior slots would still produce black windows. The plan's M3 gate:
- removing sagittal costs the weak targets (Lateral OA, PF OA, Lateral Meniscus, MCL) **less than 0.01**: a coronal/axial specialist is plausible;
- it costs **0.03 or more**: kill M3.

## Endpoints, decisions and health

- **Primary:** masked report macro on the fixed 870 at K42, with the pinned own224 endpoint truth, masks and 843 groups. Paired bootstrap (2,000 draws, seed 42) against in-orig (`in-orig-final-download/oof_fold0.csv`) and against own44 K42, with per-target results. Secondary: the mean of LM, Lateral OA, PF OA, Medial OA and MM.
- **M1 kill:** macro Δ vs in-orig with CI lower ≤ 0 **and** five-target secondary Δ < +0.005. TTA results are reported, but do not decide M1.
- **M2 kill:** K42 macro < in-orig + 0.015 (that is, < 0.92183). Promotion to a full-data fit needs ≥ 0.925.
- **Gold-58 sign check: not available.** The 58 gold studies are excluded from the staged cache, so a sign check would need a separate gold cache build. It is not run here.
- **Health stop:** epoch-1 K16 < 0.80 sends SIGTERM to that arm's controller, which finalizes its backup. The VM-side autopilot enforces this.

## Execution

- **Session:** `raptor64-orig-it2-20261014` (A100 40 GB), with a 40-minute keepalive.
- **Bundle:** v4, the v3 tools plus the iteration-2 code and the `in-mirror` and `in-distil` packages, with the teacher SHA pinned.
- **Supervision:** a VM-side autopilot launches the arms, applies the health stop and exports privately with `--ready-timeout 2700`. Files are then verified one by one locally.
- **Concurrency:** both arms run together if the probe allows; each was measured at 2.74 GB, with about 4 GB used during solo training.
- **Datasets:** private final datasets `ozgung/knee-raptor64-{in-mirror,in-distil}-fold0-final-20261014`.
