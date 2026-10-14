# Raptor64 original methods, iteration 3: protocol (distillation weight, label-preserving flip)

Written 2026-10-14 before launch. Decision document: `docs/research/iteration3-decision-20261014.md`. Infrastructure, endpoint and cache: [it2 protocol](raptor64-original-methods-it2-protocol-20261014.md) and [it2 result](raptor64-original-methods-it2-result-20261014.md).

## Comparator and single factors

- **Comparator: `in-distil`**, K42 0.91490 (`outputs/analysis/raptor64-20261013/it2/in-distil-final-files/oof_fold0.csv`). The recipe is in-orig (ImageNet-384, original head, `labels_v1_blend`, fixed fold-0 split, seed 42, 12 epochs, K8 training, K16 top-3 SWA selection, K42 export) plus 0.5 × plain BCE toward the pinned Dread v4_full K42 teacher (`dab0621a…`) on all 12 targets.
- A test proves the in-distil trainer, entry and recipe built from current code are byte-identical to the bundle v5 package that was run.

| Arm | Single factor vs in-distil | Everything else |
| --- | --- | --- |
| `in-distil-w1` | teacher weight 0.5 → **1.0** | identical, same teacher pin |
| `in-distil-flip` | **label-preserving** anatomical mirror, p = 0.5, applied to training samples only | identical, weight 0.5 |

**The flip.** It is `mirror_volume`: slice order is reversed inside each sagittal slot (0–17, 18–31), and image width is flipped in the coronal and axial slots (32–63); the mask is mirrored identically. The label permutation is the identity, nothing is masked, and the report targets and teacher targets are unchanged. Mirroring changes apparent laterality, not anatomy, and all 12 targets are defined anatomically, which corrects the it2 M1 error. The flip decision draws one Python random number per training item; the window count and sampler are otherwise unchanged. There is no TTA export: the protocol lists mirror TTA as secondary-only, and it is not produced.

**Tests:** `tests/test_raptor64_variants.py` (`FlipDistilTests`) checks that:
- labels and teacher targets are bit-identical under forced flip and no-flip;
- the images equal the historical sampler run on mirrored volumes;
- the training window count is K8, and evaluation items are unchanged;
- flipping twice gives the identity;
- the policy pins `DISTIL_WEIGHTS` = {in-distil 0.5, w1 1.0, flip 0.5} and `FLIP_SPEC` = {p 0.5, perm identity, masked [], tta_export False}.

`tests/test_raptor64_e2e.py` runs the generated trainer and export on CPU for both arms. `tests/test_colab_raptor64_vm.py` checks the controller's arm isolation now parses arm tags at boundaries, because `raptor64-in-distil` is a prefix of `raptor64-in-distil-w1`. All raptor64 suites pass (137 tests).

## Endpoint and pre-registered kill rules

- **Endpoint:** masked report macro AUC on the fixed 870 at K42, with the paired whole-component bootstrap (843 components, 2,000 draws, seed 42). Also reported: per-target deltas, and the five-target secondary (Lateral Meniscus, Lateral OA, PF OA, Medial OA, Medial Meniscus). Contrasts are against **in-distil** (primary), in-orig and own44 K42.
- **Either arm:** Δ vs in-distil with CI lower bound ≤ 0 closes that factor.
- **`in-distil-w1`:** Δ < +0.003 also closes distillation scaling as saturated.
- **Full-data bar, unchanged:** the best arm must reach K42 ≥ 0.92183 (in-orig + 0.015). Below that, no own arm enters any entry.
- **Exposure:** the teacher saw fold-0 development studies, so all distil arms carry second-order exposure. Gold-58 is unavailable, because it is not in the staged cache.
- **Health stop:** epoch-1 K16 < 0.80 sends SIGTERM to that arm's controller.

## Execution

- **Session:** `raptor64-orig-it3-20261014` (A100), with both arms concurrent.
- **Bundle:** v6, `e1c60001…`. Packages: in-distil-w1 `47196f43…`, in-distil-flip `441436db…`. Controller: `e6454bd0…`.
- **Teacher:** the it2 file is uploaded and its SHA checked against the pin.
- **Supervision:** VM-side autopilot with a 40-minute keepalive and `--ready-timeout 2700`, followed by local per-file verification.
- **Datasets:** private final datasets `ozgung/knee-raptor64-{in-distil-w1,in-distil-flip}-fold0-final-20261014`.
