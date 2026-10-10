# CoAtNet224 partial audit — October 10

Completed kernel `ozgung/knee-tracka-coatnet224-fold01-20261009` v3 was independently inspected by Sol: 56 checks passed, zero failures. Run/package/notebook hashes, labels and evaluation truth, folds/exclusions, exact heldout coverage, ten epochs, report-only top3 selection, and per-target AUC/support agree. Selected checkpoint bytes are not yet locally verified. Evidence is in `outputs/kaggle/trackA-coatnet224-fold01-20261009-result/trackA/` and its corresponding build package.

| Fold | ConvNeXt round0 | CoAtNet224 | Difference | Selected SWA epochs |
| --- | ---: | ---: | ---: | --- |
| 0 | 0.880946 | 0.901842 | +0.020896 | 8, 9, 7 |
| 1 | 0.882341 | 0.902008 | +0.019667 | 9, 8, 7 |

These are masked raw-report macro AUCs, using the unweighted mean over targets with both classes present. All twelve targets are defined. Largest target gains: MCL +0.05462/+0.07739 and medial meniscus +0.04497/+0.04634. Fold1 PF OA falls 0.00035; all other fold-target comparisons improve. Detailed class support remains in each run receipt. Gold outcomes were not used for this audit or selection.

The result supports the combined CoAtNet224 recipe. Architecture, resolution, microbatch/accumulation, gradient checkpointing and window chunking differ from ConvNeXt256, so backbone causality is not isolated. Training uses `labels_v1_blend`, not raw hard labels; epoch selection uses raw report targets.

Generic timm initialization and original blended labels avoid the known round1 global-OOF bootstrap exposure. However, heldout labels select epochs and SWA; these predictions are selection-biased, not untouched outer-fold teacher evidence. Do not automatically construct another bootstrap generation.

Both logs include a scheduler-before-optimizer warning. Scheduler advancement follows `GradScaler.step` unconditionally; accepted/skipped AMP updates are not recorded. Fold1 has 869 full microbatches per epoch, so accumulation2 leaves one microbatch without an optimizer step. Preserve this already-started fixed family for comparison; correct and instrument these behaviors in a separately identified future recipe, not silently mid-family.

Next: finish the memory-corrected folds2/3 and fold4, verify checkpoint bytes, then implement separately pinned CoAtNet224 image-only inference. Current inference contracts are ConvNeXt256-specific. Require strict checkpoint loading, actual224 window/probability parity, missing-plane behavior and an offline CUDA rehearsal before entry. Higher report validation does not establish public score improvement over the existing0.950 entry.
