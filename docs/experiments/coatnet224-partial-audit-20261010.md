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

## Fold4 and independent parent verification

At03:36UTC, fold4v2 is COMPLETE. A fresh parent audit reproduced folds0/1/4 metrics and source/data/selection bindings: **57 checks PASS**, zero failures, with all12 targets defined. Evidence and reusable audit: `outputs/analysis/coatnet224-parent-audit-20261010/audit.json` and `audit.py`. Fold4 report macroAUC **0.906643**, versus ConvNeXt round0 **0.889545** (+0.017099); selected SWA epochs[8,9,7].

Fold4 gains are concentrated in MCL (+0.075437), lateral meniscus (+0.043370), medial meniscus (+0.036652), and Baker’s cyst (+0.032998). Five targets decline slightly: lateralOA −0.001094, PFOA −0.000861, effusion −0.004699, synovitis −0.006269, fracture −0.007243. Class supports are recorded in the aggregate audit; no gold outcomes were inspected. The macro gain does not imply uniform improvement.

Parent is verifying selectedcheckpoint bytes for0/1/4, with three bounded range downloads (~293MB each;20GiB free disk checked beforehand). Separate native224 rehearsal implementation is active in an isolatedworktree. This is preparation only; noCoAtNet CUDAqualification orentry is claimed. Fable reused the completedfold4 slot for a paired bootstrap-label arm `ozgung/knee-tracka-coatnet224-r1-fold01-20261010`v1; it and original-label23v2 areRUNNING. No duplicatequeue launch.

## Rehearsal preparation complete

Selected weights0/1/4 all match recordedSHA, each292826563bytes and486finite tensors. Architecture224, target order and selectedSWAmetadata passed weights-only deserialization checks. Evidence: `checkpoint-review.json` and per-fold transfer receipts beside the parentaudit. Staging: `outputs/kaggle/own-coatnet224-inputs-20261010/`.

Added `knee/coatnet224_rehearsal.py`, `scripts/build_coatnet224_rehearsal.py`, and focused tests. This is a separately pinned copy of the existing qualified rehearsal, preserving the ConvNeXt files byte-for-byte. It admits only nativeCoAtNet224 with the reviewed original-label, source, fold, exclusion, tenepoch and gold-independent selection bindings. It retains strict checkpoint loading, finite tensors, exact raw/corpus/window/probability/repeat comparisons and16reader stresses. Subset contracts cannot claim fullfive qualification.

Parent focused verification: **8 tests PASS, no skips**, including actual pinned224 window math and immutable build rejection. Synthetic smoke PASS (`torch_forward=false`). Full ML-environment discovery:352tests,291passed,60skipped for optional evidence/backends, one pre-existing multiprocessing test blocked by sandbox `torch_shm_manager: Operation not permitted`; that exact test then passed with the required process permission. Logs remain beside the audit. No architecture forward pass or CUDA qualification is claimed.

Built only: `outputs/kaggle/own-coatnet224-rehearsal-fold014-20261010/`, folds0/1/4, complete_family=false. NotebookSHA `09ae6f54d979b56a8644891605cf4eaa72f0d4fc7c5ef1f747c0acfb96665496`; contractSHA `1ffaf9753272873da56fa5237c07e68edd27adde12db33b33a9faa4e51b73eae`. Build used base56516d0 plus the reviewed new source files before their parentcommit; exact module hashes are embedded. No GPUjob launched because both slots have active fits. When23completes, collect/audit/verify its weights and build fullfive directly before further longfits; subset is preparation evidence only. Final-submission wrapper still needs a separate pinnedCoAtNet implementation after rehearsal.
