# Source-span and depth feasibility — October12

This CPU-only pilot checks three image representations on the same deterministic32 nongold, nonquarantine studies:44 slices across15–85% source extent,44 across2–98%, and96 across2–98%. All retain336-pixel storage,140mm crop, the same series routing, ordering, intensity algorithm and image-only reader. Selected-slice intensity statistics may change. The96 arm uses slots26/22/18/12/18; this is a combined depth/span intervention, not an exact reproduction of every public preprocessing choice.

The control reader previously matched the published44 corpus exactly on5studies/25slots. This run expands the exact pixel/mask comparison to32. Mandatory checks also include source AST equivalence except the declared constants/allocation, pinned source/corpus/metadata/codecs, unique frozen cohort, finite own224 K8/K16 inputs, save/reload equality, first2 repeated decodes for all3arms, masks/partial/missing routing and resource-error propagation. Full-empty inputs fail. Control mismatch is preserved as MISMATCH, operational failures as ERROR; neither qualifies training.

No models, labels as outcomes, training or score calculation. Gold exclusion uses cell presence only. Only aggregate receipts persist; ephemeral image files are removed. Normalized ordinal source extent is reported; physical coverage cannot be inferred from it. The legacy sampler can cross slots or include masked neighbors; both counts are reported rather than silently repaired.

Implementation: `knee/source_span_pilot.py`, `scripts/build_source_span_pilot.py`, `tests/test_source_span_pilot.py`. Parent reviewed source/partial-frame behavior and reused existing offline-codec packaging. Required CPU resources are existing Kaggle competition/corpus mounts; no local raw images or bulk download. Runtime is expected in minutes from previous5/32study pilots, not a hard spending cap; actual CPU platform12h limit applies.

Parent checks:16 focused testsPASS/0skip with existing `/private/tmp/own-rehearsal-test-deps` dependency overlay; bare ML environment ran11PASS/5backend skips. Pinned-source synthetic DICOM ordering, partial failure,96allocation and sampler tests were exercised using the overlay. Dependency-free CPU smokePASS; no claim of real32study parity until remote receipt is independently reviewed.

Next, if mechanics pass: prepare remote96 cache staging/resume identity and reconcile historical own224 training control before one96wide fit. Same tenepochs, labels_v1_blend, K8train/K16 selection, report-selected top3SWA; matchedK16 primary and frozenK94 secondary. Source-span/depth feasibility does not prove improved accuracy.

## Actual result — October12 14:13 UTC

Private CPU kernel `ozgung/knee-source-span-pilot-20261012` v1/kernel138339609 COMPLETE. Runtime338.91s (5.65min), peak hostRSS1,022,160,896bytes. All32control studies/160slots exactly match mounted44 pixels and masks. Three representations completed; repeated decoding for first2 of each arm, save/reload, finite K8/K16 tensors, source/membership/pins passed. Parent aggregate consistency audit560checksPASS; this is not a second raw-DICOM execution. ReceiptSHA48655a7ef2863e2038beb9d7247f91afb7f9aba1c8153813b0cdaf03fb400e45.

All arms had6missing slots and no partial or empty selected slots.44narrow and44wide each had1368valid/decoded frames;96wide2990. Deterministic K8/K16 cross-slot window counts respectively: narrow64/32, wide64/32,96wide0/32. Masked-window counts12/14,12/14,7/14. These describe the tested evaluation sampler, not randomly sampled training bags or anatomical validity. Ephemeral image files total664,762,112bytes and were removed; no persistent images exported by this pilot.

Arm loop times: narrow60.28s, wide27.21s,96wide44.54s. Later arms benefit from warmed source storage; these are not cold full-corpus runtime estimates. Mechanics now support next private96cache generation. No accuracy metric, training or submission arose from this pilot. Evidence `outputs/analysis/source-span-pilot-20261012/{terminal-collection-v1.json,parent-review-v1.json}`.
