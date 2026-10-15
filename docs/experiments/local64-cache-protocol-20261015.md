# Local64 X1a cache protocol

Accepted October 15, 2026, Claude 18:31 shared handoff; implementation in progress. Outcome: four private/offline CPU Kaggle cache packets, preserving exact Dread 64 source-slot semantics while changing pixels to S6-derived local views. No cache or anatomy qualification has executed. Claude owns all pushes/collectors and X1b Colab control; Codex owns implementation integration and acceptance. [Recipe](../research/finetune-recipe-20261015.md), [shared agreement](../claude-astra-handoff.md).

## Frozen population and processing contract

Reuse source96/raptor64 historical membership and order: 4407 historical studies minus 58 gold and 4 quarantine yields 4345 eligible (3475 training/870 fold0). Four strided partitions 1087/1086/1086/1086; shard 0 also emits 58 gold under a separate scoring-only directory/index. Five disjoint logical outputs, four jobs. Gold magnitudes and reports never enter pixels; membership uses presence only. Historical folds preserved; patient identities are unavailable, so patient-level independence cannot be established. No new split is claimed.

Use exact Dread `slot_plan` (sag 18/14, cor 12/8, ax 12; 6–94% raw-frame span) and series/fluid-substitution selection. Run unchanged `raptor_input.prepare_study` to retain its source availability mask/counters, discard its global pixels, then map raw filenames one-to-one to S6 native rows. Dark local pixels do not imply a missing source frame. The extra decode pass is measured overhead, not hidden in estimated throughput. Unresolved filenames/ordering/geometry fail closed; no representative-series or thin-stack slab substitution.

Pinned S6 native/side/anchor closure from the qualified publication bundle, anatomy 5904dd16…/side classifier/vendored MONAI. Invoke the side path used by `CStudies.__getitem__`; `rows()` alone omits it. Preserve canonical orientation with no additional flip. Render raw selected native slices through the unchanged physical crop, INTER_AREA resize, series_companded window/intensity and uint8 rounding primitives at 384, then central 352 crop. Sagittal joint 100 mm; coronal joint 130 mm (explicit new selector); axial PF 100 mm with S6 patella-or-joint-offset policy. This is an adapter contract, not byte parity with slab-based S6 bags.

Missing/incompatible FrameOfReferenceUID blocks shared-LPS placement and causes explicit field-centre local fallback, even if K0 fails. Record bank method separately from effective crop placement: a learned bank output blocked by geometry must not count as a successfully placed learned crop. No hidden global-view fallback.

Store 64×352×352 uint8, original 64 availability flags, exact source/native slot metadata, anchor/fallback/side decisions and hashes. The genuine sagittal segmentation mask is **one diagnostic plane**, with source series/native index and exact crop geometry; it may be outside the selected Dread frames, stated explicitly. Never broadcast it over 32 sagittal slots. Dense per-slot R3 supervision is deferred.

## Acceptance and K0

Mechanics requires complete immutable source/assets/config and membership bindings, image-only staging, explicit CPU auxiliary models, finite-output hooks, unchanged model state, exact raw/native frame mapping, NPZ dtype/shape/mask and content hashes, file coverage, disk/RSS/runtime receipts and separate gold metadata. Loader must accept 352 and match the pinned 384 training-window interface with masks preserved. No frozen global 336 loader or original inference source may change.

Synthetic acceptance covers membership/disjointness, raw/native mapping, selected indices, crop/mask alignment, dark-local availability, schema/hash tampering, 352 loader parity, runtime source APIs and fail-closed K0. It does not qualify real decode, anchor placement, speed or anatomy. Collector must distinguish complete cache mechanics from advancement; any incomplete/unknown review blocks training.

Real K0 uses the same frozen 32-study qualification cohort and displays **every selected sagittal/coronal/axial frame**, paginated. Show matched native source, retained crop boundary and actual 352 pixels; mask overlay separately bound to its diagnostic source. Review artifact must bind source/assets/contract and montage hashes and record reviewer/date, with unresolved decisions explicit. Bounds from recipe §3.2:

- Learned anchor path exercised on at least 60% of studies; report model exercise and effective learned placement separately.
- Field-centre fallback at most 30% of unique Dread-selected sagittal series. Do not dilute with unselected source series or duplicate slots.
- Zero wrong-side/axis projections.
- At most 2/32 studies with any selected sagittal crop cutting visible joint margin.
- No axial PF local lacking the patella where its matched source shows it. Missing planes are N/A, not automatic successes.

K0 failure changes the view through a separately versioned build (recipe first proposal 110 mm sagittal/140 mm coronal); never relax the recorded bound. Original M4 anatomy failure remains closed. No paired fit before all shard mechanics, real K0 counts and signed review pass.

## Resources and recovery

Expected 4–6 h/shard and 25–30 GB total are recipe estimates, not measured CPU results. Double decoding and diagnostic segmentation can increase them. Respect actual account/platform limits; no assistant-selected experiment time budget. Kaggle's [notebook documentation](https://www.kaggle.com/docs/notebooks) lists a general 12 h CPU session limit (checked October 15); owner reconciles current limits/jobs/CPU capacity and disk before launch. Hidden competition inference has its separate actual 9 h limit.

Keep full caches remote; local starter budget 2 GiB remains. Only bounded synthetic/source checks and small metadata locally. CPU packets private/offline, enable_gpu=false, title=id; normal push has no `--timeout`. No provisioning or new terms. Builder must pin mounted data/auxiliary asset versions and omit reports/labels from staged inference inputs. Package no private identifiers in committed source. Use atomic per-study outputs and index checkpoints for recoverability; verify exact config/source/membership/content before any resume, never blindly retry ERROR.

Final packet names, hashes, commands, measured resources and actual job/version/session IDs will be appended after sealing/owner claim. Codex has launched no jobs. Claude must record X1b Colab session/run/log/checkpoint plus stop/resume instructions separately; owner-reported intent is not independently verified liveness.
