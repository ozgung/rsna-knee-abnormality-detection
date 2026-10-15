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

## Sealed preparation checkpoint — October 15, 19:08 UTC

**Prepared as a secondary lane; no remote jobs launched.** Claude's 19:00 user-reported main-lane pivot prioritizes S6/labels-v2. Core helper dc0dd14 integrated as a53df3d; parent acceptance/recovery fixes committed as **ce8a978**. Full staging/trainer promotion remains disabled. This completes packet preparation; real cache and anatomy qualification remain pending owner execution.

Packets: `outputs/kaggle/local64-cache-20261015/shard{0..3}/`, each 814,070 bytes, 232 hash-bound sources including 194 vendored MONAI files, pinned Docker, private/offline CPU, literal title=id, no runtime flags. Source bundle SHA **d24d6fd969ada1cf30e5525242de0ad8dbfa3997c1fb91d4f2f9c8d455a0fed2**. Full notebook/metadata/contract/receipt hashes in [sealed index](../../outputs/kaggle/local64-cache-20261015/SEALED-BUILD.json).

| Shard | Training studies | Notebook SHA256 |
| --- | --- | --- |
| 0 | 1087 | `74a968110ec1ad9d2974014d146030aea5cff8b38b0caf437fa7f5cbf1bbacfc` |
| 1 | 1086 | `955a0f8aff17776517da23a2c715ef7f153d7400d705b918514110e1cabd38aa` |
| 2 | 1086 | `aa214ac552a92da7532b9391ac304e4d0a07b78bfeb0c4967301a67de7bdf3f3` |
| 3 | 1086 | `20e5b54fd7447475bce297b7b5291650e7ae5425ac4cf65a5955728db7633fd9` |

Shard 0 additionally emits a separate 58-study scoring cache. Frozen review membership was independently checked: all 32 are eligible training studies, split 9/8/5/10 across shards. Every selected frame is paginated with matched source and retained-crop outline; a separate genuine diagnostic-mask overlay has its own source/native index and artifact hash. No mask broadcasting.

**Acceptance:** all **14 focused ML checks PASS**, including actual pinned historical StudyWindows/public K42 352→384 tensor equality, K8 training interface, mask preservation, frame remapping, full montage/diagnostic artifact hashes, exact population binding, metadata-only collection and resume tampering. Bare focused checks OK, 11 optional skips. Twelve synthetic aspect/spacing/edge-center rendering cases match the pinned S6 pixel primitive byte-for-byte. Four separately unpacked packets compile and import with exact source hashes, settings, partitions and no private IDs embedded. Default smoke and installed PyTorch CPU forward smoke PASS, synthetic only.

Evidence: [focused tests](../../outputs/analysis/local64-20261015/targeted-tests.log), [window/render proof](../../outputs/analysis/local64-20261015/pinned-render-primitive-proof.json), [packet audit](../../outputs/analysis/local64-20261015/sealed-packets-acceptance.json), [frozen membership](../../outputs/analysis/local64-20261015/frozen32-membership-proof.json), [CPU smoke](../../outputs/analysis/local64-20261015/torch-cpu-smoke.log).

Broad checks were run and **did not pass**: bare 1322 tests, 22 errors/757 skips, all 22 optional-dependency errors reproduced on pre-change d04a426 (18 NumPy, three pydicom, one torch). Installed-ML 1350 tests: four failures, six errors, 162 skips. Nine existing staging cases reproduce on d04a426 because this Mac has 3.6 GiB free against their unchanged 5 GiB preflight reserve; the remaining existing Versia multiprocessing case is blocked by sandbox `torch_shm_manager` permission. No unrelated reserve/threshold change. [Bare comparison](../../outputs/analysis/local64-20261015/baseline-error-comparison.json), [ML comparison](../../outputs/analysis/local64-20261015/baseline-ml-error-comparison.json), [ML full log](../../outputs/analysis/local64-20261015/ml-suite.log). Focused checks cover the final local64 changes; no overall-suite PASS is claimed.

**Owner execution, only if claimed after reconciling main-lane resources:**

```bash
kaggle kernels push -p /Users/ozgun/Dev/rsna-knee-abnormality-detection/outputs/kaggle/local64-cache-20261015/shard0
kaggle kernels push -p /Users/ozgun/Dev/rsna-knee-abnormality-detection/outputs/kaggle/local64-cache-20261015/shard1
kaggle kernels push -p /Users/ozgun/Dev/rsna-knee-abnormality-detection/outputs/kaggle/local64-cache-20261015/shard2
kaggle kernels push -p /Users/ozgun/Dev/rsna-knee-abnormality-detection/outputs/kaggle/local64-cache-20261015/shard3
```

No `--timeout`, duplicate jobs or local full-cache downloads. Claude posts actual accepted versions/source/settings, job status and log/result paths. Collect only contract/index/receipts and K0 PNGs into `outputs/kaggle/local64-cache-20261015-result/shardN/local64_cache/`; `scripts/collect_local64_cache_metadata.py` defaults to metadata-only validation. `--verify-pixels` is for a remote staging host with all NPZ files, not this laptop. Both modes keep training promotion false until full loader/stage integration, immutable file/access verification and real K0 review pass.

Recovery: workers write atomic NPZ/index checkpoints, private failure details, source decode counts and disk/RSS/runtime measurements. In-place recovery verifies exact contract, source membership, NPZ file/raw hashes and montage hashes, then rebuilds only derived image links. A new Kaggle run resets working storage; attachment/transport resume is not implemented, so preserve partial outputs and request a separately sealed recovery build rather than blind retry. No job currently needs stop/resume. Actual CPU auxiliary forward, real DICOM/native geometry, anatomy, throughput and full staging remain unverified.
