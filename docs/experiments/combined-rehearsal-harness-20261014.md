# Prospective combined rehearsal evidence harness — October 14, 2026

This is a dependency-free preparation and receipt checker for A′, with the separate B1′ screen supported. It executes no model, fusion, prefetch worker, training, GPU job or submission. It changes no existing worker, source pin or projection. The [component acceptance contract](entry-component-acceptance-20261014.md), [axial protocol](raptor64-axial-pf-it4-protocol-20261014.md) and [sagittal protocol](lane2-sagittal-joint-crop-protocol-20261014.md) remain authoritative.

The actual arm image-only interface, rank-population/tie/rounding arithmetic, missing-plane reader definitions and combined projection evaluator have not been handed over. **Every successful check returns `PREPARATION_ONLY`, `entry_qualified: false`, and `scientific_qualified: false`.** Neither a caller's `status: PASS`, a complete synthetic packet nor a bound receipt can change that result. Supplied scientific metrics are checked for the prospective gates; they are not recomputed from predictions. A hash proves file identity, not scientific validity or owner approval.

## Commands and result

Run from the repository root, keeping packets, actual identifiers, weights and large artifacts under ignored `outputs/`:

```bash
python3.12 scripts/check_combined_rehearsal.py outputs/analysis/combined-rehearsal-20261014/packet.json
# In an installed ML environment, optionally run the existing base verifier:
python scripts/check_combined_rehearsal.py outputs/analysis/combined-rehearsal-20261014/packet.json --verify-base
```

Exit **1** means evidence rejection, **2** means preparation checked but qualification blocked. There is deliberately no exit-0 qualification branch. JSON output contains aggregate checks, stable failure codes, counts, packet hash and blockers; it does not print study identifiers, artifact paths or backend diagnostics. The CLI fails closed on malformed nested structures as well as explicit validation failures. Python callers use `knee.combined_rehearsal.check_packet` and should treat every exception as rejection.

The module snapshots every declared artifact, checks SHA256 and rechecks all files plus the packet before returning. JSON duplicate keys and nonfinite constants are rejected. Files are read only. Source/weight hashes and reader fingerprints need independent owner reconciliation; the checker does not assign itself provenance authority.

## Prospective packet schema v1

This small explicit interchange contract is pending owner handover, not a claimed existing lane export interface. `schema_version` is integer `1`. File references below are keys in `artifacts`, whose entries are exactly `{"path": "relative-or-absolute-path", "sha256": "64-lowercase-hex"}`. Relative paths resolve against the packet directory. JSON artifacts contain the described objects; opaque source/weight files are still hash-bound. Never copy private packets into tracked documentation.

| Packet field | Required contents |
| --- | --- |
| `purpose`, `evidence_kind`, `candidate` | `preparation` or `acceptance`; `synthetic` or `real`; `A_prime` or `B1_prime` |
| `official_sample` | Bound original official visible sample CSV. Its header must equal the existing dependency-free official `UID`/`LABELS` constants in `knee.public_s6_qualification`; an arbitrary 13-column sample cannot redefine names/order. |
| `study_registry` | JSON with `study_ids` in the frozen rehearsal's exact 39-study order and `visible_ids` in the official sample's exact three-study order. All IDs are unique. This registry must be independently bound to the archived base ledger by the parent. |
| `components` | Nonempty list. Each object has `name`, `targets`, `weight`; complete acceptance adds `handover`, `gates`, `missing_plane_policy`, `records` artifact references. |
| `pending` | Preparation packets explicitly include `worker_interface`, `rank_and_missing_plane_arithmetic`, `combined_projection_policy`; additional missing handovers may be named. No model execution is attempted. |
| `base` | Acceptance only: `receipt`, `binding`, `csv`, `projection_inputs`, `projection_result` references. `csv` is the existing three-row base rehearsal output. |
| `base_predictions`, `combined_predictions`, `final_csv` | Acceptance only: full39 baseline/combined CSVs and the visible-three final CSV. All use the official header. Exact row order and finite probabilities in `[0,1]` are required. |
| `runtime` | Acceptance only: `measurement`, `combined_projection_policy`, `combined_projection_result` references. |
| `prefetch` | Optional acceptance-only paired-output/conditions packet described below; absence preserves the preparation of the serial route. |

Only `ax-crop` → `PF OA` and `sag-crop` → `Lateral Meniscus`, optionally `ACL`/`Medial Meniscus`, are admitted. The primary target comes first and is mandatory; target sets are disjoint and every admitted target has fixed weight `0.3`. No fusion function, rank population or tie policy is supplied by this harness. All unmodified CSV tokens and every unavailable component's affected target tokens must equal the base strings exactly. The visible CSV must equal the corresponding combined rows, including formatting.

### Component receipt artifacts

`handover` is a JSON object with `schema_version: 1`, matching `arm`, `mode: full_data`, `input_kind: images`, exact `supported_targets`, a `tensor_digest`, and `gate_sha256`. It references `source`, `reader_config`, `weights`, `dataset_manifest`, `initializer_provenance`, `swa_registration`, `evaluation_inputs`, `arithmetic_contract`, `worker_interface`, `repeat_policy`. These artifacts are mandatory even though their owner-defined semantics still need acceptance. Cache-to-image reader equivalence must be supplied by the owner in the arithmetic/reader handover; this checker does not derive it from the training cache.

`dataset_manifest.study_count` is a positive integer. `swa_registration` contains matching `study_count`, `full_data: true`, three distinct nonnegative `fixed_swa_epochs`, and the actual `weights_sha256`. The count must be reconciled to the actual full-data set; the checker does not hardcode historical corpus counts or claim that an arbitrary matching declaration proves full-data coverage.

`evaluation_inputs` references `predictions`, `masks`, `ids`, `split`, `labels`, `teacher_predictions`, `control_predictions`, `best_epoch_predictions` and has `rank_weight: 0.3`. `gates` binds its SHA through `evaluation_sha256` and contains exactly the admitted `targets`. Each target has numerical `delta`, `ci_lower`, `transfer_delta`. Primary also has `arm_auc`, `swa_delta`, `best_epoch_delta`, `crop_blend_auc`, `full_blend_auc`, `transfer_ci_lower`.

The checker requires primary Δ≥.010, positive CI, arm AUC≥.80, positive SWA/best-epoch signs, crop blend better than full control, and nonnegative transfer Δ. Optional sagittal targets require their own positive CI and transfer gate plus primary admission. The primary integration lower-bound screen is .0017 for A′ and .0072 for B1′. Evaluator comparators, grouping, bootstrap, masks, teacher exposure, pretrained provenance and independently recomputed CI remain scientific acceptance work; these scalar checks do not qualify an arm.

`missing_plane_policy` is the explicit prospective object with `arm`, `plane` (`axial`/`sagittal`), `unavailable_reason: missing_plane`, `output: preserve_base_strings`, `other_errors: fatal`. This is a declaration to reconcile with the delivered reader/fusion, not an implemented fallback. Reader-specific mask shape/meaning is deliberately unresolved.

`records` is a JSON list in exact full39 order. Each record has `study_id`, `input_kind: images`, `input_sha256`, boolean nonempty `plane_present_mask`, `state_before`, `state_after` (equal to the handover tensor digest), and `error: null`. `state: present` requires an available mask, `reason: null`, 12 finite `probabilities` and actual `repeat_probabilities`, exact `repeat_input_sha256`, and separately bound `output_sha256`/`repeat_output_sha256`. Each output SHA is over its supplied vector's UTF-8 `json.dumps(vector, separators=(",", ":"), allow_nan=False)`; this is serialization, not invented inference rounding. Numerical equality or tolerance for the new arm is **not** imposed: the actual owner-reviewed `repeat_policy` and its replay remain blocked. `state: unavailable` requires an all-false mask, `reason: missing_plane`, the documented plane and `probabilities: null`. Programming, resource and unknown errors remain fatal. Full39 masks/inputs and genuine repeat execution still need owner verification against actual image workers.

### Base and runtime

`--verify-base` lazily imports **the existing** `knee.public_r7_recovery_entry.require_rehearsal`. It passes the bound receipt, binding and existing base CSV to that verifier; there is no alternate base success flag or injectable verifier. It then calls that module's existing `projection(**projection_inputs)` and requires exact equality to both `projection_result` and `receipt.projection`. Projection inputs contain exactly `cfg`, `phases`, `reports`, `setup`, `merge_seconds`. Import/backend failures are rejection, never a skipped success. Default use does not import NumPy, Torch or the public model tree.

The parent has separately verified the archived v7 base. That receipt's immutable archive/slim inference closure and frozen payload are the authority; do **not** replace them with the expanded current training package or demand byte identity of every current package helper. This checker does not materialize or execute that closure. A future actual coordinator must use the archive and retain full worker/builder pins, guards and compiled control. Historical saved verification is not a fresh combined run.

The archived receipt does not directly record its merge interval. A reconstructed diagnostic projection is not actual timing evidence and must not be used to fabricate `projection_inputs`; supply measured same-session inputs for a fresh acceptance packet or leave that replay pending.

`measurement` declares `scope: combined_same_session`, `start_seconds`, `end_seconds`, `elapsed_seconds`, contiguous named `segments` spanning the entire wall interval, measured `peaks` and explicit `limits` for `host_rss_bytes`, `cuda_bytes`, `output_disk_bytes`, plus positive `minimum_disk_free_bytes`. Peaks must remain within supplied limits; the parent must reconcile those limits to the actual platform. Segments include setup/prefill, evaluation, repeat/check work and drain/cleanup; no isolated-forward subtraction is used by the checker.

`combined_projection_policy` has explicit `base_function: knee.public_r7_recovery_entry.projection`, `measurement_scope: combined_same_session`, `owner_handover: true`, nonempty owner `formula`, and `evaluator_source` reference. `combined_projection_result` binds `policy_sha256`, `measurement_sha256`, `base_projection_sha256` and positive `conservative_seconds`≤8.3h. Measured elapsed preserves the actual **9h** runtime ceiling; 8.3h remains a projection gate. The checker does **not** execute an arbitrary owner-supplied formula or subtract isolated arm time from a historical base projection. Independent replay of a delivered combined policy is always a blocker, preserving the unchanged base formula and its non-bound limitations.

### Optional prefetch receipts

`prefetch` references `serial`, `prefetch`, `parity_policy`, and `astra_conditions`. Both paired JSONs have `input_kind: images`, matching `assets_sha256`, `assignment_sha256`, `compiled_control_sha256`, `device_identity_sha256`, and exact all39 `records`. Each has actual 12-number `outputs`, a dictionary of nonempty numerical-array `intermediates`, `study_id`, matching `input_sha256`. Fingerprints or three visible outputs are insufficient. `parity_policy` explicitly supplies owner `output_atol` and per-intermediate `intermediate_atol`. Output tolerance cannot exceed the existing frozen R7 maximum `1e-3`; the parent must reconcile the correct tighter same-board/compiled-control tolerance. Unknown intermediate policy is explicitly blocked even if its supplied numbers agree. No default tolerance is invented. Supplied values are compared recursively and must be finite; output probabilities must be in `[0,1]`.

The five required `astra_conditions` keys each reference a receipt with matching `condition`, nonempty `evidence_artifacts`, and `observations`:

- `ownership`: global-quiescent scratch gate, single producer, 39 reset barriers.
- `shutdown`: zero live threads after teardown, cancellation-aware waits, preserved BaseException classification, unknown CUDA errors fatal.
- `cuda_precision`: serial warmup, explicit device/stream, auxiliary autocast disabled, completed CPU-only handoff.
- `paired_actual39`: 39 studies, identical compiled control, actual values rather than input hashes alone.
- `wall_and_resources`: positive actual paired serial/prefetch elapsed; prefetch wall matches the combined measurement and improves over serial.

Exact observation field names are in `prefetch_evidence`; the synthetic fixture illustrates this prospective contract. These receipts summarize all five Astra conditions, but observed booleans and references cannot establish concurrency safety. Owner source review, targeted cancellation/scratch/state/error tests, actual paired CUDA execution, existing source/state/schema/ledger checks, wall/resource reconciliation and pin acceptance remain mandatory. `PREFETCH_ASTRA_OWNER_ACCEPTANCE_NOT_REPLAYED` stays blocking even if every receipt is supplied.

## Verification and next action

Implementation: `knee/combined_rehearsal.py`; CLI: `scripts/check_combined_rehearsal.py`; synthetic fixtures/regressions: `tests/test_combined_rehearsal.py`.

Focused tests cover tamper and during-check mutation, wrong header/order/coverage, duplicate studies, protected decimal tokens, missing-plane contradictions, fatal errors, state/repeat/image-only claims, unqualified scientific/full-data receipts, fixed targets/weights, resource/projection omissions, unchanged-base verifier dispatch/replay and actual39 prefetch parity/condition omissions. They do not run a real arm, prove independent scientific qualification or reproduce real GPU resources.

Helper verification: `python3.12 -m unittest discover -s tests -p test_combined_rehearsal.py -v`: **30 tests PASS / 0 skips**. `python3.12 -m knee.smoke`: **PASS**, synthetic CPU path, `torch_forward: false`, no training. Real base verification in this helper suite is mocked to test dispatch/rejection; it is not an additional real v7 verifier run. The parent owns the separately checked archived receipt.

Next: parent accepts this narrow preparation layer, reconciles actual lane exports to a frozen packet, and supplies the workers, arithmetic/projection replay and scientifically verified handover before any combined rehearsal or entry promotion. Parent owns shared STATUS/handoff updates, integration and launches; this helper edits only its four assigned new files.
