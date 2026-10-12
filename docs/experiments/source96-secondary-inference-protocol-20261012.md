# Source96 secondary inference: K16 parity, K94, K94 + L/R mirror TTA (2026-10-12)

Status: implemented and tested on synthetic data on CPU. Not run on the real 96-slot cache, not run on a GPU, not competitively evaluated. The mirror orientation is an **unverified** assumption (see Verification and open items); runtime is unmeasured. Code: `knee/source96_secondary_inference.py`; tests: `tests/test_source96_secondary_inference.py`.

## Outputs and their role

One frozen checkpoint (top-3 SWA CoAtNet `RaptorClassifier`, 224, trained at K8, selected at K16), one pass over the same fold-0 studies, three separate prediction sets:

| Output | Windows | Mirror | Purpose |
|---|---|---|---|
| `k16_plain` | 16 | no | Same arithmetic as the primary export; parity check (default tolerance 1e-4 against `oof_fold0.csv`, same-hardware FP16) |
| `k94_plain` | 94 | no | Every interior triplet of the 96-slot stack |
| `k94_mirror_tta` | 94 + 94 | yes | Mean of plain and L/R-mirrored K94 probabilities (equal weights); the two views are also stored. The mirror is the anatomical L/R mirror under the assumed standard orientation (unverified) |

All three are secondary diagnostics. The frozen K16 primary endpoint governs promotion. The fold-0 development set is exposed (selection used it), so differences between arms are descriptive only: no epoch, target, weight or blend selection may use them. Adopting K94 or mirror for any submission would be a separate, pre-declared decision. A correct L/R mirror would leave labels unchanged (each structure keeps its identity, a medial structure stays medial); that holds only if the orientation assumption below is right, which has not been checked.

## Mirror derivation per plane (assumed standard orientation, unverified)

This section derives the mirror from the reader code plus the standard-IOP assumption. No DICOM header was inspected: the cache stores no `ImageOrientationPatient`, so nothing below is a measurement. Every receipt carries `mirror_orientation_verified: false`.

Layout (`knee/source_span_pilot.py:18`, same list in `knee/source96_cache.py:20-21`): `SLOTS96 = Sagittal/fluid 26, Sagittal/non-fluid 22, Coronal/fluid 18, Coronal/non-fluid 12, Axial 18`, i.e. slots at indices 0-25, 26-47 (sagittal, 48 slices), 48-65, 66-77 (coronal, 30), 78-95 (axial, 18); depth 96. The module derives its layout from that constant and refuses to import if it changes.

Reader convention (the cache reader is the pinned public reader, SHA `859aae12...` = `knee/corpus44_parity.py:20`, re-parameterised only by slots, depth and span in `knee/source_span_pilot.py:31-70`; local copy `outputs/research/2026-10-08/raptor-audit/knee-mri-twelve-findings-from-a-single-model.py`):
- Slices of a series are sorted by `dot(IPP, cross(row_cosines, col_cosines))`, ascending (`:268`, `:275`): the stack axis is the series normal. Slot slices are then picked with `linspace` over that order (`:319-320`).
- Pixels are used as stored: crop a centred square of 140 mm and resize to 336 (`:286-290`); there is no transpose, rotation or IOP-based reorientation. Array axis 1 is DICOM rows, axis 2 (width) is DICOM columns, which run along the first IOP triple.
- Under standard IOP (axial `1,0,0,0,1,0`; coronal `1,0,0,0,0,-1`; sagittal `0,1,0,0,0,-1`) this gives:

| Plane | Stack (slice) axis | Image width axis | L/R mirror under the assumption |
|---|---|---|---|
| Sagittal | patient L/R (normal along x) | A/P | reverse the slice order of that slot; slice content unchanged |
| Coronal | A/P | L/R | flip width; slice order unchanged |
| Axial | S/I | L/R | flip width; slice order unchanged |

Under this assumption a uniform horizontal flip would be wrong for sagittal (it would flip A/P). Reversal is per slot, never over all 48 sagittal positions, because the two sagittal slots are different series. The mirror is applied to the uint8 volume and the mask before windowing and normalisation (mirror the stack, then normalise; `mirror_volume` in `knee/source96_secondary_inference.py`), and window centres are re-planned from the mirrored mask by the unchanged eval policy.

Raptor comparison (untrusted reference text, read only: `_mirror_tri`, lines 270-276 of the scratchpad `velciraptor.py`): centre slot decides, sagittal reverses the 3 channels at the same centre, coronal/axial flip width. `test_full_valid_synthetic_reflected_centre_comparison_with_public_raptor_rule` is a full-valid synthetic reflected-centre comparison: with every slice populated, it compares that shortcut's window with the triplet at the reflected centre of the mirrored volume. They coincide for every window that lies inside one series (86 of 94 K94 centres) and for the four straddling windows that are purely coronal/axial (centres 65, 66, 77, 78), so 90 of 94; they differ by construction for the four windows centred at 25, 26, 47, 48 that involve a sagittal slice across a series boundary, which are already cross-series mixtures in the plain pass. This is not a proof that the sampled bags of the two constructions are identical: with missing or partial planes the mirrored mask re-plans the centres (different valid range, different `linspace`), and that case is not covered by the comparison. The `RaptorClassifier` head pools windows with a softmax over the window axis (`train_knee.py:353`), so window order does not matter. The study-level form was chosen because it is the direct statement "this study with the assumed L/R axis reflected, through the same pipeline", including partial-mask planning.

## Missing-slot policy

- Cache masks are 1 where a slice has any nonzero pixel (`pilot.validate_arrays`). A missing series zero-fills its whole slot; a failed frame leaves one zero slice (`:310-315`, `:336`).
- Mirror permutes volume and mask together: masked slices stay zero and masked, valid slices stay valid; a plane with no valid slices stays empty; nothing is written into masked positions. Pixel/mask consistency holds after mirroring (tested, including a partially valid sagittal slot, whose valid run moves to the other end of that slot).
- Centres use the pinned `StudyWindows._centers` eval policy unchanged (valid min/max, neighbours in range, `linspace().round()`, duplicates kept, fewer than 3 valid slices falls back to slices 0-2). Masked and cross-series triplets are kept as in the primary path and counted in the receipt (`masked_triplets`, `boundary_triplets`, `duplicate_centers`, `fallback_studies`); a second, independent implementation of the policy must agree with the pinned sampler or the run stops.
- Whole-empty or pixel/mask-inconsistent studies stop the run (fail-closed receipt).

## Checkpoint binding

The frozen primary is bound and qualified before anything opens the corpus. In the CLI the order is: (1) `bind_checkpoint` loads the SWA file (`weights_only=True`), checks arch, res 224, target order, top-3 `swa_over` and finite weights, hashes the file, then calls the existing `export_guard(..., secondary=True)` (`knee/source96_training.py`): ten completed epochs, `run_fold0.json` has `export.k_eval == 16`, its `checkpoint_sha256` equals this file's SHA-256 and its `selected_epochs` equal the checkpoint's; (2) the pinned trainer definitions are loaded and their digest enforced; (3) only then `guard.staged_corpus()` and `guard.verify_export_inputs` (which also opens the corpus) run; (4) `run_secondary` binds the checkpoint again immediately before inference. A missing, K16-mismatched, SHA-mismatched or tampered primary record raises `ValueError` before step 3, and `test_cli_invalid_primary_cannot_open_the_corpus` spies on `staged_corpus`, `verify_export_inputs` and the loader's `open_corpus` to prove none is called. The aggregate `receipt.json` records the file SHA-256, the primary record's SHA, selected epochs and a model state digest before and after the run; each of the three outputs carries the same `checkpoint_sha256`, and each `.npz` embeds it. The file is re-hashed after inference.

K16 parity against `oof_fold0.csv` is on by default and fails closed (`PARITY_FAIL`, outputs kept for inspection). `saved_primary` validates the reference before it is compared: exact header, well-formed rows, no duplicate validation rows (a duplicate cannot hide behind a complete ID set), exactly the evaluation ID set (no missing, no unexpected IDs), and every probability finite and within [0, 1]. A corrupt reference raises `ValueError` even with `--no-enforce-parity`, leaves a `FAIL` receipt and keeps the already written outputs.

## Runtime (unmeasured)

Forward-window counts per study: K16 = 16 windows; K94 = 94 windows (94/16 = 5.875 times the windows of K16); the mirrored K94 view is 94 more. One pass producing all three outputs forwards 204 windows per study (12.75 times the windows of the primary K16 alone); `k94_mirror_tta` on its own is 188 (11.75 times); the plain K94 view is shared with `k94_plain`. These are window-count ratios, not time estimates.

No wall-time estimate is made. No 96-slice or A100 secondary throughput has been measured, and the only nearby measurement, the old 44-slice K16 diagnostic (283.49 s including setup), does not transfer: different slice count, setup share and hardware. Time is not linear in forward windows alone: single-thread CPU window construction (resize and normalisation per window), the per-study mirrored-volume copy, lazy cache reads and the K94 one-study-per-forward grouping all contribute. Runtime will be recorded from the first receipt, which writes `elapsed_seconds`, per-arm `seconds` (window construction plus forward) and `peak_host_rss_bytes`, plus peak CUDA memory on CUDA. K94 views run one study per forward with `--k94-win-chunk 16` (memory bounded below the primary's 64-window forward); K16 uses the primary grouping (4 studies, `win_chunk` 0).

## Invocation (not executed)

Same staging environment as training (`SOURCE96_STAGE_ROOT`, `SOURCE96_MANIFEST_SHA256`, `SOURCE96_CONTENT_SHA256`, `SOURCE96_STAGE_RECEIPT_SHA256`); the checkout that provides `knee/` must match the revision that built the training package, and this module is not part of that immutable package.

```bash
PYTHONPATH=<checkout> python -m knee.source96_secondary_inference \
  --work /content/trackA96 --fold 0 \
  --trainer-source /content/trackA96_pkg/train_knee_patched.py --trainer-sha256 <manifest.json patched_sha256> \
  --labels <labels_v1_blend.parquet> --truth <train_labels_v1.parquet> \
  --out /content/trackA96/secondary96 --device cuda --k94-win-chunk 16
```

`--device` is explicit (`auto` and silent CPU fallback are refused). Development membership is verified with `verify_export_inputs` (870 ordered studies) before inference. Outputs: `receipt.json` (aggregate, no identifiers), and private `k16_plain`, `k94_plain`, `k94_mirror_tta` as `.npz` and `.csv`.

## Verification and open items

- 67 tests, all synthetic (run with the byte-pinned trainer source present: `SOURCE96_TRAIN_SOURCE=... python -m unittest tests.test_source96_secondary_inference`): asymmetric per-plane marker fixtures, mirror twice is identity, missing and partial planes, K16/K94 plans, same-checkpoint binding, guard refusals, parity pass/fail, a tiny CPU model through the real module path, and the same end-to-end tests against the byte-pinned public `StudyWindows`/`RaptorClassifier` source when present locally. Added in the review round: `saved_primary` rejection tests (duplicate, malformed, unexpected/missing/extra IDs, non-finite, out of range, bounds accepted), a corrupt-reference run that must fail closed and keep artifacts, CLI orchestration tests (guard before corpus; invalid primary or unpinned trainer cannot open the corpus), and receipt/artifact checks for the unverified orientation flag. Deliberate code mutations (wrong sagittal rule, mask not following data, wrong coronal/axial axis, TTA equal to plain, plain-mask planning, skipped guard) failed the earlier 47-test suite; for this round, restoring the corpus-first CLI order and restoring the dict-collapsing `saved_primary` each fail the new tests.
- **Not verified, and blocking any TTA interpretation: the mirror orientation.** The cache stores no IOP and there is no local DICOM, so "image width is patient L/R for coronal/axial and the slice axis is patient L/R for sagittal" rests on the standard-IOP assumption and the reader code, not on measurement. The receipt therefore carries `mirror_orientation_verified: false`, `mirror_orientation_assumption` and `orientation_audit_required_before_tta_interpretation`; the `k94_mirror_tta` output entry repeats the flag and its `.npz` embeds it. Required next action: a label-blind CPU header audit (headers only, no labels read; Kaggle or Colab) over the training series that tallies, per plane (Sagittal, Coronal, Axial), the dominant axis (largest absolute component, with its sign) of the IOP row direction cosines, of the IOP column direction cosines and of their cross-product normal. Oblique series (largest absolute cosine below a pre-declared threshold, e.g. 0.9) and nonstandard series (dominant axes other than the assumed Sagittal row A/P, column S/I, normal L/R; Coronal row L/R, column S/I, normal A/P; Axial row L/R, column A/P, normal S/I) are counted separately and never folded into the standard bucket, and the number of studies with any non-matching selected series is reported. Signs are tallied for completeness; a width flip or slice-order reversal is the same operation for either sign, so the dominant axes decide. Until that table is recorded no `k94_mirror_tta` number may be interpreted; non-matching series or studies are then excluded or stratified, and a plane whose dominant axes differ materially needs its mirror rule re-derived rather than reused. The centred crop can offset a mirrored width by at most one pixel.
- Not exercised: real 96-slot cache loading through the lazy proxies, CUDA FP16, the `weights_only=True` load of the real source96 SWA file (the own224 diagnostic loads the same format that way), agreement with the packaged `tracka_entry.py secondary` K94 CSV (expected within FP16 batching noise), and any runtime or memory measurement on a real GPU.
