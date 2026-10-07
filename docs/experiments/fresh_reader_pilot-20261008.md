# Fresh ImageNet ConvNeXt pilot — 2026-10-08

Implemented, source-reviewed and tested on synthetic inputs; **not yet trained or run on remote DICOMs**. Parent owns launch and STATUS updates. This pilot is separate from both the existing toy network and frozen DINO model.

## Predeclared experiment

Hypothesis: fine-tuning an ImageNet ConvNeXt-Tiny encoder with six MRI sequence slots and adjacent-slice inputs improves confidence-masked weak-validation BCE over training-only prevalence. It is a feasibility pilot, not a claim that four epochs can reproduce the public 0.929 reader or beat 0.950.

- Initialization: timm `convnext_tiny.fb_in1k`, fresh ImageNet weights. No public knee checkpoint is read. Public goodpjw2008 model/preprocessing source is supplied separately and SHA-256 checked; Apache-2.0 attribution and license accompany the bundle.
- Data: original Steven v1 `llm_labels_full.csv`, SHA-256 `96444cd3c7d396bae4a71454317d9cbe0689759b5278b30939781e04d0774537`. Other public label versions are rejected.
- Split: reuse `knee.experiment.choose_studies` with original 2,000/300 limits and seed42, then keep whole report groups up to512 train/200 validation. No reduced-holdout study is reassigned to training. All gold and normalized-report relatives are excluded by `knee.validation`; selected-series PatientID headers additionally exclude train/validation/gold overlaps transitively. Header identity remains limited by anonymization and within-series consistency.
- Real metadata check: before image-header exclusions, exactly512 train/200 validation; all712 assignments match the executed baseline `features.npz`. Gold58 excluded. Baseline assignment fingerprint `19683d3463f0a299abaaafc92ef4fd88d72b25017e6a22ad085cecc96f16d09f`. No private IDs or report text are committed.
- Inputs: six slots, four deterministic three-slice windows per slot,224px. Public preprocessing canonicalizes at384px/0.4mm then uniformly thins to at most64 slices; windows are adjacent in that declared thinned volume. Stored windows are area-resized to224 while retaining the physical field of view. The public 0.92 crop and shared rotation/scale/shift/gamma training augmentation follow. Centers are fixed across epochs in this small pilot; no claim of true native-slice adjacency after thinning.
- Supervision: exact0.5 is unknown. Confidence is `2*abs(y-0.5)`; loss averages each target's normalized weighted BCE, then averages supported targets, matching `knee.features.masked_loss`. Unsupported targets are listed. Image normalization uses fixed ImageNet constants. Only training labels fit the prior and output bias.
- Training: seed42, batch2, encoder LR1e-4, head LR5e-4, AdamW weight decay0.02, cosine schedule, FP16 with GradScaler, gradient clip3, EMA0.998 with warmup. Overflow skips also skip scheduler/EMA updates. Gold is never evaluated. Checkpoint selection uses EMA weak-validation macro BCE only.
- Reporting: training-only prevalence on identical validation rows, per-target BCE/confidence support, and explicitly thresholded weak-label AUC with unknown0.5 masked. These are weak supervision diagnostics, not independent clinical/gold AUC. Scores on this200-study subset cannot be numerically compared to old300-study baseline metrics without reevaluation on identical rows.

## Resource gates and outputs

1. CPU64-study cache pilot: label-value-blind hashed study selection, at most30min. Header scans and decode loops check the deadline. Every decoded selected series must have zero errors, at least3 slices and nonzero pixels; a study must have a valid slot. Synthetic tests check masks, neighboring channels and zero padding; real geometry/runtime remain to be rehearsed.
2. Before expansion, project actual per-study bytes and decode time to the complete split using1.25× disk/1.5× time margins. Defaults:2GiB compressed cache,1GiB free-disk reserve,60min full-cache job. A failed gate requires revising the plan; it never expands automatically. Full cache runs on CPU and reuses hash-verified pilot arrays.
3. GPU job starts only from a complete cache with recomputed plan digest and verified array hashes. Time64 forward/backward batches without optimizer updates; restore RNG afterward. Measure validation throughput separately, then preselect1–4 epochs that fit90min with1.5× margin. Stop if even one epoch is projected not to fit. No full-training loop starts before this gate.
4. Checkpoint model, EMA, optimizer, scheduler, scaler, Python/NumPy/Torch/CUDA RNG, shuffled order/cursor, metrics and cumulative elapsed time at epochs and graceful budget stops. Resume requires identical runner SHA, sources and plan; budget does not reset. The previous `best.pt` must accompany `last.pt` and is copied forward even if no new epoch improves. Abrupt process termination can lose work since the previous completed checkpoint.

Private outputs contain IDs/labels/arrays and must remain private. Cache: `fresh_reader_cache/{plan-private.json,cache-index-private.json,cache-pilot.json,cache-complete.json,arrays/}`. Training: `fresh_reader_training/{run-contract.json,gpu-gate.json,last.pt,best.pt,training-report.json}`. Generated notebooks record early failures in `/kaggle/working/fresh-reader-failure.json`. No cache transfer to Colab is needed for this path.

## Commands for parent execution

Builds only; these commands do **not** launch remote work. Run from the project root after integrating the commit, using the audited source directory shown. The CPU reference image is pinned to the public notebook's Python3.11/3.12-compatible decoder environment. Training internet is enabled solely to obtain fresh ImageNet initialization; inference packaging is a later task.

```bash
python3.12 scripts/build_fresh_reader.py \
  outputs/research/2026-10-08/convnext-audit \
  outputs/kaggle/fresh-cache-pilot knee-fresh-cache-pilot-20261008 \
  --stage cache-pilot
```

Parent can then launch that private CPU notebook, inspect its `cache-pilot.json`, and copy the exact generated `plan-private.json` fingerprint into the next build. Do not interpret CPU pilot statusPASS alone as permission to expand: require `gate.passed == true`.

```bash
python3.12 scripts/build_fresh_reader.py \
  outputs/research/2026-10-08/convnext-audit \
  outputs/kaggle/fresh-cache knee-fresh-cache-20261008 --stage cache \
  --cache-kernel ozgung/knee-fresh-cache-pilot-20261008 \
  --plan-fingerprint EXACT_REVIEWED_FINGERPRINT

python3.12 scripts/build_fresh_reader.py \
  outputs/research/2026-10-08/convnext-audit \
  outputs/kaggle/fresh-train knee-fresh-train-20261008 --stage train \
  --cache-kernel ozgung/knee-fresh-cache-20261008 \
  --plan-fingerprint EXACT_REVIEWED_FINGERPRINT
```

To resume, rebuild the training stage with the same complete cache and `--resume-kernel ozgung/PRIOR_TRAINING_NOTEBOOK`. Its output must contain both own `last.pt` and the matching `best.pt` if a best epoch exists. Keep runner bytes unchanged. A90min budget stop has little remaining allowance by design; extending the budget requires a newly declared experiment rather than silently resetting this run.

For an existing environment, equivalent entry points are `python -m knee.fresh_reader cache-pilot --data DATA --weak-labels V1_CSV --sources SOURCES --output CACHE`, `cache` with the same arguments, and `train --cache CACHE --sources SOURCES --output TRAINING [--resume LAST_PT]`.

## Verification

Twelve focused regression tests cover group/split preservation, label-blind selection, plan/source tampering, cache time/disk gates, synthetic masks/windows/resume/strict decoding, exact unknown masking, sparse-target timing, macro-loss parity, optimizer/RNG replay and compiled private notebook settings. Default dependency-free tests skip optional ML paths cleanly. Full ML-environment suite:72/72 passed; dependency-free suite:33 passed/39 optional skips; both dependency-free smoke invocations passed. `git diff --check` passed. An independent source reviewer found no remaining launch blockers after the resume/digest/metric fixes; this is not CUDA or real-DICOM certification.
