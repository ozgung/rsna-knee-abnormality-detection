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


## Setup follow-up — 2026-10-08

Replaced global `inputs.rglob` discovery with exact modern/classic mounts for the named decoder and Steven datasets and own notebook cache/resume outputs. Competition image directories are never enumerated by attachment discovery. Missing/ambiguous mounts, mutated plan digests, other owners and incomplete/mismatched own `last.pt`/`best.pt` pairs fail before training. The existing running CPU64 notebook is unchanged; rebuild subsequent stages. Data-plan configuration and fingerprint semantics are unchanged.

New notebooks emit setup/decoder/cache-copy/stage progress. Attachment setup is capped at300s, pip at120s, and ImageNet import/download/CUDA initialization at300s. A POSIX stage deadline starts before attachment setup and bounds the full CPU or GPU stage using the original1800/3600/5400s limits; nested setup deadlines cannot extend it. HTTP download/metadata timeouts are set before importing timm/HuggingFace. These bounds apply to newly built Kaggle notebooks, not to the already-running CPU64 version.

Five additional dependency-free regression tests forbid `rglob`, `glob`, `iterdir`, `os.walk` and `os.scandir` while resolving synthetic attachments, including distractors under a competition image subtree. They also reject ambiguous/mutated plans and wrong-owner/incomplete resumes, and exercise blocking/nested setup deadlines. No remote job was launched or altered in this fix.

Follow-up verification:77/77 ML-environment tests passed; dependency-free38 passed/39 optional skips; both smoke checks and `git diff --check` passed. The real metadata plan equals the running-pilot implementation exactly before header identity checks.

Deadline cleanup correction: a parent run exposed exit142 when nested cleanup rearmed an already-expired outer alarm for1µs, allowing it to arrive after restoration of the default SIGALRM handler. Expired enclosing deadlines now propagate a synchronous `TimeoutError` and are never rearmed. A repeated subprocess regression verifies1,000 nested unwinds per suite without process termination or residual timers. After this fix:78/78 ML tests pass; dependency-free39 pass/39 optional skips; both smoke checks pass. No remote jobs changed.

## Explicit CPU time reassessment after the real pilot

The actual64-study pilot decoded311 series with zero failures, storing119,629,145bytes. It passed decoding but **failed the original3600s time gate**: decode308.34779231s, total675.318511887s, projected full712 decode with1.5× margin5145.55378417s. Original `cache-pilot.json` remains unchanged.

The parent authorized a full-cache-only7200s cap. `--cache-seconds 7200` is a separate operational override, restricted to stage`cache` and values≤7200; it does not modify the plan, split, inputs or original3600s configuration field. The revised gate adds measured pilot nondecode overhead366.970719577s, giving5512.52450375s (~91.9min), and retains the original2GiB cap. Projected bytes with1.25× margin are1,663,592,797.65625, below2GiB. Existing64 arrays are size/hash verified and reused. `cache-revised-gate.json` records the original failed gate, original pilot-file SHA, override and revised decision. Both revised time and disk checks must pass; a failed original gate without an explicit override remains rejected.

```bash
python3.12 scripts/build_fresh_reader.py \
  outputs/research/2026-10-08/convnext-audit \
  outputs/kaggle/fresh-cache knee-fresh-cache-20261008 --stage cache \
  --cache-kernel ozgung/knee-fresh-cache-pilot-20261008 \
  --plan-fingerprint 1bd346c05c0aac20f54b82b790e356d3d6c1b95606accaa4a5d1da7e3e421942 \
  --cache-seconds 7200
```

The following GPU stage consumes the completed cache under the same fingerprint and retains its original5400s training limit. Synthetic regression checks verify the failed receipt and plan bytes are preserved, cached arrays are reused, the builder exposes its effective7200s timer, and invalid-stage/time/disk overrides are rejected. No remote job was launched by this implementation task.

Override verification:79/79 ML-environment tests pass; dependency-free40 pass/39 optional skips; smoke and `git diff --check` pass.


## Disposable CUDA32 diagnostic

Before the full cache finishes, the separate `probe` stage reads only32 training studies already present in the CPU64 cache. Selection is the first32 cached training rows in the existing label-blind hash order; all validation arrays are excluded. The recomputed complete plan fingerprint, external source hashes, pilot receipt, selected array sizes/hashes, shapes, masks, finite positions and labels must pass. The original failed full-cache time gate is retained as evidence; no cache expansion occurs.

Fresh generic ImageNet ConvNeXt-Tiny initialization, seed42, batch2, the same confidence-masked macro BCE, augmentation, AdamW, FP16 GradScaler, gradient clipping and EMA memory footprint exercise the intended CUDA path. The fresh source head has zero bias; it fits no prior outside these32 studies. At most64 successful updates (96 attempts including scaler warmup skips) run inside a600-second total stage cap, including setup and initialization. The loop reserves time for a final diagnostic. Initial and final losses use the same32 unaugmented training studies and online weights; a loss reduction is not a quality score or selection criterion. PASS means at least8 successful updates and changed encoder/head parameters; PARTIAL records insufficient updates, and failures retain their error.

`fresh_reader_probe/probe-report.json` records the selected-set fingerprint, full config, source/revision/package/device details, initial and final per-target training BCE and confidence support, finite gradients, AMP overflow/skips, forward/backward and complete-step durations, parameter changes, peak CUDA memory and wall time. No checkpoint or trained weights are saved or reused. The notebook is private and attaches only `ozgung/knee-fresh-cache-pilot-20261008`; internet is enabled for generic ImageNet initialization. It attaches no competition data or public knee checkpoint.

The probe and main training now hash the actual loaded generic encoder tensors using scheme `fresh-encoder-state-v1`: sorted names; length-prefixed JSON name/dtype/shape; length-prefixed contiguous CPU bytes. This produces a deterministic SHA256 without another weight file or network request. Main training preserves that initialization digest in its run contract and checkpoints.

```bash
python3.12 scripts/build_fresh_reader.py \
  outputs/research/2026-10-08/convnext-audit \
  outputs/kaggle/fresh-cuda-probe knee-fresh-cuda-probe-20261008 --stage probe \
  --cache-kernel ozgung/knee-fresh-cache-pilot-20261008 \
  --plan-fingerprint 1bd346c05c0aac20f54b82b790e356d3d6c1b95606accaa4a5d1da7e3e421942
```

Build only; parent launches after review. CUDA is unavailable locally, so unit checks establish selection, hash/mask/attachment boundaries and error reporting, not GPU trainability or throughput.

Probe verification:84/84 ML-environment tests pass; dependency-free42 pass/42 optional skips; both dependency-free smoke invocations and `git diff --check` pass. Five probe regressions cover label-independent training-only selection, private attachment settings, deterministic tensor hashing, selected-array integrity without reading validation arrays, and explicit CUDA-unavailable failure reporting.
