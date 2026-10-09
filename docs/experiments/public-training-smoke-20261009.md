# Public Raptor training smoke — prospective protocol

This is a build-only, training-only wiring experiment. The hypothesis is that the
published Raptor MIL recipe can perform finite CUDA updates on the public corpus
and move its encoder and nonbias head from generic ImageNet initialization. It
makes no generalization, leaderboard, corpus-ceiling or model-quality claim.
Parent reviews, integrates and launches; the existing TRAIN04 recovery continues.

## Fixed experiment

- Original audited 2,000-study training membership only; choose 40 by the fixed
  SHA-256 salt `public-raptor-smoke40-20261009-v1`. All 300 original development,
  300 sealed, 58 gold and four gold-related quarantined studies stay excluded.
  Assert transitive component separation using the immutable private identity
  manifest. Its SHA is `21c71745703d742585684298494a6269ae4217fadc5fc4c26f3bf0d10a289467`.
- Generic `convnext_tiny.fb_in22k_ft_in1k`, resolution256, ImageNet normalization,
  eight windows, batch8, accumulation1, two epochs (ten attempted updates), seed42.
  One explicit CUDA device, FP16 autocast/scaler, workers0, no ROI or chunking.
  There is no CPU fallback or replacement of skipped updates.
- Reuse exactly five definitions from public `nartaa/rsna-knee-hpo-assets/train_knee.py`:
  `_TwoPartVols`, `StudyWindows`, `collate`, `build_backbone`, `RaptorClassifier`.
  Full-file SHA `d39966791ef1c48d9317656e8346c60cae1c064192c87b3f350e298ec4c567e8`;
  never import/execute its top-level code, unrestricted weight loader or gold main.
- Preserve the public dense positive-weighted BCE, per-finding attention MIL,
  intensity jitter, AdamW (backbone3e-5/head1e-3, wd.02), norm3 clipping and
  OneCycle (ten-step horizon, pct_start.15, published final-step boundary).
  Scheduler advances only after a successful AMP optimizer step. No EMA.
- The pinned `labels_v1_blend.parquet` SHA is
  `d2cdb8a37e0c4c491758d17c96e3bd4081f06b257c7849608df88590368e94e2`.
  Only selected training target rows are returned by the Parquet filter. The
  whole file's IDs are checked as identity metadata; reserved target rows are
  not used. Silent cells are soft model-prior supervision, not asserted negatives.
  The source author compared label variants on gold: upstream gold independence
  is not claimed. No gold-picked per-target variant is used.

## Assets and scientific limitations

Mount the two CC0 dread corpus datasets; do not download the22GB volumes locally.
Pin public ID/mask bytes, retain part1-then-part2 order, check shapes and types,
and record the selected40 pixel fingerprint at first remote observation. This
is not a pre-known full-volume hash. The identity manifest comes only from the
completed private `ozgung/knee-sealed-identity-full-20261008` kernel; the large
TRAIN04 cache and competition CSV/report files are not attached.

The published44slice/336px/140mm/15–85% corpus and public sampler are unchanged.
The sampler can cross series boundaries and masked gaps; this is documented,
not silently repaired in one path. The previously qualified64slice/6–94% reader
is not this corpus. Real DICOM-to-corpus parity is a later image-only submission
qualification task, not a prerequisite67study evaluation during every training
step. No mirror or target relabeling is added.

The pinned Kaggle Docker image and torch2.10.0+cu128/torchvision0.25.0+cu128,
timm1.0.26, numpy2.0.2, pandas2.3.3 must match. Record actual Python, PyArrow,
sklearn, CUDA/cuDNN and GPU. Internet is enabled only for generic ImageNet assets;
no report/image processing service is called. Use an isolated `/tmp` cache and
record downloaded generic artifact hashes/config plus the complete initial
model tensor hash. These initialization hashes are first-observed provenance,
not an assertion of a previously matched TRAIN04 initialization.

## Measurements and acceptance

Evaluate only these same40 training studies with deterministic16window inputs,
before fitting and after each epoch. Report dense weighted/unweighted BCE,
entropy, positive/negative thresholded-soft support and per-target AUC (undefined
single-class targets explicit; macro over defined targets only). These are
training diagnostics, never independent validation. Compare against the same40
training-label mean and the weighted-loss optimum constant; no minimum AUC or
loss-improvement gate selects a checkpoint.

PASS requires completed two epochs/ten attempts, at least one successful update,
finite accepted preclip gradients, real encoder/nonbias-head movement and an
exact same-process checkpoint save/reload prediction comparison. AMP overflow
updates only its scaler, not model/optimizer/scheduler. Record all ten step norms,
clipping/AMP counters, epoch losses, final movement, walltime and memory. Temporary
fitted checkpoints stay under `/tmp` and are deleted; only aggregate JSON is
exported. There is no resumable fit or reusable trained checkpoint in this smoke.
Any error stops the run and writes an allowlisted exception class and phase;
exception text/identifiers, reports and predictions are not exported.

The experiment bound is40studies/twoepochs, not an assistant spending limit.
The only elapsed-time abort is the actual12h Kaggle training-platform ceiling.
No timing forecast is an acceptance gate; estimate a later fit from measured
throughput. No model-quality inference follows from passing these checks.

## Build, validation and parent launch handoff

From the integrated repository, with a local Parquet reader available:

```bash
PYTHONPATH=/tmp/coverage-inference-test-deps:/tmp/public-training-smoke-deps \
.venv/bin/python scripts/build_public_training_smoke.py \
  outputs/kaggle/public-training-smoke-20261009 \
  knee-public-training-smoke-20261009 \
  --public-source outputs/analysis/public-train-recipe-20261009/assets/train_knee.py \
  --manifest outputs/kaggle/sealed-identity-full-result/sealed_identity/sealed-manifest-private.json \
  --label-table outputs/analysis/public-train-recipe-20261009/assets/labels_v1_blend.parquet \
  --corpus-metadata outputs/analysis/public-train-recipe-20261009
```

The builder requires a fresh destination, checks actual selected metadata/labels,
and embeds only source, fixed settings, hashes and aggregate counts. It never
launches. Use the authoritative post-integration build; ignored draft builds
pre-date any later source review and must not launch.

```bash
PYTHONPATH=/tmp/coverage-inference-test-deps:/tmp/public-training-smoke-deps \
KNEE_REFERENCE_PARENT=/Users/ozgun/Dev/rsna-knee-abnormality-detection \
.venv/bin/python -m unittest discover -s tests -p 'test_public_training_smoke.py' -v
```

After parent verifies current quota, unique slug, source/attachment hashes and
private/internet/GPU metadata, the exact launch command is:

```bash
/Users/ozgun/miniconda3/bin/kaggle kernels push \
  -p outputs/kaggle/public-training-smoke-20261009
```

This command is guidance, not an action performed here. Retrieve only
`public_training_smoke/smoke-report.json`; verify `require_result(report, contract)`
against the immutable local build and platform COMPLETE status. Preserve failures.
The user-authorized next experiment uses the full eligible pool with group-disjoint
public folds, excluding gold and related studies; historical reserved roles here
apply only to this fixed smoke. Fable owns full-fold orchestration, with a separate
prospective protocol and gold-free checkpoint selection. Do not launch duplicate folds.
