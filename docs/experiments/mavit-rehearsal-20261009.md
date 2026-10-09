# MaViT image-only qualification — October 9, 2026

Implemented, locally tested on synthetic DICOMs and pinned source fixtures; **real 67-study CUDA qualification is RUNNING**. This adds an independent public model component rehearsal only. It does not train, blend, submit, or claim predictive quality above the scored mirror. The prospective reason is a distinct encoder that may complement existing models; image quality, statistical independence and training lineage remain unverified.

The [public notebook](https://www.kaggle.com/code/hengck23/mavit-288-baseline-01-1-5sec-study-1gpu) uses the [hengck23 dataset](https://www.kaggle.com/datasets/hengck23/hengck23-rnsa-knee), with MIT declared. Its checkpoint passed the separate CPU audit: 341/341 exact state entries, all finite, strict load, 79,601,008 bytes, SHA256 `7acd2a990a89c7a7d73ec0488badc5058a61b0c77142742c532ebc196749bdbc`. The builder requires the authentic CPU receipt SHA256 `8d694907b0cf70504da2b33644db1ea3a18a057365987bfa59b4f8ac55a2d1e6` and unchanged CPU-audited helper sources. Published training cohort/label provenance is unknown. The original published speed claim is not a measured result here.

## Frozen active recipe

Use the CPU-audited original model definitions and complete `model_state_dict`, `weights_only=True`, exact state keys/shapes/dtypes, `strict=True`, `pretrained=False`; no external weights or unsafe pickle allowlists. Image inference calls `model.infer`, eval/no-grad, one study per batch, CUDA FP16 autocast, sigmoid inside autocast then float32 CPU. No augmentation, TTA, rank transform, additional normalization, FP32 retry or CPU fallback.

Six slots in public order: Sagittal(0,0), Sagittal(1,1), Axial(0,0), Axial(1,1), Coronal(0,0), Coronal(1,1), for fluid/fat flags. Preserve the first metadata row per complete key, original CSV order, filename tie-breaking and physical-position conventions. All selected-series files are decoded; no per-frame salvage or candidate retry. Preserve unique-modal-shape filtering and source geometry/spacing defaults. Eight rounded centers in the clipped 20–80% depth interval take adjacent triplets; position is center/D, including after right-knee normalization. Right sagittal reverses depth and channel order; right axial/coronal reverses width; explicit laterality precedes the source geometric fallback.

Full-series 1/99 percentiles precede sampling/cropping. Clip/scale and truncate to uint8 **before** scipy `ndimage.zoom(order=1)` to 288. Crop 130 mm only for nonsquare input and the original strict pixel-size bounds. Preserve the public denominator floor and intentional repeated centers for usable short series. Reader I/O uses 32 threads.

Policy `mavit-active-reader-v2` fixes a reproduced initial-review defect: the original `list(ex.map(...))` stopped observing after its first exception and could hide later submitted worker failures. The original285-test logs are preserved; the updated inverse-AST proof covers the single worker-collector replacement as well as catch/check insertions.

The active reader is extracted by exact source hash and explicit definition inventory, excluding unused reader variants and notebook top-level code. An inverse-AST regression proves all original scientific expressions remain unchanged. The independent comparator adds the five fatal-exception prefixes and an exception-observation-only worker collector. Both paths drain every submitted future before accepting a malformed-view failure; fatal/resource errors take precedence, then unknown programming errors. Successful result order and executor/thread count are unchanged. Production adds nonmutating validity checks and a narrowly declared selected-view catch. The original public model/helper files are unchanged.

## Declared malformed-input extension

Mask a selected view for missing/unreadable pixels, fewer than three modal-shape frames, ambiguous modal-shape ties, nonfinite/unsafe pixel shape, or invalid accepted geometry/spacing. Retain other usable views; do not resample remaining files or try duplicate alternatives. Preserve the source mask for a black view if another view is usable. A whole-empty or whole-black study is fatal. Unsupported metadata/target schemas remain fatal.

Production's inner catch hook recognizes malformed DICOM/geometry exception families and otherwise re-raises unknown programming errors. Known missing/malformed geometry defaults remain source-compatible. MemoryError, TimeoutError, CUDA OOM, relevant resource OSError and OpenCV no-memory errors escape all five public catch regions in both paths. Decoder/reference fallbacks to pydicom are part of the original reader and are counted separately from prohibited prediction/device/precision fallbacks.

Aggregate counters report selected/missing/masked/black views, completed-series decoded files, shape filtering, defaults, decoder fallbacks/errors, laterality ambiguity, and resource/whole-empty errors. Failed series may have decoded some files before failure; the completed-series file counter does not claim those partial reads. This handling of malformed inputs is an inference extension, not trained or validated pixel parity. Healthy studies require exact uint8 volume, bool mask and float32-position equality against the independent public reader; same-model probabilities and within-run repeat errors must each be ≤1e-6. Strict-ineligible malformed studies are counted explicitly.

## Offline package and limits

The notebook attaches only competition data, the published MaViT dataset, and the existing ConvNeXt dataset for three SHA-pinned cp312/pure-Python codecs. It reads official test IDs/target schema and train/test series metadata, never train labels, reports or the published gold CSVs. Cohort selection is the existing SHA256(seed:StudyInstanceUID) order with seed20261008:64 training studies plus the three visible test IDs. Source/bundle, checkpoint, codec, ordered-cohort and visible CSV hashes are recorded. Only the three visible probabilities are exported; train predictions are represented by an aggregate matrix hash.

Use the pinned existing Python3.12 CUDA image with audited torch2.10.0+cu128, timm1.0.26, einops0.8.2, numpy2.0.2, torchvision0.25.0+cu128, pydicom3.0.2 and pandas2.3.3. SciPy is **first-observed** under that fixed image and recorded in the receipt; any future production implementation must pin that exact observed version. Do not infer the remote version from local synthetic tests.

Fresh child-process flags are explicitly declared: cuDNN benchmark=False, deterministic=False, cuDNN TF32=True, matmul TF32=False, torch CPU threads=4. These are the qualification baseline, not an assertion that the public notebook prescribed them. The 32,400-second (actual9h) timer starts before notebook bootstrap/attachment setup. The parent uses process-group timeouts and kills remaining descendants even after normal child exit. No ETA-based abort or arbitrary memory batch adjustment. Successful qualification requires parent completion, so a child's PASS cannot survive failed cleanup as a qualifying receipt. Elapsed setup/read/inference/public-comparison/repeat times, GPU allocated/reserved peaks, host RSS and disk are recorded; the 1,300-study p95 projection is informational only.

## Build and checks

After integration, build from the primary repository; do not upload the local pre-review smoke build:

```bash
.venv/bin/python scripts/build_mavit_rehearsal.py \
  outputs/kaggle/mavit-image-rehearsal-20261009v1 knee-mavit-image-rehearsal-20261009v1 \
  --source-dir outputs/analysis/post-fusion-research-20261009/mavit/dataset-code \
  --cpu-report outputs/kaggle/mavit-checkpoint-audit-20261009-result/mavit-checkpoint-audit/checkpoint-audit.json
```

This command only builds an immutable private/offline GPU notebook and contracts. Parent review/launch is separate. Intended outputs are `mavit-rehearsal/mavit-audit.json` and `mavit-rehearsal/visible-predictions.csv`; failures contain aggregate type/counts only. `require_rehearsal` rejects incomplete, old, wrong-source/contract/checkpoint/environment, failed-stress, nonfinite, over-budget and resource-failed receipts. No checkpoint is copied locally.

Focused fixture command:

```bash
KNEE_REFERENCE_PARENT=/Users/ozgun/Dev/rsna-knee-abnormality-detection \
  /Users/ozgun/Dev/rsna-knee-abnormality-detection/.venv/bin/python \
  -m unittest discover -s tests -p test_mavit_rehearsal.py -v
```

Checks cover inverse AST, actual tiny DICOM decoding/intensity/orientation/laterality, square/nonsquare uint8 zoom, all-six-slot mixed-decoder parity, malformed/missing/black views, exact first-series behavior, fatal catch regions and counters, concurrent first-unreadable/later-fatal-or-unknown errors, descriptor/thread/process-group cleanup, CPU receipt/source binding, offline notebook/import, metadata privacy, checkpoint/device rejection and inference call contract. The CPU inference-contract mock verifies argument/call order only; it is not a CUDA numerical test. Runtime also executes16 tiny production-path stress cases before real images. Real probabilities, remote decoder/torch/scipy compatibility, throughput and memory still require the reviewed GPU rehearsal.

Local check logs/build smoke are ignored under `outputs/analysis/mavit-rehearsal-20261009/` in the primary repository. No changes to existing qualified modules, active training sources or STATUS; no commits or remote actions performed by this implementation task.

Parent integrated as678d31c and launched `ozgung/knee-mavit-image-rehearsal-20261009v1`v1 at~03:20UTC; API RUNNING03:22UTC. Build and launch receipts are under the paths above. This is not yet a successful image qualification.
