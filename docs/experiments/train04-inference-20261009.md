# TRAIN04 own-model offline inference

**Implemented and synthetic-tested; real checkpoint export, CUDA inference and67-study raw-MRI qualification have not run.** This package preserves the selected own EMA model and image preprocessing from `coverage_training.py`/`coverage_cache.py`; neither training source nor the active controller changes. It adds no training or model blending.

Healthy inference uses the same pinned goodpjw2008 Apache-2.0 `preprocess.py` and `knee.py`, canonical384-pixel volume followed by OpenCV `INTER_AREA` resize to256, up to64 retained slices per selected series, six fat-suppression/plane slots, and the unchanged `sample_windows(k=12, train=False)` triplets/masks/positions. Stable series-ID ties and capped file counts match cache creation. The selected EMA state is loaded strictly into the original ConvNeXt-Tiny ImageNet22k→1k architecture with `pretrained=False`, CUDA float16 inference, no TTA, no probability fallback and no report input. The public knee checkpoints present beside codec wheels are never opened.

The separately versioned `train04-coverage256-partial-v1` corruption policy accepts a selected series with unreadable files only when at least three decoded frames remain nonzero after resize. Unusable selected slots are masked; another candidate is never tried. Black frames are retained in their original positions for otherwise usable volumes. Whole-empty studies fail. Exactly one public per-file exception handler receives a fatal-error prefix so MemoryError, TimeoutError and OpenCV `StsNoMem` escape the entire catch, including list extension; inverse-AST regression confirms scientific statements are unchanged. Source files themselves remain byte-identical. **This corrupt-input policy extends training-cache acceptance and is not yet validated on real corrupt MRI.** Counters distinguish missing slots/directories, unreadable files, partial/failed/short/empty series and black frames.

The local exporter requires a completed12-epoch own training report, matching run/checkpoint/private selected evidence, exact2000/200 cache plan and receipt hashes, recorded initialization, source/configuration and minimum-BCE epoch. It reproduces selected BCE from logits, loads checkpoint tensors with `weights_only=True`, verifies finite tensors and strict model keys/shapes, then writes only `weights.pt` and a non-sensitive `manifest.json`. Study IDs, weak labels, logits, optimizer state and private evidence are not exported. No raw image cache is copied.

Export also requires the original saved training build directory. It reads the notebook's literal `CODE`, `EXTERNAL` and `CONTRACT` assignments without execution, verifies every bundled source hash, checks the archived runner's static configuration and initialization, and binds the build's canonical source digest/revision to the run receipt. The original TRAIN04 runner SHA is explicitly pinned; a future corrected objective needs separate review. Current training source changes do not invalidate this archived reference, while changed cache preprocessing does require review. The exported manifest records build/notebook/source hashes and the original revision. No hash is rewritten or inferred from the current training module.

After the parent retrieves a completed own result, run from the integrated repository; substitute its actual result directory and a new asset destination:

```bash
.venv/bin/python scripts/build_coverage_inference.py export \
  outputs/research/2026-10-08/convnext-audit \
  outputs/kaggle/train04-reference-20261009-result/coverage_training \
  outputs/kaggle/train04-cache-full-v2-result/coverage_cache \
  outputs/train04-inference-assets \
  --training-build outputs/kaggle/train04-reference-20261009

.venv/bin/python scripts/build_coverage_inference.py build \
  outputs/research/2026-10-08/convnext-audit \
  outputs/train04-inference-assets outputs/kaggle/train04-inference-rehearsal \
  knee-train04-inference-rehearsal-20261009 \
  --weights-dataset ozgung/knee-train04-own-ema-20261009 \
  --cache-kernel ozgung/knee-train04-cache-full-20261008
```

These commands do not upload or launch anything. The private own-weights dataset reference is prospective; the parent must create/attach the exact exported pair separately. Keep the exporter directory limited to those two files. Export requires local Torch/timm and the pinned public source; timm is currently absent from the local primary environment. Real selected tensors are not yet available.

Rehearsal attaches competition images, the own weights dataset, the existing pinned Python3.12 codec wheels, and the full cache kernel. It selects64 cached training-set studies by a fixed UID hash over the exact cache index, without reading labels/reports (the pool includes the development rows; no metric is evaluated). Together with the three visible test studies this is67 studies. Every study is reread and predicted again. All64 cached cases must qualify as healthy and match selected series, complete uint8 volumes,12 triplets, masks and positions exactly; cached-input and reread predictions must agree within1e-6. The receipt includes deterministic synthetic production stress checks, actual CUDA/RSS/output resource measurements, input counters and conservative1300-study timing estimates. Estimates are informational; the only runtime deadline is the competition's9-hour limit.

A full entry requires a successful matching `rehearsal.json`:

```bash
.venv/bin/python scripts/build_coverage_inference.py build \
  outputs/research/2026-10-08/convnext-audit \
  outputs/train04-inference-assets outputs/kaggle/train04-inference-entry \
  knee-train04-inference-entry-20261009 \
  --weights-dataset ozgung/knee-train04-own-ema-20261009 \
  --rehearsal-report outputs/kaggle/train04-inference-rehearsal-result/coverage_inference/rehearsal.json
```

Full mode has no cache/training-kernel attachment. Internet is disabled. Qualification binds own weights/manifest, all bundled runtime and builder hashes, public sources, corruption policy, Python/Torch/timm and all three codec hashes; old receipts cannot qualify changed code. Actual test IDs come from `test.csv` in its exact order, while target names/order come from the official sample header and must match the model. Sample rows never determine hidden coverage. The writer requires unique exact IDs and12 finite probabilities per row and exclusively creates `submission.csv`.

Implementation files are `knee/coverage_inference.py`, `scripts/build_coverage_inference.py` and two focused test modules. No model/array/generated notebook is retained from synthetic fixtures. Test-only OpenCV4.12.0.88 was installed under `/tmp/coverage-inference-test-deps` to exercise pixel tests; the primary environment and repository dependencies were unchanged. GPU throughput, actual own weight compatibility, observed visible predictions, hidden-test robustness and an eligible submission remain pending parent-run qualification. No job, dataset upload, submission or commit was performed by this implementation task.

Verification:11/11 focused tests PASS with real OpenCV and pinned public preprocessing exercised on synthetic pixels; all209 fixture-enabled ML tests PASS, default smoke PASS, source compilation PASS. Full log: `outputs/analysis/train04/coverage-inference-provenance-tests-20261009.log` in the primary workspace. Existing TRAIN04 training/cache/controller files are unchanged.
