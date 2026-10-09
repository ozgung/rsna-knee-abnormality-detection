# Frozen Versia fusion: common rehearsal and full packaging

Implemented locally; the new composite has **not yet run on CUDA or been submitted**. Parent owns review, launch, receipt retrieval and submission. Existing scientific modules, checkpoints and scored notebooks are unchanged. The frozen specification predates the newly scored mirror entry; no target weight is selected from our labels or scores.

The fixed [published source](https://www.kaggle.com/code/evgendvorkin/rsna-versia-5) snapshot is `outputs/analysis/mirror-fusion-spec-20261009/spec.json`, SHA `61931a5687982af68f50df34a6a67fbbc76ff875869758f08e0d8b3dac41a4cd`. Mirror retains native320/K94, two-view mean probabilities and its reviewed partial-input policy. ConvNeXt retains16 windows, paired fat/nonfat candidate-list copying before best-series selection, three pinned folds, batch4/four workers and the normal last remainder. The official test CSV order determines batching and mirror tie breaking; source metadata order still resolves equal series-size candidates.

For mirror probabilities M and ConvNeXt probabilities C, the exact formula is `R((1-w)*R(float32(Q(M))) + w*R(C))`. Q is NumPy's default double argsort after clipping, divided by `max(1,N-1)`; it breaks mirror ties. R is ascending pandas percentile rank with average ties. Ordered weights for ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion and Fracture are `.25,.20,.20,.08,.25,.08,.08,.04,.03,.35,.30,.08`. Published constant-column checks remain; there is no missing-arm fallback. Final rank rows may legitimately equal1 across all targets.

The builder requires both authentic67-study receipts and reconstructs their original source archives. The mirror's historical all-module bundle remains bound to its archived hash; adding this new orchestration does not silently replace that hash. Shared scientific files must match both archives/current bytes. A separate qualification fingerprint binds new runtime/build bytes, original receipts, public sources/weights/codecs, frozen formula and common package versions. The final builder accepts only a successful common-visible result with that exact fingerprint.

The mirror and ConvNeXt run in separate sequential processes. The process group, including workers, is terminated on failure/timeout and cleaned after successful exit before the other arm starts. The outer nine-hour deadline begins before source extraction, imports, mount resolution, codec installation and both models. Each child receives the same absolute deadline. Mirror uses benchmark/matmul-TF32 true; ConvNeXt uses false; both record deterministic false and cuDNN-TF32 true as in the qualified environment. CPU workers start before ConvNeXt CUDA model construction. A fresh destination and exclusive CSV creation reject stale outputs. Resource/model/whole-empty failures abort; no batch, model, precision or CPU retry exists.

Common-visible production remains one ConvNeXt batch of3. An additional repeated batch2 forward is **diagnostic only**: finite outputs and within-shape repeat ≤1e-6 are required; differences from batch3 are recorded without changing its predictions. This matters because the authentic Versia67 batch1 diagnostic changed some ranks (max probability difference1.15633e-4); production therefore preserves batch4/final remainder. The common run also compares exact visible ConvNeXt tensors to the protected public source and states healthy eligibility separately. Both components repeat within-process at absolute1e-6. Locally, the full builder compares each common raw CSV with its immutable67-component CSV: mirror absolute1e-4, ConvNeXt absolute1e-6, zero relative tolerance and all pairwise visible ranks/ties unchanged. Mirror cross-process precision tolerance is inherited from the reviewed standalone policy; it is not evidence of hidden rank invariance.

No labels, reports or prediction fixtures are embedded. The only attachments are competition images/metadata and the two public weight/codec datasets. Mirror's custom `other` license includes publisher competition-use permission; this is not a claim of explicit host/prize clearance. ConvNeXt source/weights retain goodpjw2008 Apache-2.0 attribution; Versia recipe attribution is evgendvorkin.

## Build commands (no launch)

Run from the reviewed/integrated repository with its existing interpreter. Use fresh destinations; keep archived originals intact. The following common arguments are identical for visible and full builds:

```bash
PROJECT=/Users/ozgun/Dev/rsna-knee-abnormality-detection
KAGGLE_EVIDENCE="$PROJECT/outputs/kaggle"
FUSION_ARGS=(
  --mirror-build "$KAGGLE_EVIDENCE/mirror-rehearsal-20261009"
  --mirror-report "$KAGGLE_EVIDENCE/mirror-rehearsal-20261009-result/reference-rehearsal/rehearsal.json"
  --mirror-csv "$KAGGLE_EVIDENCE/mirror-rehearsal-20261009-result/reference-rehearsal/visible-predictions.csv"
  --public-source "$PROJECT/outputs/research/2026-10-08/mirror-audit/rsna-knee-0949-anatomical-mirror.py"
  --versia-build "$KAGGLE_EVIDENCE/versia-convnext16-rehearsal-20261009"
  --versia-report "$KAGGLE_EVIDENCE/versia-convnext16-rehearsal-20261009-result/versia-rehearsal/versia-audit.json"
  --versia-csv "$KAGGLE_EVIDENCE/versia-convnext16-rehearsal-20261009-result/versia-rehearsal/visible-predictions.csv"
  --spec-dir "$PROJECT/outputs/analysis/mirror-fusion-spec-20261009"
)
"$PROJECT/.venv/bin/python" scripts/build_versia_fusion.py \
  "$KAGGLE_EVIDENCE/versia-fusion-visible-20261009" knee-versia-fusion-visible-20261009 \
  "${FUSION_ARGS[@]}"
```

Parent may launch the reviewed visible package privately/offline using the existing authorized GPU. No launch is performed by this implementation task. Retrieve `versia-fusion/` from its result, preserving `fusion-audit.json`, `visible-predictions.csv`, and both component subdirectories' raw CSV/audit files. Runtime logs expose only aggregate progress/failure types. `runtime-input.json` contains source/contracts/mount paths, no labels, images or reference predictions. Then build the full entry:

```bash
"$PROJECT/.venv/bin/python" scripts/build_versia_fusion.py \
  "$KAGGLE_EVIDENCE/versia-fusion-submission-20261009" knee-versia-fusion-submission-20261009 \
  "${FUSION_ARGS[@]}" --submission \
  --visible-results "$KAGGLE_EVIDENCE/versia-fusion-visible-20261009-result/versia-fusion"
```

The full notebook still requires its own successful visible execution and exact official output review before parent submission. Actual hidden test coverage comes from `test.csv`, not the three-row sample submission. The full output is `/kaggle/working/submission.csv`; detailed raw outputs and aggregate receipts remain under `/kaggle/working/versia-fusion/`.

## Local checks

The10 new tests cover public-source rank/tie equality, ID coverage/alignment/order, missing/failed arms, stale outputs, source/receipt tampering, within/cross-process precision, full offline packaging/import, true process-group cleanup, and actual four-worker batch remainder/order for N1 through9. They use synthetic predictions or read-only archived receipts; no real model runs. Existing component synthetic DICOM/resource/parity tests remain in the full fixture suite.

```bash
PYTHONPATH=/tmp/coverage-inference-test-deps \
KNEE_REFERENCE_PARENT=/Users/ozgun/Dev/rsna-knee-abnormality-detection \
KNEE_VERSIA_SPEC=/Users/ozgun/Dev/rsna-knee-abnormality-detection/outputs/analysis/mirror-fusion-spec-20261009 \
KNEE_REFERENCE_SOURCE=/Users/ozgun/Dev/rsna-knee-abnormality-detection/outputs/research/2026-10-08/mirror-audit/rsna-knee-0949-anatomical-mirror.py \
KNEE_RAPTOR_SOURCE=/Users/ozgun/Dev/rsna-knee-abnormality-detection/outputs/research/2026-10-08/raptor-audit/knee-mri-twelve-findings-from-a-single-model.py \
/Users/ozgun/Dev/rsna-knee-abnormality-detection/.venv/bin/python -m unittest discover -s tests -v
```

Local four-worker tests need the normal process/IPC permission allowed by the reviewed shell execution; restricted sandbox IPC can fail independently of code. Final fixture-enabled suite: **262/262 PASS**; dependency-free smoke PASS (`torch_forward:false`); source compile and diff checks PASS. The full log is `outputs/analysis/versia-fusion-20261009/tests-full.log`. Parent and independent read-only source reviews found no blocker; real common CUDA probabilities/runtime remain untested. A temporary build-only smoke artifact resides in primary ignored `outputs/analysis/versia-fusion-20261009/build-smoke-v1/`; it predates final review and must not be launched. No private/generated artifacts are stored outside primary ignored outputs after temporary test directories are removed.
