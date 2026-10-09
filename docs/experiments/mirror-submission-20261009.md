# Standalone native320 mirror entry

October 9, 2026. Packaging is implemented and locally synthetic-tested. The actual 67-study mirror rehearsal is running separately; this document does **not** claim that its receipt, checkpoint, visible entry, hidden runtime, or score has passed. Parent owns retrieval, qualification, launch and submission. The public author's 0.949 is not our result; the scored 0.937 reader remains unchanged.

The standalone entry preserves the [nartaa source](https://www.kaggle.com/code/nartaa/rsna-knee-0949-anatomical-mirror), accuracy checkpoint, native320/K94 windows, equal original/anatomical-mirror probabilities and reviewed `mirror-study-lut-partial-v1` input policy. No model math, preprocessing, alternative-series retry, fp32 retry or fabricated predictions are introduced. The [published weights](https://www.kaggle.com/datasets/nartaa/rsna-knee-publication-swa-weights-20261007) have dataset license `other` and explicit publisher competition-use permission. The accepted rules2.6 reading is not explicit host/prize clearance; see [the rehearsal record](mirror-rehearsal-20261009.md).

The builder parses the preserved rehearsal notebook's literals without executing it. It reconstructs the original complete code cell and verifies source-map hash, build receipt, contract, revision, public-source extraction, publisher manifest pin, original entry/builder hashes, and offline metadata. Every needed scientific dependency must still match its archived bytes. Only that archived dependency closure plus the new standalone module/entry is packaged. The original rehearsal bundle and new entry bundle retain separate hashes: adding this packaging does not pretend the 67-study run used the new full repository bundle, and unrelated training modules cannot alter the image pipeline silently.

A valid new 64-training-plus-3-visible receipt is mandatory even for building the visible entry. It must pass the existing exact checkpoint/source/policy/environment/sampling/stress/strict-coverage qualifier. Strict healthy comparison counts are copied honestly; partial-input comparisons are not relabeled strict. The report's visible CSV hash and original revision must match. The first standalone version processes only the three actual visible IDs and repeats each two-view inference. Building the full entry then requires that version's PASS report and exact CSV, matching IDs and target order against the 67-study CSV and cross-run absolute probability error at most 1e-4 with unchanged visible per-target rankings and ties. A difference fails qualification for inspection; the builder never alters probabilities to force parity. Prediction fixtures are compared locally and are not attached or embedded in either notebook.

The full runtime reads only `test.csv`, `test_series.csv`, the `sample_submission.csv` header and raw test MRI. It preserves actual test order, verifies official 12-target order against the model, checks finite [0,1] probabilities and rejects fallback-like rows. Selected missing/unreadable inputs retain the reviewed masks/counters; whole-empty and resource failures remain fatal. The actual 9-hour timer begins before bundle writing/imports and covers attachment resolution, offline codec installation, model initialization and inference. Sequential sample timing estimates are informational. Receipts include source/weight/codec pins, environment, elapsed/setup time, input counters, host/CUDA memory, disk, ID/target fingerprints and CSV hash. No label/report dataset, training notebook or extra model is attached; the ConvNeXt dataset supplies only the three pinned cp312/pure-Python codec wheels.

Run these commands from the integrated primary repository only after fetching authentic result files. They build locally and never upload or execute a GPU job:

```bash
.venv/bin/python scripts/build_mirror_submission.py \
  outputs/kaggle/mirror-visible-20261009 knee-mirror-visible-20261009 \
  --rehearsal-build outputs/kaggle/mirror-rehearsal-20261009 \
  --rehearsal-report outputs/kaggle/mirror-rehearsal-20261009-result/reference-rehearsal/rehearsal.json \
  --rehearsal-csv outputs/kaggle/mirror-rehearsal-20261009-result/reference-rehearsal/visible-predictions.csv \
  --public-source outputs/research/2026-10-08/mirror-audit/rsna-knee-0949-anatomical-mirror.py
```

After the parent runs that private offline visible version and retrieves its outputs:

```bash
.venv/bin/python scripts/build_mirror_submission.py \
  outputs/kaggle/mirror-submission-20261009 knee-mirror-submission-20261009 \
  --rehearsal-build outputs/kaggle/mirror-rehearsal-20261009 \
  --rehearsal-report outputs/kaggle/mirror-rehearsal-20261009-result/reference-rehearsal/rehearsal.json \
  --rehearsal-csv outputs/kaggle/mirror-rehearsal-20261009-result/reference-rehearsal/visible-predictions.csv \
  --public-source outputs/research/2026-10-08/mirror-audit/rsna-knee-0949-anatomical-mirror.py \
  --submission \
  --visible-report outputs/kaggle/mirror-visible-20261009-result/mirror-submission/mirror-audit.json \
  --visible-csv outputs/kaggle/mirror-visible-20261009-result/mirror-submission/visible-predictions.csv
```

All build directories must be new. The full notebook writes `/kaggle/working/submission.csv` only after complete validated inference. Parent must still verify the full version's ordinary visible execution before authorizing its hidden rerun. No fusion is implemented.

Focused checks use synthetic probabilities/receipts and the genuine archived source build; a synthetic receipt is never remote qualification. Tests cover executable tampering, changed scientific source, obsolete 24-study receipts, missing visible evidence, CSV/target/ID/value tampering, actual test-order selection, official series schema, exact two-view production calls, resource failure without output/fallback, exhausted-deadline prevention before model loading, offline attachments, size limits and fresh-interpreter isolated bundle import. Existing source fixtures cover healthy/partial/missing/corrupt geometry and pixel behavior through the unchanged production module.

```bash
PYTHONPATH=/tmp/coverage-inference-test-deps \
KNEE_REFERENCE_PARENT=/Users/ozgun/Dev/rsna-knee-abnormality-detection \
KNEE_REFERENCE_SOURCE=/Users/ozgun/Dev/rsna-knee-abnormality-detection/outputs/research/2026-10-08/mirror-audit/rsna-knee-0949-anatomical-mirror.py \
KNEE_RAPTOR_SOURCE=/Users/ozgun/Dev/rsna-knee-abnormality-detection/outputs/research/2026-10-08/raptor-audit/knee-mri-twelve-findings-from-a-single-model.py \
/Users/ozgun/Dev/rsna-knee-abnormality-detection/.venv/bin/python -m unittest discover -s tests -v
```

Local verification:236/236fixture-enabled tests PASS, including six new standalone tests; the existing17-case mirror input stress suite also passes. Archived executable reconstruction, synthetic visible/full builds, fresh-interpreter bundle import, in-memory source compilation, whitespace checks and dependency-free smoke PASS (CPU selection, no model forward). Full test log: `outputs/analysis/mirror-submission-20261009/tests.log`. No commit, remote launch, model download, upload or submission was performed by this implementation task. Generated tests use temporary directories; durable aggregate logs live in primary ignored `outputs/analysis/mirror-submission-20261009/`.

## Cross-process precision correction

The first visible run `ozgung/knee-mirror-visible-20261009`v1 completedPASS in50.9997s, with exact within-process repeats. Its original full-build qualification **failed** the assistant-selected1e-6cross-process guard: maximum difference1.8775463e-5, mean3.2368633e-6, all visible rankings unchanged. Preserve this failure; do not retroactively pass it. Archived source, decoder, scientific modules, checkpoint, codecs, target order, T4 and package versions match; summed visible input counters match exactly. Rehearsal-only CPU/reference checks and synchronization/allocation history differ. CUDA benchmarking algorithm selection is a plausible explanation, not established causation.

Prospective correction: permit absolute cross-process probability error≤1e-4 (no relative term), require identical visible per-target ordering and ties, retain1e-6within-process repetition and exact source/weights/pixel/window/schema requirements. This is a numerical precision policy, not a public-score optimization. It does not establish hidden rank invariance. Build a fresh visible package because qualification/runtime fingerprints changed; old visible evidence cannot qualify the new package. If this guard fails, investigate rather than progressively relaxing it. Scientific inference functions and all model values remain unchanged. Regression tests reject rank flips even below1e-4.

## Submitted version

Entry56978342 acceptedPENDING onOctober8at23:44:10UTC: `ozgung/knee-mirror-submission-20261009`v1, build85f3de4. Freshvisible qualification45.2801s and finalfullvisible45.9177s PASS. Both independently satisfy≤1e-4against the immutable67rehearsal CSV (8.6725e-5 and6.6340e-5 respectively), with identicalranks/ties. Their pairwisedifference1.29223e-4 is within the corresponding2e-4trianglebound; the fixedreferencepolicy was not relaxed. FinalCSV SHA7943d3acd89246fd5355607346ac06201d356f4881895d8fdf679bc175c143fc. Finalreceipt`outputs/analysis/mirror-submission-20261009/final-visible-verification.json`; hiddenresultpending, no scoreclaim.

October9at02:06:44UTC: API confirmsentry56978342COMPLETE, publicscore**0.949**, emptyerror. Thisisourreproducedresult, +0.012vs62-windowblend0.937. Receipt`outputs/analysis/mirror-submission-20261009/entry-latest.json`. Targetstrictlyabove0.950remainsunmet; frozenVersiacandidatewasrecordedbeforethisscore.
