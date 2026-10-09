# Standalone MaViT production packaging — October 9, 2026

Implemented, reviewed and qualified on visible data; **entry56989116 is pending; hidden execution and public score remain unverified**. This is one public MaViT model with raw sigmoid outputs, not a fusion or training experiment. Training lineage, predictive quality and independence remain unknown. No claim above the current0.949 best follows from successful execution.

The authentic [MaViT67 rehearsal](mavit-rehearsal-20261009.md) passed162.977s on a TeslaT4:67/67 exact healthy volume/mask/position comparisons, exact same-model public probabilities, repeat error0,16stress checks, and zero masked/decoder/resource failures. It observed SciPy1.16.3, CUDA12.8 and cuDNN91002. Source/rehearsal revision `678d31c`; complete strict CPU checkpoint SHA256 `7acd2a990a89c7a7d73ec0488badc5058a61b0c77142742c532ebc196749bdbc`. Authentic build/result are under `outputs/kaggle/mavit-image-rehearsal-20261009v1{,-result}/`; parent independently verified them. No repeat67 run is needed.

## Unchanged science and separate lifecycle

New orchestration imports the unchanged archived `mavit_input.prepare`, `mavit_rehearsal.load_model` and `mavit_rehearsal.predict`. Preserve six public slots, first metadata candidate, eight central triplets, full-series1/99 normalization, uint8-before-scipy resize288,130mm-only-nonsquare crop, right-knee normalization and center/D positions. Preserve the reviewed v2 malformed-view policy, drained worker errors, whole-empty/resource failures and no alternative series. The exact model uses eval/no-grad, batch1 CUDA FP16 `model.infer`, sigmoid inside autocast then float32 CPU. No TTA, rank transform, precision retry, device fallback or substituted predictions.

Visible and full production share identical runtime source and qualification fingerprints. Every actual test ID is processed in official `test.csv` order, with two calls to the same frozen predictor for an explicit within-run repeat check. No batching/chunk alternatives. Inputs are only official test ID/schema, test series metadata and raw test DICOMs; no training metadata, labels, reports or gold CSVs are opened. Sample-submission rows do not determine coverage.

The builder parses only literal archived source/contract assignments. It verifies the archived notebook/contract/bundle/metadata hashes, reconstructs the complete original executable exactly, checks every frozen dependency byte against the reviewed archive, then applies the67 receipt qualifier and exact visible-CSV hash/revision binding. New inference files cannot invalidate the old complete-bundle hash: they are bound separately in the new qualification. No archived notebook code is executed locally.

SciPy1.16.3 is pinned from the successful67 receipt alongside the other observed package versions. Preserve the qualified fresh-process flags: cuDNN benchmark=False, deterministic=False, cuDNN TF32=True, matmul TF32=False, torch threads4 and reader I/O threads32. Scientific model, decoder, checkpoint, codec and helper bytes are immutable. Publisher [dataset/source](https://www.kaggle.com/datasets/hengck23/hengck23-rnsa-knee) declares MIT; lineage remains unknown.

## Qualification and output safety

First build/run visible-only production. Full-entry packaging requires its completed parent receipt and exact three-row CSV, then compares against the **same immutable67 CSV**. Both cross-run and within-run absolute tolerances are1e-6, without a relative term. No automatic relaxation, retry or score-based choice. A failure remains a failure pending investigation; no cross-run or hidden ranking invariance is assumed.

Reference probabilities and real IDs are used only for local comparison, never embedded as remote fixtures. Output/schema/order hashes and finite `[0,1]` checks bind each receipt. Full inference reuses the same path for any actual test count. Local synthetic receipt tests are labeled fixtures and are not CUDA or model evidence.

The setup-inclusive32,400s timer starts before notebook bootstrap. Setup and inference run through the frozen process-group timeout/cleanup helper, retaining the actual9h limit without estimate-based vetoes. There is one fresh CUDA child process; no device/batch fallback. Elapsed setup/read/inference/repeat time, CUDA allocated/reserved peaks, host RSS and disk usage are recorded.

Only a successful parent can release `/kaggle/working/submission.csv`. The target is created exclusively. Any BaseException during copy, post-copy deadline/validation, report writing or later parent completion removes only this run's owned output; preexisting files are preserved. Reviewed regressions cover partial-copy I/O errors, interruption and receipt-write failure. A child PASS with `parent_complete=False` cannot qualify a full build.

## Exact build commands

Run from integrated primary source after review. These commands build private/offline notebooks; they do not launch or submit.

```bash
.venv/bin/python scripts/build_mavit_submission.py \
  outputs/kaggle/mavit-visible-20261009v1 knee-mavit-visible-20261009v1 \
  --rehearsal-build outputs/kaggle/mavit-image-rehearsal-20261009v1 \
  --rehearsal-report outputs/kaggle/mavit-image-rehearsal-20261009v1-result/mavit-rehearsal/mavit-audit.json \
  --rehearsal-csv outputs/kaggle/mavit-image-rehearsal-20261009v1-result/mavit-rehearsal/visible-predictions.csv
```

Retrieve the completed visible result into `outputs/kaggle/mavit-visible-20261009v1-result/`. Only after the1e-6 check passes:

```bash
.venv/bin/python scripts/build_mavit_submission.py \
  outputs/kaggle/mavit-submission-20261009 knee-mavit-submission-20261009 \
  --rehearsal-build outputs/kaggle/mavit-image-rehearsal-20261009v1 \
  --rehearsal-report outputs/kaggle/mavit-image-rehearsal-20261009v1-result/mavit-rehearsal/mavit-audit.json \
  --rehearsal-csv outputs/kaggle/mavit-image-rehearsal-20261009v1-result/mavit-rehearsal/visible-predictions.csv \
  --submission \
  --visible-report outputs/kaggle/mavit-visible-20261009v1-result/mavit-submission/mavit-audit.json \
  --visible-csv outputs/kaggle/mavit-visible-20261009v1-result/mavit-submission/visible-predictions.csv
```

Both notebooks attach only competition data, the public MaViT dataset, and the existing ConvNeXt dataset for the three pinned codec wheels. They attach no report/label assets or prediction fixtures. The full package writes `mavit-submission/{mavit-audit.json,submission.csv}` and the released root `submission.csv`; visible mode writes `mavit-submission/{mavit-audit.json,visible-predictions.csv}` only. Failures produce aggregate type/counts without IDs or paths in logs.

## Local verification and remaining boundary

Focused tests cover immutable archive reconstruction/tamper, observed SciPy, isolated no-ML import, receipt mixing,1e-6 rejection, official ordering/coverage, source-function reuse, test-only file access, arbitrary test counts, fresh output ownership and failure rollback. Mocked orchestration tests verify call/ID order only, not CUDA numerics. Existing reader/model tests remain unchanged.

```bash
KNEE_REFERENCE_PARENT=/Users/ozgun/Dev/rsna-knee-abnormality-detection \
  /Users/ozgun/Dev/rsna-knee-abnormality-detection/.venv/bin/python \
  -m unittest discover -s tests -p test_mavit_submission.py -v
```

Ignored local checks and smoke builds: `outputs/analysis/mavit-submission-20261009/`. Rebuild after integration; do not upload pre-review smoke packages. No commits, launches, submissions, checkpoint/image downloads or STATUS edits were performed by this implementation task. Parent owns the actual visible/full execution, parity review and any subsequent submission.

## Real standalone qualification

Parent integrated five exact reviewed files as1581c7c;294/294 fixture tests passed with zero skips, plus synthetic smoke/compile/import and independent8focused tests. The independent review reproduced and cleared a late-output cleanup defect; original1e-6 tolerances and scientific bytes were unchanged.

Visible notebook `ozgung/knee-mavit-visible-20261009v1`v1 completedPASS51.158seconds. Independent17check review confirmed archived source/build/receipt/environment and official3ID/12target schema. Its CSV is byte-identical to the immutable67 result; repeat error0. ReportSHA `df3d1021e5a8cf549bb83777f59afb2334d10c234964dd4fcfb706ac00a33c26`.

Full-mode notebook `ozgung/knee-mavit-submission-20261009`v1, built fromc62d631, completedPASS51.311seconds on the visible data. Parent confirmed exact immutable67 predictions, repeat0, exact root/nested CSV and official coverage. Full receiptSHA `cac1a7ac53e3ec4734ce66a4d7687665af8ec346e28e8cd3b0bfbbe8b44c7921`; finalCSV SHA `3073b05627cee80ed9a3c745d354fd4cfc1cc862a23425caab838640fb48bec6`.

Both modes use qualification `8dafd8783324474399b43d641b0c12c1a4c6f47721be75c5301b4ca71d3fc8ee` and sourcebundle `c4bdbfa2e607e204b497f8fdba1c274dfdd95511d2b387ddd69003f178617349`. Full notebookSHA `4bed725af0e1409d5eab9fbaf36184859e9344ba4bd1216a47af3aaf73f7ff75`. Results `outputs/kaggle/mavit-{visible-20261009v1,submission-20261009}-result/`; reviews `outputs/analysis/mavit-submission-20261009/`.

Prospective accuracy comparison: exact raw-probability standalone versus immutable mirror0.949, conditional on pending fusion not already achieving>0.950. No local gold/sealed evaluation or blend-weight search; unknown public training lineage limits independent validation claims. October9at03:36UTC Kaggle allowed four additional entries. Full-mode visible success does not establish hidden execution or accuracy.

## Competition entry

Independent full-mode18check review passed, then official SDK confirmed remote current_version_number1 and parent compared the retrieved remote executable exactly with the reviewed build. Submitted immutable `ozgung/knee-mavit-submission-20261009`v1 as **entry56989116** at2026-10-09 03:40:40.807UTC. API03:40:53UTC PENDING, empty score/error; three daily submissions remain. Entry receipt `outputs/analysis/mavit-submission-20261009/entry-latest.json`; independent full review `independent-full-review.json`, remote version/executable review `remote-version-review.json` in the same directory. Fusion56986393 remains pending; best mirror0.949 unchanged.

The CLI version-suffixed pull returned403; supported owner/slug retrieval succeeded, and its metadata version1 and exact executable were separately verified through the SDK. No retry submission or source change occurred. Wait for the actual public result before deciding on the prospectively described mirror–MaViT blend.
