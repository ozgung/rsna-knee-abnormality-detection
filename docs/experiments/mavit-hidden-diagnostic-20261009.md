# MaViT hidden-failure follow-up, October 9, 2026

Entry **56989116** failed with a generic unhandled error and no score. Its original source, receipts and submitted notebook are preserved. The actual hidden cause is unknown. A local tiny-DICOM reproduction found one concrete acceptance gap: a malformed US element raises pydicom `BytesLengthException`, aborting a study even when another selected view is healthy.

The separately reviewed `mavit-active-reader-v3` policy recognizes exactly that pydicom exception class as malformed input. The unusable selected view is masked; the healthy view is unchanged. There is no alternate-series retry, per-frame salvage, fabricated prediction or precision/device fallback. Fatal resource/deadline errors and unknown programming errors still propagate. Healthy scientific expressions and public source remain unchanged. The real-byte case is also runtime stress case 17.

The new authentic 67-study v3 rehearsal passed in **191.947 seconds**: 67 exact public tensor/probability comparisons, all 17 stresses, repeat error 0 and the original visible CSV unchanged. Receipt SHA256 `e51e51273867c13885d4d6c84fcf824a417857cc654bcf21e8e7d770a3e0c364`. This is evidence for the narrow repair, not proof that the hidden failure is resolved.

## Separate broad diagnostic

The new diagnostic runs the same qualified reader, checkpoint and `model.infer` on the **existing TRAIN04 2,000-study training membership**, using raw competition MRI. It repeats each prediction once, stops at the first error, and writes only an aggregate `mavit-diagnostic/diagnostic.json`. It never exports probabilities, arrays, images or weights, and does not compute predictive metrics. No training, fusion or submission is performed.

Membership is taken only from `coverage_cache/plan-private.json` in the already completed `ozgung/knee-train04-cache-full-20261008` attachment. The entire file is checked against SHA256 `5d05d478dcb63a90a322671f3553cd70c09319ac6685b9059e2dbeb6dec4a3cc` and its declared fingerprint `3f4f0230aff05c8ade9b78ff6e898571839349e3cca41a86c6ded491c7408a32`; only each record's `uid` and `split` are then accessed. The original JSON physically contains unused weak labels; they are not inspected, selected on, persisted, or passed to inference. The 200 development IDs are used only to verify disjoint membership. No development, sealed or gold images or labels are read, and no cache arrays are used.

Training IDs are sorted by SHA256 of `mavit-broad-train2000-v1:20261009:<uid>` with UID as the deterministic secondary key. The ordered cohort hash is `7a9545f5ae5f79c133f27a20d758f0590da9f04789ef3ca721e8bb44e1a87da9`; exact coverage is 2,000 studies and 11,145 series. Metadata candidate order is preserved. No study IDs are embedded in the generated notebook or emitted in diagnostic output.

The builder requires the authentic v3 report, archived source/build/CSV binding, observed SciPy **1.16.3**, exact checkpoint/codecs and unchanged backend flags. Old v2 evidence is rejected. Inference stays batch 1 CUDA FP16 with the existing 1e-6 repeat guard and all original schema/whole-empty guards. A fresh process and the reused process-group cleanup cover the actual nine-hour platform limit including setup. Runtime estimates are informational only.

Reports contain current phase and one-based cohort ordinal, completed/finite-call counts, current tensor shapes/dtypes and present-view count, input-policy counters, repeat error, timing/RSS and CUDA memory. Failure evidence contains an allowlisted error class and known-source function/line, never an exception message, file path, identifier or frame locals. A resource snapshot failure does not replace the original error. Progress is flushed every 32 completed studies and at completion. A FAIL remains a FAIL; there is no automatic retry or relaxation.

## Build after integration

From the primary repository, using its existing environment:

```bash
.venv/bin/python scripts/build_mavit_diagnostic.py \
  outputs/kaggle/mavit-train2000-diagnostic-20261009 \
  knee-mavit-train2000-diagnostic-20261009 \
  --rehearsal-build outputs/kaggle/mavit-image-rehearsal-v3-20261009 \
  --rehearsal-report outputs/kaggle/mavit-image-rehearsal-v3-20261009-result/mavit-rehearsal/mavit-audit.json \
  --rehearsal-csv outputs/kaggle/mavit-image-rehearsal-v3-20261009-result/mavit-rehearsal/visible-predictions.csv \
  --plan outputs/kaggle/train04-cache-full-v2-result/coverage_cache/plan-private.json
```

The builder attaches only the competition, the original MaViT dataset, the existing codec dataset, and the specified private cache notebook for membership. Outputs remain private and ignored. The implementation agent does not launch or submit this package; the parent reviews, integrates and launches it. No real broad diagnostic has been run at implementation time.

Focused local checks cover UID/split projection without accessing supervision fields, exact train order/disjointness, malformed membership, authentic v3 versus old/tampered receipts, isolated offline import, and mocked production call identity/order. Failure fixtures cover decode/inference errors, resource/deadline errors, repeat drift/nonfinite error, safe telemetry, output isolation, arrays unchanged, resource-snapshot failure and exhausted budget. Existing real-byte/source-AST/fatal-concurrency tests remain part of the full suite. Local mocks do not establish GPU or hidden-test success.

Local verification before handoff: **301/301 full fixture tests passed**, no skips; the six focused diagnostic tests, dependency-free smoke and `git diff --check` passed. The genuine-v3 local build/import smoke produced a roughly 200 KB notebook without running inference. Logs and the non-launchable local smoke build are in ignored `outputs/analysis/mavit-hidden-failure-20261009/`; parent rebuilds from its committed revision before launch.

## Launch

Parent integrated as `f85e2ec` and reran all301 tests successfully in15.560s, plus synthetic smoke and diff checks. Launched `ozgung/knee-mavit-train2000-diagnostic-20261009` version1 at approximately08:00UTC, October9; CLI confirms RUNNING. Contract SHA256 `7faba8d7a73216ef19056b17b9cf28f514aa6ef999446ed5e8ef3953b7ef6594`. Result is pending. Expected active processing45–60minutes is informational; no assistant-imposed runtime budget.

## Result and repaired submission — 09:28 UTC

The diagnostic completed **PASS**: 2,000 studies, 4,000 finite prediction calls and 325,435 decoded files in **2,702.783 seconds**. There were 9,726 selected views, 2,274 missing views, 101 unknown lateralities and zero masked/decoder/resource/whole-empty failures. Repeat error was exactly zero. Peak allocated CUDA memory was 854,832,640 bytes and host RSS 2,847,539,200 bytes. The parent validated the exact contract and report; report SHA256 is `627d76550924c528f84b93330d4d655111566e39ee2cf72f4ff09e40c2f5e774`.

New private offline notebooks `ozgung/knee-mavit-visible-v3-20261009` v1 and `ozgung/knee-mavit-submission-v3-20261009` v1 completed in 50.434 and 54.441 seconds. Both passed all production guards and matched the original reference CSV exactly: `3073b05627cee80ed9a3c745d354fd4cfc1cc862a23425caab838640fb48bec6`. The parent checked the published executable, actual version 1, GPU/private/offline flags and attached sources before submission. Build revision is `2e59f75`; final notebook SHA256 `a23ffbaf39c999508d44f2a6f07f630a73243e4ee9ab0d38041b5c9471ee0c08`.

Repaired **entry 57003785** was submitted at **2026-10-09 09:28:35.770 UTC** and is pending. Two daily submissions remain. Original entry 56989116 remains failed and preserved. The broad PASS supports retrying the qualified narrow repair; it neither establishes the original hidden cause nor guarantees the new hidden rerun.

Receipts are in `outputs/analysis/mavit-hidden-failure-20261009/`: `broad-result-review.json`, `final-production-review.json`, `remote-final-review.json`, `entry-v3-latest.json`. Full outputs are under `outputs/kaggle/mavit-{train2000-diagnostic,visible-v3,submission-v3}-20261009-result/`. No runtime changes were made during this requalification; the existing 301-test result remains applicable. Official rules were fetched at09:24UTC and match the07:44 snapshot.
