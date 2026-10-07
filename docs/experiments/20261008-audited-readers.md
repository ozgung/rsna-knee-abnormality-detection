# Audited readers and fresh training pilots — 2026-10-08

User objective: target public macro AUC strictly above0.950; long and concurrent
training/submissions authorized, with research, bounded pilots and resource gates.
No promise that this target is attainable. Existing Colab credits only, no purchases.
See [provenance audit](provenance-audit-20261008.md) and
[experiment strategy](score-strategy-20261008.md).

## Decisions before new results

- Defer submissions using the OAI-derived0.949 checkpoint while its model-versus-data
  eligibility remains unresolved. No host ban was found; this is a conservative
  experiment choice, not a claim of illegality.
- Establish the standalone public ConvNeXt reader as a strong reproducible reference.
  Then rehearse the original CC0 Raptor. One prospective blend, if both pass, will use
  equal50/50 per-target percentile ranks with average ties. No gold/leaderboard search
  for weights or target-specific constants.
- New training starts from generic ImageNet ConvNeXt, preserving the original grouped
  weak split, rather than claiming an independent holdout for public knee checkpoints.
  CPU64-study cache gate precedes a512-train/~200-weak-validation GPU pilot capped at
  90minutes. Fixed224px, six slots, adjacent windows, exact0.5 unknown masked,
  confidence-weighted BCE and EMA. Runtime/cache gates precede expansion.
- Resource observation23:36UTC: KaggleGPU28.91h/TPU20h; refreshOctober10 00:00UTC.
  ColabUI351.8units, zero sessions. Reserve at least8KaggleGPUh for offline qualification.

## ConvNeXt reference

Public source/weights: goodpjw2008/rsna-knee-2-5d-convnext-reader, Apache-2.0.
Our builder pins the inspected model/preprocessing source hashes and adds strict
selected-series decode/empty guards, safe checkpoint loading and actual-test schema
coverage. Public knee checkpoints are diagnostics, not independent validation models.

- Rehearsal v1 failed before model execution because generated notebook lacked a
  kernelspec. Fixed and added a packaging regression; source`fdb0f60`.
- Private notebook`ozgung/knee-public-reader-rehearsal-20261008` v2 COMPLETE.
  Offline16 label-blind train studies+3 visible test studies; sample fingerprint
  `d1efdc9c1144dddaad7361e94a9f24852d27a3c15efcbdec72810dd73c54c3a6`.
  **19studies,90selectedseries,24legitimately missing slots,0decodefailures,0emptystudies**.
  Repeated prediction max error0; serialization roundtrip max error0. The latter is
  input persistence parity, not an independent raw-versus-canonical-cache pipeline test.
- Runtime161.42s, peakCUDA1,283,609,600bytes, processRSS2,542,764KiB.
  Per-study mean5.796s projects to7,535s/1,300studies before setup/enumeration;
  projection is not a hidden-test measurement. Ample margin against9h, with strict
 28000s runtime guard and external28800s notebook cap.
- Actual environment: Python3.12, torch2.10.0+cu128, timm1.0.26, NumPy2.0.2,
  pandas2.3.3, pydicom3.0.2, pylibjpeg2.1.0, OpenCV4.13.0, twoTeslaT4.
- Submission notebook`ozgung/knee-public-reader-submission-20261008` v1 COMPLETE,
 63.94s, exact visible prediction parity, checkpoint hashes identical to rehearsal.
  Entry**56924358** submitted**2026-10-07 23:50:09UTC**, PENDING at recording.
  Preserve this version and its attachments while scoring. Poll with
  `kaggle competitions submissions rsna-knee-abnormality-detection --format json`.
- Ignored receipts: `outputs/kaggle/public-reader-{rehearsal,submission}-result/`.
  Build sources/contracts: sibling package directories without`-result`.

## Raptor reference

CC0 plain checkpoint292,829,402bytes, independently verified SHA
`89606f05849838529e1b4658d28fb049623205d9853371548403bca460361ded`.
Safe local load confirmed486state tensors, expected CoAtNet architecture/res384 and
12-target order. Gold metadata is not used for selection. Source geometry preserved:
64slices/five slots,140mm crop→336px,42adjacent windows→384px, noTTA.

`ozgung/knee-raptor-rehearsal-20261008` v1 launched approximately23:48–23:49UTC; at most16train+3visible
test, one-hour external cap. Source`659bcbd`, contractSHA
`d8fedf9a1d7dc9849ab10fd67bb1e981ce96d628eb4e4bceaab20e448a84801b`.
Status was RUNNING at23:49UTC; output expected`reference-rehearsal/rehearsal.json`.
Poll via`kaggle kernels status`/`logs`; stop via Kaggle notebook running-job controls
if needed. No training or blending has run yet at this checkpoint.

## Checks

Public-reader packaging3/3 passed; parent fullML suite63/63 passed before integrating
the reference module; dependency-free smoke passed. Reference-agent tests69/69 passed
with both audited source snapshots enabled. Full integrated suite will be rerun after
the next implementation checkpoint. All generated notebooks compile. These checks
do not establish model quality or a score above0.950.
