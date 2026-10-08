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
Completed PASS:19studies,1,156unique slices,0fallback; decodedpixelerror0,
windowerror1.1920928955078125e-6,repeaterror0. Runtime98.71s; p95projection+25%
7,053.68s/1,300studies. PeakCUDA8,247,278,080bytes; RSS2,106,449,920bytes.
Actual Python3.13.15,torch2.11.0+cu128,timm1.0.29,T4. Receipts under
`outputs/kaggle/raptor-rehearsal-result/reference-rehearsal/`.

## Fixed blend and own-training pilot

- Fixed50/50 per-target average-tie percentile-rank blend builder integrated`57bb203`.
  Both components are sequential; exact sources/checkpoints and all actual test IDs.
  A commonPython3.12 environment must pass visible reread checks and local comparison
  to prior component outputs before an entry. Projected combined runtime16,472.59s
  with margin from separate environments, not a hidden-test measurement.
- `ozgung/knee-reference-blend-rehearsal-20261008`v1 launched00:03UTC October8,
  **FAILED before inference**: dataset contains bothcp311 andcp312 codec wheels,
  while guard counted all five instead of selecting three compatible wheels.
  Failure at8.34s, no predictions. Logs retained in
  `outputs/kaggle/reference-blend-rehearsal-result/`. Fix/requalification in progress.
- `ozgung/knee-fresh-cache-pilot-20261008`v1 launched~23:57UTC October7,source`1f1d788`.
  CPU64study, internal1800s/external2100s. RUNNING00:06UTC. Require measured
  `gate.passed` before expanding; see [pilot protocol](fresh_reader_pilot-20261008.md).
  Device metadata follow-up integrated`367ac30`for future builds; originalCPU source
  remains fixed. No CUDA training yet.

## Checks

Parent integrated ML suite95tests:92exercised PASS,3optional source/receipt skips.
Previous dependency-free smoke PASS. Independent review found no training source
blocker after resume/digest/macro-loss fixes. Blend component contract/rank tests
passed; actual mixed-ABI asset directory exposed a missing packaging case, now
being covered. These checks do not establish model quality or score>0.950.

## 00:13 UTC checkpoint

Codecfix`50f2803`selects exactlythethreeauditedcp312/purePythonwheels whileignoring
cp311copies. FullparentMLsuite96/96PASS withauditedsources/receipts enabled.
Blendv2rehearsalPASS79.99s,zeroerrors/fallback,rawrepeat0,bothcomponentrankorders
andfinalequalrankblendmatchoriginalstandalonevisibleCSVsexactly. ConvNeXtprobability
maxdifference2.812e-8; Raptor2.576e-4 acrosssoftwareversions,consistentwithsmallFP16
variation,notclaimed1e-6cross-environmentparity. Estimated1,300runtime22,340.51s
with25%marginfromonlythreevisiblewarmupstudies; peakCUDA8.181GB.

Finalblendnotebookv1launched00:10UTC,source`50f2803`,preservesqualifiedcodeenvironment,
threewheelhashes andfourcheckpointidentities. VisibleauditprintedPASS104.80s;
competitionentrynotyetsent. Evidenceunder`outputs/kaggle/reference-blend-*-result/`.

CPU64isdecoding/writingarrays. Nextstagebuildernowusesexplicitsource/cachepaths
andboundedsetup(`b736da6`),butparentfullsuitecaughtnestedSIGALRMcleanup race(exit142).
Fixrequiredbeforeusingnewbuilder; currentCPU64doesnotusethisnewhelper. NoGPUtrainingyet.

At 00:14 UTC, the final blend notebook was COMPLETE. Saved submission SHA
`2aa5bde70b03e720b21435b9137d9b7560ff4256c4b6ae7059e12538f1fb9d25` matched
the rehearsal exactly. **Entry 56925712** submitted at **00:13:49 UTC**, PENDING;
four daily entries remain after the UTC reset. Standalone entry 56924358 also pending.

Nested timeout cleanup fixed in `9196c69`: expired enclosing timers are not rearmed.
Parent full suite **102/102 PASS**, including 1,000 nested expirations across subprocesses.
CPU64 has reached 40 cached studies with no reported failure. Next CPU duration will
be determined from the completed measured gate; training remains unlaunched.

## Completed CPU64 pilot and revised CPU resource budget

CPU64 v1 completed on October8 around00:16UTC. Data checks **PASS**:64studies,
311selected series,73missing slots,0failures;119,629,145compressed bytes.
Decode/write time308.35s; total preprocessing675.32s, including366.97s of metadata
and header work. The initial broad attachment search added startup latency outside
that measurement; later builds resolve explicit mounts and bound setup.

The original one-hour gate **failed on time**: projected712-study decoding with
1.5×margin5,145.55s, storage with1.25×margin1,663,592,798bytes. The storage gate passed.
Under the user's explicit authorization for long runs and dynamic planning, the
next full-cache CPU cap is **7,200s**. Revised projection adds the measured nondecode
cost: **5,512.52s**. The512/200split, image recipe,2GiBstorage limit and90minGPU budget
remain fixed. The original failed gate is preserved; an explicit revised gate is
required before expansion. Hash-verified pilot arrays will be reused.

Final private plan fingerprint:
`1bd346c05c0aac20f54b82b790e356d3d6c1b95606accaa4a5d1da7e3e421942`.
All770selected/gold studies had header identities; no additional weak exclusions.
Anonymization still limits patient-level guarantees. Pilot cache contains45training
and19weak-validationstudies. Receipts under`outputs/kaggle/fresh-cache-pilot-result/`.
Two label-blind arrays were downloaded for local QA: hashes, uint8shape6×4×3×224×224,
finite positions and zero padding passed. Visual inspection retained expected planes
and knee anatomy. Gallery`outputs/analysis/fresh-pilot-baselines/cache-gallery.png`.

The existing frozen models were evaluated without refitting on the exact200-row
fresh-pilot validation cohort: single weakBCE**0.44760065**, macro weakAUC**0.85756882**;
fixedfour-head weakBCE**0.44628505**, macro weakAUC**0.86033998**. All12AUCs defined.
These baselines used2,000trainingstudies versus512in the new pilot; this is a matched
validation cohort, not a matched training budget. No gold labels or metrics used.
Aggregate per-target evidence and hash checks are in
`outputs/analysis/fresh-pilot-baselines/actual-200-baselines.json`.

## CUDA trainability diagnostic launched

`ozgung/knee-fresh-cuda-probe-20261008` v1 launched at **00:33 UTC**, source `bf2cbe5`.
Private; only the CPU64 notebook output attached. Fresh generic ImageNet initialization,
32 hash-selected cached training studies, at most64 successful updates /96 attempts,
600-second whole-stage cap and900-second external cap. No validation/gold use, no
checkpoint persistence or reuse. Initial/final unaugmented training loss is a numerical
diagnostic, not evidence of generalization. Report includes actual encoder state SHA,
AMP skips, finite gradients, timing and peak memory. Independent source review found
no launch blocker; parent integrated suite **108/108 PASS**.

The saved receipt confirms **PASS** in98.21seconds:65attempts,64successful updates,
one AMP overflow safely skipped. Same32training-row macroBCE fell from0.73460954
to0.46728107, and sampled encoder/head parameters both changed. Peak allocated CUDA
memory3,405,646,848bytes; reserved4,473,225,216bytes. Actual worker had two TeslaT4s;
device0 was used. Python3.12.13, torch2.10.0+cu128, timm1.0.26. The initial generic
ImageNet encoder SHA is`b1d4d0986c66bb1d02eddf7bf4bceea7fde12aced5def666c28d3b89951b569e`.
Initial cold evaluation took31.03s versus1.75s warm; later optimizer steps were
approximately0.24s, so these short warm-cache timings do not replace the full-run
throughput gate. Receipt and full log:
`outputs/kaggle/fresh-cuda-probe-result/fresh_reader_probe/probe-report.json`.
This establishes trainability only, with no saved diagnostic weights or held-out result.

Full CPU cache `ozgung/knee-fresh-cache-20261008` v1 remains running since00:21UTC,
source`ca6ff83`, internal7200s/external7500s. Two log-stream requests returned transient
server500 errors; both CLI status and browser active-events panel still report RUNNING.
Both reference entries remain pending. At00:32UTC, Kaggle quota28.72GPUhours remained.

## Prospective promotion rules before the first held-out fit

Declared at00:46UTC before observing the fresh model's validation results. The exact
512-training-row prevalence baseline is weak BCE **0.59717834** on the fixed200 rows.
The stronger frozen baseline used2,000 training rows: BCE **0.44628505**, weak AUC
**0.86033998**. The following thresholds govern spending, not statistical significance
or expected public score. Epoch selection remains exclusively weak BCE.

- Near-reference performance from512 rows (selected-best BCE≤0.456285 and all-target
  weak AUC≥0.850340) supports one larger-data pilot with the recipe otherwise fixed.
- A weaker but still improving result (BCE≤0.537460, AUC≥0.80, epoch4 is BCE-best,
  epoch2-to4 BCE gain≥0.01, and improving fixed-training diagnostic) supports one
  newly declared12-epoch experiment on the same512/200 split, capped at90 minutes
  after throughput measurement. The completed four-epoch cosine schedule must not
  be silently extended or reset through resume.
- Numerical failure, failure to beat prevalence, or falling training loss with
  flat/worsening validation calls for diagnosis before scale. Four epochs alone
  cannot establish that the architecture is unsuitable.

Compare selected predictions with both frozen baselines using aligned study groups,
paired uncertainty and per-target support. This is a reused development cohort with
report-derived labels, not an independent clinical test. Before larger-data training,
reserve approximately300 eligible previously unused groups outside the entire original
2,000/300 cohort as a sealed weak proxy. No candidate evaluation on that proxy until
the modeling choices are fixed. Larger remote caches need a measured allowance beyond
2GiB; the local2GiB limit and at least8Kaggle GPU hours reserved for qualification stay.
