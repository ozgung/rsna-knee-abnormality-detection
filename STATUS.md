# Project status and session handoff

Last updated: **2026-10-08 (Europe/Istanbul)**. Read this first, then [README](README.md) and [AGENTS](AGENTS.md). Evidence: [first experiment](docs/experiments/20261008-first-baseline.md).

## Current snapshot

- Objective: finish every 01 task through a scored first submission, with optional second-entry improvement. User authorized autonomous replanning, subagents, submissions and existing Colab credits; no purchases. Branch `codex/first-submission`, local Git only, no remote.
- Real data: 4,407 training studies, 24,371 series, 58 fully labeled studies, 12 targets. Four nongold report duplicates of gold excluded. Exact schema comes from official files. Data and generated/private artifacts remain ignored.
- Trained frozen DINOv2-Small baseline: 2,000 weak train / 300 weak validation / 58 gold; 7,074 series and 56,592 slices decoded with zero errors/missing planes. Extraction 2,807.88s. Gold macro AUC **0.720149**, prevalence **0.5**; all 12 defined. Weak BCE **0.463229**, prior **0.593802**. Source `359c831`.
- Training-only normalization and report/patient exclusions verified. Observed PatientID hashes are unique per study and disjoint across folds, but anonymized identity semantics limit repeat-patient guarantees. Public v1 weak-label prompt independence from gold is unverified; gold AUC is exploratory. Exact 0.5 labels are masked.
- First offline rehearsal passed: 58 gold rereads match cached predictions within **1.788e-7**; repeated visible-test predictions identical; zero fallback. Measured throughput extrapolates to 31–44 minutes for ~1,300 studies, not a hidden-test runtime measurement.
- Colab T4 comparison completed in 19.63s. Selected equal average of attention seeds42/43/44 + meanmax seed42 by weak-validation BCE **0.459576**. No gold-based blend selection. Results retrieved; session terminated and `colab sessions` confirmed none active.
- Competition pages checked October8: entry/merge October15, final October22, 23:59UTC; offline notebook <=9h. Two of five daily submissions used; first COMPLETE with public score0.756, second pending. Initial Kaggle allocation30 free GPU hours. Local cache remains under initial2GiB; no full-corpus download.
- `knee/model.py` remains an unrelated untrained toy shape check. The trained implementation is in `knee.features`, `experiment`, `runtime` and `ensemble`.

## Prioritized queue

| ID | State | Acceptance / next action |
| --- | --- | --- |
| PM-01 | done | Compact handoff, durable rules, reproducible commands. |
| SETUP-01 | done | Local Git and task branch; no remote. |
| COLAB-01 | done | CUDA checks and bounded head comparison completed; runtime stopped. |
| ENV-01 | done | Locked Mac environment; CPU/MPS smoke passed; separate actual CUDA environments recorded. |
| DATA-01 | done | Official rules/schema/counts, metadata and supervision audit verified. |
| VAL-01 | done | Grouped exclusion, train-only statistics, label masks, per-target support/AUC and prevalence baseline. |
| PIPE-01 | done | DICOM geometry/intensity/mask tests plus complete bounded real-data extraction and resource measurements. |
| TRAIN-01 | done | Baseline trained, preserved and evaluated; artifacts retrieved. |
| SUBMIT-01 | done | First entry56922773 COMPLETE; public score **0.756**, verified22:42UTC. |
| SUBMIT-02 | active | Entry56923149 submitted after all offline checks passed; wait for score. |
| REF-02 | ready after this run | Audit the published Pilkwang v15 weight manifest/provenance, then reproduce its exact offline inference as a separate reference. Retain attribution and preprocessing; compare against verified public0.891. Recheck resource budget and restore CLI login first. |
| TRAIN-02 | proposed | Compare a partially fine-tuned DINO encoder under the same split, then sequence/crop coverage separately. Stop prioritizing head-seed sweeps; predeclare metric and bounded compute before launch. |

## Active remote jobs

- **First competition entry56922773**, submitted2026-10-07 22:13:58UTC; **COMPLETE, public score0.756**, verified22:42UTC via UI and CLI. Private `ozgung/knee-dino-submission-20261008` v1, frozen source`efad291`. Poll `kaggle competitions submissions rsna-knee-abnormality-detection --format json`. Preserve this scored version.
- **Second competition entry56923149**, submitted2026-10-07 22:36:29UTC; PENDING. Private `ozgung/knee-ensemble-submission-20261008` v1, source`5f66959`, bundleSHA`97f06aa2db0373c46f6cb2593c03d001cc4fca4eec7319cbab5288351ad7376e`. All-four equal ensemble selected only by weak BCE. Notebook and rehearsal COMPLETE; no visible notebook jobs running. Preserve its inputs while scoring.
- Ensemble gold parity error **1.043e-7**, repeat test identical, no errors/fallback. Gold AUC **0.720727**; 58-study inference100.94s, peakCUDA252,470,784bytes across rehearsal. Visible3 inference26.82s including load. Reports in `outputs/kaggle/{ensemble-rehearsal-result,ensemble-submission-result}`. Visible CSV SHA`be43e19b805c97174f9d0bc051b15f0f170f51c670f7739c0446004ab2398ac8`, header/actual ID order/probabilities verified.
- Ensemble inputs: competition + vanilla DINO + original training notebook + private `ozgung/knee-own-heads-20261008` v1. Dataset contains exactly four own trained heads; no study IDs, prediction fixtures, labels, reports or images. Rehearsal computes its reference from the existing Kaggle feature cache. Internet disabled.
- No Colab session or local training job active. Existing user-authorized350-credit allocation expiresOctober18; remaining balance/provider cutoff unverified. No purchases.

## Artifacts and reproducibility

- Baseline `outputs/experiments/dino-frozen-20261008/artifacts/`: features80,772,745bytes, extraction/audit, attention weights/training/gold predictions, hash-verified Linux codec wheels. Private training notebook `ozgung/knee-dino-train-20261008` v1 COMPLETE. Extraction has no intermediate checkpoint.
- First rehearsal and visible submission: `outputs/kaggle/{rehearsal-result,submission-result}/`; first visible CSV SHA`ae6e03fbe3b3c6c95ee7da5084413c4915268d39e6739a0dd0366a16775d045b` checked against inference report.
- Colab comparison: `outputs/experiments/head-comparison-20261008/comparisons/`; manifest, checkpoint hashes and input ZIP integrity verified. Input/archive/logs in `outputs/colab/`. CLI0.6.0 required finalizing chunk uploads and root-path reassembly; two attempts stopped at missing-input guard before training. Successful runner/log preserved.
- Research: `outputs/research/2026-10-08/research-brief.txt`, primary source snapshots, original pipeline diagram. Two real preprocessing galleries under `outputs/kaggle/gallery-result/`, visually checked (three planes,24slices, anatomy retained; no identifiers/report text drawn).

## Decisions, blockers and latest checks

- No current approval blocker. Kaggle CLI authentication expired around22:50UTC after the successful first score check; the in-app browser remains authenticated and confirms the second hidden-test notebook is running. Renew normal CLI login before the next CLI training workflow; do not extract browser cookies. Kaggle rejected embedded weights because notebook source exceeded1MB. Automatic approval review rejected a separate upload containing a private prediction fixture; removed the fixture, verified a weights-only payload and uploaded that safer dataset successfully. No rejected payload was uploaded.
- Preserve baseline and fixed weak-holdout selection. Do not choose candidates or blending from gold/public leaderboard. Public methods support future partial DINO tuning, adjacent-slice channels and finding-specific pooling, but their gold-selection/provenance limitations matter; avoid opaque weights.
- Offline inference reads images/series/test IDs only, uses actual hidden test IDs rather than visible sample rows, checks finite probabilities and exact target order; prior fallback is explicit and bounded. Raw image symlinks are removed even on failure.
- Latest checks: **60/60 ML-environment tests pass**, dependency-free smoke PASS; generated notebooks compile, private/offline settings and weight payload verified; `git diff --check` PASS. Dependency-free suite:26 passed,34 optional ML skips; smoke passed. No full pipeline claim rests on synthetic tests alone.
- Next action: wait for second-entry scoring and compare with first public score0.756. Every01 task is complete. All47 ignored agent-output files preserved with hash manifest under `outputs/agent-archives/20261008/`; all three managed worktrees confirmed archived. No further training planned in this run. Descriptive paired gold bootstrap puts ensemble-minus-first95% interval at[-0.00978,0.01182]; no demonstrated gold improvement.

Metadata follow-up: non-fat-suppressed sagittal coverage4,266/4,407 studies and coronal3,406/4,407; axial857/4,407. Current one-series-per-plane selection usually discards this complementary input. Fluid_Sensitive and Fat_Suppression flags are identical in all24,371 rows and do not distinguish T1 from PD/T2; recover DICOM contrast evidence before naming sequence types. Aggregate: `outputs/analysis/20261008/sequence-coverage.json`. This motivates sequence coverage ahead of a larger encoder; no third run launched.

User challenged the0.756 score against Pilkwang0.891. Live v15 notebook/logs confirm20 models,336px/130mm, six slots,12 slices and10 windows/member; our frozen224px/three-series representation is much smaller. Barun ResNet34 v3 was now inspected:46 labeled training studies,12 validation, best local AUC0.6086, no demonstrated public score. Revised TRAIN-02 toward reference reproduction and encoder adaptation; exact causal contributions remain untested. Details and primary links are in the experiment record.
