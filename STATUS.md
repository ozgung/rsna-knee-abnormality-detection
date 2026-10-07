# Project status and session handoff

Last updated: **2026-10-08 (Europe/Istanbul)**. Read this first, then [README](README.md) and [AGENTS](AGENTS.md). Detailed current experiment: [first baseline](docs/experiments/20261008-first-baseline.md).

## Current objective

Complete all 01 tasks through the first scored Kaggle submission. User explicitly authorized autonomous replanning, submission, subagents, and Colab on October 8; optional second-submission improvement. No questions needed for routine choices. Branch `codex/first-submission`; local Git only, no remote. Main experiment source `359c831`.

## Verified snapshot

- Official CSVs: **4,407 studies, 58 gold-labeled studies, 24,371 series, 12 targets**. Four nongold studies share normalized reports with gold and are excluded. Metadata split: 4,345 train / 58 gold / 4 excluded; actual bounded run selects 2,000 train + 300 weak validation + 58 gold before DICOM patient exclusions. Audit: `outputs/validation/20261008/`.
- Competition access works; no terms accepted during this run. Live official pages verified via `kaggle competitions pages -c rsna-knee-abnormality-detection --content --format json`. Entry/merge deadline October 15, final October 22, both 23:59 UTC. Notebook-only submission, internet disabled, <=9 hours. Five submissions remained at start; 30 free Kaggle GPU hours.
- Implemented: deterministic bounded DICOM loading; geometry/intensity/padding masks; report/patient group exclusion; training-only feature normalization; frozen DINOv2-Small features; trainable attention/meanmax heads; weak-label training and explicit gold metrics; offline image-only inference. `knee/model.py` remains the unrelated untrained toy smoke architecture.
- Public labels: Steven Lee Hans v1, exact 0.5 masked. Gold is excluded from our training and epoch selection. Upstream weak-label prompt independence from gold is **unverified**, so gold performance is exploratory. No CSV patient identity; useful DICOM PatientID hashes participate in exclusions, but missing/anonymized identities limit guarantees.
- GPU probe completed on Tesla T4: DINO forward passes; all 30 sampled headers have geometry and PatientID (five distinct sampled IDs). Main extraction has decoded its first 1,100 studies without an empty study (~1,399 seconds); training/model quality still pending.
- Local environment: Python3.12.9, torch2.14.1, numpy2.5.3, pandas3.0.6, pydicom3.0.2, sklearn1.9.1; locked macOS arm64 environment. Prior CPU/MPS synthetic checks passed. Kaggle: Python3.13, torch2.11+cu128, transformers5.16.1, numpy2.1.3. Exact remote versions will be retained in experiment outputs.
- Raw metadata/weak labels are in ignored `data/`; no full raw corpus downloaded. Local cache remains below initial 2 GiB; ~31 GiB free before this run. Generated notebooks, private hashes, features and weights remain ignored.

## Work queue

| ID | State | Acceptance / remaining work |
| --- | --- | --- |
| PM-01 | done | STATUS/README/AGENTS handoff established. |
| SETUP-01 | done | Local Git and task branches; no remote. |
| COLAB-01 | done | Authenticated T4 CUDA smoke and tests; connection session terminated. |
| ENV-01 | done | Locked local environment; explicit CPU/MPS checks passed. |
| DATA-01 | done | Current official rules/schema/counts and metadata audit verified. |
| VAL-01 | done | Grouped split, leakage exclusions, missing-label policy, per-target supports and AUC aggregation; constant gold AUC0.5. |
| PIPE-01 | done | Geometry/intensity/mask tests and real-image CUDA extraction passing; complete resource aggregate captured as part of TRAIN-01. |
| TRAIN-01 | active | Retrieve completed bounded run, inspect patient exclusions, per-target metrics vs prevalence, checkpoint and resource measurements. |
| SUBMIT-01 | blocked | After trained artifact: internet-off gold parity + repeated visible-test rehearsal, pure test-only notebook, submit and wait for scored result. |

## Active jobs and artifacts

**Kaggle training:** private `ozgung/knee-dino-train-20261008`, version1, RUNNING. Source `359c831`; source bundle SHA256 `64a0f7501c8d4097999c62b5b81d764b5a7b4c4409561c42f99237fe999d0bb3`. Command: `kaggle kernels push -p outputs/kaggle/train --timeout 7200`. Extraction cap5400s; head cap900s/80epochs/patience12. Progress: `kaggle kernels logs ozgung/knee-dino-train-20261008 --follow`; local follower session38149. Status: `kaggle kernels status ozgung/knee-dino-train-20261008`.

Outputs `/kaggle/working/artifacts/{features.npz,extraction.json,patients-private.json,attention/head.pt,attention/training.json,attention/gold_predictions.npz,wheels/,wheel-hashes.json}`. Retrieve with `kaggle kernels output ozgung/knee-dino-train-20261008 -p outputs/experiments/dino-frozen-20261008`. No intermediate extraction checkpoint; interruption requires a rerun. Runtime exits at completion or timeout. To stop early use Kaggle notebook Stop; stopping the local log follower does not stop remote compute.

**Probe:** private `ozgung/knee-probe-20261008` v1 failed before code (missing kernelspec, fixed); v2 COMPLETED, ~630s. Aggregate/log retrieved at `outputs/kaggle/probe-result/`. Slow recursive file discovery in probe was replaced with direct attachment paths in actual experiment.

**Gallery:** private CPU notebook `ozgung/knee-gallery-20261008` v1 COMPLETED (~34s notebook runtime). Two fixed gold examples visually checked: 24 slices each, all planes, zero errors; joint anatomy retained. Images/audit/log in `outputs/kaggle/gallery-result/`; no identifiers or report text drawn. Public-source brief and original diagram saved in `outputs/research/2026-10-08/`.

**Colab:** no runtime allocated this session. Existing user-authorized 350-credit allocation expires October18; actual balance/provider cutoff unverified. Target useful work and artifact retrieval by October17; no purchases authorized. Optional lightweight head comparison can use Colab after the feature cache exists. Release idle runtimes.

## Working decisions

- Freeze independently pretrained DINOv2-Small and train a small finding-specific head first: permits a bounded experiment and affordable offline inference. Reject opaque competition-finetuned public checkpoints and gold-adjusted v2/v4 weak labels.
- Hash-order whole report groups for weak train/validation selection. Apply transitive patient/report exclusions after selected DICOM headers become available. Fit feature statistics/priors only on training. Select epochs only by weak-validation BCE; report gold as exploratory.
- Protect first submission; compare other heads/seeds only on the same cache and select by weak-validation BCE. Do not use leaderboard feedback to retrofit the first result.
- Separate pure test-only submission entry from gold/visible rehearsal. Rehearsal must compare raw-image predictions to cached-feature gold predictions and verify repeated inference. Codec wheel hashes checked before offline installation; model uses local files only.
- User explicitly authorized concurrent subagents this session; isolated managed worktrees used for loader, validation audit, and read-only research. Keep ordinary future work to one active implementation task unless requested.

## Latest verification

2026-10-08: `.venv/bin/python -m unittest discover -s tests -v`: **44 passed**. `python3.12 -m knee.smoke`: PASS (dependency-free synthetic). Entry scripts compile; `git diff --check` passes. Independent synthetic CPU training/checkpoint reload passed for attention and meanmax (max prediction discrepancy <=1e-6), including empty gold study and missing labels. Independent mocked hidden-ID replacement test passed exact actual-test coverage/order and rejected checkpoint target mismatch. These are not real-data quality or offline Kaggle success claims.

Frozen first-submission/rehearsal bundles use source0cdcac7, internet disabled, no weak-label dataset attachment; their only notebook input is the private training run. Do not rebuild them while separately considering ensemble code.

Latest changes: added offline artifact discovery/hash checks, pure inference and independent rehearsal entries, hidden-ID/mask regressions, and bounded optional head comparisons. Dependency-free full suite passes with26 optional ML skips; ML environment runs all44 tests. Prepared a private CPU image gallery for visual QA. Training run remains in progress. Concrete next action: finish TRAIN-01, launch offline rehearsal with the completed training notebook attached, then SUBMIT-01 through a scored result. Save aggregate experiment evidence and update this handoff before ending.
