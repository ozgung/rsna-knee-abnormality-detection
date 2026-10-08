# Project status and session handoff

Updated **2026-10-08, 01:23 UTC** (Europe/Istanbul). Read this first, then [README](README.md) and [AGENTS](AGENTS.md).

## Current snapshot

- Target: public macro AUC **strictly above 0.950**. Not achieved. User authorizes long and concurrent local/Kaggle/Colab runs, subagents and promising submissions, with research and small pilots first. Existing credits only; no purchases or questions for this run.
- Branch `codex/first-submission`; local Git only, no remote. Original 01 tasks and second submission are complete. Current best scored entry is **0.757**; two stronger entries failed hidden reruns with generic notebook exceptions. Author-reported public scores are not our results.
- Competition data: 4,407 training studies, 24,371 series, 58 fully labeled studies, 12 targets. Four nongold report duplicates of gold excluded. Raw/private/generated artifacts stay ignored. No full raw-corpus download.
- Rules checked October 8: five entries/day; entry/merge deadline October 15 and final October 22, 23:59 UTC; offline inference <=9 hours; hidden test approximately 1,300 studies. Recheck before later calendar planning.
- Main records: [current campaign](docs/experiments/20261008-audited-readers.md), [provenance audit](docs/experiments/provenance-audit-20261008.md), [strategy](docs/experiments/score-strategy-20261008.md), [fresh pilot](docs/experiments/fresh_reader_pilot-20261008.md), [original baseline](docs/experiments/20261008-first-baseline.md).

## Current queue

| Task | State | Acceptance / next action |
| --- | --- | --- |
| All original 01 tasks; SUBMIT-02 | Done | Preserve the trained baseline, four-head ensemble, parity evidence and scored versions. Stop head-seed sweeps. |
| REF-02 | Repaired; qualifying | Hidden reruns failed generically. Shared usable-partial-series defect repaired at `67a016f`; new 64+3-image/seven-stress rehearsal running. Require old/new visible probability parity before resubmission. |
| TRAIN-02 cache | Running | Complete the 712-study CPU cache within the explicit 2-hour cap and unchanged 2 GiB limit. Require complete coverage and hash checks. |
| TRAIN-02 CUDA diagnostic | PASS | 32 cached training studies, 64 successful updates in 98.21s; finite training loss decreased. No validation, gold or saved trained weights. |
| TRAIN-02 fit | Prepared, gated | Cache-only notebook built at `outputs/kaggle/fresh-train`, source `fa8082d`. Complete cache -> measured CUDA gate -> at most 4 epochs / 90 minutes, 512 train / 200 weak validation. Select only by weak BCE; fixed train64 diagnostics and selected logits support paired comparison. |
| TRAIN-03 | Conditional | Scale only if actual learning and held-out improvement justify it. Benchmark transfer/hardware before Colab expansion; reserve >=8 Kaggle GPU hours for qualification. |

## Live jobs and resources

| Job | State and reproducibility |
| --- | --- |
| Entry **56924358** | Hidden-run **Notebook Threw Exception**, confirmed UI/API error field at01:05–01:07 UTC; no score. API enum COMPLETE alone is misleading. `ozgung/knee-public-reader-submission-20261008` v1, submitted October 7 at 23:50:09 UTC. Standalone public ConvNeXt three-fold reader; notebook COMPLETE and exact visible parity. |
| Entry **56925712** | Hidden-run **Notebook Threw Exception**, confirmed UI/API error field at01:05–01:07 UTC; no score. API enum COMPLETE alone is misleading. `ozgung/knee-reference-blend-submission-20261008` v1, submitted October 8 at 00:13:49 UTC. Fixed 50/50 percentile-rank blend; notebook COMPLETE, source `50f2803`, internal 28,000s / external 28,800s. Four daily entries remained after submission. |
| Full CPU cache | `ozgung/knee-fresh-cache-20261008` v1 launched **00:21 UTC**, source `ca6ff83`. Command: `kaggle kernels push -p outputs/kaggle/fresh-cache --timeout 7500`. Effective internal cap 7,200s; reuses CPU64 arrays. Build-contract source SHA `bd1a5733a89d5650e698555a05cb0ffab5a5b1eb37235d410bc329832b192e31`. Expected `fresh_reader_cache/cache-complete.json`, `cache-revised-gate.json`, private plan/index/arrays; early failure `fresh-reader-failure.json`. |
| ConvNeXt repaired rehearsal | RUNNING, `ozgung/knee-public-reader-rehearsal-v3-20261008` v1 launched01:23UTC, source `67a016f`. Command `kaggle kernels push -p outputs/kaggle/public-reader-rehearsal-v3 --timeout 3600`. Internal3500s cap;64label-blind train+3visible, seven synthetic production-path stresses, fixed old weights. Expected `reader-audit.json`, `reader-weight-hashes.json`, visible `submission.csv`. Entry SHA `450feb0a5090d427ee754bfba6695804d978b5687f0fc1fb13ad9878300db863`. |
| CUDA32 diagnostic | **COMPLETE/PASS**, `ozgung/knee-fresh-cuda-probe-20261008` v1, launched **00:33 UTC**, source `bf2cbe5`. Command: `kaggle kernels push -p outputs/kaggle/fresh-cuda-probe --timeout 900`. 600s cap, only CPU64 cache input, generic ImageNet download. Build source SHA `eb8556b5a206e80a671d58b8d87a647d713ecc1c9ac37a5a0ed8f9cedfbfe589`. Receipt: `outputs/kaggle/fresh-cuda-probe-result/fresh_reader_probe/probe-report.json`. |

- Poll notebook status with `kaggle kernels status OWNER/SLUG`; live progress with `kaggle kernels logs -f OWNER/SLUG`; retrieve selective outputs with `kaggle kernels output --file-pattern`. Scoring: `kaggle competitions submissions rsna-knee-abnormality-detection --format json`. Stop remote jobs through notebook running-job controls if necessary. No recurring automation exists.
- Kaggle CLI authentication works. Quota checked October 8 around 00:32 UTC: GPU 28.72 hours, TPU 20 hours; refresh October 10 at 00:00 UTC. Colab UI: 351.8 units, no sessions; historical October 18 expiry is unverified. No Colab/local training active. The bounded CUDA diagnostic is the only training job launched so far.
- Mac free disk was 31 GiB; local project artifacts approximately 550 MiB before two small QA arrays. Keep large caches remote. The full raw corpus is outside the local workflow.

## Completed campaign checks

- ConvNeXt rehearsal: 19 studies / 90 series, zero decode or empty-study failures, exact repeat and serialized-input parity. Runtime 161.42s, peak CUDA 1.284 GB. Source `fdb0f60`; Python 3.12 / torch 2.10 / timm 1.0.26. Standalone visible entry matched exactly. Receipts: `outputs/kaggle/public-reader-{rehearsal,submission}-result/`.
- Plain CC0 Raptor rehearsal: 19 studies, 1,156 unique slices, zero fallback. Pixel error 0, window error 1.1921e-6, repeat error 0. Runtime 98.71s, peak CUDA 8.247 GB. Checkpoint SHA `89606f05849838529e1b4658d28fb049623205d9853371548403bca460361ded`. Receipts: `outputs/kaggle/raptor-rehearsal-result/reference-rehearsal/`.
- Shared-environment blend rehearsal v2: PASS, 79.99s, repeat error 0, no failures. Visible rankings and final blend exactly match ranks from both original component CSVs. Raptor probability drift across software versions max 2.576e-4; cross-environment pixel identity was not established. Conservative three-study warmup projection 22,340.51s with margin, not measured hidden runtime. Peak CUDA 8.181 GB. Final notebook 104.80s; CSV SHA `2aa5bde70b03e720b21435b9137d9b7560ff4256c4b6ae7059e12538f1fb9d25`. Receipts under `outputs/kaggle/reference-blend-*-result/`.
- CPU64 v1 COMPLETE: 64 studies / 311 selected series, 73 missing slots, zero failures; 119,629,145 bytes. Decode 308.35s, preprocessing total 675.32s. Original 1-hour gate failed on time; storage passed. Measured revised projection including overhead is 5,512.52s and 1.664 GB with margins. Explicit 2-hour override preserves the original failed receipt, data plan and GPU budget; never claim the original gate passed.
- Fixed plan fingerprint: `1bd346c05c0aac20f54b82b790e356d3d6c1b95606accaa4a5d1da7e3e421942`. All 770 selected/gold studies had header identities, with no additional exclusions. Final split 512/200; CPU64 contains 45 train / 19 weak validation. Two label-blind cached arrays passed hashes, shape, padding and visual QA. Receipts: `outputs/kaggle/fresh-cache-pilot-result/`; gallery: `outputs/analysis/fresh-pilot-baselines/cache-gallery.png`.
- Exact 200-row baseline comparison, without refitting: single frozen DINO BCE **0.44760065**, weak AUC **0.85756882**; four-head BCE **0.44628505**, weak AUC **0.86033998**. All 12 AUCs defined. Baselines trained on 2,000 rows versus 512 planned here. Evidence: `outputs/analysis/fresh-pilot-baselines/actual-200-baselines.json`.
- CUDA32 diagnostic PASS: 98.21s, 65 attempted / 64 successful updates, one AMP overflow safely skipped. Same-training-row BCE 0.73460954 -> 0.46728107; both encoder and head changed. Peak allocated 3.406 GB / reserved 4.473 GB. Actual T4x2 worker, device0 used; Python3.12.13 / torch2.10.0+cu128 / timm1.0.26. Initial ImageNet encoder SHA `b1d4d0986c66bb1d02eddf7bf4bceea7fde12aced5def666c28d3b89951b569e`. Cold evaluation31.03s, warm1.75s; later steps approximately0.24s. Trainability only, no generalization estimate; no diagnostic weights saved or reused.

## Decisions and limitations

- Defer OAI-derived 0.949 weights while eligibility remains unresolved; no explicit host prohibition was found. Audited ConvNeXt uses competition images, ImageNet and public report labels; original Raptor is CC0 with ImageNet plus competition training per its author.
- The submitted blend was declared before results: global 50/50 per-target percentile ranks with average ties. No gold or leaderboard search for blend weights. Public knee checkpoints have no independent holdout here.
- Fresh pilot: original Steven v1 labels, original grouped assignments then subset 512/200, gold/report/patient exclusions, six slots and four fixed three-slice windows at 224px. Unknown 0.5 labels masked; confidence-weighted per-target macro BCE; EMA; training-only prior. No gold evaluation. Patient anonymization and unverified upstream prompt independence remain limitations.
- The old frozen encoder and three-series representation discarded useful sequence/slice information. Its public improvement from extra heads was only 0.001. Encoder adaptation and multi-sequence inputs are the current priority; their separate causal effects remain untested.
- No current approval blocker. No purchases, host messages or credential extraction. Earlier rejected private prediction-fixture upload never happened; safer weights-only upload succeeded. `knee/model.py` is still an unrelated untrained toy.

## Latest verification and next action

- **135/135 full ML tests pass**, with audited sources/receipts enabled; dependency-free smoke PASS. Log: `outputs/research/2026-10-08/tests-resilience-integrated.log`. Real CUDA diagnostic PASS; full fitting and resume remain unverified.
- Fixed two observed packaging issues: mixed Python-ABI codec selection and a nested SIGALRM cleanup race. The latter passed 1,000 nested expirations across subprocesses. Current data-plan-preserving budget override is `ca6ff83`.
- Evidence/packaging implementation `fa8082d` integrated and independently reviewed (32/32 focused checks). No recipe, split or selection change. Paired analysis `44591a3` integrated; reproduces old real-cohort scores exactly. Resilience repair `67a016f` integrated and independently reviewed (24 focused tests); no implementation active. Requalification is now first priority. Preserve worktrees' ignored evidence before archive. Earlier 47 outputs are under `outputs/agent-archives/20261008/`.
- Next: inspect the CUDA diagnostic and full-cache result/revised gate; launch bounded fitting only after complete cache checks. Full-cache stream is now live: setup7.32s, cache456 at2519.94s preprocessing, checked01:23 UTC; local stream session38256. Earlier500/empty logs resolved without restart. Fallback to a GPU worker was considered but not launched; receipt `outputs/research/2026-10-08/cache-worker-investigation.json`. Submission error receipt: `outputs/research/2026-10-08/submission-errors-0107.json`. Both saved images retain the exact expected digest; hidden environment/traceback unavailable, so environment mismatch is unproven. Paired-analysis usage: `docs/experiments/paired-weak-validation.md`. Target >0.950 remains unmet.
