# Project status and session handoff

Last updated: **2026-10-08 (Europe/Istanbul; 00:14 UTC)**. Read this first, then [README](README.md) and [AGENTS](AGENTS.md).

## Current snapshot

- Objective: audit stronger public references, then target public macro AUC **strictly above 0.950**. Not achieved. User authorizes long/concurrent local, Kaggle and Colab training, subagents and promising submissions; use small pilots and resource gates first. Existing credits only, no purchases or questions for this run.
- Branch `codex/first-submission`, local Git only, no remote. All original 01 tasks and second submission completed. Own frozen-DINO scores: **0.756**, then **0.757**. Stronger ConvNeXt entry pending; do not confuse public authors' scores with ours.
- Real data: 4,407 training studies, 24,371 series, 58 fully labeled studies, 12 targets. Four nongold report duplicates of gold excluded. Exact schema from official files. Private/generated artifacts ignored. Full raw corpus is not downloaded.
- Frozen DINO: 2,000 weak train / 300 weak validation / 58 gold, zero decode failures across56,592 slices. Single-head weak BCE0.463229 vs prior0.593802; gold AUC0.720149. Equal four-head blend weak BCE0.459576, gold0.720727; descriptive paired gold gain interval[-0.00978,0.01182]. Stop head-seed sweeps.
- Split excludes report/patient relatives; anonymized PatientID semantics limit repeat-patient guarantees. Exact0.5 weak labels masked. Public v1 prompt independence from gold unverified; gold is exploratory. Public knee checkpoints do not have an independent holdout here.
- Rules/pages checked October8: entry/mergeOctober15, finalOctober22,23:59UTC; five submissions/day; offline inference<=9h, hidden approximately1,300studies. Recheck before calendar planning. No terms accepted by agent.
- Current campaign: [audited readers](docs/experiments/20261008-audited-readers.md), [provenance](docs/experiments/provenance-audit-20261008.md), [strategy](docs/experiments/score-strategy-20261008.md), [fresh pilot](docs/experiments/fresh_reader_pilot-20261008.md).

## Prioritized queue

| ID | State | Acceptance / next action |
| --- | --- | --- |
| PM/SETUP/ENV/DATA/VAL/PIPE/TRAIN/SUBMIT-01 | done | First trained baseline, real-data audits, offline parity and scored entry preserved. |
| SUBMIT-02 | done | Four-head ensemble public0.757; no further head-seed sweeps. |
| REF-02 | active | Separate ConvNeXt and plain Raptor rehearsals passed. Await standalone ConvNeXt score. Common-environment fixed50/50 rank-blend rehearsalv2 passed; ranks exactly match earlier component results. Fixed blend entry56925712 is PENDING; both scored candidates now awaiting evaluation. |
| TRAIN-02 | active pilot | Fresh ImageNet ConvNeXt CPU64 cache job running. Require measured gate.passed before expanding to512train/200weak-validation cache. Then CUDA throughput gate and<=90min/4epochs; select only by weak BCE. |
| TRAIN-03 | conditional | Scale only if pilot beats training-only prior and throughput justifies it. Measure compact-cache transfer before Colab expansion; preserve >=8KaggleGPUh for qualification. |

## Live jobs and resources

- **Entry56924358 PENDING** as of00:00UTC October8. `ozgung/knee-public-reader-submission-20261008`v1; submittedOctober7 23:50:09UTC. Exact Apache-2.0 public ConvNeXt threefold reader. Notebook COMPLETE and visible parity exact. Preserve version/attachments. Poll `kaggle competitions submissions rsna-knee-abnormality-detection --format json`.
- **CPU64 pilot RUNNING**, launchedOctober7 ~23:57UTC: `ozgung/knee-fresh-cache-pilot-20261008`v1, source`1f1d788`. Command `kaggle kernels push -p outputs/kaggle/fresh-cache-pilot --timeout 2100`. Internal1800s; expected `fresh_reader_cache/cache-pilot.json` and private plan/index/arrays; early failure `fresh-reader-failure.json`. No GPU training yet. Require `gate.passed == true`, then exact plan digest for next stage. Full commands in pilot protocol.
- **Blend entry56925712 PENDING; final notebookv1 COMPLETE:** `ozgung/knee-reference-blend-submission-20261008`, launched00:10UTC, source`50f2803`, external28800s/internal28000s. RuntimePASS104.80s onvisible3; saved CSV exactly matches rehearsal, SHA`2aa5bde70b03e720b21435b9137d9b7560ff4256c4b6ae7059e12538f1fb9d25`. Entry submitted00:13:49UTC; fourdailyentriesremaining. QualificationSHA`f77fc5da880386f5a1f34f7f8386f5e043aeace3c4cf3149e54dc7187af8b5c8`. Rehearsalv1 failedbeforeinference due bothPythonABIs counted; fixedexact3cp312wheel selection+regression. v2rehearsal79.99s,zeroerrors/fallback,repeat0,projected22,340.51swithmargin,peakCUDA8.181GB. Cross-environment probability maxerrorConvNeXt2.812e-8/Raptor2.576e-4; allvisible ranks and finalblend exactly unchanged. Rawcross-environmentpixelidentity was not established. Receipts `outputs/kaggle/reference-blend-rehearsal-v2-result/reference-blend/`.

- Poll jobs with `kaggle kernels status OWNER/SLUG`; read `kaggle kernels logs`; completed output via `kaggle kernels output`. Stop through notebook running-job controls if needed. Bounded jobs stop internally/external timeout; no recurring automation exists.
- CLI authentication works. AtOctober7 23:36UTC: KaggleGPU28.91h,TPU20h; refreshOctober10 00:00UTC. ColabUI351.8compute units and zero sessions; expiryOctober18 historical/unverified. No local/Colab training active. Last freeMacdisk31GiB; initial localcache2GiB. Four lifetime entries; fourdailyremaining at00:13UTCafterreset.

## Qualified reference evidence

- ConvNeXt source`fdb0f60`, notebook rehearsalv2 COMPLETE (v1 kernelspec failure fixed):19studies,90selectedseries,24 legitimate missing slots,0decode/emptyfailures; repeat and serialization parity0. Serialization is persistence parity, not independent raw/cache equivalence. Runtime161.42s,peakCUDA1.284GB; mean inference projects7,535s/1,300 before setup. Python3.12,torch2.10,timm1.0.26. Submission visible3 COMPLETE63.94s, exact sameweights/predictions. `outputs/kaggle/public-reader-{rehearsal,submission}-result/`.
- Plain CC0 Raptor rehearsalv1 COMPLETE:19studies,1,156unique slices,0fallback; decodedpixelerror0,windowerror1.1921e-6,repeaterror0. Runtime98.71s; p95projection+25%=7,053.68s/1,300;peakCUDA8.247GB. Python3.13,torch2.11,timm1.0.29. Contract`d8fedf9a1d7dc9849ab10fd67bb1e981ce96d628eb4e4bceaab20e448a84801b`; checkpointSHA`89606f05849838529e1b4658d28fb049623205d9853371548403bca460361ded`. `outputs/kaggle/raptor-rehearsal-result/reference-rehearsal/rehearsal.json`.
- Original entries:56922773 COMPLETE0.756 (`knee-dino-submission-20261008`v1);56923149 COMPLETE0.757 (`knee-ensemble-submission-20261008`v1). Baseline artifacts `outputs/experiments/dino-frozen-20261008/artifacts/`; Colab comparison `outputs/experiments/head-comparison-20261008/comparisons/`. Details in [first experiment](docs/experiments/20261008-first-baseline.md).

## Working decisions and limitations

- Live community audit found accessible0.949 notebooks above Pilkwang0.891, both sharing an OAI-derived CoAtNet. OAI/model eligibility is unresolved, **not explicitly prohibited**; defer that checkpoint for now. ConvNeXt dataset Apache-2.0 and complete source; plain Raptor CC0 with ImageNet and competition training per author. Do not claim their gold scores are independent.
- Predeclared one50/50 per-target percentile-rank blend with average ties, no gold/public weight search. Both readers run sequentially with actual hiddenIDs and strict failures. Estimated full blend4.58h with margin from separate environments; common-environment rehearsal required.
- Fresh ImageNet pilot preserves original2,000/300 assignments then subsets512/200; originalStevenv1 SHA pinned. No gold evaluation; patient/report exclusions beforecache. Four fixed windows per six slots at224px, unknown0.5masked, confidence-weighted per-target macroBCE, EMA. CPU measured time/disk gates precede fullcache; CUDA64batch gate precedes at most4epochs/90min. Resumable own optimizer/RNG/order/checkpoints with cumulative budget; new source binds resume. Latest device metadata integrated`367ac30`; direct attachment search/setup deadlines integrated`b736da6` for future builds. RunningCPU64 remains source`1f1d788`. Nested deadline cleanup fixed`9196c69`,1000subprocess expirations exercised before next build.
- Prior low score is consistent with frozen224px three-series representation versus tuned multi-sequence public encoders; exact causal effects untested. Nonfat-suppressed sagittal/coronal coverage4,266/3,406studies suggests complementary inputs; metadata fluid/fat flags alone cannot distinguish T1/PD/T2.
- No current approval blocker. No purchases, fullrawdownload, hostmessages or credential extraction. Earlier private prediction-fixture upload was rejected and never uploaded; safer own weights-only artifact succeeded.
- `knee/model.py` remains an untrained toy; real pipelines are separate. Unit tests do not establish competitive quality or a full pipeline.

## Latest verification and handoff

- Aftercodecfix: full ML suite96/96PASS withallthreeauditedsource/receipt environment variables enabled; dependency-free smoke previouslyPASS. Aftersetupchange`b736da6`, suite abortedexit142 atnesteddeadlinecleanup test; fix`9196c69`verifiedbyfullsuite102/102PASS including1000nestedexpiry checks. CUDAtraining/resume stillunverified.
- Latestpassinglog`outputs/research/2026-10-08/tests-integrated-sources.log`; failingtimeoutlog`tests-final-setup.log`. Fixedblend notebookcompiles,private/offline,exactlytwo modeldatasets. Localmatched-baseline helper preserved`outputs/analysis/fresh-pilot-baselines/compare_frozen_on_plan.py`, awaiting actualremoteplan; no refit/golduse.
- Three managed agent worktrees active: `model-provenance`, `score-strategy`, `reference-rehearsal`; all commits integrated, independent blend review in progress. Preserve ignored evidence before archive. Prior47agentoutputs already preserved at`outputs/agent-archives/20261008/`.
- Next: inspectCPU64receipt and resource gate; inspect shared-environment blend parity; launch fullcache or fixedblendentry only after their checks. Await standaloneConvNeXt score. Continue authorized experiments; target0.950 remains unachieved.
