# Project status and session handoff

Last updated: **2026-10-08 (Europe/Istanbul; 00:04 UTC)**. Read this first, then [README](README.md) and [AGENTS](AGENTS.md).

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
| REF-02 | active | Separate ConvNeXt and plain Raptor rehearsals passed. Await standalone ConvNeXt score. Common-environment fixed50/50 rank-blend visible rehearsal packaging fix pending; require pass and cross-run component parity before one prospective blend entry. |
| TRAIN-02 | active pilot | Fresh ImageNet ConvNeXt CPU64 cache job running. Require measured gate.passed before expanding to512train/200weak-validation cache. Then CUDA throughput gate and<=90min/4epochs; select only by weak BCE. |
| TRAIN-03 | conditional | Scale only if pilot beats training-only prior and throughput justifies it. Measure compact-cache transfer before Colab expansion; preserve >=8KaggleGPUh for qualification. |

## Live jobs and resources

- **Entry56924358 PENDING** as of00:00UTC October8. `ozgung/knee-public-reader-submission-20261008`v1; submittedOctober7 23:50:09UTC. Exact Apache-2.0 public ConvNeXt threefold reader. Notebook COMPLETE and visible parity exact. Preserve version/attachments. Poll `kaggle competitions submissions rsna-knee-abnormality-detection --format json`.
- **CPU64 pilot RUNNING**, launchedOctober7 ~23:57UTC: `ozgung/knee-fresh-cache-pilot-20261008`v1, source`1f1d788`. Command `kaggle kernels push -p outputs/kaggle/fresh-cache-pilot --timeout 2100`. Internal1800s; expected `fresh_reader_cache/cache-pilot.json` and private plan/index/arrays; early failure `fresh-reader-failure.json`. No GPU training yet. Require `gate.passed == true`, then exact plan digest for next stage. Full commands in pilot protocol.
- **Blend visible rehearsal FAILED before inference:** `ozgung/knee-reference-blend-rehearsal-20261008`v1, source`57bb203`, external/internal3600s. QualificationSHA`86f3456e0f385414fdaa528c30d1f56501337fad8ff583160ab450472994ee96`. Failed at8.34s because codec guard counted bothcp311/cp312 wheels. Fix selects exactthreecp312-compatible packages; regression and v2 retry pending. No training or predictions.
- Poll jobs with `kaggle kernels status OWNER/SLUG`; read `kaggle kernels logs`; completed output via `kaggle kernels output`. Stop through notebook running-job controls if needed. Bounded jobs stop internally/external timeout; no recurring automation exists.
- CLI authentication works. AtOctober7 23:36UTC: KaggleGPU28.91h,TPU20h; refreshOctober10 00:00UTC. ColabUI351.8compute units and zero sessions; expiryOctober18 historical/unverified. No local/Colab training active. Last freeMacdisk31GiB; initial localcache2GiB. Three lifetime entries; daily count must be rechecked after UTC reset.

## Qualified reference evidence

- ConvNeXt source`fdb0f60`, notebook rehearsalv2 COMPLETE (v1 kernelspec failure fixed):19studies,90selectedseries,24 legitimate missing slots,0decode/emptyfailures; repeat and serialization parity0. Serialization is persistence parity, not independent raw/cache equivalence. Runtime161.42s,peakCUDA1.284GB; mean inference projects7,535s/1,300 before setup. Python3.12,torch2.10,timm1.0.26. Submission visible3 COMPLETE63.94s, exact sameweights/predictions. `outputs/kaggle/public-reader-{rehearsal,submission}-result/`.
- Plain CC0 Raptor rehearsalv1 COMPLETE:19studies,1,156unique slices,0fallback; decodedpixelerror0,windowerror1.1921e-6,repeaterror0. Runtime98.71s; p95projection+25%=7,053.68s/1,300;peakCUDA8.247GB. Python3.13,torch2.11,timm1.0.29. Contract`d8fedf9a1d7dc9849ab10fd67bb1e981ce96d628eb4e4bceaab20e448a84801b`; checkpointSHA`89606f05849838529e1b4658d28fb049623205d9853371548403bca460361ded`. `outputs/kaggle/raptor-rehearsal-result/reference-rehearsal/rehearsal.json`.
- Original entries:56922773 COMPLETE0.756 (`knee-dino-submission-20261008`v1);56923149 COMPLETE0.757 (`knee-ensemble-submission-20261008`v1). Baseline artifacts `outputs/experiments/dino-frozen-20261008/artifacts/`; Colab comparison `outputs/experiments/head-comparison-20261008/comparisons/`. Details in [first experiment](docs/experiments/20261008-first-baseline.md).

## Working decisions and limitations

- Live community audit found accessible0.949 notebooks above Pilkwang0.891, both sharing an OAI-derived CoAtNet. OAI/model eligibility is unresolved, **not explicitly prohibited**; defer that checkpoint for now. ConvNeXt dataset Apache-2.0 and complete source; plain Raptor CC0 with ImageNet and competition training per author. Do not claim their gold scores are independent.
- Predeclared one50/50 per-target percentile-rank blend with average ties, no gold/public weight search. Both readers run sequentially with actual hiddenIDs and strict failures. Estimated full blend4.58h with margin from separate environments; common-environment rehearsal required.
- Fresh ImageNet pilot preserves original2,000/300 assignments then subsets512/200; originalStevenv1 SHA pinned. No gold evaluation; patient/report exclusions beforecache. Four fixed windows per six slots at224px, unknown0.5masked, confidence-weighted per-target macroBCE, EMA. CPU measured time/disk gates precede fullcache; CUDA64batch gate precedes at most4epochs/90min. Resumable own optimizer/RNG/order/checkpoints with cumulative budget; new source binds resume. Latest device metadata integrated`367ac30` for future builds, existingCPU64 remains source`1f1d788`.
- Prior low score is consistent with frozen224px three-series representation versus tuned multi-sequence public encoders; exact causal effects untested. Nonfat-suppressed sagittal/coronal coverage4,266/3,406studies suggests complementary inputs; metadata fluid/fat flags alone cannot distinguish T1/PD/T2.
- No current approval blocker. No purchases, fullrawdownload, hostmessages or credential extraction. Earlier private prediction-fixture upload was rejected and never uploaded; safer own weights-only artifact succeeded.
- `knee/model.py` remains an untrained toy; real pipelines are separate. Unit tests do not establish competitive quality or a full pipeline.

## Latest verification and handoff

- Before blend integration: ML suite84tests,82exercised PASS/2optional source skips; dependency-free smoke PASS. Freshpilot independently reviewed; all resume/digest/macro-loss fixes addressed. CUDA training and resume still unverified.
- Blend isolated-agent suite80/80 with real snapshots/receipts; parent integrated suite95tests:92exercised PASS,3optional source/receipt skips (`outputs/research/2026-10-08/tests-blend-integrated.log`). Generated notebook compiles,225,109bytes,private/offline with exactlytwo model datasets.
- Three managed agent worktrees active: `model-provenance`, `score-strategy`, `reference-rehearsal`; all commits integrated, independent blend review in progress. Preserve ignored evidence before archive. Prior47agentoutputs already preserved at`outputs/agent-archives/20261008/`.
- Next: inspectCPU64receipt and resource gate; inspect shared-environment blend parity; launch fullcache or fixedblendentry only after their checks. Await standaloneConvNeXt score. Continue authorized experiments; target0.950 remains unachieved.
