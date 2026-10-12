# Project status and session handoff

Updated **2026-10-12 12:37 UTC (15:37 TRT)**. Read this snapshot, then [shared handoff](docs/claude-astra-handoff.md). Rules: AGENTS.md; setup: README.md. [Previous exact recovery/history](docs/status-archive/status-before-20261012-recovery-terminal.md).

## Current snapshot

- Best public score is now **0.954**, submission **57096002**, independently confirmed by SDK at12:14:55UTC with no error and381317 output bytes. Previous0.950 submission56986393 remains a fallback. Top-ten target: the tenth displayed row was 0.963 on October 11 at 20:06 UTC (rounded scores/ties).
- **New capped-recovery rehearsal v1/kernel 138259814 COMPLETE; 95 independent checks PASS.** Both arms produced all 39 genuine main predictions; four separate typed-failure/next-healthy/state injections PASS. Actual guarded mirror10/native2 PASS. All healthy raw probabilities and visible CSV match strictv4 exactly. Runtime 1,573.63 seconds / 26.23 minutes. Hidden projections: base 5.97 h, conservative 8.19 h; larger sensitivities exceed 9 h and are not bounds. [CPU/runtime result](docs/experiments/s6-capped-recovery-result-20261012.md).
- **Recovery submission57096002 scored0.954**, a +0.004 public improvement. This establishes completion and public performance of the combined recovery method; hidden recovery counts and the cause of prior strict failures remain unknown. Ordinary entry passed26 checks and all39 rehearsal inputs remained unchanged. Preserve exact v1/kernel138266997.
- CPU recovery sensitivity: all 13 frozen scenarios PASS; worst macro delta +.006614209, paired 2,000-bootstrap CI[+.002551465,+.010174061], no supported-target damage. Independent 22 checks PASS. This is exposed synthetic sensitivity, not natural-failure accuracy or a public-score forecast.
- Fixed 870-study development comparison: original 0.951330909 → fixed 70:30 blend 0.962963989, delta +.011633080, all 12 targets improved; paired CI[+.009787471,+.013714779]. Public training/report/gold exposure and unavailable patient identities prevent independent-generalization claims. [Comparison](docs/experiments/s6-fixed-evaluation-result-20261011.md).
- Strict S6 submission **57086107** and full384 **57083497** failed with generic unhandled hidden-rerun errors, empty scores/output. No traceback or established failing phase. Preserve both; never blindly resubmit. [Official failure receipt](outputs/analysis/s6-hidden-failure-20261012/official-state.json).
- Actual GPU allowance12:14UTC: **43.191879/45h used,1.808121h remaining, zero reserved**; four submissions remaining today. No new compute launched. Opus relays user clarification: existing Colab credits are usable and do not generally expire October16; verify actual account/runtime availability before launch. No purchases or new terms are authorized. Wall time, not an assistant-imposed reserve, controls prioritization.

## Ready queue and ownership

| Owner/task | State / concrete next action |
| --- | --- |
| Parent — recovery entry | Completed and submitted once as 57096002. Preserve exact accepted v1; visible qualification 26 PASS. |
| Parent — balanced blend | Deferred after Opus review: exposed870 cannot select generalizing weights. Fresh Heliosli public source already uses50:50 with the same S6 bundle; this is not a new method. Preserve diagnostic receipts, no submission. |
| Opus — next-direction review | Read12:34 findings; parent replied12:37 accepting validation correction and requesting sourced correlation/rules claims plus one concrete training hypothesis against rejected own384/R2. Shared file remains coordination channel. |
| Parent — Colab training | Next: freeze a reproducible independent-reader pilot with validation excluded from encoder training as well as head fitting. Public frozen-encoder head is not clean validation. Resolve label/native-view hypothesis and grouping before launch; no job launched. |
| Parent — S6 reproduction | Sol source audit finds missing per-study likelihood-ratio/state targets, exact membership/init pin/full flags. Extra exact S6 seeds/folds are not currently reproducible. A new recipe must be named and evaluated as such. |
| Parent — deferred alternatives | Full384 extra views have positive exposed evidence but add runtime. Own224/384/R2/public-adaptation rejected recipes stay closed. MRI checkpoint leads still need release/license verification. |

## Active job and recovery

**Pure entry:** `ozgung/knee-public-s6-recovery-entry-20261012`, **version1 / kernel138266997**, ordinary COMPLETE 04:53 UTC after exactly one push; hidden submission57096002 COMPLETE/public0.954 confirmed12:14UTC. Committed build revision **82111ffb52f54bc2ea3eb50f40a3b9c234573f56**; immutable build `outputs/kaggle/public-s6-recovery-entry-20261012-reviewed-v1/`,569,811 bytes / 232 sources. Notebook SHA256 **bd1d1d37899ca40130d4298f12361539a80414cacf0a23663252927cc888125e**; production bundle **b681951999bb086d71b55224c35ef23c68ff939acc200f09f8c7cd121a7cb8e9**. Parent 11 pure-build and16 accepted-source checks PASS. Source and private/offline/T4 settings match the reviewed build.

Launch already performed once: `/Users/ozgun/miniconda3/bin/kaggle kernels push -p outputs/kaggle/public-s6-recovery-entry-20261012-reviewed-v1 --accelerator NvidiaTeslaT4`. Logs `outputs/analysis/s6-capped-recovery-20261012/push-entry-v1.log`; binding/accepted metadata/source `outputs/analysis/s6-capped-recovery-20261012/pure-entry/launch-v1.json` and adjacent files. Visible qualification passed 26 checks; submitted once as 57096002. Do not push again or resubmit.

Runtime root `/kaggle/working/s6-recovery-entry/`: per-board ledgers/receipts/logs; `s6-recovery-entry-project/runtime-config.json`; root `/kaggle/working/submission.csv`. At terminal state run:

```bash
/Users/ozgun/miniconda3/bin/python outputs/analysis/s6-capped-recovery-20261012/pure-entry/collect_terminal_v1.py
.venv/bin/python outputs/analysis/s6-capped-recovery-20261012/pure-entry/review_visible_v1.py
```

Collector pins exact accepted source/version and downloads small outputs only to `outputs/kaggle/public-s6-recovery-entry-20261012-result-v1/`; collector/auditor exercised on actual completed v1:26PASS/0FAIL; receipt bfe29e321e91a491a0c7472eea7befc1f31eb6fa7258f297c0e57257f5ce77d4. No need to repeat unchanged visible checks. Preserve failures/partial artifacts; no blind restart. The nine-hour alarm remains active. No deadline substitution, global resource recovery or hidden-test human labels. The fixed3% recovery union cap uses actual test.csv population; only classified data-local whole-arm failures may be filled from global genuine arm means before whole-population ranks.

**Qualified rehearsal:** `ozgung/knee-public-s6-recovery-rehearsal-20261012`, **v1/kernel 138259814**, COMPLETE 04:46 UTC. Source 42d4b3d; build `outputs/kaggle/public-s6-recovery-rehearsal-20261012-reviewed-v1/`; notebook78b3be843d0e6a8b7f5827449ce43fa4fcf53bbfb1655586e1a7c940a97f1ec1. Parent95 independent checks PASS; receipt **e41bcec3a0b593f9031f976e9b780a0a62f85e390056ddae1252b6c9a60e859b**. CSV **04d8f0bb193601edb241b326f7bd0e15c6b8a9f64fb025bb6447902809f8e1e0**, exact strictv4 parity. Small outputs `outputs/kaggle/public-s6-recovery-rehearsal-20261012-result-v1/`; audit `outputs/analysis/s6-capped-recovery-20261012/parent-terminal-review-recovery-v1.json`. Actual simultaneous RSS peaks old 6.00 GB / S6 8.74 GB; minimum free disk 20.35 GB. Old-arm scratch/side cleanup fields are not applicable, not observed teardown. Four typed-failure/next-healthy/state gates are separate from genuine main rows.

**Failed strict entries:** S657086107 submitted once from `ozgung/knee-public-s6-entry-20261011` **v2/kernel138226549**, accepted21:40:07UTC October 11. Full38457083497 also failed. SDK confirms empty score/generic unhandled rerun/zero output; timing includes queue and does not locate cause. Strictv4 all 39/35 checks and purev2 all 3/21 checks passed healthy mechanics but do not qualify recovery. Exact sources, commands, failure observations and collectors are retained in the linked archive. Never submit these failed versions or the rehearsal.

## Working decisions and evidence

- New recipe**s6-fixed70-30-capped-arm-mean-v1** is a prospectively qualified method. Public single-arm fallback does not prove its branch executed in a scored notebook. No broad error-class/message catch: audited source/class/site/cause typing guards errors before upstream stringification; unknown/programming/generated-artifact/resource errors remain fatal. Actual native warnings/skipped companions may accompany source-valid images. All raw data remains immutable and outside Git.
- Runtime implementation preserves strict/frozen CPU bytes and successful numerical code: parent 58 focused tests / no skips, 18 integration checks, guarded CPU mirror10/native2, smoke/diff PASS. NeedMore parser flow allowed only at original pinned boundaries; two pydicom3.0.2 validation raise sites bound to file/function hashes. Actual host 95 checks now qualify both-board inference/state/logits/recovery/partial-input behavior. These do not establish clinical generalization or natural-failure frequency. Broad legacy-suite failures remain recorded; no full-suite PASS claim.
- CPU13-scenario freeze SHAe11e611c1089f63bd2bfd90b7659ba276bf5b7c9ddf97d18c178df8922b99def; result2b3f43c8e8872c00a211211020798ba899ff38f94b29de84265a771dcc2a2198. Protocol/result/per-target supports in [result](docs/experiments/s6-capped-recovery-result-20261012.md) and [protocol](docs/experiments/s6-capped-recovery-protocol-20261012.md). Unknown report findings remain masked; patient grouping limitations persist.
- Independent 1,300-study fake orchestration, ledger, resource and teardown audit PASS, real readers/GPU mocked. Small ledger growth does not explain real GB memory; one-study windows and hashing copies remain plausible risks. [Resource audit](outputs/analysis/s6-resource-audit-20261012/report.md). Optional incremental hash prototype 9 layouts PASS, production unchanged to isolate recovery.
- Full384 improved two exposed development folds, +.002646205/+.001816829, but failed hidden execution. Native-crop/S6 improvement is confounded with readers/sampling; use matched views/coverage ablations for later causal comparisons. [Fold0](docs/experiments/public-fullfield-result-20261011.md), [fold1](docs/experiments/public-fullfield-replication-result-20261011.md), [rehearsal](docs/experiments/fullfield-image-only-rehearsal-20261011.md).
- [MRI transfer research](docs/research/kaggle-mri-transfer-20261010.md), [public refresh](docs/research/knee-public-refresh-20261011.md), [eligibility](docs/research/public-source-eligibility-20261011.md), [visual review](docs/research/pipeline-review-20261011.md), [29-page PDF](outputs/briefings/knee-pipeline-review-20261011/knee-pipeline-review.pdf), [domain checkpoint audit](docs/research/mri-domain-checkpoint-audit-20261011.md). No new weights fetched; domain leads remain unverified.
- Official rules/code refreshed October 12: five submissions/day, offline≤9h, public free assets subject rules, no human test labeling. Aggregating solely image predictions has no inspected prohibition (interpretation, not host approval). Dates verified October 11: entry/mergeOctober 15 and finalOctober 22,23:59UTC. Approximately 1,300 hidden studies; actual test.csv authoritative. T4x2 shares4 CPUs / 29 GB RAM; ordinary 12 h does not override competition 9 h.
- Parent owns commits. Git has no remote. Preserve unrelated user AGENTS.md changes and Fable untracked files. All subagents completed; no agent or controller should restart jobs automatically. Use existing resources/actual account limits, no self-imposed hour caps.

## Latest checkpoint

**2026-10-12 12:32 UTC:** verified public0.954, reviewed Opus's two proposals and credit-expiry correction, and posted an independent next-direction response. Timeout contingency deferred now that this entry scored; no hidden failure cause inferred. Sol's cached-source feasibility audit found exact S6 training inputs/settings missing. One50:50 blend diagnostic was frozen before calculation and passed: .962963989→.967675464, all12 positive, no damage, paired2,000-component CI[+.003666794,+.005915891]. Independent pairwise AUC reproduced all24 baseline/candidate target values exactly. This is exposed development evidence, not a public forecast. Parent recommends one qualified50:50 public follow-up plus a controlled Colab frozen-feature head pilot; Opus response and training protocol remain pending. No new training, inference or submission.

Evidence: `outputs/analysis/next-direction-20261012/{training-feasibility.md,balanced-blend-protocol.json,balanced-blend-result.json,balanced-independent-check.json}`. Diagnostic protocolSHAfaad2580867ea3744753bf7d201f57329f523f4d5162ff034790c3fc5f855801; CPU0.414s. Official scored result `outputs/analysis/s6-capped-recovery-20261012/pure-entry/scored-result.json`. Current baseline .954 is preserved; no changes to submitted source or recovery policy. Direct Claude UI denied access; cooperation continues in shared handoff, awaiting Opus reply.

Submit command already executed once:

```bash
/Users/ozgun/miniconda3/bin/kaggle competitions submit rsna-knee-abnormality-detection \
  -k ozgung/knee-public-s6-recovery-entry-20261012 -v 1 -f submission.csv \
  -m 'Fixed 70:30 original320/S6 rank blend; classified data-local recovery with global arm means and 3pct union cap; dual-T4 39-study rehearsal and exact visible parity passed'
```

Reconcile through `api.competition_submissions('rsna-knee-abnormality-detection')`, selecting **ref 57096002** and checking `status`, `public_score`, `error_description` and `total_bytes`. Initial receipt `outputs/analysis/s6-capped-recovery-20261012/pure-entry/submission-v1.json`; command log adjacent. Do not infer success from ordinary notebook COMPLETE or empty-score evaluation COMPLETE. Previous failure versions remain preserved.

**12:37 UTC research checkpoint:** read Opus12:34 review and refreshed public notebook listing/source. Heliosli latest is an existing50:50 same-bundle recipe; goodpjw latest better-ranked stage4 is private, public copy falls back to70:30. Sources/evidence and ownership in shared handoff12:37. No new job or submission.
