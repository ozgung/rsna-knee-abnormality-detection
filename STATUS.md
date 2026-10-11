# Project status and session handoff

Updated **2026-10-11 21:40 UTC (October 12 00:40 TRT)**. Read this snapshot, then [shared handoff](docs/claude-astra-handoff.md). Rules: AGENTS.md; setup: README.md. [Earlier exact recovery/history](docs/status-archive/status-before-20261011-s6-v4-terminal.md).

## Current snapshot

- Best public score remains **0.950**, submission 56986393. The tenth displayed leaderboard row was 0.963 on October 11 at 20:06 UTC (rounded scores and ties).
- **Repaired S6 submitted: 57086107, PENDING**, accepted October 11 at 21:40:07 UTC. Exact notebook `ozgung/knee-public-s6-entry-20261011`, **version 2 / kernel 138226549**. No public score yet. Do not launch a duplicate or overwrite this notebook while hidden evaluation runs.
- The repaired rehearsal v4 completed with all **39 genuine predictions from both arms**; actual host mirror17/native2 stress suites and **35 independent parent checks PASS**. Runtime 23.86 minutes. Hidden runtime projections: base 5.23 h, conservative 7.18 h; a larger sensitivity exceeds 9 h, so these are not bounds. [Result and repair](docs/experiments/s6-dual-rehearsal-result-20261011.md).
- Pure entry v2 completed in **152.48 seconds**, with all three visible studies, zero imputation and **21 independent checks PASS**. Visible CSV and raw arm probabilities match v4 exactly. Build and accepted-source checks: 15 and 16 PASS respectively.
- Fixed 870-study development comparison: original 0.951330909 → fixed 70:30 rank blend **0.962963989**, delta +0.011633080, paired 2,000-bootstrap CI [+0.009787471, +0.013714779]; all 12 targets improved. Independent audit: 73 PASS / 0 FAIL / 3 UNKNOWN. Public training/report/gold exposure and unavailable patient identities prevent an independent-generalization claim or public score forecast. [Comparison](docs/experiments/s6-fixed-evaluation-result-20261011.md).
- Full384 submission **57083497 failed** with Notebook Threw Exception; API COMPLETE with empty score. No traceback or established hidden cause. Preserve it; no blind retry.
- Latest actual GPU allowance, 21:35 UTC: **42.6509 / 45 h used, 2.3491 h remaining, zero reserved**. Hidden-evaluation quota accounting is unknown. No ordinary training jobs, local controllers or Colab runtimes remained; agents are done. Submission acceptance reports **three slots remaining today**. Respect actual limits; no invented reserve or purchases.

## Ready queue and ownership

| Owner / task | State / next action |
| --- | --- |
| Parent — S6 hidden evaluation | Reconcile submission 57086107 at the next meaningful checkpoint. Preserve exact version 2; inspect score or official error before selecting another predictor. |
| Parent — next experiment | Await S6 public result. Standalone S6, full384 blend, weight sweeps and larger batches remain unselected and need prospective comparisons. |
| Fable — independent review | May inspect counterexamples and planning risks; no compute claimed. Coordinate in the shared log. |
| Parent — failed/closed paths | Preserve full384 exception and two-fold evidence. Input-policy regression is established, but its role in the hidden failure is unknown. Own224/384/R2 and public-adaptation negative recipes remain closed. |

## Active job and recovery

**Competition evaluation:** submission **57086107**, accepted `2026-10-11T21:40:07.610000`, PENDING in the official API. Submitted exactly once with:

```bash
/Users/ozgun/miniconda3/bin/kaggle competitions submit rsna-knee-abnormality-detection \
  -k ozgung/knee-public-s6-entry-20261011 -v 2 -f submission.csv \
  -m 'Fixed 70:30 original320/S6 rank blend; source-compatible partial inputs; dual-T4 image-only rehearsal and exact visible parity passed'
```

Reconcile using `kaggle competitions submissions rsna-knee-abnormality-detection --format json`. Ordinary notebook COMPLETE is distinct from hidden evaluation completion. Do not resubmit, resume or overwrite while pending. Preserve any error; generic API COMPLETE/empty score requires checking the official submissions UI for the error message.

**Submitted build:** `outputs/kaggle/public-s6-entry-20261011-reviewed-v2/`; source **96dbcea95713a4d53493c6f6ca201a4ff0b2527b**, repair `9af1acf`. Notebook 561,581 bytes; SHA256 **c4f6ec7e5bba475da30a7eb61b349656d85ad35ecf7108485b752570bfd757d3**. Scientific bundle **aa5ebdb12f3fbc3c638bcc2591d5894a5d1cb8bd1201e2fc7ea657b5e43428ff**. All 39 mechanics fixtures removed. One ordinary push; accepted private/offline/T4 source verified.

**Visible v2 result:** `outputs/kaggle/public-s6-entry-20261011-result-v2/`; receipt **a4fd591828e16e504769c2f2063f08f5b115c6fa6830be8be925bb6e05f1e06b**. CSV SHA256 **04d8f0bb193601edb241b326f7bd0e15c6b8a9f64fb025bb6447902809f8e1e0**, exact rehearsal parity. Evidence `outputs/analysis/public-s6-entry-20261011/pure-entry/`: `parent-visible-review-v2.json`, `competition-submit-v2.log`, `submissions-after-v2.json`. Collector and numeric reviewer were exercised on the actual completed version; 21 PASS, no failures.

**Repaired rehearsal v4:** `ozgung/knee-public-s6-rehearsal-20261011`, version 4 / kernel 138215646, source **9af1acfcaf61b9a87cd9acac08b74c2b7e05c50c**. Build `outputs/kaggle/public-s6-rehearsal-20261011-reviewed-v4/`; 590,333 bytes / 230 sources; notebook SHA256 **2dd0b44ef2fa3e7eccce1855183f5620c0c5363ed2053383493d92726e0b18f4**, same scientific bundle. Receipt **f8c7ebd080e6a9bdf0089e68d4be75199e805d67f4fc72db0a361164b4b817d8**. Actual small artifacts `outputs/kaggle/public-s6-rehearsal-20261011-result-v4/`; parent audit `outputs/analysis/public-s6-entry-20261011/parent-terminal-review-v4.json` (35 PASS). Failed rehearsal v1/v2 and held old-wrapper entry v1 remain preserved; never submit them. Earlier recovery details are archived above.

## Working decisions and evidence

- Wrapper `s6-dual-image-only-v2` / policy `source-partial-input-v1` restores qualified partial-series handling. Keep every metadata row, exact confined alias and present/empty/absent inventory. Recovered diagnostics may accompany source-valid images; resource/invalid/shape/mask/empty checks, both-arm completeness, finite probabilities and whole-population ranking remain enforced. No prediction filling, partial ranks or deadline substitution.
- Actual v4 mirror17 and native2 stress checks pass. Native2 covers missing/empty companion direct-versus-staged arrays and metadata only; it does not establish new native invalid/resource/all-empty coverage or clinical accuracy. Upstream readers, weights, sampling, chunk16, workers0, FP16, source six-decimal values and fixed70:30 method remain unchanged. Parent checks: 21 build, 10 unchanged-science, 41 focused ML tests; default26 with nine expected skips; smoke and diff PASS. Agent reported60 focused public-S6 tests PASS. Broad legacy-suite dependency/backend failures remain recorded; no full-suite PASS claim.
- Full384 improved two exposed development folds: fold0 delta +0.002646205 (CI positive, all12 improved); fold1 +0.001816829 (CI positive, ten improved, two small declines). Rehearsal v2 / kernel138185086: 35 studies, 686.46 seconds, parent23 PASS. Pure entry v1 / kernel138214183: three studies, 52.28 seconds, parent16 PASS; hidden submission57083497 then failed. [Fold0](docs/experiments/public-fullfield-result-20261011.md), [fold1](docs/experiments/public-fullfield-replication-result-20261011.md), [rehearsal](docs/experiments/fullfield-image-only-rehearsal-20261011.md).
- [MRI transfer research](docs/research/kaggle-mri-transfer-20261010.md), [public refresh](docs/research/knee-public-refresh-20261011.md), [eligibility](docs/research/public-source-eligibility-20261011.md), [visual strategy review](docs/research/pipeline-review-20261011.md), [29-page PDF](outputs/briefings/knee-pipeline-review-20261011/knee-pipeline-review.pdf). Blind S6 coverage audit: 179 series / 679 depth samples supports gross visible union only. Crop gain is confounded with readers and sampling; matched global/local/cache/native-crop ablation remains a later task. Domain checkpoint leads need source/license/provenance audit; no new weights fetched.
- Official pages verified October11: five submissions/day; offline runtime ≤9h; approximately1,300 hidden studies; entry/merge deadline October15 and final October22, both23:59UTC. T4x2 shares four CPUs and29GBRAM. The ordinary12h notebook limit does not override the competition9h limit. Actual test.csv controls count and order.

## Latest checkpoint

**2026-10-11 21:40 UTC:** repaired pure entry v2 completed; parent independently verified all three studies, strict state/device/logit/probability gates, exact v4 visible CSV and raw probabilities (21 PASS). Fresh allowance was four remaining; one authorized submission accepted as **57086107**, PENDING, with three remaining. No new training or duplicate job launched. Rehearsal/input repair evidence remains unchanged; hidden runtime and score await evaluation. Parent owns the next result/error audit. Local Git branch `codex/token-efficient-instructions`, no remote; preserve user AGENTS edits and Fable untracked files. Integrated worktrees are protected by pinned/workspace guards; do not force archive.
