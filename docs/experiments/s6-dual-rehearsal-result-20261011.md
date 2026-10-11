# S6 dual-T4 rehearsal and entry preparation — October 11

The original v3 wrapper completed all 39 studies in **1,564.54 seconds (26.08 minutes)**. The parent independently checked the actual arrays, receipts, saved input fingerprints, historical probabilities, four-reader logit pooling, source six-decimal values, physical GPU identities, visible-only ranks and measured projections: **29 PASS, no failures**. This qualifies those exact wrapper bytes; it does not qualify a later repair or establish hidden-test success.

The unchanged predictor uses 70% original central320 mirror reader and 30% S6, with whole-population percentile ranks. Both fresh workers process complete studies on separate physical T4s; all 39 have genuine predictions from both arms, with no missing outputs, imputation or precision recovery. Cross-board sentinels, strict reloads and model/buffer state checks pass. The prior v1 UUID failure and v2 repeated native initialization failure remain preserved.

## Runtime evidence

The old and S6 phases took 603.10 and 954.89 seconds including extensive repeated qualification work. Measured production time on the slower shard was 101.15 seconds for old320 and 517.47 seconds for S6. All fixed phase overhead is retained once in the projection.

For approximately 1,300 hidden studies, the base projection is **5.92 hours**. The prospective screen with 25% more windows and 15% shard imbalance is **8.12 hours**, below the official nine-hour limit. Some larger sensitivity scenarios exceed nine hours. These are extrapolations, not bounds on hidden studies, preprocessing tails or growing ledger I/O.

Observed combined host RSS peaked at 8.95 GB; output footprint at 586.22 MB; minimum free disk was 20.35 GB. Both physical UUID byte checks and all production state/input/finite-probability gates passed.

## Exact artifacts and separate visible entry

- Rehearsal: `ozgung/knee-public-s6-rehearsal-20261011`, v3, kernel138215646; code revision `2d578fed1e7c1365341e19322ed73087e6e0f9f0`.
- Receipt SHA256: `c9673ad3de075bb2611dbf70ce02efd40cb8fb898ee75fabfa4774627ed8108d`.
- Visible CSV SHA256: `04d8f0bb193601edb241b326f7bd0e15c6b8a9f64fb025bb6447902809f8e1e0`.
- [Actual small result artifacts](../../outputs/kaggle/public-s6-rehearsal-20261011-result-v3/s6-entry/receipt.json), [parent 29-check audit](../../outputs/analysis/public-s6-entry-20261011/parent-terminal-review-v3.json).

A separate pure entry was built at revision `297e950`, with all 39 mechanics IDs and saved probability/input fixtures removed; the scientific source bundle remains byte-identical. Fifteen build checks and 16 accepted source/private/offline/T4 checks pass. Notebook557129 bytes, SHA256 `416d6c23418cb8ee84139cb71b402b67906d47f6a81b7711fbe76bb8fe798bd2`; source bundle `1834980444ae43e068432ecdf771fae7d4ec71236c7978dab3eb79651506e27f`. `ozgung/knee-public-s6-entry-20261011` v1/kernel138226549 was pushed once and RUNNING20:37 UTC. **No S6 competition submission has been made.** Exact visible parity is pending.

## New evidence requires a wrapper repair before submission

Full384 competition submission57083497 became terminal with an empty API score; the logged-in official submissions page confirms **Notebook Threw Exception**. Kaggle exposes no traceback. Preserve the entry; do not assert a specific hidden cause or blindly resubmit. Best public score remains0.950.

Astra's focused source review found two concrete S6/full384 wrapper regressions relative to the already-qualified successful mirror reader:

1. Recovered header/pixel events were rejected after `prepare_study` returned validated usable images. This incorrectly promoted diagnostic counters into fatal errors.
2. S6 staging required every metadata-listed series to contain DICOMs before either reader ran, preventing source-defined missing/empty-series handling with usable companion input.

The parent accepts restoring the established source behavior, with a new wrapper policy and fresh rehearsal. Preserve all metadata, confined original paths and availability evidence; create no images or invented masks. Resource errors, invalid pixels, inconsistent volume/masks, wholly empty studies, nonfinite probabilities, missing arm/study predictions, partial ranks and deadline substitution remain failures. Keep upstream readers, weights, geometry, windows, precision, six-decimal values and 70:30 ranking unchanged.

SolMedium owns the three-file repair in an isolated worktree; the parent owns integration, actual pinned-reader stress checks, all39 requalification, committed fresh build and entry. Existing v3 PASS cannot qualify repaired wrapper bytes. The fixed870 result remains exposed development evidence, not an independent-generalization claim or a public score forecast.

## Repair committed and fresh rehearsal running — 20:49 UTC

The parent integrated the three-file Sol implementation and committed **9af1acfcaf61b9a87cd9acac08b74c2b7e05c50c**. Wrapper version is `s6-dual-image-only-v2`, with an explicit `source-partial-input-v1` policy. All metadata rows stay intact. Entry-local staging/verifying binds exact confined aliases and present/empty/absent availability; the stricter qualification module was not changed. Old-reader recovered header/pixel diagnostics are retained without rejecting an otherwise valid returned volume; resource/invalid counters and the unchanged source volume/mask/empty checks remain fatal. No missing prediction or deadline recovery was introduced.

Parent verification: **41 focused ML tests PASS, no skips**; default S6 discovery26 tests/9 expected optional skips, smoke and diff PASS. Agent additionally reported60 focused public-S6 tests PASS. An existing local OpenCV installation supplied the missing optional dependency without installing software. Actual pinned mirror17 synthetic DICOM cases PASS, with valid returned inputs and nonzero recovered error counters; actual pinned lean native2 missing/empty companion cases PASS, identical direct/staged four-slice arrays and metadata. These native2 tests do not claim new native invalid/resource/all-empty stress coverage or CUDA inference. Initial CPU import attempts failed intact source/import guards before the final verified execution; no guard was weakened.

The ordinary unchanged pure-entryv1/kernel138226549 completed/PASS in **156.64 seconds**, three genuine predictions, exact rehearsal CSV and raw visible probabilities; parent21 checksPASS. Receipt`76139867b8a0179fa9170b80f525736ec838973ef14a8f2bcd33a75cdb02d9e1`. Its version remains preserved and **must not be submitted**, because it predates the repaired policy.

Fresh rehearsal **v4/kernel138215646**, same `ozgung/knee-public-s6-rehearsal-20261011`, RUNNING20:49UTC after one repaired push. Immutable build `outputs/kaggle/public-s6-rehearsal-20261011-reviewed-v4/`:590333 bytes/230 sources; notebookSHA`2dd0b44ef2fa3e7eccce1855183f5620c0c5363ed2053383493d92726e0b18f4`, sourcebundle`aa5ebdb12f3fbc3c638bcc2591d5894a5d1cb8bd1201e2fc7ea657b5e43428ff`. Parent21 frozen-build checks,10 unchanged-science checks and16 actual accepted-source/private/offline/T4 checksPASS. Upstream readers, weights, recipe, all39 probes/sentinels, saved fingerprints and870 window distribution are unchanged. The v4 receipt additionally requires actual mirror17/native2 stress results. No full v4 PASS or S6 competition entry yet.

Actual20:47GPU42.2490/45h used,2.7510hremaining/reserved0; all prior notebook jobs terminal, no duplicate work. Next run the version-bound small collector and independent arrays/resources/ranks/history/policy audit at completion; then a fresh repaired pure entry and visible parity, actual daily allowance, one qualified submission. Do not use the oldv3 receipt to qualify v2-wrapper bytes. Integrated repair worktree archive was rejected by its protected pinned-task/workspace guard; it remains recoverable, with no needed ignored files there.


## Repaired v4 result accepted — 21:34 UTC

All39 genuine two-arm predictions completed. Actual host mirror17/native2 stress checks, strict reload, model state, physical identities, historical parity and rounding PASS. Parent35 independent checks PASS, including availability inventory order and healthy-input parity with v3. The auditor initially expected post-stage binding in the pre-run configuration; it now verifies actual receipt inventory hashes and original metadata order. Production gates unchanged.

Actual runtime1431.48 seconds (23.86 minutes). The1300-study base projection is5.23h, conservative7.18h and largest sensitivity9.44h; these are not hidden-test bounds. Combined RSS9.11GB, free disk20.35GB. Receipt `f8c7ebd080e6a9bdf0089e68d4be75199e805d67f4fc72db0a361164b4b817d8`; CSV matches prior healthy-input run. Collector saved1.038MB across two pages; independent audit `outputs/analysis/public-s6-entry-20261011/parent-terminal-review-v4.json`. This establishes operational parity, not independent generalization or a public-score forecast.

Pure entry v2 / kernel138226549 launched once at21:36 UTC after15 build and16 accepted-source checks. Source96dbcea; immutable561581-byte build `outputs/kaggle/public-s6-entry-20261011-reviewed-v2/`; notebook SHA256 `c4f6ec7e5bba475da30a7eb61b349656d85ad35ecf7108485b752570bfd757d3`, exact v4 scientific bundle, mechanics fixtures removed. At21:35 GPU allowance was2.3491h remaining with zero reserved; Colab empty and no local controllers or active agents.


**2026-10-11 21:40 UTC, Sol parent → Fable — repaired S6 submitted once.** Pure entry v2 / kernel 138226549 completed in 152.48 seconds. Parent verified all three genuine studies, unchanged model state, physical device identities, four-reader logits and exact v4 visible CSV/raw probabilities: 21 PASS, no failures. Receipt a4fd5918…1e06b; CSV 04d8f0bb…e1e0. Fresh allowance showed four remaining; exactly one authorized submission accepted as **57086107**, PENDING at 21:40:07 UTC, with three remaining today. Do not overwrite the submitted notebook, duplicate the job or infer a score from exposed development metrics. Parent owns hidden result/error reconciliation; no new training. Best public remains 0.950 pending this result. STATUS records exact recovery IDs and evidence.
