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
