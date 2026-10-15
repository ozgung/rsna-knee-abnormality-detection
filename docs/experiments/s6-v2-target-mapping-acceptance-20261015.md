# S6-v2 target mapping acceptance

October 15, 2026, 19:29 UTC. **Target CSV mapping and selected trainer snippets PASS; no complete VM/cache/training qualification.** Claude owns the main-lane implementation, Colab and commits. Codex performed a read-only independent replay after reporting split, prior-fit, quarantine and exposure issues in the draft.

The target schema/order comes from the real official submission header. All four CSVs cover the official 4407 unique studies in exact order: 4345 eligible non-gold/non-quarantine rows, 58 gold rows with blank targets and is_gold=1, and four quarantined rows blanked with review_holdout=1. All strict targets are blank; all eligible E_prior targets are finite probabilities. Recomputed each of 52,140 eligible target cells per file from the source tables, without calling the producer's mapping functions. J0 averages available labels_v1_blend/JEV values; V2 uses the declared own224-w30 table. All values match at the declared six-decimal serialization.

Fold-0 imputation statistics use only 3475 eligible non-fold0 rows; full-data statistics use 4345. No cell actually needs prior fill in either arm. Consequently fold-0 and all-data CSVs have identical values/hashes despite their different prior-fit receipts. Before native-cache filtering, fold0 has 870 eligible studies; actual VM cache/train/dev membership still needs its own proof.

| CSVs | SHA256 |
| --- | --- |
| targets_J0_f0.csv / targets_J0_all.csv | `0345cf239352a340b2c70470d396c43b6497ae06a50d719efb666056a34f5e8b` |
| targets_V2_f0.csv / targets_V2_all.csv | `e2283884e3b445eba0c62aa3f011811aef3e9fd1d3732259fb80de9e75ac0377` |

[Independent CSV replay](../../outputs/analysis/s6v2-acceptance-20261015/target-csv-replay.json), [reproducible audit](../../outputs/analysis/s6v2-acceptance-20261015/audit_targets.py). Input tables and folds are SHA-bound to the producer receipts; identity/quarantine manifest and fold file match their existing pinned hashes. No raw identifiers or report text in this note.

The published local trainer **7dd52945700f1c63d76a183ee96e0a747e746e12593abe7399316de576787431** is reproduced byte-for-byte by the reviewed patch **8088c66553fbe5d0a2112de500df8a7f775371b04071c6768ae0de0a4e2f2f72** over pinned heliosli source **14767c7a8f0daa986477511fe9a83aefe032a208ae978fabf15a5f896992ac80**. Source ae18d2e1 is the unavailable nartaa wrapper, not the executed trainer; no identity transfer is claimed.

Executed the actual patched AST snippets on synthetic records/tensors: pool construction excludes gold and quarantined rows; fold assignments consume CSV values rather than SHA1; E_prior consumes dense targets with unit cell weights; model bias initialization uses the fold's training indices only. [Trainer consumption proof](../../outputs/analysis/s6v2-acceptance-20261015/trainer-target-consumption.json). This does not run the full model or VM environment. Require the actual CLI/config to select this trainer and E_prior, exact memberships after cache filtering, matching arms/seed/config, and immutable artifact receipts before interpreting a fit. Partial-fold exports are not full OOF.

Gold-58 selected V2 among about ten label variants and remains selection-exposed. Its paired delta/bootstrap and promotion threshold are development heuristics. Multifold OOF teacher/calibration lineage can also expose the secondary report-fold comparison; do not describe that as independent validation. Gold exclusion from gradient training does not erase selection exposure. A label mapping pass is not a model gain or competitive result.

Owner-reported Colab setup: session s6v2a, config ~/.config/colab-cli/sessions-s6v2.json, A100, /content/setup.log, as recorded in outputs/s6v2/STATUS.md. Codex has not independently verified VM liveness, cache completeness or a fit; no duplicate execution/collection.
