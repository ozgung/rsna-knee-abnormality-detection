# Main lane: S6 ConvNeXt retrained on labels v2 — protocol (pre-registered Oct 15, 2026, ~19:20 UTC)

Approved by the user as the main lane on Oct 15. The R1 local-view fine-tune and `na-orig` continue as secondary work.

## Why this lane
- **A top-10 team says labels are the lever.** azamat1ch, whose team is rank 9 at 0.965, wrote in thread 748295: "Short answer is labels", then "AI, not by hand. And don't rely on reports alone", and "only the competition data".
- **nartaa's public S6fall recipe reaches 0.950 from an ImageNet init.** Thread 749359 and the `nartaa/rsna-knee-s6fall-weights` run_config describe it: heliosli's `compact_train.py`, ConvNeXt-small with spatial attention and window MIL, v4 views at 320/288 px, 3 slices at 3.5 mm, JEV soft labels with prior fill, lr 1e-4 / head 1e-3, wd 0.05, 16 epochs, batch 2, 64 train windows, gold-58 excluded. It was trained on a laptop GPU.
- **Report-label scores don't track the board.** nartaa's fold-0 S6 scored 0.907 OOF against report labels and 0.948 public. Our own224 scored 0.902 and 0.914 public. Gold-58 does track the board: their S6 scores 0.931 and their CoAtNet 0.919.
- **Our labels v2** (`labels_v2_base_own224_w30`, see `labels-v2-20261015.md`) score 0.920 on gold-58, against 0.904 for the best public table (JEV hybrid) and 0.901 for `labels_v1_blend`.

## Design (one factor: the training targets)
| Arm | Targets | Everything else |
|---|---|---|
| **J0 (control, replication)** | nartaa's JEV-blend soft labels with prior fill and weight-normalised BCE, reconstructed as closely as the public assets allow | nartaa's S6fall run_config exactly, but with `only_folds 0` and `all_data false` (folds_v0), seed 42 |
| **V2 (test)** | labels v2 (`labels_v2_base_own224_w30`), mapped into the same targets schema with the same prior-fill and weighting rules | identical to J0 |

- **Gold-58 is excluded from training in both arms.** It is checked by ID before launch.
- **Fold 0 of `folds_v0`** is held out in both arms (about 855 studies).

## Endpoints
- **Primary:** gold-58 macro AUC (12 findings, gold labels), V2 − J0, with a paired study-level bootstrap (2,000 draws, seed 42) and per-finding Δ.
- **Secondary:** fold-0 OOF macro against report labels (descriptive only, since it is known not to track the LB), per-finding AUCs, and the Spearman correlation of V2 with J0 on fold 0.

## Pre-registered rules
1. **J0 sanity, which gates reading V2.** J0 must land within ±0.02 of nartaa's gold-58 0.931 and ±0.01 of their fold-0 OOF 0.907. Outside those bands, the pipeline does not reproduce theirs: debug before interpreting V2.
2. **Promote V2 to an all-data fit** if gold-58 Δ ≥ +0.005 and fold-0 OOF Δ ≥ −0.010. With n = 58, the macro SE is about 0.017, so this is a point-estimate rule and is stated as such.
3. **Kill V2 as a label source** if gold-58 Δ ≤ −0.010.
4. **All-data V2:** nartaa's all-data config (16 epochs, `no_final_predict`), then a leaderboard probe.
   - **Probe frame:** S6fall solo is public 0.950. Option (a) submits V2-all solo against 0.950 under tie-or-better. Option (b) puts V2-all in place of R7 in A′ after a fresh 2×T4 rehearsal; one S6 model costs about a quarter of R7. The frame is chosen when V2 is promoted, before any score is seen.
5. **Iterate the labels (v3):** V2's fold OOF predictions become the image teacher for labels v3, if time allows.

## Compute and owners
- **Colab A100 sessions:** Claude runs these. Units are not binding. Each run uses its own CLI config.
- **Pixel cache:** heliosli's `px_cache.py`, views v4 at size 320 with SKM-TEA anchors, built for the 4,263 pool studies plus gold-58. They're staged on the VM by range-reading the competition archive, the method used for the R224 Colab runs; the laptop never relays images.
- **Codex:** reviews the targets mapping and later builds the entry. It does not run Colab.
- **Timeline target:** fold-0 pair read out by Oct 16 ~18:00 UTC; all-data fit and probe by Oct 18; last hidden submission Oct 20 16:00 UTC.

## Limits
- Gold-58 is small, and nartaa selected their models on it, so it is selection-exposed for public weights. It is clean for our arms.
- J0 may not reproduce nartaa exactly, because the JEV reconstruction and code versions may differ.
- A label gain on gold-58 is not proven to be a leaderboard gain.
