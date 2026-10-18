# Next direction for the last nine days — 2026-10-13, about 15:00 UTC

Author: Fable (Claude), strategy review. Read-only: no compute, submission, commit, STATUS or handoff-log edit. Leaderboard and notebook sources were pulled with the Kaggle CLI into ignored `outputs/analysis/next-direction-20261013/` and not executed. Labels: [M] measured, [P] public claim by another participant, [E] my estimate.

## 1. The landscape moved under us today

Public leaderboard at about 14:50 UTC, 5,921 teams [M]: rank 1 = 0.966; rank 10 = 0.963 (16 teams at 0.963 or above); 51 teams at 0.960 or above; 141 at 0.957; **637 at 0.956**; 1,206 at 0.954. Our 0.954 now sits behind roughly 700 to 1,200 teams. Yesterday the crowd was at 0.954.

Cause: heliosli released **R7** (`heliosli/rsna-knee-compact-r7cnxfsss`, files dated Oct 11 19:59 UTC, manifest built Oct 12 03:51). heliosli's notebook "0.957 latest dataset score in one click" runs the public Raptor arm (nartaa r384 SWA, 140 mm, 320 crop, **K94, L/R mirror**, i.e. our original320 recipe exactly) plus R7's four folds at fixed **0.5/0.5** per-finding percentile ranks (OA findings also 0.5), with an 8.3 h deadline guard. Forks score 0.956 [P]; one fork submitted 40/60, 50/50 and 60/40 and all three scored 0.956 (submissions 57116807, 57119321, 57119457) [P], so the blend ratio is flat at display resolution.

R7 against S6 from the two manifests [M]: same arm (`E_lrprior`), same backbone (convnext_small), same size 384 / crop 352, four folds, same anchor weights (`5904dd16…`), same `side_cls.pt`, same views v4 and 3.5 mm physical spacing. New: `fluid_stacker.json`, `state_stacker.json`, `compile: true`, `flip_tta: false`, a `compact_infer.py` source, "fastMRI auxiliary training" [P]. Runtime should therefore match S6 (4.73 h of our 5.97 h base) [E]. It is a drop-in for our S6 slot once the inference code and stackers are carried over.

Second public lever, LB-tested by another participant [P]: nartaa's dataset holds an unused second checkpoint `…_c08_r224_swa.pt` (224 px from a 0.8 centre crop, 0.945 alone). Mixed at 35% of the Raptor slot it moved rank 437 to 320 inside the 0.954 block and 233 to 156 inside the 0.956 block. Inside a block of about 500 teams spanning 0.001, that is roughly +0.0002 real [E]: sub-display, but ties are broken by unrounded score and then by submission time.

## 2. Gap arithmetic

From the new base 0.957 to the tenth row 0.963 is +0.006. With the blend tracking the strong arm at slope about 0.6, that needs Raptor-class +0.010 alone, or a third arm far above the 0.93 class (a 0.93 arm adds at most about +0.002). No lever in hand delivers +0.006 in nine days. Realistic target: **0.957 very likely, 0.958 plausible, 0.960 perhaps one chance in four, 0.963 under one in twenty** [E]. The campaign should bank 0.957 fast (block position depends on submission time), then stack the small, cheap, LB-judged increments.

Budget: five submissions/day, about 7 h to score, so two or three informative probes per day at most; last new candidate Oct 20; hidden limit 9 h on 2×T4; Kaggle GPU about 16 h until Oct 17 (a rehearsal costs about 0.5 h); Colab A100 is not the constraint.

## 3. Ranked workstreams (expected public gain per wall-clock day)

| # | Workstream | Expected gain | Wall-clock | Owner |
| --- | --- | --- | --- | --- |
| 1 | R7 replaces S6 in the capped-recovery entry (A) | +0.002 to +0.003 displayed | 1 day | Codex builds, Claude reviews |
| 2 | nartaa 224 checkpoint at 35% of the Raptor slot (A) | +0.0002 to +0.001, mostly sub-display | 0.5 day after WS1 scores | Codex |
| 3 | B refresh: Dread v4_full + R7, no OAI | B 0.940 to about 0.943–0.945 | 0.5 day, same code as WS1 | Codex |
| 4 | goodpjw 2.5D reader as bounded per-finding residuals (A) | 0 to +0.002 | 1.5 days, runtime risk | Claude designs rule, Codex packages |
| 5 | dr-full as a B third arm, gated on dr-orig | B +0.001 to +0.003; A about 0 | 1.5 days (3.7 h A100 fit + export + entry) | Claude |

**WS1 — R7 swap.** Hypothesis: the 0.957 public recipe with our recovery wrapper scores at least 0.956, probably 0.957, because the Raptor arm is byte-equivalent to ours and R7 is S6-shaped. Cheapest decisive test: visible-3 output parity against heliosli's notebook output, the existing 39-study dual-T4 rehearsal (about 0.5 Kaggle GPU h), then one submission. Fixed-870 is useless here: R7 trained on all labelled studies. Compute: one rehearsal, one hidden run. Kill: displayed below 0.956 or a hidden failure; fall back to 57096002. Carry heliosli's `compact_infer.py`, both stacker JSONs and the deadline guard unchanged; keep the recovery policy identical so WS1 isolates the arm swap. Submit by about Oct 14 12:00 UTC: earlier entries rank first inside the block.

**WS2 — 224 checkpoint.** Hypothesis: a same-family model at a different field of view adds rank diversity to the Raptor slot. Test: LB only, as a second submission after WS1 has scored, with `A224 = 0.35` fixed in advance (the published value) and the stage guarded by the clock. Runtime about +0.4 h [E] at 224 px. Kill: displayed drop against WS1. OAI-trained, so A only.

**WS3 — B refresh.** Same build as WS1 with Dread v4_full K62 in the Raptor slot, as in B0. One submission. Flag before freezing B: R7 lists fastMRI auxiliary training; fastMRI is a free individual data-use agreement, which reads as compatible with the host's "no institutional review, free of charge" wording, but the dataset licence shows "other". Verify both before Oct 21; if judged risky, B falls back to Dread + S6 (0.940, submission 57119585).

**WS4 — reader residuals.** Hypothesis: a different family (ConvNeXt-Tiny 2.5D, 0.929 alone) helps where goodpjw's own per-finding weights are large: Baker's 0.35, Contusion 0.30, then Medial OA and ACL 0.25; flat 30% hurt (0.948) [P]. Rule fixed in advance from that table, no LB iteration. Test: a rehearsal that projects total runtime; then one submission after WS1/WS2 settle. Runtime adds about 1 h [E] on a 5.97 h base whose conservative projection is already 8.19 h, so it needs a heliosli-style deadline guard and may need K94 Raptor to drop to K42 (nartaa measured K94 to K24 at −0.006, so do not drop below 42). Kill: projection above 8.3 h, or no displayed gain.

**WS5 — dr-full.** dr-orig (Dread init, 64-slot, original head) ends about 17:10 UTC with an exposed fold-0 metric. Gate: K42 macro at least 0.935 with no target more than 0.01 below own44; then build `dr-full` on all 4,345 studies with pre-registered SWA epochs (dr-orig's selected epochs), about 1,100 s/epoch × 12 [E from 880 s measured on 3,475], and add it to B only, at a fixed 0.2 of the Dread slot. It is no-OAI, so it belongs to B. Expected A contribution is at most +0.001, so do not spend an A submission on it.

## 4. Stop doing

- **Raptor inference-geometry probes.** nartaa's own LB ablation covers them: K94 to K24 −0.006 (K94 already visits every position of the 96-slice volume, so nothing denser exists), 304/288 crops −0.001/−0.002, width-flip TTA −0.001, full-field 384 then 320 crop 0.948 to 0.948, mirror +0.001. Our fold-0 full384 +0.0026 was exposure, and full384 failed hidden execution twice. Expected gain at most +0.001; no submissions on this.
- **Fine-tuning Raptor from nartaa weights on our folds.** Our two-epoch adaptation lost 0.003 on exposed dev; nartaa's BN-replay retrain lost 0.004 on the LB. A converged all-data SWA model has no new information to learn from the same studies. Closed.
- **Blend-weight or per-target weight searches**, on exposed 870 or on the LB: three ratios gave identical 0.956. Use only fixed, pre-declared weights.
- **New own families as A candidates.** raptor64 in-orig is +0.001 and not significant; the plane head is null; relabelling, E4 stacking and source96 are closed. An own arm in the 0.91 to 0.92 public class adds under +0.001 to a 0.957 blend.
- **Native ROI joint crop.** v2 fell back on 100% of 384 slots; a per-series redesign, pair training, replication and five folds need about six days for at most +0.001 on A. Codex may continue it as a bounded learning lane, but it stops if no qualified reader exists by Oct 16, and no multi-fold expansion runs before Oct 17.
- **in-plane K42 recovery:** skip. A tight K16 null (±0.0044) cannot be overturned by K42, and in-orig itself is not significant.

## 5. Selection A/B

Keep two selections: **A = best LB** (OAI-trained Raptor stays legal until the host rules otherwise; the whole top of the board shares that exposure) and **B = no-OAI** (Dread + R7). Move both to R7. Spend about 80% of effort on A (WS1, WS2, WS4) and 20% on B (WS3, WS5). Do not add an own-model-heavy third selection; there are only two slots and no own arm earns one.

## 6. Calendar (UTC)

| Date | Action |
| --- | --- |
| Oct 13–14 | WS1 build, visible parity, rehearsal, submit by about 12:00 Oct 14. Claude: dr-orig K42 result, R7 stacker code review, WS4 residual rule. |
| Oct 14–15 | WS1 scores. Submit WS2 and WS3 (two hidden runs). Verify fastMRI/R7 licence wording. |
| Oct 15–17 | WS4 rehearsal with runtime projection; submit if under 8.3 h. dr-full fit if gated; B three-arm rehearsal. Keep about 10 Kaggle GPU h for rehearsals. |
| Oct 17–20 | Read results, at most two refinements per day, no weight iteration. Last new candidate Oct 20. |
| Oct 21–22 | Freeze. A = best LB; B = best no-OAI. Full-length retry only, starting by about 12:00 Oct 22. |

## 7. Limits

Public claims about R7, the 224 checkpoint and the 40/60 to 60/40 probes are other participants' submissions, not ours. The Kaggle discussion listing could not be refreshed through the CLI or web search this session; the forum view is the October 12 dump. Expected gains are estimates from blend slope and public block positions, not forecasts of the private score; the public/private split and site holdout remain unstated, so a shake-up can reorder everything above.

Evidence: `outputs/analysis/next-direction-20261013/lb/` (leaderboard CSV), `kernels/` (pulled notebook sources, unexecuted), `r7meta/` (R7 and S6 manifests).
