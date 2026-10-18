# Strategy revision — 2026-10-17 (data as of 17:06 UTC)

Author: Fable (Claude), read-only research. No GPU, Kaggle push, submission, commit, STATUS or handoff edit; this file is the only repo output. Pulled with the Kaggle CLI into `~/.claude/jobs/74614526/tmp/strat-1017/` (`kernels/`, `forums/`, `lb/`, `efficiency/`, `ds-neronrx/`, `ds-dkim54/`); downloaded code was dumped to text and read, never executed. Repo evidence: r3 and A′ qualification receipts, visible-run receipts, labels docs, handoff log (cited inline). Labels: **[measured]** = a number in a receipt, CSV, API response or downloaded file; **[literature]** = another participant's claim; **[estimate]** = my judgement.

## Verdict

1. **Labels v2 did not produce a leaderboard-visible gain and cannot reach the final as a label result.** The table beat `labels_v1_blend` on gold-58 (+0.019, selection-exposed); the model trained on it beat its control by +0.006 [−0.007, +0.020] on gold-58 and tied it on the public board (0.950 = 0.950) [measured]. The best reading is "+0.000 to +0.001 on an S6-class model", below the board's resolution. Top teams confirm the *direction* (image pseudo-labels, "not reports alone": azamat1ch rank 2, colum2131 rank 39 [literature]) but they iterate it over strong single models; heliosli's R6 targets are already a sixth pseudo-label round (`plr6`) [measured, manifest]. Our one-round table cannot out-label R6 in three days. **No label work on the critical path.** One optional background bet is listed in §1 with a kill rule; it has no slot before Oct 19.
2. **A‴ did not time out because na-full is slow.** On the 39-study rehearsal and on the three visible studies, na-full K62 one-view costs *less* per study than the R384 arm it replaced (5.13 vs 5.82 s; 8.2 vs 9.7 s on the big visible study) and R7/R224 were identical; the conservative projection was 8.32 h against A′'s 8.59 h, and A′ finished in ≤ 7.6 h [measured]. The hidden run still exceeded 9 h, so it was ≥ 18 % slower than A′ on a strictly smaller workload. That points to a stall or a slow hidden host, not to windows. **The fix is structural: a deadline guard with per-study fallback, so a run can never be killed.** Fewer windows, dropping R224 or a prefetch worker only add headroom. na-full itself is worth keeping only as a *small unique member* inside the R6-flip build (§2, §4), not as A‴.
3. **Nothing public is above 0.958.** The 0.958 block grew to 865 teams (ranks 143–1007); bronze (top 10 % ≈ 614) sits inside it, so a verbatim copy submitted now lands at about rank 1007 and the private board sorts exact copies by time. goodpjw's new version 8 adds two measured negatives: gold-58-fitted per-finding weights score 0.957 and 0.955 against heliosli's 0.958, and S6 next to R6 costs 0.001 [literature]. A new public R6 fine-tune on pseudo-labels (neronrx, team at rank 970, 0.958) did not lift its author above the block [measured, LB]. New threads are external-data pressure on the hosts, unanswered. Details in §3.
4. **Plan: parallel, paired probes around one build.** Today: A′-w (weights only, in progress). Oct 18: our R6-flip controller as *one package with a configurable na-full arm*, submitted as a pair, R6F-pure (na-full weight 0) and R6F+ (na-full 0.2 of the Raptor half), so the pair measures na-full's marginal effect and R6F+ breaks the exact tie with the public copies. Oct 19: two weight variants of the same package (heliosli's V2 weights; R224 → na-full swap). Oct 20: one confirmation slot. Expected: 0.958 for the R6F family (P ≈ 0.6–0.85 by variant), 0.957 for A′-w. Schedule and owners in §4.
5. **One decision is needed before the R6F rehearsal:** how the 8.3 h projection gate applies to a deadline-guarded compact stage. Proposal in §4.3: unguarded stages keep the conservative projection (≤ 3.5 h), the guarded stage is bounded by its deadline by construction, and the receipt must show ≥ 97 % projected compact coverage. Without this, every R6F variant fails the 8.0 h waiver rule on paper while being safer than A′ in practice.

## 0. Board at 17:06 UTC (6,145 teams) [measured, `lb/*17T17:06:52.csv`]

| Display | Teams | Ranks | | Display | Teams | Ranks |
|---|---|---|---|---|---|---|
| 0.969 | 1 | 1 | | 0.960 | 29 | 66–94 |
| 0.968 | 3 | 2–4 | | 0.959 | 48 | 95–142 |
| 0.966–0.967 | 5 | 5–9 | | **0.958** | **865** | **143–1007** |
| 0.965 | 3 | 10–12 | | 0.957 | 224 | 1008–1231 |
| 0.961–0.964 | 53 | 13–65 | | **0.956 (us, 1335)** | 444 | 1232–1675 |

Since 09:45 the 0.958 block gained 112 teams and we slid from 1244 to 1335 without a new score. Bronze needs rank ≤ 614: a 0.959 display is safe, a 0.958 display is a coin flip decided by the fourth decimal, and an exact copy of the public recipe is behind every earlier copy. The private board applies the same tie rule, so **uniqueness of the prediction vector is worth about P(bronze) 0 → 0.5 at equal expected score** [estimate]. Any non-zero weight on a member nobody else has (na-full, V2-S6) breaks the tie; the member's quality then decides which side of the block we land on.

Submissions: 27 total, one today (A‴ 00:25 UTC), so 4 slots remain today, 5 on each of Oct 18–20 [measured, `kaggle competitions submissions`]. Hidden walls so far: A′ ≤ 7.65 h (11:54 → ≤ 19:34), B0′ ≤ 7.2 h, solo pair ≤ 2.35 h, A‴ killed at 9 h [measured, submission and snapshot times].

## 1. Labels verdict

### 1.1 Evidence

| Layer | Comparison | Result | Source |
|---|---|---|---|
| Label table | `labels_v2_base_own224_w30` vs `labels_v1_blend`, gold-58 | 0.920 vs 0.901, +0.019 [+0.005, +0.034]; chosen among ≈ 10 variants on gold-58 (selection-exposed); predeclared variant +0.014 [−0.001, +0.028] | `labels-v2-20261015.md` [measured] |
| Fold-0 model | V2 vs J0 (same trainer, folds_v0), gold-58 stride 1 | 0.9294 vs 0.9230, +0.0064 [−0.0067, +0.0197], P(Δ>0) 0.83; OOF +0.007 (report reference, teacher-contaminated); strict verdict BLOCKED, post-hoc override | `s6-labels-v2-main-lane-result-20261015.md` [measured] |
| Public board | V2-all solo vs nartaa S6fall solo, identical wrapper | **0.950 vs 0.950** (57285381 / 57285382) | STATUS [measured] |
| Fine-tune, not labels | B0′ (dr-full, Dread-init fine-tune, `labels_v1_blend` + Dread teacher) vs B0 | 0.944 vs 0.940 | STATUS [measured] |
| Public peers | neronrx "hive": heliosli R6 fold weights fine-tuned 14 epochs on `labels_pseudo_v1`, fold-0 val macro 0.921 (870 studies); team at rank 970, 0.958 | own-label R6 fine-tune did not leave the block | `ds-neronrx/log_fold0.txt`, LB [measured] |
| Public peers | heliosli R6 manifest targets `probe_targets_opusv11b_rr6_plr6_cdr.csv` | a sixth pseudo-label round already feeds the public compact model | R6 manifest [measured] |
| Forum | azamat1ch (rank 2): "Short answer is labels… AI, not by hand. And don't rely on reports alone"; colum2131 (rank 39): "pseudo-labels are generated from the images, with the image model taking the same input as at submission time" | direction confirmed, magnitude not | 748295, 735304 [literature] |

Reading: the two displayed scores tie at three decimals, so |Δ| < 0.001 on about 390 public studies (if 30 %) [estimate]; the gold-58 +0.006 is inside that band once the 0.017 macro SE is considered. The B0′ gain came from fine-tuning a weak member, not from labels v2. The label gain is real on paper, invisible on the board, and bounded for S6-class models by what the compact slot can contribute: goodpjw measured S6 next to R6 at −0.001 (v4) and nothing (R7 next to R6, v26) [literature]. A better S6 therefore has no place in a 0.958-class blend; only a better *R6-class* model would matter, and R6 already uses six teacher rounds.

### 1.2 Options, value and time [estimate unless noted]

| Option | What it needs | Wall time from a 20:00 UTC start | Expected effect | Verdict |
|---|---|---|---|---|
| (a) Iterated image teacher (noisy student round 2) | 4 V2 fold fits for a full OOF teacher (cache restore 7 min + ≈ 3 h each, parallel A100 sessions), relabel on CPU (1 h), S6 all-data fit 3.3 h (11,984 s measured), export + strict CPU check 1 h, solo entry rebuild + rehearsal + visible 3 h | ≈ 12 h → earliest hidden score Oct 18 ≈ 22:00 UTC | S6 solo +0.000 to +0.001; blend value nil next to R6 (goodpjw v4) | only as the "all-ours compact slot" hedge (§4, P7); background, no slot before Oct 19 |
| (b) Ensemble teacher from the public models' train-set predictions | 20–26 h of T4 (R384/R224/R7 over 4,407 studies), new teacher mode | not before Oct 19 | in-sample echo (R7 trained on all non-gold studies; Dread echo measured 0.95–0.99 agreement) unless R7's fold split is reconstructable; `compact_train.py` uses `sha1 % folds`, so a fold-restricted teacher is possible but needs the worker changes of §6 of the labels doc | no |
| (c) Silent-cell handling | CPU only | 2 h | +0.013 on the label table for Fracture/PF OA, gold-derived and circular; cannot be validated in time | no |
| (d) LLM re-extraction "not from reports alone" | new prompts with image evidence = (a) with a VLM teacher; or adopt a new public table (`omerzalman/rsna-knee-11-label-evidence-fusion`, 60 % coverage, "97.13 % accuracy on the expert dataset" = gold-verified, exposed, 11 labels without Synovitis; `haungjiawei/rsna-knee-train-v2-code-and-labels`; `kritikabenjwal/rsna-knee-labels-soft-v2`) [measured, dataset listings] | a fit per table | untested; the authors sit at 0.728, 0.891 and unknown on the board | no |

**Recommendation.** Treat gold-58 as spent and labels v2 as a documented, non-promotable result. The only label-adjacent asset that still matters this week is the V2-all checkpoint as a *unique S6-class member* (0.950, qualified, entry exists) for the all-ours hedge. If a Claude lane and Colab are otherwise idle, (a) may run in the background with this kill rule: stop unless the fold-0 round-2 model beats V2's fold-0 OOF (0.9180 against `labels_v1_blend` ≥ 0.5, stride 1) by ≥ +0.005 on the same reference by Oct 18 20:00 UTC.

## 2. A‴: why it timed out and how to make a na-full entry submittable

### 2.1 Measured costs: the workload was not larger than A′'s [measured]

| Arm (slower board) | A′ v2 qualification (39 studies) | A‴ r3 (39 studies) | Visible 3 studies, A′ → A‴ |
|---|---|---|---|
| Old arm production per study | R384 two-view 5.82 s (116.3 s / 20) | **na-full K62 one-view 5.13 s** (103.5 s / 20; includes the per-study state digest) | 9.65 / 5.42 / 9.66 s → **8.20 / 4.85 / 8.21 s** |
| Old arm fixed overhead per board | 504.6 s | 302.1 s | — |
| R7 production per study | 23.95 s (479.1 s / 20) | 23.6 s (471.6 s / 20) | identical studies, 302–320 s first study (297 s compile check) |
| R224 production per study | 3.33 s | 3.38 s | 6.8 / 3.3 / 6.5 → 7.0 / 3.4 / 6.7 s |
| Conservative projection | **8.594 h** (30,936.8 s) | **8.318 h** (29,943.5 s) | — |
| Hidden wall | **≤ 7.65 h, COMPLETE 0.956** | **> 9 h, killed** (GetSubmission: "exceeded the allowed runtime") | — |

Sources: `outputs/kaggle/ws2-r224-v2-qual-20261015-result/ws2-a-prime/**/receipt.json`, `outputs/kaggle/nafull-aprime-rehearsal-r3-20261016-result/nafull-aprime/**/receipt.json`, both `-probe-waiver-*-visible/` trees, `official-submission-detail-20261017-1227.json`. `fp32_recoveries` is 0 in every receipt; the entry-mode na-full records carry no reference pass (`production 8.20 = seconds 7.02 + state digest`), so no rehearsal-only work ran per study.

### 2.2 Diagnosis [estimate, ranked]

1. **A stall, not slowness.** Our pipeline has no per-study, per-arm or overall timeout and no fallback; a worker stuck on one study (DICOM decode, a CUDA/inductor subprocess, a `multiprocessing` join) runs until Kaggle's 9 h kill, and Kaggle reports that as "exceeded the allowed runtime". A‴ differs from A′ only in the old-arm worker and its reader (the B0 raptor64 reader, `public_s6_recovery_b0`), which completed the same hidden set twice (B0 0.940, B0′ 0.944), so a reproducible per-study fault is unlikely; a non-reproducible hang is the simplest explanation of "≥ 18 % slower on a smaller workload".
2. **Hidden-host variance.** Session speed moved +9 % between B0 and B0′ rehearsals; actual/conservative ratios of completed runs range 0.77–0.89. A‴ needed > 1.08. A host 20 % slower than A′'s would do it alone; unusual, not impossible. A′'s own margin to 9 h was only 1.4 h.
3. **Projection blind spots** (shared with A′, so they explain the thin margin, not the difference): R7's per-study state digest (≈ 2.5 s, 0.45 h per board) is counted once; ledger I/O grows over 650 rows; the three visible studies cost 1.6–1.7× the rehearsal mean in every arm, and the error text says the hidden set "can be larger/different", so the 1.25 window multiplier may be thin if the hidden set resembles the visible test.

Without hidden receipts none of this can be proven; what *is* proven is that na-full's per-study cost is not the cause.

### 2.3 Fixes, with runtime and score [estimate from measured inputs]

| Fix | Runtime effect | Output effect | Effort | Note |
|---|---|---|---|---|
| **F1 Deadline guard + per-study fallback** (heliosli's pattern: compact stage `--deadline T0 + 8.3 h`, unscored studies take the Raptor-half rank; Raptor arms with 2 h caps; never abort, always write) | bounds wall at ≈ 8.45 h by construction | 0 when nothing is skipped; ≈ −0.0005 per 10 % of studies skipped in the compact stage | compact_infer/probe_dual already implement it; our controller must keep it and add it to our own arms (per-study watchdog → `data_local_failure` under the 3 % cap) | **mandatory for every new entry**; converts a lost slot into a slightly degraded score |
| F2 Run the Raptor arm concurrently with R7 on each board | hides ≈ 0.9 h (R7 is 48 % CPU-bound) | 0 (same per-study computation) | medium; CUDA 6.2 + 5.9 GB per board fits a 15 GB T4; host RSS +2.3 GB per board | needs a paired rehearsal to measure contention |
| F3 Ledger-light R7 (skip per-study input hashing 3.0 s and state digest 2.5 s) | −0.8 to −1.0 h real | 0 | ≈ 20 lines | weakens receipts; acceptable for an entry, not for a rehearsal |
| F4 K42 instead of K62 | −0.2 h | ≈ 0 (Dread K42 vs K62 was within noise) | config | small |
| F5 Drop R224 | −0.6 h point, −0.71 h conservative | −0.0003 to 0 (A → A′ showed no displayed change) | config | best trade if headroom is needed |
| F6 Lane 3b prefetch worker | −1.1 to −1.5 h conservative on paper | 0 by construction | P3 measured 0.888 against the ≤ 0.85 criterion → "output-preserving, no gain" | not a reliable lever this week |
| F7 Run na-full inside a compact_infer-style controller | n/a | n/a | na-full is a Raptor-architecture model with its own reader; it cannot use compact_infer. F2 is the equivalent idea | — |

**Is na-full worth it?** As A‴ (na-full replacing R384): expected 0.955–0.957, P(≥ 0.958) ≈ 0.1; its solo quality is unmeasured (na-orig fold-0 0.941 is exposed; the one-view R384 lineage is 0.945–0.947 [literature]). Not worth a two-day repair. As a **0.1–0.2 weight unique member in the R6-flip build**: expected effect −0.0003 to +0.0003 on the display, zero on runtime beyond 1.0 h (650 × 5.13 s + 302 s per board), and it breaks the exact tie with ≈ 865 copies. That is the use that pays. The fine-tune evidence (B0′ +0.004) came from a weak base; na-full's base is already strong, so expect diversity, not lift.

## 3. Public landscape, last 24 h [measured source; scores literature]

| Item | What it is | Against blend-mine | Relevance |
|---|---|---|---|
| `goodpjw2008/rsna-knee-does-s6-help-next-to-r6-lb-0-957` **v8** (Oct 17 14:17) | v6 = heliosli weights with Fracture 0.8 / Effusion 0.7 / ACL 0.3 → **0.957**; v7 = every finding at its gold-58 optimum, clipped 0.1–0.9 → **0.955**; v4 = S6 at 0.3 of the compact slot → 0.957 | two measured negatives: do not tune per-finding weights on gold-58; do not add S6 next to R6 | confirms heliosli's weights are public-LB-tuned ("chosen on much more data"); our V2-S6 adds value only as a tie-breaker, never at 0.3 |
| `heliosli/…-0-9586` (04:21, 137 votes) | unchanged since the morning scan: V2 weights {Lateral OA .15, PF OA .65, Synovitis .85, Baker's .30} + goodpjw reader at 1/10 | +0.0006 claimed | the cheapest public +0.0004; private transfer unknown |
| `kaiwalyaatulraut/rsna-knee-competition-solution` (12:38, 13 votes) | goodpjw 0.958 recipe (rostigor R6, flip, A224 .35) | copy | none |
| `kiyoshiohno/rsna-knee-template` (10:36, 45 votes; author kiyoc at rank 899, 0.958) | thin launcher that executes a pipeline held in a versioned dataset (`rsna_knee_manifest.json`, `rsna_knee_pipeline.py`); attaches R7, R6-w1, nartaa, orthofoundation | copy with a different packaging | none |
| `lavinwins/top-1-gold` (rewritten 12:58) | now a self-contained "train your own 5-fold model on T4" notebook ("rearranging the public models' predictions is exhausted") | pivot away from blending | signal that the block is saturating |
| `prvsiyan/the-bee-s-knees-final-rsna-push` (108 votes) | readable port of nartaa's mirror reader, 0.949 solo | below | none |
| `ryanholbrook/rsna-knee-abnormalities-efficiency-lb` (Kaggle staff, 597 votes, refreshed Oct 16 21:22) | Efficiency Track leaderboard (prizes $7k/$6k/$5k per the rules). Top 3 are overall ranks 10, 2-class and 1; we are **efficiency rank 4804** with the selected A (0.956, 7 h) | separate track | not reachable with a 7–8 h blend; a 1 h solo S6 would sit around rank 100–400 (0.950 entries: best 54, median 433) — no medal value, ignore |
| `neronrx/mycelis-rsna-knee-4fold-hive` (dataset, 08:53, 778 MB, 252 downloads) | 4 ConvNeXt-small 384 folds initialised from heliosli's R6 `E_lrprior_s0_f*.pt`, 14 epochs on `labels_pseudo_v1`, k=8/12 windows, fold-0 val 0.921 with OOF logits for 928 studies; the log's "gold" column (0.58–0.67) looks like a broken evaluation | author's team at rank 970, 0.958 | an R6 fine-tune on pseudo-labels that did not move the block; its OOF files are a possible extra teacher column, untested, licence unknown |
| `dkim54/rsna-knee-models-v11i` (945 MB) | own 5-fold ConvNeXt-small 384 with TTA; author at rank 1916, 0.952 | below | none |
| Discussions 750236, 750257–750260 (Oct 17 11:59–13:42) | "what will be checked, and for whom" on external data; four yes/no questions on nartaa OAI weights, R7 fastMRI, SKM-TEA anatomy, RadImageNet; all at −4 to −22 votes, no host reply | — | provenance pressure is rising but the hosts have not moved since Oct 14; Oct 20 recheck stands |
| 749570 | the top public notebook is inference-only; the public board (≈ 135 teams per 0.001) cannot validate per-label weights | — | supports treating every per-finding weight as unvalidated |
| 750295 | omerzalman's label dataset (above) | — | none now |
| 748295 (37 comments) | the labels hints quoted in §1 | — | direction only |

Nothing new claims ≥ 0.959 publicly; the only ≥ 0.958 teams with public code remain tom99763's (0.960, rank 77, private weights). Top of the board: rank 1 0.969 (also efficiency rank 3), azamatt 0.968, colum2131 0.962 — all with own single models and image pseudo-labels [literature, LB].

## 4. Plan, Oct 17–20

### 4.1 Compositions

All R6F variants are **one package** (worktree `.claude/worktrees/r6-controller-20261017`, frame `knee/public_r6flip.py`, profile `knee/public_r6_profile.py`), with the na-full arm and the weight map as sealed config, so one rehearsal qualifies the family. Stage order fixed for robustness: setup → R384 two-view (our old320 worker) → na-full K62 one-view (A‴'s worker, weight may be 0 → arm skipped) → R224 one-view (crop224 worker) → compact R6 flip (heliosli `compact_infer` v4c `--flip-tta --loader-workers 2`, `--deadline T0 + 8.3 h`, per-study Raptor fallback, never abort) → per-finding percentile-rank blend → CSV. Raptor half with na-full weight w: `(1−w)·(0.65 rank R384 + 0.35 rank R224) + w·rank(na-full)`.

| Id | Composition | Expected display | P(≥ 0.958) | Unique? | Runtime (point → actual ×1.17; guard bounds the wall) |
|---|---|---|---|---|---|
| **P1 A′-w** | A′ (57229227) with Raptor-half weights {Lateral OA .2, PF OA .6, Synovitis .8, Baker's .35}, all else 0.5; same members, sources, runtime | 0.957 | 0.15 (P(≥ 0.957) ≈ 0.6) | yes (our ports of R7/R384/R224) | identical to A′: 6.5 → 7.6 h; **no guard**, P(timeout) ≈ 0.1–0.15 after A‴ |
| **P2 R6F-pure** | public recipe in our controller: R384 .65 + R224 .35, R6 flip, heliosli v1 weights, w = 0 | 0.958 | 0.8–0.85 (every known runner shows 0.958) | **no** — expected exact tie with ≈ 865 copies | 0.1 + 1.05 + 0.6 + 4.2 (3.8–4.6) ≈ 6.0 → 7.0 h |
| **P3 R6F+** | P2 with na-full at w = 0.2 (R384 .52, R224 .28, na-full .20) | 0.958 | 0.6–0.7 | yes | + 1.0 h → 7.0 → 8.2 h; compact stage has ≈ 5.5 h before the guard against a 4.9 h need (×1.17) — expect ≤ 3 % fallback on a slow host |
| **P4 R6F+V2w** | P3 with heliosli's V2 weights {Lateral OA .15, PF OA .65, Synovitis .85, Baker's .30} | 0.958–0.959 | 0.7 (P(0.959) ≈ 0.2) | yes | as P3 |
| **P5 R6F-swap** | R224 dropped, na-full at .35: Raptor half .65 R384 + .35 na-full | 0.958 | 0.55 | yes, most decorrelated | 0.1 + 1.05 + 1.0 + 4.2 ≈ 6.4 → 7.5 h |
| P6 na-full solo | K62 one-view alone (B0 worker), information only | 0.945–0.950 | 0 | — | ≈ 1.3 h; needs a solo wrapper (Codex 0.5 day); **optional** — the P2/P3 pair already measures the marginal effect that matters |
| P7 all-ours compact slot (hedge) | Raptor half of P5 + V2-all S6 with `--flip-tta` instead of R6 | 0.956–0.957 | 0.2 | yes, no R6/R7 dependence | 0.1 + 1.05 + 1.0 + ≈ 2.7 ≈ 4.9 → 5.7 h; only if the provenance recheck on Oct 20 makes heliosli's weights a liability |

P(≥ 0.958) for P3–P5 is P2's probability minus the risk that na-full's rank noise crosses the 0.9575 rounding edge [estimate]. Per-finding weights (v1 and V2) are public-LB-tuned; goodpjw's v6/v7 show that moving them costs 0.001–0.003, so no weight invention of our own.

### 4.2 Schedule (UTC; Codex integrates and accepts, Claude builds controller/GPU and submits)

| When | Action | Owner | Slot | Result expected |
|---|---|---|---|---|
| Oct 17 20:00–23:00 | **P1 A′-w**: sealed entry, visible run, submit | Claude (build in `.claude/worktrees/aprime-w-20261017`), Codex targeted acceptance | 1 of 4 today | score Oct 18 04:00–06:00 |
| Oct 17 evening → Oct 18 12:00 | **R6F package**: controller around `compact_infer`/`probe_dual` with observer hooks, na-full arm from the A‴ packet, blend from `public_r6flip.py`, deadline/fallback semantics, synthetic/CPU tests; Codex: strict asset/CPU qualification of R6 weights (already staged under `outputs/research/public-0958-20261017/r6-v1-assets/`), gate decision of §4.3 | Claude + Codex | — | package ready |
| Oct 18 12:00–16:00 | one 39-study 2×T4 rehearsal with na-full at 0.2 (covers P2–P5), collection, acceptance; entry builds for P2 and P3 | Claude GPU (≈ 0.7 h quota), Codex acceptance | — | receipts, projection split (§4.3) |
| Oct 18 16:00–20:00 | visible runs, **submit P2 and P3 together** | Claude | 2 | scores Oct 19 00:00–04:00 |
| Oct 19 06:00–12:00 | read P2/P3: if P3 ≥ P2 keep na-full; **submit P4 and P5** (config variants, visible runs only) | Claude, Codex acceptance | 2 | scores Oct 19 14:00–20:00 |
| Oct 19 (parallel, optional) | P6 na-full solo if Codex has time; P7 hedge build only if the provenance signal changes | Codex build, Claude run | 0–1 | — |
| Oct 20 ≤ 16:00 | provenance recheck; one confirmation slot: the best variant with the unused knob (e.g. P4 weights + P5 swap, or P7) | Claude | 1 | score Oct 20 night |
| Oct 21 | final selection (§4.4) | user | — | — |

Quota: rehearsal 0.7 h + visible runs ≈ 0.15 h each + optional P6 ≈ 0.5 h → ≈ 3 h of the 60 h T4 week; Colab is needed only for the optional labels bet of §1.2(a). Slots used: 6–8 of ≈ 18 available.

### 4.3 Gate decision needed before the rehearsal (consequential)

The frozen projection (`setup + overheads + Σ production × 1300/39 × 1.25 × 1.15`) applied to the compact stage gives ≈ 8.3 h for P2 and ≈ 9.4 h for P3 on paper, so every R6F variant fails the user's "waive only if projection < 8.0 h" rule — yet a guarded stage cannot exceed its deadline. Proposed reframing, to be accepted by Codex and the user before the rehearsal:

1. Unguarded stages (setup, R384, na-full, R224, stage 3, CSV) keep the ordinary conservative projection and must sum to **≤ 3.5 h**.
2. The compact stage is bounded by its deadline (T0 + 8.3 h); the receipt must show **projected coverage ≥ 97 %** of 1,300 studies under the conservative scaling, and the rehearsal must exercise the fallback path on at least one synthetic study.
3. Worst-case wall = 8.3 h + stage 3 + write **≤ 8.45 h by construction**; actual < 9 h stays enforced; no unchanged resubmission of a run that hit the guard with < 90 % coverage.

A′-w is outside this rule (no guard, A′'s own waiver as decided by the user).

### 4.4 Final picks for Oct 21 (rule fixed now)

- **Final 1 = the highest displayed R6F variant that carries na-full (P3/P4/P5), provided it displays ≥ P2.** If every na-full variant displays below P2, final 1 = P2 (a 0.958 floor, tie-tail position) only if ≥ 0.958; otherwise A′-w if ≥ 0.957, else A′.
- **Final 2 = the best remaining variant with a different knob** (if final 1 is P3, prefer P5 or P4; if final 1 is P2, prefer the best unique variant even at 0.957). Kaggle scores the better private result of the two, so decorrelation beats a second copy of the same composition.
- A timeout or ERROR is a failed probe; a guarded run that reports > 10 % compact fallback is a degraded probe and is not selected over an undegraded one at equal display.

## 5. Risks and limits

- Everything in §4.1's expected columns is an estimate built on three-decimal public displays and participants' version ledgers; the public/private split is unpublished and the per-finding weights may give back ≈ 0.001 privately (shared by the whole block, so relative position is unaffected).
- The R6F runtime rests on our measured R7/compact costs and `compact_infer`'s loader design, not on a run of the public controller at scale; the rehearsal decides. If the rehearsal shows the compact stage above 5.2 h point, drop R224 (P5 structure) before dropping the guard margin.
- Provenance is unchanged: R6 carries OAI through nartaa's checkpoints and an unresolved `r6fm` tag; the Oct 20 recheck can still demote the whole R6F family, in which case P7 is the only unique fallback and it is not built yet.
- A‴'s hidden diagnostics are unavailable; §2.2 is inference. Nothing here was executed; this file qualifies no entry.

## Evidence

- Leaderboards: `lb/rsna-knee-abnormality-detection-publicleaderboard-2026-10-17T17:06:52.csv`, `lb/submissions.csv`, `efficiency/full_leaderboard.csv` (notebook output of ryanholbrook's tool).
- Kernels (text dumps and metadata): `kernels/<owner>__<slug>/`; forum threads: `forums/*.txt` (748295, 735304, 743416, 747658, 748181, 749570, 749623, 749858, 750236, 750257–750260, 750295).
- Datasets: `ds-neronrx/{knee.py,log_fold0.txt,log_fold1.txt,oof_fold0.csv}`, `ds-dkim54/models.json`; listings via `kaggle datasets files`.
- Repo receipts: `outputs/kaggle/nafull-aprime-rehearsal-r3-20261016-result/nafull-aprime/`, `outputs/kaggle/ws2-r224-v2-qual-20261015-result/ws2-a-prime/`, `outputs/kaggle/{ws2-r224,nafull-aprime}-probe-waiver-*-visible/`, `outputs/analysis/na-full-aprime-rehearsal-20261016/r3-real/official-submission-detail-20261017-1227.json`.
- Prior analysis: `public-0958-scan-20261017.md` (recipe and runtime model), `runtime-headroom-plan-20261015.md` (projection formula and blind spots), `labels-v2-20261015.md`, `s6-labels-v2-main-lane-result-20261015.md`, `na-full-aprime-swap-scope-20261016.md`.
