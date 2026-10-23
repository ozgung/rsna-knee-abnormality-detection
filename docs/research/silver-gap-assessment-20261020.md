# What it takes to reach 0.960: gap assessment, 2026-10-20 (data as of 15:10 UTC)

Author: Claude (Fable max, solo). Read-only review: no training, kernel push, submission or VM was started; this note, the scan files under `outputs/research/public-scan-20261020/` and the board CSV under `outputs/analysis/lb-20261020/` are the only outputs. Labels: **[measured]** = a number in a receipt, CSV, API response or downloaded file; **[literature]** = another participant's claim; **[estimate]** = my judgement. Times UTC. Deadline Oct 22 23:59 UTC; the plan's last new submission is Oct 22 10:00 UTC, about 43 h from now.

## Verdict

1. **The public lineage ends at 0.958.** Every public notebook at or above 0.956 is the same nartaa Raptor + heliosli compact recipe; nothing public claims 0.959 (scan of 60 score-sorted and 80 recently-run kernels, 168 discussion topics). The 211 teams at 0.959 or better all run private models. Copying has nothing left to give: the two public knobs we do not run (R224 mirror, MIL reader at 1/10) are worth about +0.0003 to +0.0005 [estimate].
2. **The top 10 (0.967 to 0.969) are single own models or small own ensembles trained on image-informed, iterated pseudo-labels, without external data.** Rank 2 (azamat1ch): "Short answer is labels", "AI, not by hand. And don't rely on reports alone", competition data only; rank 1 is also efficiency rank 3, so its inference is fast; optimo's 0.960 was one model [literature]. hengck23's "Tricks" thread (745861) adds the label-side craft: let the LLM abstain and drop those cells, train only on high-agreement cells (one participant removed 25 % of samples and gained on CV and LB), add definite negatives, few-shot on gold-58 with at most about 10 parameters. That is a multi-week programme; it is not reachable in 43 h.
3. **Our own evidence points the same way, at a smaller scale.** One round of image-teacher relabelling (labels v3 = report consensus + R6 out-of-fold teacher at weight 0.30) moved the own S6 solo from 0.950 to 0.955 (gold-58 0.934 → 0.938, exposed) [measured]. That is the largest own-model step in the project and the only lever that produced a displayed gain outside the public recipe. Our 3-member own compact is now R6-class on gold-58 (0.936 vs 0.937) and better than R7 as a solo (0.955 vs 0.949).
4. **The only ≥ 0.959 shot before the deadline is a dual compact that pairs R6 with our own members** ("P9": compact slot = ½ rank(R6) + ½ rank(mean of own S6 members), Raptor half and v1 per-finding weights exactly as P2). Lane B measured the dual-compact effect with R7 as the partner at +0.0029 [+0.0019, +0.0039] out of fold (single-fold columns, an upper bound), and our partner is both stronger than R7 and less correlated with R6 (blend-level ρ to P2: P7′ 0.967 vs P6s′ 0.991). Expected display 0.958 to 0.960; P(≥ 0.959) about 0.35 to 0.45, P(≥ 0.960) about 0.2 [estimate]. Runtime is the binding constraint (§3).
5. **A second label round (labels v4) is the real lever but barely fits.** It needs out-of-fold own teacher columns (four fold fits), a relabel, all-data fits, export and an entry: about 20 h of wall clock with no VM loss, against a record of one VM loss per lane. P(a scored entry by Oct 22 10:00) about 0.3 [estimate]. Run it only as a parallel bet with a hard kill, never on the critical path.

## 1. Board and medals (Oct 20 15:06 UTC) [measured, `outputs/analysis/lb-20261020/`]

| Display | Teams | Ranks | | Display | Teams | Ranks |
|---|---|---|---|---|---|---|
| 0.969 | 1 | 1 | | 0.961 | 26 | 67–92 |
| 0.968 | 3 | 2–4 | | 0.960 | 44 | 93–136 |
| 0.967 | 6 | 5–10 | | 0.959 | 75 | 137–211 |
| 0.966 / 0.965 | 5 / 5 | 11–20 | | **0.958** | **1,387** | **212–1,598** |
| 0.964 / 0.963 / 0.962 | 13 / 17 / 16 | 21–66 | | 0.957 / 0.956 | 174 / 276 | 1,599–2,048 |

6,270 teams. Medal lines by Kaggle's rule: gold ≈ 23 (0.965, with 0.964 on the edge), silver ≈ 314 (any 0.959 display; the first 100 or so 0.958 entries by time), bronze ≈ 627 (inside the 0.958 block). Since Oct 19 the 0.958 block grew by 128 teams and the 0.959 tier by 11.

Private reading (plan §1): per-entry sampling SD about 0.0023; byte-identical public copies share one private draw, and P2 differs from them by a residual with SD about 0.0003 minus the two missing knobs. Silver on the private board therefore needs a true edge of about +0.0015 to +0.002 over the public recipe, not a lucky public display [estimate].

## 2. What we tried, and what each result says

| Entry (public) | What changed against the public recipe | Reading |
|---|---|---|
| P2 57340072 **0.958** | our controller around the public recipe (no R224 mirror, no reader) | reproduces the block |
| P3 / P4 / P5 0.957 | + na-full at 0.1 to 0.35 of the Raptor half, V2 weights, R224 → na-full | no displayed gain; inside noise |
| A′-w 0.957, A′ 0.956 | R7 instead of R6 flip | R6 flip > R7 by about 0.001 |
| P6s′ 57414571 0.957 | ½ R6 plain + ½ R7 plain (shared decode), no R224 | OOF said +0.003; display −0.001 (noise, missing R224, possible guard cut) |
| P7′ 57432784 0.955 | compact = own 3-member S6 mean rank, no heliosli bundle | own compact 0.003 below R6 in the blend |
| C1 57432734 0.955 | own S6 solo, labels v3 seed 43 | +0.005 over the labels-v2 solo (0.950) |
| Lane D solo 57426580 0.954 | S6 on labels v3 + 745 fastMRI exams, P2-teacher targets | +0.004 over v2 solo; fastMRI itself not separable from v3 |
| V2 / S6fall solos 0.950 | labels v2 vs nartaa's labels, same wrapper | labels v2 = nartaa's labels on the board |

Supporting measurements:

- **Lane B out of fold (4,349 studies, single fold models)** [measured]: dual (½ R6 plain + ½ R7 plain) 0.9095, R6 flip 0.9066, R6 plain 0.9057, R7 plain 0.9051. Flip − plain = +0.0009 [+0.0005, +0.0014]. The gain of a second compact model is 3× the gain of flip TTA. Caveat: deployed entries average four fold models first, which captures part of the variance the dual removes; the four-model gold-58 read (n = 58) has the opposite sign with a CI including 0.
- **Weak findings are the same in every model** [measured]: OOF R6 flip PF OA 0.884, MCL 0.883, Lateral OA 0.887, Synovitis 0.887, Effusion 0.889 against MM 0.946, Baker's 0.927, ACL 0.922, LM 0.921; gold-58 R6 flip Synovitis 0.860, Lateral OA 0.865, Fracture 0.888, PF OA 0.918. The domain note's targets (PF OA, Lateral OA, LM) stand; nothing we built moved them specifically.
- **P7′ = C1**: the Raptor half added nothing visible to our compact, while it adds about +0.006 to R7 (0.949 → 0.956) [literature]. Either noise (two 0.003-SD reads), heliosli's per-finding weights misallocating for a non-R6 compact, or overlap between our compact and Raptor. Not resolvable without an out-of-fold Raptor column, which does not exist (Raptor trained on every study).
- **Similarity** [measured, 39 studies]: ρ(P6s′, P2) 0.991, ρ(P7′, P2) 0.967; the own compact is the only decorrelated component we have.
- **Runtime** [measured]: P2 point 24,930 s (hidden ≤ 8.1 h); the R6 flip stage is about 5 h of it (31 s/study/board, decode-dominated; R6 plain = 70 % of flip); Raptor half about 1.7 h; P7′ point 16,369 s (hidden ≤ 4.5 h) of which the 3-member own stage is about 2.8 h. Lane A's shared decode ran R6 + R7 on windows rendered once (592 s vs 749 s separately per 39 studies).
- **Labels** [measured]: labels v3 gold-58 0.9254 vs v2 0.9202 (exposed); teacher weight 0.30 was a-priori. R7 plain and R6 flip out-of-fold columns exist for every pool study; a dual teacher column (0.9095 vs 0.9066 against the weak truth) is available on CPU.

## 3. What is required, by horizon

### 3.1 Before Oct 22 10:00 UTC: P9, dual compact with our own members

- **Composition.** Raptor half as P2 (R384 two-view 0.65 + R224 0.35, v1 per-finding weights, unchanged and pre-registered). Compact slot = ½ rank(R6) + ½ rank(mean rank of own members {v3-s43, student, v3-s42, plus any new seed}). Stage order: Raptor arms, R6, own members last under the 8.3 h guard; a trimmed tail falls back to R6-only compact ranks (the P6s pattern). Salvage-before-constant as in every entry.
- **Runtime fit** (point hours; actual ≈ ×1.17): P2 6.9 = setup 0.1 + R384 1.05 + R224 0.6 + R6 flip 5.1. Own stage from P7′: 2.8 h for 3 members (decode ≈ 1.6 h of it).
  - (b) **no new decode code**: R6 plain (3.6) + 2 own members (2.4), R224 dropped → 7.15 h point, ≈ 8.4 h actual: the guard trims about 1 to 2 % of the last stage. With 3 members 7.55 h point, ≈ 6 % trimmed. Buildable from the existing `p6s` (dual compact, guarded stage order) and `p7` (s6c worker) variants: new variant plus tests and one rehearsal, 4 to 6 h [estimate].
  - (a) **shared decode between R6 and the S6 members** (one DICOM decode, two renders at 384 and 320 px): own stage ≈ 0.4 h per member → R6 plain + R224 + 3 members ≈ 6.6 h point, or R6 flip without R224 ≈ 7.5 h. Lane A's R6 + R7 shared decode took about 5 h to build and test; the second render size is new and unverified. Worth attempting in parallel; the entry must not depend on it.
- **Members.** Seeds 44 and 45 on labels v3 (all-data S6, 3.3 h each alone on an A100, 7.2 h as a pair) add variance reduction worth about +0.001 on the compact [estimate]; they are ready by about 20:00 UTC today if started now, with export and the strict CPU check by 21:00. Flip TTA on own members (+0.0009 class for R6 out of fold) only if shared decode lands.
- **Expected effect.** +0.001 to +0.003 true over P2 [estimate: dual gain upper bound +0.003 with a weaker, more correlated partner; minus 0.0003 to 0.0005 if R224 is dropped]. Display 0.958 to 0.960; P(≥ 0.959) 0.35 to 0.45; P(≥ 0.960) about 0.2.
- **Validation without the board.** No out-of-fold column exists for own all-data members, so the Lane B rule cannot be applied to P9. Pre-register instead: rehearsal PASS on every non-projection gate, `fallback_studies` 0 on the visible run, per-finding Spearman of the P9 compact against the R6-flip compact reported (expect 0.93 to 0.97), gold-58 of the P9 compact read once (R6 flip 0.937, own set 0.936; expect the dual ≥ 0.935).
- **Schedule.** Build and tests by 22:00 today; rehearsal (0.8 h), visible run and submission by Oct 21 03:00; scored by about 11:00. Second variant (extra seeds or shared decode) submitted by Oct 21 18:00, scored by Oct 22 02:00. Slots: 1 left today, 5 on Oct 21, 5 on Oct 22.
- **Selection rule (proposed amendment to plan §4).** P9 displays ≥ 0.959 with every gate passed → final 1 = P9, final 2 = P2. P9 displays 0.958 → final 1 = P2, final 2 = P9 (lower ρ than P6s′, real upside). P9 ≤ 0.957 → the current plan (P2 + P7′ or P6s′).

### 3.2 In parallel, if the user wants to spend the units: labels v4

Teacher = out-of-fold own S6 (four fold fits on labels v3, 4 × 3.3 A100-h, about 7 h wall on two sessions) averaged with the existing R6-flip and R7 out-of-fold columns; relabel on CPU (1 h, the labels v3 construction with the teacher swapped, weight 0.30 fixed); two all-data fits (3.3 h each, one session each); export and CPU check (1 h); C-solo probe and a P9 member swap. Earliest export about Oct 21 12:00 with no loss. Expected solo +0.002 to +0.004 over v3 [estimate]; as a P9 member +0.0005 to +0.001. Kill: fold fits not all done by Oct 21 04:00, or exports not done by Oct 21 18:00. Colab balance about 370 units [estimate: 499 at Oct 19 22:09 minus about 24 A100-h at 5.3 units/h], enough for all of §3.1 and §3.2 (about 35 A100-h).

### 3.3 Not worth a slot now

Public knobs (R224 mirror, MIL reader): +0.0003 to +0.0005, user declined copying. Per-finding weight changes: unvalidatable (goodpjw's gold-58 refits scored 0.955 and 0.957). omerzalman's "98 % / 99 % accuracy" label datasets (750334): partial coverage, gold-verified, the claimed +0.02 to +0.1 AUC comes from weak baselines. 3DReasonKnee (hengck23, 745861): OAI-derived boxes, licence unchecked, no time to use.

### 3.4 What 0.965+ would have needed (for the write-up)

Two to three pseudo-label rounds with image models on the exact inference input (colum2131), abstention and agreement filtering of LLM labels (hengck23's thread), per-finding localisation for PF OA, Lateral OA and LM (domain note items 2 and 3), 4 to 5 fold own ensembles with flip TTA, and a fast single-model inference path (the top team is efficiency rank 3). No external data was needed by the teams that said so. Our labels v3 → 0.955 solo is one round of that loop.

## 4. Evidence

Board: `outputs/analysis/lb-20261020/rsna-knee-abnormality-detection-publicleaderboard-2026-10-20T15:06:38.csv`. Kernels and topics: `outputs/research/public-scan-20261020/{kernels-by-score,kernels-by-daterun,kernels-by-votes}.csv`, `topics-p1..8.csv`, `disc-{745861,748111,750529,750334,751021}.txt` (parsed from the site's JSON endpoint). Our record: `STATUS.md`, `docs/experiments/{final-selection-20261020,lane-b-oof-20261019,lane-c-seeds-20261019,lane-d-fastmri-20261019,labels-v2-20261015}.md`, `docs/research/{final-push-plan-20261019,strategy-revision-20261017,public-0958-scan-20261017,original-methods-plan-20261014,knee-domain-research-20261014}.md`, `outputs/laneb-20261019/analysis/summary_final.json`, `outputs/analysis/final-selection-20261020.json`, `kaggle competitions submissions` (15:08 UTC).
