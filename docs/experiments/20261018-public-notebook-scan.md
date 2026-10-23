# Public notebook scan — 2026-10-18

Revised 2026-10-18 evening after merging the live campaign branch: the first version was
written from the stale October 9 `main` and assumed our best entry was 0.950. Our best is
now **R6F-pure v2, submission 57340072, public 0.958** (our own controller around the
public recipe), with A′-w at 0.957 and A′ at 0.956; see the
[handoff log](../archive/handoff/claude-astra-handoff.md) and the
[October 17 strategy revision](../research/strategy-revision-20261017.md). The listing,
provenance and measurement sections below are unchanged evidence; the consequences and
queue sections are rewritten for the actual position.

Read-only research from a cloud session without Kaggle CLI credentials. The score-sorted
Code listing (300 entries, pages 1–6) and the competition metadata were read through
Kaggle's anonymous web endpoints; eleven notebook sources were downloaded through the
public script-content endpoint and inspected without executing them. Raw evidence:
`outputs/research/2026-10-18/public-scan/` (listing, competition JSON, discussion topics,
downloaded sources; ignored by Git). No model ran, no weights were downloaded, no
submission was made.

## Verified competition facts (competition JSON, 2026-10-18 13:05 UTC)

| Field | Value |
| --- | --- |
| Final submission deadline | **2026-10-22 23:59 UTC** (02:59 Istanbul on October 23) |
| Entry / team-merger / notebook-publishing lock | 2026-10-15 23:59 UTC, all passed |
| Daily submissions / final selections | 5 per day / 2 scored |
| Teams | 6,174 |
| Public leaderboard share | 30 % of the hidden test (N = 1,284 studies per nartaa's receipts; roughly 385 public studies) |

Hosts (Christopher Carr, RSNA, October 14, topic 743416): they will not designate named
datasets. External data that needs **only registration and acknowledgement of licence
terms** meets rule 2.6; data needing an institutional agreement does not. The OAI
geo-restriction question (China cannot register) was raised again on October 14 and 16
and has no host answer. Nothing here changes the October 9 eligibility decision; it adds
that fastMRI/fastMRI+ and SKM-TEA (registration plus click-through) fall on the permitted
side of the host's stated test, with the same unresolved geo caveat as OAI.

## Score-sorted public notebooks, October 18

Counts over the 300 listed entries: 8 at 0.958, 7 at 0.957, 42 at 0.956, 4 at 0.955,
24 at 0.954. Our R6F-pure v2 (0.958) sits inside the 0.958 block, which held 865 teams at
ranks 143–1007 on October 17 17:06 UTC with bronze (top 10 %, about rank 614) inside it;
A′-w (0.957) sits in the 224-team block below. The explainer notebook reports place 206
of 6,035 for 0.957 on October 15, before the block grew.

| Score | Notebook | Date | What it is |
| --- | --- | --- | --- |
| 0.958 | [goodpjw2008 · CoAtNet + R6 + 2.5D MIL](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958) v27 | Oct 16 | nartaa 384 (mirror) + nartaa 224 (mirror added) in the Raptor slot, 0.65/0.35; heliosli R6 with flip TTA; the ConvNeXt reader we already use at 1/10 of its old weights; heliosli's per-finding Raptor weights. Fully documented version history. |
| 0.958 | [heliosli · blend mine in](https://www.kaggle.com/code/heliosli/blend-mine-in-private-lb-luck-to-all-upvoters) and [best public dataset new version](https://www.kaggle.com/code/heliosli/this-is-best-public-dataset-new-version-0-9586) | Oct 15–17 | The origin recipe: Raptor half = 0.65·rank(384) + 0.35·rank(224, plain); compact half = R6 (4 folds, flip TTA, own Effusion/Baker's stackers); per-finding Raptor share Synovitis 0.85, PF OA 0.65, Baker's 0.30, Lateral OA 0.15, else 0.5 (the goodpjw2008 copy uses 0.8/0.6/0.35/0.2). Author reports 0.9586. |
| 0.958 | kaiwalyaatulraut, kiyoshiohno, lavinwins "Top 1% Gold" | Oct 17–18 | Copies of the two above; one with no attached datasets (a stale listing or a script-only copy). |
| 0.958 | [lavinwins · RSNA Gold Medal](https://www.kaggle.com/code/lavinwins/rsna-gold-medal) | Oct 17 | Different lineage: pjmathematician's D4/community stack datasets (Raptor maxspan/native384, RadImageNet heads, pilkwang DINO weights). 6,005 lines, 20+ checkpoints; not inspected in depth. |
| 0.957 | [goodpjw2008 · Does S6 help next to R6?](https://www.kaggle.com/code/goodpjw2008/rsna-knee-does-s6-help-next-to-r6-lb-0-957) | Oct 17 | Two measured negatives: adding S6 beside R6 costs 0.001; per-finding weights refitted on Gold-58 score 0.957 and 0.955 against 0.958. |
| 0.957 | [bliverrigpdoge · R7 + Raptor + knee-centred ConvNeXt](https://www.kaggle.com/code/bliverrigpdoge/r7-raptor-knee-centred-convnext) | Oct 14 | heliosli R7 recipe plus a private ConvNeXt-S/EfficientNetV2-S ensemble at 0.2. Private weights; forks fall back to 0.956. |
| 0.957 | [laimerbotrolenk · four model ensemble](https://www.kaggle.com/code/laimerbotrolenk/rsna-four-model-ensemble) | Oct 15 | R7 0.3 + S6 (nartaa all-data) 0.2 + Raptor 384/224 mirror; the listed version has no measured score of its own. |
| 0.957 | [leoprovorov · 0.957 explained](https://www.kaggle.com/code/leoprovorov/rsna-knee-0-957-explained-why-the-open-ensembl) | Oct 16 | Analysis notebook with the submitted recipe appended (Raptor 384 + 224 mirror 0.65/0.35; R7 0.6 + S6 0.4; 50/50). Useful noise estimates, see below. |
| 0.956 | [heliosli · R7 + Raptor one click](https://www.kaggle.com/code/heliosli/0-957-latest-dataset-score-in-one-click), [varshithakolupuri · unused checkpoint](https://www.kaggle.com/code/varshithakolupuri/rsna-knee-one-unused-checkpoint-lifts-the-0-956), [evgendvorkin · Versia 7](https://www.kaggle.com/code/evgendvorkin/rsna-versia-7) and ~40 forks | Oct 11–15 | R7 + nartaa 384, 50/50 (0.956); the 224 checkpoint at 35 % of the Raptor slot keeps 0.956 but moves rank 233→156. Versia 7, the successor of the Versia 5 recipe behind our 0.950 fusion, dropped our ConvNeXt reader for heliosli's R7. |
| 0.950 | [nartaa · S6 all-data solo](https://www.kaggle.com/code/nartaa/rsna-knee-s6fall-solo) | Oct 14 | One ConvNeXt-small, no Raptor in the score path (CoAtNet only as a per-study fallback). Weights CC BY-NC 4.0, training code Apache-2.0, trained on competition studies and public label tables only. |

## The new components, with provenance

| Component | Dataset | Training data and terms | Evidence |
| --- | --- | --- | --- |
| nartaa CoAtNet 224 (`..._c08_r224_swa.pt`, 0.8 centre crop of the 140 mm field resized to 224, 0.945 alone) | `nartaa/rsna-knee-publication-swa-weights-20261007`, **already attached to our fusion** | Same OAI-assisted recipe and terms as the 384 checkpoint we use (`other`, competition use permitted; SWA epochs ranked on Gold-58 per nartaa's October 17 disclosure) | nartaa's 0.945 script, 53 s visible run; 23 s as stage 1b in the 0.958 notebooks |
| heliosli R6 (`fe055406`, ConvNeXt-small, 4 folds, 288 px cache, centre crop 256, full-field + 100 mm knee-centred views, Effusion/Baker's state stackers) | `heliosli/rsna-knee-r6-frozen-fe055406-w1` or `rostigor/rsna-knee-r6-fe055406` (byte-identical; the unsuffixed heliosli dataset now holds R7) | Competition studies with report labels; crops placed by `anatomy.pt`, an SKM-TEA V-Net in `heliosli/rsna-orthofoundation-fold0-epoch7-20261004`; no fastMRI entry in its manifest (edteoh's audit, topic 749858). **No licence stated** on the heliosli datasets (nartaa, topic 750180). Trainer and worker sources are embedded in the notebooks with SHA pins. | Stage runs through heliosli's `probe_dual.py`/`compact_infer.py` with an 8.3 h deadline and mean-fill fallback; visible 3-study stage 91 s including setup |
| heliosli R7 (`8c7b666e`, crop 352, fastMRI+ auxiliary task) | `heliosli/rsna-knee-compact-r7cnxfsss` | As R6 plus fastMRI images/boxes as an auxiliary loss | Measured worse than R6 with flip TTA by its users (0.956 vs 0.957) |
| nartaa S6 all-data (`convnext_small.fb_in22k_ft_in1k`, 320 px cache/288 crop, 3-slice windows at 3.5 mm, SKM-TEA anchors, JEV-blend soft labels, 16 epochs, no SWA/EMA, Gold-58 excluded) | `nartaa/rsna-knee-s6fall-weights` (CC BY-NC 4.0; `training_code/` Apache-2.0 with argv, input hashes, pool and label scripts) | Competition data plus CC0 public label tables; attaches heliosli's anatomy checkpoint | 0.950 alone; S6 next to R6 measured −0.001 |
| Flip TTA on the compact model | heliosli worker flag `--flip-tta` | Mirrors the windows, swaps medial/lateral outputs back, averages logits, MCL from the plain pass | R7 without flip 0.956 → R6 with flip 0.957 (one factor not isolated) |

The compact stage installs pinned wheels offline (`timm 1.0.25`, `pydicom 3.0.1`), needs
`anatomy.pt`, and its code tree is roughly 300 KB of SHA-pinned Python. Every 0.956–0.958
notebook runs it as a child process with a hard deadline; none reports the hidden-set
stage time, only that the 9 h limit was met.

## What the notebooks actually measured

- The whole 0.950 → 0.958 band is **one lineage**: nartaa's two CoAtNet checkpoints plus
  one heliosli ConvNeXt-small, rank-blended per finding with weights fixed by the authors.
  Forty-two of the 300 listed notebooks print 0.956 and most differ by glue only.
- Ordered gains reported by goodpjw2008 from their own submissions: nartaa 384 alone
  0.949 → + S6 0.954 → + R7 0.956 → 224 checkpoint at 35 % of the Raptor slot 0.956 with
  a better rank → R6 with flip TTA instead of R7 0.957 → heliosli's per-finding Raptor
  weights 0.958. Each step is one submission, not a controlled replicate.
- Their reader, the ConvNeXt we blend at full weight, **stopped adding on these bases**:
  0.956 base + reader at full weight scored 0.954; at 1/4 weight 0.954; they keep it at
  1/10. Our fusion gives it up to 0.35 per finding next to the 384 mirror only, which is
  the 0.950 configuration they also reported.
- Two negatives on the same base: S6 beside R6 (−0.001) and Gold-58-fitted per-finding
  weights (−0.001 to −0.003). Gold-58 has 9 to 26 positives per finding; the differences it
  shows are noise.
- leoprovorov's noise estimate: with about 385 public studies the standard deviation of the
  macro score is roughly 0.003, larger than the spread of the top 400 teams. Differences of
  0.001 on the public board are not evidence; rank inside a tie block follows the
  unrounded score and is the only finer signal.
- Discussion 748111 shows a single-model 0.960 at the top of the board; the author says it
  is one model and gives no recipe. Not reproducible from public material.

## Where our R6F-pure v2 differs from the public 0.958 notebooks

| Knob | Public 0.958 notebooks | Our R6F-pure v2 (57340072) | Public evidence on the difference |
| --- | --- | --- | --- |
| Raptor arms | nartaa's scripts run as published; goodpjw2008 v27 adds the anatomical mirror to the 224 arm | our own A′ ports: R384 two-view mirror, R224 one view, 0.65/0.35 | None isolated. goodpjw2008 v21–24 already had the 224 mirror at 0.957 and v25 reached 0.958 by the weights, so the mirror on 224 is unmeasured. Costs about +0.6 h (R224 3.3 s/study doubled). |
| Per-finding Raptor share | V2: Synovitis 0.85, PF OA 0.65, Baker's 0.30, Lateral OA 0.15 (heliosli 0.9586 reported); v1: 0.8/0.6/0.35/0.2 (goodpjw2008 v25/26, 0.958) | v1 | Both display 0.958; the V2 claim is +0.0006 unrounded, inside public noise but what the block's rank is sorted by. Zero runtime cost, so P4 is the right place to take it. |
| Compact slot | R6 `fe055406`, flip TTA, own stackers | same bundle, same worker v4c with flip, 2 loader workers | Identical by construction (manifest 6508ce69… pinned). |
| Extra members | goodpjw2008 reader at 1/10 (reported "at no cost", never a measured gain); S6 beside R6 −0.001; R7 beside R6 no change | none | Adding a second compact-class model does not help. The only public precedent for a same-family Raptor-half member is the 224 checkpoint at 0.35: no display change, rank 233→156. That is the case closest to na-full at 0.2 (P3). |
| Failure policy | mean-fill fallback inside compact stage; blend over each model's non-failed rows | deadline guard at T0 + 8.3 h, salvage of on-disk ledgers before any constant fill (the v1 0.500 was the pre-salvage path) | v2 scored 0.958, so the full pipeline ran within 9 h on the hidden host; no hidden timing is visible. |

## Consequences for the remaining days

1. **The public recipe is exhausted at 0.958 and the block is the medal line.** Nothing
   public claims 0.959 or better with code; the 0.960 single model has no recipe. Within
   the block, rank follows the unrounded score and Kaggle orders exact ties by submission
   time. Our predictions already differ from every copy (own Raptor ports, v1 weights), so
   the question is not tie-breaking but which side of 0.9575–0.9585 we land on, which no
   public measurement resolves. The public board's macro SD of about 0.003 on roughly 385
   studies says the same: no 0.001 step on the board is evidence; the private 70 % has
   SD about 0.002.
2. **P4 (V2 weights) is free and consistent with the public evidence**: heliosli chose
   them on the public board and every notebook that kept them displays 0.958 or reports
   0.9586; the two attempts to refit weights elsewhere (Gold-58) lost 0.001–0.003. Take
   V2 verbatim, declare before the run, change nothing else in that probe.
3. **P3/P5 (na-full in the Raptor half) have no public analogue that lost.** The 224
   checkpoint at 0.35 is the only same-family addition measured and it was neutral on
   display with a better rank. Expect display 0.957–0.958 and treat P3 against P2 as the
   one paired measurement we get; do not read 0.001 as a verdict either way.
4. **Do not spend a slot on the 224 mirror, the 1/10 reader, S6/V2-S6 beside R6, or any
   Gold-58-fitted weights.** Each is unmeasured or measured negative on the public
   base, and the mirror also costs runtime against a controller that has already hit a
   hidden exception once.
5. **Provenance stands as decided on October 9 and rechecked October 16.** R6 and
   `anatomy.pt` carry no licence; the host's October 14 test (registration-only external
   data permitted, nothing named) is met on paper by SKM-TEA and fastMRI; the OAI
   geo-restriction question is open. The October 20 recheck remains the only scheduled
   action; P7 (all-ours compact slot with V2-all S6) is the hedge if that recheck turns
   against heliosli's weights.
6. **Final selection on October 21** follows the strategy revision's rule: final 1 = the
   best na-full variant that displays at least P2's 0.958, else P2; final 2 = the best
   remaining variant with a different knob. The public evidence above favours P4 as the
   second knob if P3 wins final 1, because its only change is the one the public board
   was tuned on.

## Queue effect

No new task. The first version's REF-07 was our R6F-pure, already scored; its fallback
(224 mirror arm only) is withdrawn as unmeasured. P3, P4 and P5 were sealed and launched on
October 18 (handoff, 14:40 UTC) with exactly the content recommended here: P3 and P4 put
na-full at 0.20 of the Raptor half (R384 0.52, R224 0.28), P4 with heliosli's V2 weight
map, P5 drops R224 for na-full at 0.35; na-full runs last so the 8.3 h guard cuts it rather
than R6. The October 20 confirmation slot and the October 21 final rule are unchanged.
