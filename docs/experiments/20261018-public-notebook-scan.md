# Public notebook scan — 2026-10-18

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
24 at 0.954. Our best entry, 0.950, now ties 36 public notebooks and sits far below the
public medal bands (a participant snapshot on October 6 put bronze at 0.944; the explainer
notebook reports place 206 of 6,035 for 0.957 on October 15).

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

## Consequences for us

1. **A qualified 0.956–0.958 entry is a packaging task, not a modelling task.** Both
   nartaa checkpoints are in a dataset we already attach, and heliosli's R6 stage runs
   as published inside the same process model our Versia fusion already uses (separate
   child processes, hard deadline, exact CSV parity checks). The 224 arm is the cheapest
   addition (23 s visible, same reader as the mirror arm). R6 is the step that moves the
   score and is the larger integration: two more datasets, an offline wheel install, a
   worker with its own fallback policy, and no stated licence.
2. **Provenance is the open risk, not runtime.** R6 and `anatomy.pt` carry no licence;
   the host's test (registration-only external data) is met by SKM-TEA on paper, but our
   October 9 rule stands: use only published inference code and weights, claim no
   explicit host clearance, and record it. nartaa's S6 is the cleanest alternative
   (CC BY-NC weights, Apache-2.0 training code, competition data only) at 0.950 alone,
   but adding it beside R6 was measured negative, so it is a substitute for R6 only if the
   licence question blocks R6.
3. **Our ConvNeXt reader should drop to a small weight** if R6 enters; the reader's own
   authors measured it as neutral-to-negative at full weight on these bases.
4. **Do not tune per-finding weights on Gold-58 or the leaderboard.** Take heliosli's
   declared weights as the fixed public recipe, as the 0.958 notebooks do, and declare
   them before the run.
5. **Time.** Four days remain, with five entries per day and two final selections. One
   qualified rehearsal (67-study visible/stress protocol as before) plus one submission
   fits; a second variant does not unless the first fails early.

## Recommended queue for the remaining days

- **REF-07 (first):** fork our frozen fusion builder into a four-arm recipe: mirror 384
  (unchanged) + 224 crop with mirror (nartaa's 0.945 script, stage 1b as in goodpjw2008
  v27) + heliosli R6 with flip TTA (their stage cells unchanged, pinned bundle
  `fe055406`, manifest SHA checked) + our ConvNeXt reader at the published 1/10 weights.
  Blend exactly as heliosli's stage 3 (percentile ranks over each model's non-failed
  rows, Raptor half 0.65/0.35, per-finding Raptor share 0.85/0.65/0.30/0.15/0.5), declared
  before any run. Qualify on the existing 67-study protocol, then submit once.
- **Fallback if R6 cannot be qualified by October 21:** mirror 384 + 224 mirror in the
  Raptor slot at 0.65/0.35 with our current ConvNeXt weights. Expected effect is a better
  rank inside the 0.950 block, not a new displayed score.
- **Do not start:** S6 beside R6, Gold-58 weight refits, the 20-checkpoint community
  stack, or new training. None is supported by the measurements above within the time
  left.
