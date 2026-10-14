# Shake-up risk and the A/B final selections — 2026-10-13, about 16:30 UTC

Author: Fable (Claude), strategy review. Read-only: local CPU analysis only; no submission, commit, STATUS or handoff-log edit. Labels: [M] measured here or in a linked receipt, [P] public claim by another participant or official text, [E] my estimate. Receipt and script: `outputs/analysis/shakeup-risk-20261013/` (`analyze.py`, `receipt.json`).

Question answered: should Selection B move from Dread + R7 to an own-model-heavy blend so that A and B fail differently?

## 1. Shake-up magnitude

**Split.** The rules (saved copy, `outputs/analysis/s6-hidden-failure-20261012/official-pages-20261012.json`) only say the public leaderboard uses a public test set and the private leaderboard a private test set; no percentage is given, and no site holdout is stated [M]. The data page says about 1,300 test studies [P]. I therefore show two assumptions: 30/70 (about 390 public / 910 private, the common Kaggle default) and 50/50 (650 / 650).

**Sampling SD, bootstrap on the fixed 870** (weak-label truth with masks, 2,000 resamples, macro over defined targets; public arms are exposed to these studies, so levels are inflated but dispersion is a fair proxy) [M]:

| Entry proxy | Dev macro (exposed) | SD at n=870 | SD at 390 | SD at 650 | SD at 910 |
| --- | --- | --- | --- | --- | --- |
| A proxy: 0.5/0.5 Raptor + S6 rank blend | 0.9677 | 0.0024 | 0.0034 | 0.0027 | 0.0023 |
| Current entry: 0.7/0.3 Raptor + S6 | 0.9630 | 0.0027 | — | — | — |
| Own-heavy B: 0.5 A proxy + 0.5 own blend2 | 0.9487 | 0.0039 | 0.0060 | 0.0043 | 0.0036 |
| Own blend2 alone (CoAtNet-384 + R2-224, fold 0) | 0.9163 | 0.0059 | — | — | — |

Own blend2 = 0.5/0.5 probability mean of our two fold-0 held-out families (`stacking-e4` identity scores); S6 stands in for R7 (same arm, same folds; the new stackers are untested here) [E].

**Is a 0.003 to 0.006 gap noise?** It depends on whether the two entries are paired on the same studies:
- Two forks of the same public blend (0.7/0.3 versus 0.5/0.5, or A versus A + 20% own) have a paired bootstrap difference SD of **0.0006 to 0.0008** at n=870 [M], about 0.001 on a 390-study public set [E]. A 0.003 displayed gap between such entries is real and will mostly carry to private.
- One entry versus a genuinely different model at correlation 0.85 has a paired SD of about 0.0019 on 390 public studies and 0.0012 on 910 private studies [E, from SD_single × sqrt(2(1−ρ))]. A 0.006 lead (tenth row 0.963 versus 0.957) is about 3 SD and is mostly real; a 0.003 lead is about 1.5 SD and may flip.
- The public-to-private change of one entry's absolute score has SD about 0.004 (30/70) [E]. Inside the 637-team 0.956 block [M, LB 14:47 UTC] rank order will reshuffle nearly at random; no choice of A or B changes that lottery.

**Correlation of shake-ups** (bootstrap-score correlation with the A proxy, and per-study placement-error correlation) [M]:

| Entry | Boot corr with A | Study-error corr with A | Error corr with Raptor |
| --- | --- | --- | --- |
| 0.7/0.3 Raptor + S6 (the scored 0.954 build) | 0.98 | 0.98 | 0.97 |
| A + 20% own blend2 | 0.97 | 0.99 | 0.95 |
| A + 35% own | 0.92 | 0.96 | 0.95 |
| A + 50% own | 0.88 | 0.94 | 0.95 |
| S6 + 50% own (no Raptor) | 0.90 | 0.95 | 0.91 |
| 0.5/0.5 CoAtNet-384 + S6 (Dread + R7 proxy, see below) | 0.88 | 0.95 | 0.90 |
| S6 alone | 0.86 | 0.91 | 0.74 |
| Raptor alone | 0.92 | 0.93 | 1.00 |
| Own blend2 alone | 0.77 | 0.83 | 0.90 |

Two entries that share Raptor move together at about 0.92 to 0.98. Even an own-only entry keeps 0.77 to 0.83 with A: the same 4,345 studies, the same report-derived label table (nartaa `train_labels_v1` for our own models too) and the same 140 mm cache dominate the error structure. Per-target prediction rank correlation of own blend2 with Raptor is 0.91, with S6 0.79; Raptor with S6 is 0.82 [M]. Our own arms are therefore closer to Raptor than S6 is.

No Dread predictions exist on the 870, so Dread + R7 cannot be measured. Dread v4_full is the Raptor recipe trained without OAI, so its errors should sit between Raptor's (1.00) and our CoAtNet-384's (0.90) correlation with Raptor; I use the CoAtNet-384 + S6 blend as a lower bound on its correlation with A [E].

## 2. Overfitting to the public LB versus model strength

- The 0.957 recipe is fixed 0.5/0.5 per-finding percentile ranks; a fork's 40/60, 50/50 and 60/40 probes all displayed 0.956 [P]. Selection pressure on the public LB is therefore at most one bit (0.7/0.3 versus 0.5/0.5), and that choice is worth about 0.002 to 0.005 on dev with SD 0.0006 [M], so it is signal, not LB noise.
- Raptor and R7/S6 are trained on all 4,345 labelled studies and never saw test data [P]. Their public lead over our arms (0.949 and about 0.95 versus 0.914 submitted own224) is model strength, not public-set overfitting. On dev, exposure inflates them by different amounts (Raptor 0.951 dev versus 0.949 public; the 70:30 blend 0.963 dev versus 0.954 public [M]), while weak labels understate our own arms (own224 0.902 dev fold 0 versus 0.914 public [M]). Dev-level gaps are therefore unusable for selection; dev correlations and SDs are still informative.
- Conclusion: the shake-up to fear is not "public blends are tuned to public"; it is sampling noise plus any systematic ruling or run failure.

## 3. Correlated-failure modes and what an own-heavy B buys

| Failure | Hits A? | Dread + R7 | R7 alone | Own ≥50% + R7 | Own only (scored 57037296, 0.914) |
| --- | --- | --- | --- | --- | --- |
| Host rules against OAI-trained public weights (Raptor) | yes | covered | covered | covered | covered |
| Host rules against R7's fastMRI auxiliary data [P, licence "other"] | yes | not covered | not covered | not covered | covered |
| Hidden-run failure or timeout of the new build | per entry | separate build, same wrapper | smallest build (about 4.7 h) | +0.7 h at K16 × 5 folds, new reader code | already scored |
| Label or site shift in private | yes | same labels, same recipe family | same | same labels and cache; corr with A still 0.88 to 0.90 | corr 0.77 |
| Expected public score [E] | 0.957 [P] | 0.943 to 0.945 (B0 was 0.940 with S6 [M]) | 0.948 to 0.953, wide; S6 alone was never scored publicly | 0.940 to 0.946 | 0.914 [M] |

The decisive arithmetic: an own-heavy B only helps the final rank if its private score exceeds A's. The expected gap is about 0.012 to 0.017 on the public scale [E]; the paired private SD between them is about 0.002 to 0.0035 [M/E]. Sampling shake-up cannot close a 4 to 7 SD gap. Only a systematic event can, and every systematic event listed above is covered as well or better by a cheaper no-OAI B. Own-heavy B hedges nothing that Dread + R7 or R7 alone does not, and costs about 1.5 days of packaging plus a rehearsal and a hidden run.

The one scenario where an own-only entry pays is both rulings adverse (OAI and fastMRI). That is already covered at zero cost: submission 57037296 (own CoAtNet224, five folds, 0.914, no external data) is scored and can be selected on Oct 22 if a host ruling lands first.

## 4. Recommendation

**Keep A = Raptor + R7 at fixed 0.5/0.5 (WS1).** It is the measured best; dropping it costs about 0.003 for certain and no shake-up argument offsets that.

**Keep B no-OAI, but choose it by a two-candidate probe rather than by default:**
1. **R7 alone** (no Raptor, no Dread), same recovery wrapper, submitted once. It is the cheapest build, has the largest runtime margin, removes the Dread dependency, and may score above Dread + R7 (dev says S6 alone beats every Dread-class blend, but dev is exposed; the public S6-alone score is unknown [E]). It also measures R7 standalone, which the A weight reasoning lacks.
2. **Dread + R7 at 0.5/0.5** (WS3 as planned).

Pick the higher displayed score as B. Expected B: 0.945 to 0.952 [E]. Both cover the OAI ruling. Neither covers the fastMRI question, so verify R7's licence wording before Oct 21 as already planned; if judged risky, B falls back to Dread + S6 (57119585, 0.940, scored) and, if both OAI and fastMRI are ruled out, to 57037296 (0.914, scored).

**Do not build an own-heavy B.** Measured error correlation with A falls only from about 0.98 to 0.88 at 50% own weight, the expected cost is 0.012 to 0.017 against A, and it covers no failure mode that the cheaper candidates do not. If own arms earn a place anywhere, it is as a 20% rank component of A (corr 0.97, dev cost −0.006 exposed, public effect unknown), and that belongs to the A probe queue, not to B.

**Minimal work before Oct 22:** one extra hidden run (R7 alone, about 4.7 h), reusing the WS1 build with the Raptor slot disabled and the recovery policy unchanged; no new code beyond a flag; one 39-study rehearsal (about 0.5 Kaggle GPU h). Submit after WS1 scores, by Oct 15.

## 5. Limits

- Truth on the 870 is the report-derived weak-label table with masks, not radiologist labels; Synovitis (440 labelled) and Baker's (547) are half-masked, MCL has 20 positives, so per-target noise is large and SDs are proxies. Gold-58 is excluded from the 870 and has no public-arm predictions, so it was not used.
- Raptor and S6 saw the 870 during training; all dev levels for them are exposed. Dread has no dev predictions; its correlation with A is bounded, not measured. R7 is represented by S6 (same arm and folds, new stackers untested).
- The public/private split is unstated; all size-dependent numbers assume 30/70 or 50/50. Expected public scores for untested B candidates are estimates from blend slope and the scored 0.940 and 0.954 entries, not forecasts of private rank.
