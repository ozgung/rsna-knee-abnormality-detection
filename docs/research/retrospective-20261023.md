# Retrospective: final selection and strategy — 2026-10-23

Author: Claude (Opus 5.5), after the private leaderboard was published. Sources: our submissions with private scores (`kaggle competitions submissions`, saved to `outputs/postdeadline/submissions-20261023.csv`), the private leaderboard as seen by the user on Oct 23, and the earlier notes linked below. Labels: [M] measured, [E] estimate.

## 1. Outcome

- Final rank **715 / 6,425** [M]. Both finals scored 0.942 private: S4 57504011 (Raptor R384 + R224 + own-8 compact) and P2 57340072 (the public R6-flip blend) [M].
- Best private entry: **P11 57471109**, our eight own members alone, no Raptor, 0.944 (public 0.956) [M]. On the private board, the 0.944 block ends at rank 311 and 0.943 starts at 312 [M, user screenshot]. Under Kaggle's medal table for 1,000+ teams (silver at or above rank 321, bronze at or above 642), P11 as a final would very likely have taken silver [E].

## 2. What the shake-up looked like

Every entry dropped from public to private, but not by the same amount [M]:

| Family | Entries (public → private) | Drop |
|---|---|---|
| Own members on image-teacher labels (v3, v4b, v5), no public model | P11 .956 → .944; v4b-s43 solo .956 → .943; fastMRI student .954 → .943; C1 solo (v3) .955 → .942 | −0.011 to −0.013 |
| Blends built on public models (Raptor, R6, R7) | P2 and S4 .958 → .942; P10p, C1, p13nf .957 → .942; P9 v2, A′-w .957 → .941; R7 alone .949 → .934 | −0.015 to −0.016 |
| Own members on report-only labels, and public S6 weights | V2 solo .950 → .936; S6fall .950 → .935 | −0.014 to −0.015 |
| R6 + own without Raptor | P12 .950 → .936 | −0.014 |
| Runtime-trimmed entry | E1 .949 → .931 | −0.018 |

Readings:

1. **Gains confirmed offline held up on private; gains found only on the public board did not.**
   - Relabelling with image teachers (labels v3 and later) was chosen on out-of-fold and gold-58 evidence. It moved the own solo from 0.950 to 0.955 on public and from 0.936 to 0.942 on private: the whole gain carried over.
   - Raptor's advantage over our own members came from the public board alone (gold-58 put the Raptor half last, 0.924 against R6 flip 0.937 and our members 0.931–0.940). It was +0.002 on public and −0.002 on private.
2. **The shake-up differed by recipe family.** The [Oct 14 shake-up simulation](rank-value-selection-20261014.md) assumed a common shift that cancels across teams plus sampling noise, so reshuffling inside the fork block was modelled as a lottery. In fact the families moved apart by about 0.003–0.004, larger than the paired sampling SD between related entries (about 0.001–0.002 on 390 public studies; [Oct 13 note](shakeup-risk-20261013.md)). The likely cause is public-tuned choices: per-finding weights in the public 0.958 notebooks, and a single-finding probe switch in heliosli's code, both flagged in the [Oct 17 scan](public-0958-scan-20261017.md).
3. **The offline signals were right about the direction but too weak to rank blends.** Out of fold (report-derived truth, 4,345 studies): own r2 0.9232 vs R6 flip 0.9066, inflated by label-source alignment. Gold-58 (58 studies, macro SE about 0.017): own four-fold 0.940, R6 flip 0.937, Raptor half 0.924, ½ R6 + ½ own 0.941. The last of these is P12's composition, which scored 0.936 private. So neither signal alone would have chosen well.

## 3. What was knowable before the deadline

- About 1,400 teams were tied at 0.958 on public, nearly all forks of the same blends, and the forum expected a shake-up for that reason ([handoff](../archive/handoff/claude-astra-handoff.md), early research log).
- P2 was effectively that fork. As a final it could only land wherever the fork block landed.
- Raptor's per-finding weights looked tuned on the public board, and gold-58 ranked Raptor last.
- The user's standing preference was to spend effort on our own methods rather than on public-notebook tweaks.

The selection rule we pre-registered (highest public display, ties to the own-heavy entry) ranked by the public score first. Pre-registering was right; the criterion was wrong.

## 4. Rules for the next competition

1. **Build a trusted validation set early.** Aim for a few hundred studies, labelled independently of the weak-label sources and never used for tuning. Gold-58 was too small, and the out-of-fold truth shared sources with our training labels. This should have taken priority over many of the blend-variant slots.
2. **Use the public board to catch failures, not to rank entries.**
   - Compute its noise band early: the paired SD for related entries, and the SD between distinct model families.
   - Veto entries more than about 2–3 SD below the best; that catches failures like E1 (runtime trim) and P12.
   - Inside the band, treat entries as tied and rank them by offline evidence.
   - Treat a gain seen only on the public board as unconfirmed.
3. **Never pick a public fork as a final.**
   - With two picks: pick 1 is the best entry on trusted validation among those not vetoed.
   - Pick 2 is a hedge that differs from both pick 1 and the crowd.
   - S4 + P2 held nearly the same bet twice.
4. **Model the shake-up by recipe family.** Add a per-family shift term, with a larger variance for families carrying public-tuned choices, not only sampling noise around a common shift.
5. **Put effort into our own pipeline from day one.** Label quality was what transferred. Cap the slots spent on public-blend variants: about a dozen went to P3/P4/P5, A′, A′-w and the C1 weightings, chasing +0.001 on public.

Applied to Oct 22 with the evidence we then had, rules 2 and 3 veto E1 and P12 and leave 0.955–0.958 as a tie. On trusted evidence our members were at least R6-class and Raptor last, so pick 1 = P11 (or another own-only compact) and pick 2 = S4, which has public-model strength but is not a fork. That pair scores 0.944 [E, counterfactual].

Caveat: these rules come from one private outcome, and P11 − S4 = 0.002 is close to noise. Points 1, 3 and 4 do not depend on that result: the crowded fork block, the fork as a final, and the weak validation set were problems whichever way the 0.002 had gone.
