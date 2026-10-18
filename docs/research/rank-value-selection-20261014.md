# Lane 3 result: private-rank value and the pre-registered final-selection rule — 2026-10-14, about 15:50 UTC

Author: Fable (Claude) analysis agent, CPU only. Protocol: [lane3-rank-value-selection-protocol-20261014.md](../experiments/lane3-rank-value-selection-protocol-20261014.md), executed as written; no training, no submission, no entry code touched. Receipt, script and the scores-and-ranks leaderboard: `outputs/analysis/rank-value-20261014/` (`analyze.py`, `receipt.json` with input SHA-256s and seeds, `leaderboard.csv`; raw download under `lb-raw/`, ignored). Runtime 59 s; agent time about 25 min of the 4 h cap. Labels: **[measured]** = number in the receipt or a linked file; **[literature]** = official text or another participant's claim; **[estimate]** = judgement or model assumption.

## Q0. Per-target A weights: closed, no fitting

1. Exposure inflation (dev macro on the 870 minus public): Raptor alone 0.9513 − 0.949 = **+0.002**; the 0.7/0.3 build 0.9630 − 0.954 = **+0.009**; the 0.5/0.5 build 0.9677 − 0.956 = **+0.012** [measured; S6 stands in for R7 on the 870].
2. Derived R7/S6-class inflation: +0.021 (from the 0.5/0.5 build), +0.024 (from 0.7/0.3), +0.021 directly (S6 dev 0.9695 vs R7-alone public 0.949) [measured/estimate, weight-linear assumption].
3. The two components' inflations differ by about 0.018–0.022; the paired bootstrap SD of Raptor − S6 on the 870 is 0.0026, so the rule threshold is 0.0052 [measured]. **Rule 1 triggers**: any weight fit on the 870 would move weight toward the component whose level is inflated ten times more.
4. Gold-58 Hanley–McNeil SE per target, from the host's 58 gold rows (positives 9–35): at AUC 0.90 every target has SE 0.040–0.071; even at A's exposed per-target levels (0.94–0.99) 11 of 12 exceed 0.02 (ACL 0.015 is the exception) [measured]. **Rule 2 triggers.**
5. Closure: **no per-target or global blend weight is fitted on the 870 or on gold-58, under any further argument.** A stays at the fixed 0.5/0.5; A′ and B1′ keep their pre-registered 0.3 arm weights.
6. Descriptive by-product (exposed; where diversity lives, not what weight to use) [measured]: mean per-target Spearman Raptor~S6 0.82, Raptor~Dread 0.88, S6~Dread 0.79, Raptor~in-distil 0.90, S6~in-distil 0.77, Raptor~own blend2 0.91, S6~own blend2 0.79. Raptor and S6 disagree most on ACL (0.66), Contusion (0.72), Fracture (0.76), Baker's (0.82), MCL (0.83); lowest top-decile overlap (Jaccard) on Medial Meniscus 0.42, Synovitis 0.43, Lateral OA 0.50, Effusion 0.53, PF OA 0.54.

## Q1. Leaderboard density and medal lines (download 15:29 UTC, 5,983 teams) [measured]

| Display | Teams | Ranks | | Display | Teams | Ranks |
|---|---|---|---|---|---|---|
| 0.966 | 4 | 1–4 | | 0.957 | 103 | 111–213 |
| 0.965 | 3 | 5–7 | | **0.956 (A)** | **843** | **214–1056** |
| 0.964 | 7 | 8–14 | | 0.955 | 50 | 1057–1106 |
| 0.963 | 9 | 15–23 | | 0.954 | 298 | 1107–1404 |
| 0.962 | 8 | 24–31 | | 0.953 | 31 | 1405–1435 |
| 0.961 | 11 | 32–42 | | 0.952 | 17 | 1436–1452 |
| 0.960 | 16 | 43–58 | | 0.951 | 48 | 1453–1500 |
| 0.959 | 17 | 59–75 | | 0.950 | 213 | 1501–1713 |
| 0.958 | 35 | 76–110 | | **0.949 (B1)** | 93 | 1714–1806 |

- Teams at or above 0.956: 1,056; 0.957: 213; 0.958: 110; 0.960: 58; 0.963: 23. Fork-like block (0.955–0.957): 996 teams. Our API rank 677, mid-block.
- **Medals and points: verified.** The Kaggle API record (`lb-raw/competition-api.json`) has `awardsPoints = True`, category Research, reward 77,000 USD; medals follow points under Kaggle's progression rules [literature]. The overview page itself is JS-rendered and could not be read offline.
- Cuts for N = 5,983 (floor): **gold 21** (score 0.963), **silver 299** (inside the 0.956 block), **bronze 598** (inside the 0.956 block). Gold would be 22 under round-half-up. The new-entrant and merger deadline is Oct 15 23:59 UTC, so N can still rise by a few per cent; the cuts move by a few places at most.
- Correction to the plan: H1 put about 650 teams in 0.956–0.958 and the gold line near 0.958. Measured: 981 teams in 0.956–0.958 and the gold line at 0.963 (rank 21). 0.958 is silver territory (ranks 76–110), not gold.

## Q2. Shake-up simulation (10,000 draws per cell, seed 42)

Model as pre-registered: private = public (with uniform ±0.0005 display jitter) + common shift (cancels) + correlated noise; a fraction f of the 0.955–0.957 block is fork-like at ρ_f, all other teams at ρ_i; per-entry SD = the Oct 13 A50 bootstrap SD scaled to the private size (0.0025 at 650, 0.0023 at 910 for A; Raptor's 0.0034-class SD for B1); grid f ∈ {0.5, 0.7, 0.9}, ρ_f ∈ {0.95, 0.98}, ρ_i ∈ {0.80, 0.90}, splits 50/50 and 30/70. Ranges below are min–max over the 24 cells, median in bold [measured on the model; the model itself is an estimate].

| Entry | Private rank median | p10–p90 (typical cell) | P(gold ≤ 21) | P(silver ≤ 299) | P(bronze ≤ 598) |
|---|---|---|---|---|---|
| A (0.956) | 640–**654**–672 | 460–860 | 0 | 0.00 | 0.34–**0.38**–0.42 |
| A + 0.0005 | 373–**441**–495 | 290–625 | 0 | 0.06–**0.13**–0.33 | 0.75–**0.86**–0.92 |
| A + 0.001 | 205–**282**–354 | 185–420 | 0 | 0.31–**0.57**–0.83 | 0.97–**1.00**–1.00 |
| A + 0.002 | 99–**125**–179 | 95–180 | 0 | 0.94–**1.00**–1.00 | 1.00 |
| A + 0.003 | 71–**78**–100 | 62–100 | 0 | 1.00 | 1.00 |
| B1 (0.949) | 1704–**1715**–1726 | 1545–1830 | 0 | 0 | 0 |
| B1 + 0.001 | 1617–**1619**–1619 | 1460–1780 | 0 | 0 | 0 |
| B1 + 0.003 | 1437–**1444**–1445 | 1270–1600 | 0 | 0 | ≤ 0.002 |

- **Rank per true +0.001:** near 0.956, **356 places** (298–433 across the grid); near 0.949, **83 places** (80–89). The static reading is the same story: the 843-team 0.956 block is passed in one display step, whereas 0.949 → 0.950 passes about 100 teams.
- **Sensitivity (two-sample model, not pre-registered):** if the public reading also carries its own sampling error that the private set redraws (per-entry SD becomes 0.0045 at 30/70), A's median rank is 679 with P(bronze) 0.29; A + 0.001 → median 382, P(silver) 0.20 (0.10–0.53), P(bronze) 0.97; A + 0.002 → median 196, P(silver) 0.92; slope 289 places per +0.001 (232–369). B1 is unchanged in substance.
- **Paired public-reading SD of A′ − A** (A′ = A with one target's column replaced by 0.7 × rank(A) + 0.3 × rank(arm); arm stand-in = in-distil SWA fold-0 export; study bootstrap at the public size) [measured]: one target (PF OA or Lateral Meniscus) **0.00041–0.00044 at n = 390**, 0.00032–0.00034 at 650; both targets 0.00063 / 0.00048; whole-component (843) bootstrap at 870 agrees with the study bootstrap to 0.00001. Fork bound (A70 vs A50, a 12-target 0.2 change) 0.00088 / 0.00068. The protocol expected 0.0005–0.001; the single-target figure is lower because a 0.3 weight on one of twelve targets moves the macro by at most one twelfth of that target's change.

## Q3. Pre-registered decision rules (written before any A′, B1′ or arm score exists)

**Selection between A and A′ (and B1 and B1′).** A′ is selected over A if (i) all pre-registered fold-0 gates of every component in it passed, (ii) A′ is COMPLETE in the official API, and (iii) public(A′) ≥ public(A) − τ, with **τ = 0.00063** (one paired public SD at n_public = 390 for a two-target arm; 0.00041 for a single-target arm). If A′'s public reading is below A's by more than τ, A is kept: the fold-0 evidence is then contradicted, not merely unconfirmed. The same rule for B1′ vs B1.

*Operationalisation on the 3-decimal display.* The protocol's parenthetical anticipated "a tie or a 0.001 deficit does not reject A′" for τ in 0.0005–0.001. With the measured τ and the display rounding simulated (receipt `Q3.display_rule_error_rates`): under a true gain of +0.0005 the displayed difference is ≥ 0 in 93 % of readings and ≥ −0.001 in 99.8 %; under a true loss of −0.001 it is ≥ 0 in 24 % and ≥ −0.001 in **76 %** [measured on the model]. Tolerating a displayed 0.001 deficit therefore accepts a harmful A′ three times out of four, while a displayed 0.001 deficit implies an expected true deficit of about 0.001, which exceeds τ. The rounding-consistent reading of criterion (iii) is **"displayed(A′) ≥ displayed(A): a tie passes, any displayed deficit rejects."** A displayed lead of 0.001 occurs in 77 % of readings under a true +0.001 and in 1.5 % under a true −0.001, so the public reading does resolve differences at the 0.001 level, contrary to H2's premise. Recommendation to the parent: adopt the "tie or better" reading now, while no A′ score exists; record the choice in the handoff.

**Integration threshold for a passing arm.** An arm enters A′ (or B1′) only if its fold-0 CI lower bound is > 0 (lanes 1–2) and the expected rank gain of its lower-bound macro δ is ≥ 50 places. With 356 places per +0.001 near A, 50 places need macro δ ≥ 0.00014, i.e. a **primary-target AUC Δ lower bound ≥ +0.0017** (÷12 → macro). Near B1 (83 places per +0.001) the same 50 places need **Δ lower bound ≥ +0.0072**. Lane 1's gate already asks for PF OA Δ ≥ +0.010 with lower bound > 0, so any arm that passes lanes 1–2 with a lower bound ≥ +0.002 clears the A′ threshold; an arm whose only home is B1′ needs a lower bound ≥ +0.007 or stays knowledge only. Use the Δ measured on the gate-5 Dread-transfer blend when it exists, otherwise the in-distil blend's lower bound [estimate: the in-distil Δ overstates the transfer into A's stronger column].

**Failure handling.** A failed hidden run of A′ costs a slot and a day, not rank, because A remains selectable; at most one retry; no new hidden run after Oct 20 16:00 UTC. Two-final composition and the OAI/fastMRI contingencies: plan step 4, unchanged.

## Q4. Where A's components disagree (descriptive, exposed, changes no weight)

Lowest Raptor~S6 rank agreement: ACL, Contusion, Fracture, MCL, Baker's (Spearman 0.66–0.83); lowest top-decile overlap: Medial Meniscus, Synovitis, Lateral OA, Effusion, PF OA (Jaccard 0.42–0.54). Own-class arms (in-distil, own blend2) correlate 0.90–0.91 with Raptor and 0.77–0.79 with S6, so an own arm adds more diversity to the R7-only base (B1′) than to A′ — but B1′ carries about a quarter of A′'s rank value per +0.001 and no medal odds, so the natural home and the valuable home differ. For PF OA specifically, in-distil's top decile overlaps S6's by 0.33 and Raptor's by 0.45 [measured].

## Limits

- The fork correlation is assumed for every team but ours; the field's private noise uses A's SD for all teams near our scores; the private set may differ by site, which no bootstrap on the 870 captures; absolute shifts cancel only if common to all teams.
- Dev levels on the 870 are exposed for Raptor, S6 and Dread; only dispersions and correlations are used. R7 is represented by S6; the arm stand-in is in-distil's SWA export (K16), not the K42 evaluation, which is not held locally.
- Whether Kaggle breaks displayed ties on the private board by underlying score or by submission time is unverified; the simulation assumes underlying score. N and the cuts may move slightly until Oct 15 23:59 UTC.
- The medal verification rests on the API field `awardsPoints`, not on the overview page text.

## What changes for the next 48 h

1. The gold line is 0.963, not 0.958: P(gold) is 0 for every entry we can build. Silver (rank ≤ 299) needs a true +0.001 to +0.002 over A (P 0.57 and 1.00 in the protocol model; 0.20 and 0.92 in the two-sample sensitivity); bronze is a coin flip for A as it stands (0.38 / 0.29) and near-certain with +0.001.
2. Every robust +0.001 into A is worth about 300–430 places; the same gain in B1 is worth about 83 places and no medal. Spend slots and rehearsals on A′; treat B1 strictly as the OAI hedge and do not improve it for rank.
3. The public reading of A′ vs A is informative at the 0.001 level (paired SD 0.0004–0.0006): a displayed lead confirms, a displayed tie is neutral, a displayed deficit rejects.
4. Per-target weights stay closed; nobody should fit them on the 870 or gold-58.
