# Lane 3 protocol: private-rank value model and pre-registered final-selection rule, 2026-10-14

Pre-registered 2026-10-14 at about 15:30 UTC. Author: Fable (Claude), strategy. Owner of execution: a Fable analysis agent at xhigh effort, **CPU only**: no GPU, no training, no submission, no Colab, no edit to any entry code. It cannot touch lane 1 or lane 2 by construction. Labels: **[measured]**, **[literature]**, **[estimate]** as in the plan.

Inputs: [plan of record](../research/plan-update-20261014.md) sections 1–2 and step 4, [shake-up note](../research/shakeup-risk-20261013.md) and its receipt `outputs/analysis/shakeup-risk-20261013/receipt.json` (bootstrap SDs, paired SDs, correlations on the 870), the official state receipt `outputs/analysis/resume-20261014/official-state.json`, the public scores of our entries (A 0.956, the 0.7/0.3 build 0.954, B1 0.949, B2 0.948, B0 0.940, own-only 0.914 [measured]; Raptor alone 0.949 [literature]), and the public leaderboard downloaded with the Kaggle API.

## Why an analysis lane rather than a third training lane

The remaining training candidates on the 336 cache after lanes 1 and 2 are a coronal MCL arm (20 fold-0 positives, SE 0.048: unmeasurable) and a coronal Lateral OA specialist (E1 gives it no mechanism) [measured]. Both would cost a Colab session and a rehearsal slot for an undecidable result. Meanwhile three decisions with real rank value have no numbers behind them: how many places a true +0.001 buys near our score, how to choose between A and A′ (and B1 and B1′) when public readings differ by a rounded 0.001 or tie, and whether per-target blend weights for A can be chosen at all from the predictions we hold. Lane 3 answers these before any A′ score exists, so the rules cannot be fitted to the public leaderboard.

## Hypotheses

- **H1 (rank value).** About 650 teams sit between 0.956 and 0.958 [literature, LB readings], so a robust true +0.001 moves expected private rank by hundreds of places, and the bronze/silver lines (if the competition awards medals) fall inside that block. The lane replaces the readings with the downloaded distribution.
- **H2 (selection threshold).** Because A′ differs from A by a 0.3-weight term on one or two targets, their paired public difference has SD about 0.0005–0.001 at 390–650 public studies [estimate from the paired SDs of 0.0006–0.0008 at n = 870, measured]. Selecting A′ only when its public reading exceeds A's therefore discards a true gain roughly half the time; the right rule tolerates a small public deficit when the fold-0 gates passed.
- **H0 (per-target A weights, the user's example question).** Per-target weights between Raptor and R7 cannot be chosen validly from the 870 or gold-58: every component of A was trained on all 4,345 studies, with different exposure inflation per component (Raptor 0.951 dev vs 0.949 public; the 0.7/0.3 blend 0.963 dev vs 0.954 public [measured]), so a fit on the 870 would move weight toward the more over-fitted component; gold-58 has a macro SE of about 0.017 and per-target SEs far larger [measured/estimate]. Expected outcome: **closed with no fitting**, by the fixed rule in Q0.

## Baseline

The plan's current rules: finals = best A-family entry that passed all gates and best no-OAI entry; "A′ if ≥ A public"; the Oct 13 shake-up figures (single-entry public-to-private SD about 0.004 at 30/70; paired fork SD about 0.001; correlation with A 0.92–0.98 for Raptor-sharing entries, 0.77–0.83 for own-only) [measured/estimate].

## Arms (analysis questions, each with a method fixed here)

### Q0. Validity check for per-target A weights (about 30 min; fixed rule, no fitting under any outcome)

- Compute exposure inflation = dev macro on the 870 (exposed) minus public score for: Raptor alone; the 0.7/0.3 Raptor + S6 build (public 0.954); the 0.5/0.5 build (public 0.956). S6 stands in for R7 on the 870, as in the shake-up note; R7's own 870 predictions do not exist and S6 alone was never scored publicly, so the R7-class inflation is derived from the blends.
- **Rule:** if the component inflations differ by more than 2 × the paired SD of the two components on the 870 (about 0.002), per-target weight fitting on the 870 is declared invalid. For gold-58, compute the Hanley–McNeil SE of each target's AUC from its gold positives and negatives; any target with SE > 0.02 is declared unusable for weights. If either rule triggers (expected: both), write a ten-line closure and do not fit weights, per target or global, under any further argument.
- Permitted by-product, descriptive only: per-target rank correlation and top-decile disagreement between Raptor, S6 and the clean own OOF on the 870. Correlations are far less exposure-sensitive than levels; they say where diversity lives, not what weight to use. Labelled exposed.

### Q1. Leaderboard density and medal lines (about 45 min)

- Download the public leaderboard with `kaggle competitions leaderboard -c rsna-knee-abnormality-detection --download` to the ignored `outputs/analysis/rank-value-20261014/`. It is public data; store only scores and ranks, no team names in any committed artifact.
- Report: team count N; teams per 0.001 over 0.950–0.966; the ranks at 0.956, 0.957, 0.958 and 0.960; the gold, silver and bronze cut ranks under Kaggle's progression rules for N ≥ 1,000 teams (gold top 10 + 0.2% of N, silver top 5%, bronze top 10%) **after verifying on the competition page that this competition awards medals and points**; if the page cannot be read offline, mark the thresholds [unverified].
- Also record the fork share: the number of teams within ±0.001 of 0.956, which the simulation treats as the fork-like population.

### Q2. Shake-up simulation (about 1.5 h)

- Two-population model of the field relative to our entry E: fork-like teams (fraction f of the 0.955–0.957 block, private-noise correlation ρ_f with E) and independent teams (ρ_i with E). Fixed grid: f ∈ {0.5, 0.7, 0.9}; ρ_f ∈ {0.95, 0.98}; ρ_i ∈ {0.80, 0.90}; private set n ∈ {650, 910} (50/50 and 30/70 splits). Per-entry private-score noise SD = the Oct 13 bootstrap SD scaled by sqrt(870 / n) [measured basis]. Each team's private score = its public score + a true public-to-private offset drawn from a common shift (which cancels in rank) + correlated noise. 10,000 draws per grid cell, seed 42.
- Outputs for E ∈ {A, A + δ} with true gain δ ∈ {+0.0005, +0.001, +0.002, +0.003}, and for B1 and B1 + δ: the distribution of private rank (median, 10th and 90th percentiles); P(within each medal cut from Q1); the rank-per-+0.001 slope near 0.956 and near 0.949; the paired public-reading SD of A′ − A at n_public ∈ {390, 650}, derived from the paired bootstrap of a 0.3-weight single-target blend on the 870 (use the lane-1 blend evaluator's bootstrap machinery on in-distil as a stand-in for the arm; the public arms' paired SD is bounded by the fork figures already measured).
- Honest limits to state: the fork correlation is assumed, not measured, for teams other than ours; the private set may differ by site, which no bootstrap on the 870 captures; absolute shifts cancel in rank only if they are common to all teams.

### Q3. Pre-registered decision rules (the deliverable; written before any A′ or B1′ score exists)

- **Selection between A and A′ (and B1 and B1′):** A′ is selected over A if (i) all pre-registered fold-0 gates of every component in it passed, (ii) A′ is COMPLETE in the official API, and (iii) public(A′) ≥ public(A) − τ, with τ = one paired public SD from Q2 (expected 0.0005–0.001, which at 3-decimal display means "a tie or a 0.001 deficit does not reject A′"). If A′'s public reading is below A's by more than τ, A is kept: the fold-0 evidence is then contradicted, not merely unconfirmed. The same rule for B1′ vs B1.
- **Integration threshold for a passing arm:** it enters only if the fold-0 CI lower bound > 0 (already pre-registered in lanes 1–2) and the Q2 expected rank gain of its lower-bound δ is at least 50 places [estimate of the value of a submission slot and a rehearsal]; otherwise the arm is knowledge only.
- **Failure handling:** a failed hidden run of A′ costs a slot and a day, not rank, because A remains selectable; at most one retry, as in the plan. No new hidden run after Oct 20 16:00 UTC.
- The two-final composition and the OAI/fastMRI contingencies stay as plan step 4; this lane references, not duplicates, them.

### Q4 (optional, only within the time cap)

A one-table summary of where A's components disagree per target from Q0's descriptive by-product, to inform which targets a passing arm enters in A′ versus B1′ (an own arm is less correlated with S6/R7, 0.79, than with Raptor, 0.91 [measured], so B1′ is the more natural home). Descriptive; it changes no weight.

## Primary metric and pre-registered kill rule

An analysis lane has no AUC endpoint; its acceptance is a rule set with numbers and a receipt.

- **Accept** when Q0's closure, Q1's density table, Q2's rank distributions and Q3's rules are written with their inputs' SHA-256s and seeds in the receipt, and every number is labelled measured, literature or estimate.
- **Kill or degrade:** if the leaderboard download fails (API, terms or network), Q1–Q2 run on the plan's readings (top 10 at 0.963, rank 1 at 0.966, about 800 teams at ≥ 0.956, gold line about 0.958) and are labelled [estimate]; the Q3 rules are still written. Stop after **4 h of agent time**; whatever is unfinished is listed as not done.
- **Forbidden under every outcome:** fitting any blend weight on the 870 or on gold-58; reading the hidden-test or private leaderboard in any way other than the public page; changing lane 1's or lane 2's pre-registered weights or gates; editing entry code.

## Expected value

No score gain. Decision value [estimate]: the A/A′ and B1/B1′ selections involve a true difference of about ±0.001 and a public reading that cannot resolve it; a wrong pick costs a few hundred places under H1. The integration threshold prevents spending the last submission slots on arms worth fewer places than a hidden-run risk. Closing H0 saves anyone from a tempting but biased weight fit. The user also receives calibrated medal odds for the two finals, which the plan currently states only as P(gold line) about 0.05–0.12 [estimate].

## Final-entry path

No code enters A, A′, B1 or B1′ from this lane. The Q3 rules are applied by Claude and Codex on Oct 21 when the finals are selected, and the rule actually applied, with the readings it used, is recorded in STATUS and the handoff log.

## Files the analysis agent must touch

- New, ignored: `outputs/analysis/rank-value-20261014/analyze.py` (reuses `outputs/analysis/shakeup-risk-20261013/analyze.py` where possible), `receipt.json` (inputs' SHA-256s, seeds, grid, results), `leaderboard.csv` (scores and ranks only).
- New result note: `docs/research/rank-value-selection-20261014.md`, with the Q0 closure, the density table, the rank distributions, the medal cuts with their verification status, and the Q3 rules verbatim.
- One dated entry in `docs/archive/handoff/claude-astra-handoff.md`; STATUS is updated by the parent at the next checkpoint.
- Read-only inputs: the shake-up receipt and script, `outputs/analysis/resume-20261014/official-state.json`, the 870 predictions already held for Raptor, S6, Dread v4_full and the clean own OOF files, the lane-1 blend evaluator for its bootstrap functions. No file under `knee/` or `scripts/` changes.
