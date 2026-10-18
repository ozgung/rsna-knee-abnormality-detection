# Iteration 3 decision — 2026-10-14, about 07:00 UTC

Author: Fable (Claude), strategy. Read-only: no compute, submission, commit, STATUS or handoff-log edit. Labels: [M] measured in a linked receipt, [P] public claim or official text, [E] my estimate. Inputs: `docs/research/original-methods-plan-20261014.md`, `docs/experiments/raptor64-original-methods-it2-result-20261014.md` (worktree `agent-a3334099cc20effac`), handoff entries from 2026-10-14 05:12 and 05:42 UTC.

## 1. Where iteration 2 leaves us

- A (Raptor + R7) scored public 0.956, our best; B2 (Dread + R7, no OAI) 0.948; B1 (R7 alone) is resubmitted after a platform "system error" [M]. About 800 teams sit at ≥ 0.956 and the tenth row is 0.963 [M].
- `in-distil` (ImageNet init + 0.5 × Dread v4_full K42 soft labels): dev 0.9149, Δ +0.0081 [+0.0052, +0.0111] over in-orig 0.9068; the five-target secondary is +0.0117 [M]. The gain is concentrated where the plan said the model was weakest (Lateral OA +0.020, PF OA +0.016, LM +0.016, MCL +0.013) [M]. It is below the pre-registered +0.015 bar, and second-order exposure through the teacher makes it an upper bound.
- `in-mirror`: Δ −0.018, damage exactly on the side-swapped and masked targets [M]. **This was my plan error, not an execution error.** I specified a Medial↔Lateral label permutation under an anatomical mirror. A left-right mirror turns a right knee into a plausible left knee; the medial compartment stays medial, the fibula stays lateral, so every label is invariant. The only targets a mirror would permute are laterality-defined ones, and none of the 12 is. Rule for every future geometric design: state the anatomical invariant in the protocol, make the label transform the identity on anatomy-defined targets, and add a unit test asserting label invariance under the augmentation before launch (M1's identity test checked the image path only).
- E1: dropping sagittal windows costs 0.028 macro (ACL −0.122, LM −0.103, PF OA −0.041); coronal carries MCL (−0.037); Lateral OA depends on neither [M]. M3 as specified is dead.

## 2. Calibrating what an own model can be worth

- Dev-to-public offset for our own arms is about +0.012 (own224 0.902 dev → 0.914 public) [M]. The student at 0.915 dev therefore projects to about 0.925 ± 0.006 public [E], roughly Dread's 0.927 [P]. in-orig projects to about 0.918 [E].
- Blend slope about 0.6 toward the stronger arm; a third arm two to three hundredths below the blend adds at most +0.002, and goodpjw's 0.929 reader cost −0.002 at full weight [P/E].
- Runtime: B2 projected 7.21 h in its qualifying session and 8.57 h in a slow one; B3 (B2 + dr-full as a separate worker) 9.48 h conservative [M]. A combined same-process old-arm worker shares reading and preprocessing, so a second model at K24 adds about +0.3 to +0.5 h instead of +0.9 h [E]; a slow-session projection would still sit near 8.9 to 9.0 h [E].

**Ceiling for B through a 0.2 slot (0.1 overall weight).** A 0.92 to 0.93 student adds −0.001 to +0.002 to B2 [E], a quarter of the ±0.004 single-score blur [P], for a runtime gate failure risk that costs a candidate. **Replacing Dread with the student** is no better: the student is at best Dread-equal in strength and is trained to imitate Dread, so its decorrelation from Dread is modest (dr-full's Spearman against Dread is 0.83 to 0.96 [M]; the student's should be similar); expected −0.002 to +0.001 [E]. **Per-target residual** on the weak targets is not tunable without R7 OOF predictions [E]. If B1 scores ≥ 0.948, Dread at 0.5 is already dragging B and the own slot is worth even less; the B composition should then be R7-heavy and the own arm only enters if it clears the bar in section 4.

## 3. Honesty about the goal

There is no original route with a credible path to over 0.960 public by Oct 20 [E, under 1 in 50]. 0.963 needs a Raptor-class (≥ 0.945) decorrelated component, that is +0.03 over our own arms; iteration 2 bought +0.008. The two public levers of that size, OAI pretraining (+0.02 [P/E], unresolved ruling, no time) and high-resolution anchored local views (+0.02 [E], Codex's ROI lane, not yet qualified, P(0.94 public by Oct 20) about 0.1), are not reachable. **What maximises the expected final private rank** is already banked: final 1 = A at fixed 0.5/0.5 (0.956); final 2 = the best no-OAI entry (B1 if ≥ B2, else B2). Inside the 0.956 block the private rank is a ±0.004 lottery that no entry choice changes [M/E]. Iteration 3 therefore cannot move the final rank in expectation; its value is a clean, measurably better own model, a correct answer to the distillation and laterality questions, and one option-value path into B if the student clears a real bar. I recommend saying this plainly to the user and continuing the original work on that basis.

## 4. Iteration 3: two single-factor tests against `in-distil`, today on the A100

Comparator is `in-distil` K42 0.91490 on the fixed 870 (in-orig stays as a secondary comparator). Same split (3,475/870), seed 42, 12 epochs, OneCycle, K8 training, K16 top-3 SWA, K42 export, paired whole-component bootstrap (843 components, 2,000 draws, seed 42), health stop at epoch-1 K16 < 0.80. Teacher: the same pinned Dread v4_full K42 npz (`dab0621a…`). Nothing else changes. Both arms run concurrently as in iteration 2 (about 21 min/epoch each, 8 GB) [M].

**Arm `in-distil-w1`: distillation weight 0.5 → 1.0.** This is the most decisive scaling question. The plan's distillation options reduce to one: is the gain saturated at 0.5? A stronger teacher is not available in time (an R7 teacher pass needs raw DICOMs that exist only on Kaggle, about 16 T4-hours against 9 h left before Oct 17 [E]; Raptor is OAI and A-only; Dread + dr-full is a same-family ensemble worth ≤ +0.003 [E] and comes second). Dread init + distillation is exposed and uninterpretable. Temperature is second-order for per-target BCE on probabilities [E]. Config delta: `DISTIL_WEIGHT` 1.0 for a new variant key, teacher pin unchanged, the pre-registration guard updated to accept the new weight for this variant only, one regression test. Expectation: +0.003 to +0.006 over in-distil [E]; P(reaching the full-data bar) about 0.3.

**Arm `in-distil-flip`: in-distil plus label-preserving anatomical mirror, p = 0.5.** Reuse `mirror_volume` (reverse sagittal slice order, flip width in coronal and axial slots). Labels and mask unchanged, no MCL masking, no permutation; mirror TTA at export is a secondary read only. Config delta: a `flip` variant with policy `{'p': 0.5, 'perm': identity, 'masked': [], 'tta_export': False}` plus the teacher pin; tests: mirror∘mirror identity (exists) and a new label-invariance assertion. Expectation: +0.002 to +0.006 over in-distil [E]; P(CI > 0) about 0.5. If it passes, it is a recipe multiplier for the full-data student at zero runtime.

**Kill rules, pre-registered.**
- Either arm: Δ vs in-distil with CI lower bound ≤ 0 → that factor is closed. For `w1`, Δ < +0.003 also closes distillation scaling as saturated.
- **Full-data bar (unchanged from the plan):** the best arm must reach K42 ≥ 0.9218 (in-orig +0.015). Below that, no own arm enters any entry; iteration 3 ends as knowledge and the final remains A / best no-OAI.
- Above the bar: launch one full-data student (all 4,345 studies, winning recipe, pre-registered SWA epochs, about 3.7 h A100 [M]) on Oct 15; build the combined same-process old-arm worker with Dread K42 + student K24 at 0.2 of the Dread slot; one 2×T4 rehearsal on Oct 17 after the quota refresh with the existing 8.3 h conservative projection gate; one hidden run on Oct 18. Any slip past Oct 18, or a projection above 8.3 h, keeps B2/B1 unchanged.

**Runtime today:** cache restaging about 1.5 h, implementation about 2 h in parallel, training about 4.3 h, export 10 min, evaluation 15 min; results about 7 h after launch. Compute: one A100 session, plentiful. Kaggle hours untouched.

## 5. Closed and deferred

- Narrow coronal MCL/Lateral OA specialist: new localiser and cache (about 3 days), per-target gain on 2 of 12 targets gives about +0.003 own and ≤ +0.001 B [E]; MCL has 20 dev positives, so fold-0 cannot validate it. Closed for this round.
- R7 or Raptor teachers, Dread-init students, frozen backbones: closed (time, exposure or OAI).
- Codex's same-series sagittal ROI lane (M4) continues under its own Oct 16 kill date; a passing it3 recipe should be adopted there, not duplicated.
- Public inference tweaks (R7 flip TTA, R224 at 35%) are each ≤ +0.002 [P/E]; listed only so the ranking picture is honest. The user's direction is original work, and I am not proposing them.

**Risks:** second-order exposure makes every student gain an upper bound; dev-to-public offsets are from one submitted own arm; B composition depends on B1's pending score; the runtime gate has shown 1.3 h of session variance [M].
