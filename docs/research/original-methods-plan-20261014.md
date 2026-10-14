# Original-methods plan for the last eight days — 2026-10-14, about 01:00 UTC

Author: Fable (Claude), strategy design. Read-only: no compute, submission, commit, STATUS or handoff-log edit. Labels: [M] measured in a linked receipt, [P] public claim by another participant, [E] my estimate. Direction from the user: stop chasing public-notebook tweaks; spend our resources on methods that are ours, using public ideas, results and weights only as components.

## 1. What the evidence says an original method must do

- Our own ImageNet-init models sit at 0.906–0.907 dev macro on the fixed 870 (own44 K42 0.9059, raptor64 in-orig 0.9068, Δ +0.0010, CI [−0.0035, +0.0052]) [M], about 0.914–0.92 public [M/E]. The plane-aware head is null (−0.0002, CI ±0.0044) [M]; E4 stacking is null [M]; relabelling is NO-GO [M].
- The weak targets are the lateral and patellofemoral compartments and MCL: in-orig K42 Lateral OA 0.840, PF OA 0.848, Lateral Meniscus 0.862, MCL 0.878, against 0.93–0.95 for ACL, Synovitis, Baker's, MM [M]. Prediction correlations far exceed label co-occurrence (Medial/Lateral OA 0.91 vs 47% co-occurrence; Contusion/Fracture 0.92) [M]. Confusion lives in shared features, not in the output layer [M, E4].
- The Dread-init fold-0 arm scores 0.9471 [M], but Dread v4_full trained on fold 0, so that number is exposed and tells us nothing about generalisation.
- The public gap is model strength, not LB fitting: Raptor (OAI-pretrained) 0.949 alone, Dread (same recipe, no OAI) 0.927 [P], and R7, a no-OAI anchored local-view ConvNeXt at 352 px, is 0.95-class [P; B1 pending]. The two levers the public frontier used are external knee data (about +0.02) and high-resolution local views (about +0.02 over the 140 mm global design) [E].
- Blend arithmetic: a third arm in the 0.93 class adds at most about +0.002 to a 0.957 blend, and goodpjw's 0.929 reader cost −0.002 at full weight [P]. An own arm only moves A if it is decorrelated and at least 0.94 public [E].
- Validation: every public-weight model saw all 4,345 labelled studies. Fixed-870 fold-0 dev is honest only for ImageNet-init models; gold-58 is clean for Dread (trained on non-gold studies only) [M] but has a per-target SD of several hundredths [E]; the LB is ±0.004 per score [P].

Consequence: no original method can make the 0.963 tenth row in eight days [E, well under 1 in 20]. The honest aim is (a) an own component that is both measurably better on the weak targets and decorrelated enough to help B, perhaps A by a sub-display amount, and (b) methods whose validation is clean, so that what we learn is real.

## 2. Methods

### M1. Mirror-equivariant training with side-swapped labels (compartment consistency as structure)

- **Hypothesis.** The network confuses medial and lateral because nothing forces it to encode side. Anatomical mirroring (reverse sagittal slot order within each sagittal series; flip width in coronal and axial slots) with a Medial↔Lateral label permutation (meniscus, OA) and the MCL loss masked on mirrored samples makes side an explicit, contrastive signal. R7 trains with mirror 0.5 and a class permutation [M, manifest]; nartaa's inference-time mirror gave +0.001 [P]; ours is the training-time equivariant version on our own recipe, plus mirror TTA at export.
- **Entry.** It is a recipe change for every own arm, not a new arm. It enters through the M2 student.
- **Honest validation.** ImageNet init, fixed fold-0 split, paired against in-orig on the 870 with the qualified component bootstrap. Clean. Pre-registered secondary endpoint: the mean of LM, Lateral OA, PF OA, Medial OA, MM.
- **Cost.** One config delta in `knee/raptor64_training.py` (dataset augmentation + label permutation + MCL mask + mirror TTA) with a synthetic test that mirror∘mirror is identity; about 3 h of implementation, about 3 h A100 for 12 epochs solo (dr-orig ran at 880 s/epoch) [M], 15 min export.
- **Kill.** Macro Δ vs in-orig with CI lower bound ≤ 0 and the five-target secondary Δ < +0.005: drop mirror from the recipe.
- **Expected gain.** Own macro +0.002 to +0.005 [E], lateral/medial targets +0.005 to +0.01 [E]; P(CI > 0) about 0.4 [E]. Final-entry effect by itself: none; it is a multiplier for M2.

### M2. Our own distilled student: ImageNet init, Dread soft labels plus report labels, no OAI

- **Hypothesis.** Dread v4_full is a converged 0.927-class teacher on exactly our input contract (the raptor64 cache is the Dread reader) [M]. Soft targets on all 4,345 studies, including the half-masked Synovitis and Baker's rows, give denser, less noisy supervision than report labels; a student from a different init with the M1 structure should land near or above the teacher while being decorrelated from it, which is what a B third arm needs. Public weights are used only as a teacher; the student, recipe and structure are ours.
- **Entry.** Replaces dr-full in the planned B3 slot (B2 + 0.2 of the Dread slot), same reader and cache as the Dread arm, so the marginal runtime is model forward only: about +0.5 h at K42 on 2×T4 [E; the Dread K62 arm costs about 0.9 h in B2: 7.21 h vs B1 6.34 h conservative] [M]. In A only if it is at least 0.94 public [E].
- **Honest validation.** Fold-0 split, ImageNet init; the student never sees fold-0 images or labels, but the teacher did, so teacher targets on folds 1–4 carry second-order exposure (jessky's README flags this for the same construction) [P]. Report dev as "near-clean, second-order exposed". Gold-58 is clean for Dread and the student; use it as a sign check only. Final arbitration: one LB probe in B.
- **Cost.** Teacher inference on the staged cache with the pinned public Dread source: about 75 min A100 at K42 for 4,345 studies [E, from the 15 min 870-study K42 export] [M]. Loss delta: masked BCE on report labels + BCE on teacher probabilities at a pre-registered fixed weight 0.5 (teacher only where the report label is masked). About 2 h of implementation. Training about 3 h; a full-data student later about 3.3 h [M, dr-full 11,859 s].
- **Kill.** Student K42 macro < in-orig + 0.015 (that is, < about 0.922) on the 870, or gold-58 sign negative: close distillation. Promote to a full-data fit only if ≥ 0.925.
- **Expected gain.** Student 0.92–0.93 dev vs 0.907 [E]; P(≥ +0.015) about 0.5 [E]. B +0.002 to +0.005 at a fixed 0.2 slot [E]; A +0.000 to +0.001 [E].

### M3. Compartment specialists for the weak targets (lateral/PF/MCL from coronal and axial)

- **Hypothesis.** Lateral OA, PF OA and MCL are read on coronal and axial series; a reader restricted to those planes at higher effective resolution (about 100 mm crops) removes sagittal-dominated shared features and the medial/lateral confusion.
- **Critique.** The gate for this is the never-run E1 diagnostic: ablate plane groups at inference on the existing in-orig checkpoint (slots are grouped 18/14/12/8/12 and the cache stores per-slot plane [M]) and measure per-target change. If removing sagittal slots costs Lateral OA less than 0.01, a coronal/axial specialist is plausible; if it costs 0.03+, the specialist loses more than it gains. The SKM-TEA segmenter supports sagittal only [M], so coronal/axial crops need our own localiser (bone-centred crops from the coronal slab), which is new reader code plus a new cache: about 3 days to a first fold-0 fit [E].
- **Entry.** Per-target residual on the four weak targets only, fixed weight, in B; about +0.3 h runtime [E].
- **Validation.** Clean (ImageNet init, fold 0, paired per-target on the 870). The E1 diagnostic is 20 min of A100 and needs no training.
- **Kill.** E1 shows sagittal removal costs the weak targets ≥ 0.03, or no qualified coronal/axial reader by Oct 17.
- **Expected gain.** Weak-target AUC +0.01 to +0.03 if the hypothesis holds [E], P about 0.3; B +0.001 to +0.003 [E]; A none. Low probability of finishing by Oct 20.

### M4. Our own anchored local-view model (R7-style) on our grouped folds

- **Hypothesis.** The high-resolution anchored local view is the no-OAI public lever worth about +0.02 [E]. Codex's same-series sagittal ROI lane is exactly this for sagittal (100 mm native crops matched to global views, ImageNet init) and awaits CPU qualification [M]. Do not open a second lane; keep this one, add M1 mirror once qualified, and judge it on the 870 paired against in-orig.
- **Entry.** B third arm or a per-target residual, +0.5 to +1 h runtime [E].
- **Kill.** No qualified reader by Oct 16 (already agreed) or fold-0 Δ ≤ 0.
- **Expected gain.** Own +0.005 to +0.015 on sagittal targets if resolution is the mechanism [E]; reaching 0.94 public from ImageNet init in five days is unlikely (P about 0.15) [E].

### M5. Frozen public backbones under domain-aware heads: rejected

Exposure makes every dev number for a Dread or Raptor backbone uninterpretable, the head test with full training was already null, and a frozen backbone caps the ceiling below the teacher. Drop.

Not on this list: OAI pretraining is the largest lever on the board (0.949 vs 0.927 at the same recipe [P]), but it is a public method, the host ruling is unresolved, and the download, mapping and pretraining do not fit eight days. Keep the no-OAI final regardless.

## 3. Ranking and today's launches

| Rank | Method | EV on final | Clean validation | Wall-clock to first result | Start |
| --- | --- | --- | --- | --- | --- |
| 1 | M2 distilled student (with M1) | B +0.002 to +0.005 | near-clean + gold sign + LB | about 9 h | now |
| 2 | M1 mirror-equivariant recipe | multiplier for M2 | clean | about 7 h | now |
| 3 | M3 specialists, gated on E1 | B +0.001 to +0.003 | clean | E1 20 min; fit 3 days | E1 today, fit Oct 16 only if gated |
| 4 | M4 anchored local view (Codex lane) | B +0.001 to +0.003 | clean | reader pending | continue, no new lane |
| 5 | M5 frozen backbones | 0 | none | – | drop |

**Launch today, one A100 session, reusing the raptor64 bundle, cache staging and controllers** (`docs/experiments/raptor64-dread-init-protocol-20261013.md`, worktree `agent-a3334099cc20effac`):

1. Stage the raptor64 cache (about 1.5 h, as before) while implementing M1 and M2; prepare both the ImageNet-384 initialiser and the full Dread v4_full predictor (backbone and head) for teaching.
2. **E1 diagnostic (20 min, no training):** K42 export of `in-orig` with each plane group zeroed in turn (sagittal, coronal, axial); per-target deltas on the 870. Decides M3.
3. **Teacher pass (about 75 min):** Dread v4_full K42 probabilities on all 4,345 cached studies, written privately with a SHA; never used as truth, only as a training target.
4. **Arm `in-mirror`:** in-orig config with mirror p = 0.5, side-swapped labels, MCL masked on mirrored samples, mirror TTA at export. Everything else frozen: labels_v1_blend, fold-0 split, seed 42, 12 epochs, K16 selection, K42 export.
5. **Arm `in-distil`:** in-orig config plus teacher BCE at fixed weight 0.5, mirror off, so each arm changes one factor against in-orig. If the GPU probe allows two concurrent arms, run both; otherwise mirror first.
6. **Stop rules:** the existing health stops; evaluate with `evaluate_raptor64_endpoint.py` against in-orig and own44; decide M1 and M2 by the kill criteria above. If both pass, launch `in-mirror-distil` on all 4,345 studies with pre-registered SWA epochs for the B probe (Oct 15–16), then one rehearsal and one hidden run (Oct 17–18).

## 4. What this means for final selection

- **A stays the banked public recipe** (Raptor + R7 0.5/0.5, 0.957 expected [P]); an original component enters A only if it reaches ≥ 0.94 public and is decorrelated, probability about 0.1 [E]. Do not spend A probes on 0.93-class own arms.
- **B is where original work lands:** best of B1/B2/B3 today (about 0.945–0.952 [E]), plus an own student at a fixed 0.2 slot for +0.002 to +0.005 [E]. B is also the no-OAI hedge, and the student makes it partly ours.
- **Realistic landing:** public A 0.957 ± 0.004, B about 0.947–0.952; the private rank within the 0.956 block is a lottery of ±0.004 [P/E]. Original methods will not produce a top-ten finish this round; they can produce a measurably better own model with clean evidence and a B entry that is partly ours, which is what the user asked us to invest in.
- **Remaining risks:** second-order exposure makes the student's dev gain an upper bound; the mirror change touches the sampler, so the identity test and window-count check must pass before launch; a new B arm needs a full 2×T4 rehearsal against the 9 h limit; the fastMRI question on R7 still applies to every B candidate.
