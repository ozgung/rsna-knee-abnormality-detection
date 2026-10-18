# Iteration 4 protocol: axial patellofemoral arm, 2026-10-14

Pre-registered before any training. Decision trail: [cloud research](../research/knee-domain-research-20261014.md) item 3, [Claude's assessment](../research/iteration4-assessment-20261014.md), and Fable's review (summarised below).

## Hypothesis

A dedicated model on an anterior crop of the axial slots adds PF OA signal that the 64-slot global model leaves unused. In the E1 ablation, dropping axial cost PF OA almost nothing [measured]. The lever is magnification (about 1.8× at 384 input) and dedicated capacity, not pixel pitch: the cache already stores 336 px over 140 mm, about 0.42 mm/px.

## Arms (single factor: the crop)

| Arm | Input | Everything else |
|---|---|---|
| `ax-crop` | Slots 52–63, the 12 axial slices of the existing raptor64 336 cache. Anterior 55% × central 65% of each slot, bilinear upsample to 384. | Same as in-distil (see below) |
| `ax-full` (control) | Slots 52–63, the full 336 slot resized to 384 | Same as in-distil (see below) |

Shared recipe, matching in-distil:
- CoAtNet-384 with ImageNet init and per-finding attention MIL heads over all 12 targets;
- 3-slice windows over the 12 axial slices, giving 10 windows per study, all used in training and evaluation;
- report-label loss plus 0.5 × BCE toward the pinned Dread v4_full K42 teacher;
- grouped fold 0, the same seed policy, 12 epochs, top-3 SWA.

Studies with no axial series have an all-masked axial block. They are excluded from arm training, and at evaluation they fall back to in-distil alone.

## Orientation

First check whether `knee.raptor_input.prepare_study` canonicalises in-plane orientation from IOP.
- **If it does,** "anterior" is a fixed image side, and the side is documented with the code reference.
- **If it doesn't,** apply this heuristic before training: the anterior side is the one whose soft-tissue centroid lies farther from the posterior soft-tissue mass. Verify the heuristic by eye on 20 cached studies, and record failures.
- **In both cases,** train with p = 0.5 anterior–posterior flip augmentation, with labels unchanged, so that rare orientation errors are tolerated.

## Primary metric and kill rule

- **Primary metric:** PF OA AUC of a fixed 0.7 × in-distil + 0.3 × arm rank blend, against in-distil K42 alone, on fold 0 (870 studies, masked weak labels), with a paired bootstrap CI.
- **Keep `ax-crop` only if all of these hold:**
  1. PF OA Δ ≥ +0.010 with CI lower bound > 0;
  2. arm-alone PF OA ≥ 0.80;
  3. the sign of Δ agrees across the SWA export and the best single epoch;
  4. `ax-crop` beats `ax-full` on the same blended PF OA, which shows the crop is the active factor;
  5. a secondary gate before any submission: the same 0.3 blend into the pinned Dread v4_full K42 fold-0 predictions must not lower PF OA.
- **Secondary read only, not blended in the primary:** Effusion and Synovitis.
- **No weight tuning on fold 0.**
- If `ax-crop` fails, close the axial-arm idea; a native-resolution rebuild is not justified.

## Final-entry path if it passes

- **Training:** the arm is retrained on full data. It reads Raptor's already prepared axial slots, so there is no new DICOM path.
- **Blending:** it goes into A and B1 per target, PF OA only, at a fixed 0.3 rank weight.
- **Runtime:** expected T4 cost is a few minutes for about 1,300 studies, to be measured.
- **Gain:** realistic gain at entry level is macro +0.0004 to +0.0013 [estimate, Fable]. That is within public-leaderboard noise, but it is the only research item that can reach the final entries before Oct 22.

## Not run in this iteration

| Item | Reason |
|---|---|
| Item 1 | Already mostly covered by in-distil's teacher-on-masked loss; an OAI teacher would also break B eligibility. |
| Item 2 | Needs Laterality/IOP or segmentations, which are absent from the Colab cache. |
| Item 4 | Codex's M4 lane. |
| Item 5 | No per-window logits. |

A second-seed replicate of the iteration-3 winner will be decided after the iteration-3 read-out.
