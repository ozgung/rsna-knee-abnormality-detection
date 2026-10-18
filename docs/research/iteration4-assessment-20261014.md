# Iteration 4 assessment of the cloud domain research, 2026-10-14

Claude's reconciliation of [the cloud report](knee-domain-research-20261014.md) against repo facts. The decision on iteration 4 is final only after iteration-3 results arrive (due about 10:35 UTC plus evaluation).

## Bottom line

None of the five items is a route to 0.963 by itself. The report estimates a solo own-model gain of +0.002 to +0.005 macro per item on fold-0 [estimate]. Our iteration-3 decision already estimated that an own student moves entry B by at most about +0.002.

The one item that can reach the final entries directly is a **target-specific blend component**. An axial patellofemoral arm blended only into PF OA (and perhaps Effusion and Synovitis) adds a signal that the public OAI-trained arms cannot hold, because OAI has no native axial fluid-sensitive series [literature]. That makes item 3 the iteration-4 lead.

## Item-by-item reconciliation

| # | Item | Repo fact that changes it | Verdict |
|---|---|---|---|
| 3 | Axial patellofemoral native-resolution arm | E1 ablation: PF OA barely uses axial [measured]. A small model on anterior axial crops costs minutes of T4 time. | **Lead.** The blend weight of 0.3 is fixed before testing and applied per target; it doesn't consume the R7 runtime budget. |
| 2 | Side canonicalisation + compartment-aware pooling | The public S6 bundle already ships a side classifier (`side_cls.pt`, used in `knee/public_s6_entry.py`). Iteration-3 `in-distil-flip` tests mirroring without a label swap but without canonicalisation (epoch-4 K16: 0.8686 vs w1 0.8927). | **Second.** Gated on the flip result. If the flip hurts, canonicalisation becomes the explanation to test. If it helps, stage A is the flip plus a canonical frame. |
| 1 | Multi-teacher distillation that fills masked labels | **Already done in part.** `DistilBCE` applies 0.5 × BCE toward the teacher on all targets, masked ones included (`knee/raptor64_training.py:610`). `in-distil-w1` tests the weight. The nartaa teacher is OAI-trained, so a student built from it is not eligible for no-OAI entry B. | **Deprioritised.** The only new factor left is a second no-OAI teacher. Revisit only if w1 shows that distillation isn't saturated (Δ ≥ +0.003). |
| 4 | Fixed physical FOV for native crops | This is Codex's M4 containment blocker. It matches my extent-plus-margin advice from 08:41. | **Route to Codex.** Offer a fixed 120 mm sagittal FOV centred on the cartilage centroid as the simpler alternative inside the same pre-registered ≤ 2% truncation gate. |
| 5 | Two-slice-consistent meniscus pooling | Needs per-window logits. The current exports are study-level. | **Parked.** Lowest confidence. |

## Item 3 sketch (to become a protocol)

- **Hypothesis:** a native-resolution anterior-axial arm adds PF OA signal that the 64-slot global model misses.
- **Baseline:** in-distil on fold-0, K42.
- **Input:** the axial PD FS series only.
  - The crop is the anterior 55% × central 65% of the field, oriented by IOP (anterior is not assumed to be image-up).
  - Native pixel pitch, resized to 384. All axial slices up to 24 three-slice windows.
- **Model:** ImageNet-initialised, with per-finding attention MIL over the axial windows; same split, seed policy and SWA as in-distil.
- **Primary metric:** PF OA AUC of the fixed 0.7/0.3 rank blend with in-distil vs in-distil alone, with a paired bootstrap CI.
- **Kill rule (pre-registered):** keep only if PF OA Δ ≥ +0.010 with CI lower bound > 0, and neither Effusion nor Synovitis drops by more than 0.005 when blended. No weight tuning on fold-0.
- **Open checks:**
  - axial series presence and missing-axial fallback, which falls back to in-distil;
  - the IOP sign convention, reusing the orientation-audit helpers;
  - the cache budget.

## Final-entry relevance

- **A (0.956):** a PF OA-only blend at 0.3 changes one of 12 targets. Even +0.02 on PF OA adds about +0.0017 macro.
- **B1 (0.949):** the same arithmetic applies; B1 is R7 alone, so there is more room.
- **Cost:** both need a new Kaggle submission with the arm attached, about 5–10 min of extra T4 time against the measured R7 runtime.
