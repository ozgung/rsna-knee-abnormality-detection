# Lane 2 protocol: sagittal joint-line crop arm (Lateral Meniscus primary), 2026-10-14

Pre-registered 2026-10-14 at about 15:30 UTC, before any implementation or training. Author: Fable (Claude), strategy. Owner of execution: a Colab training agent of the same type as lane 1's. Lane 1 (the axial PF arm, [protocol](raptor64-axial-pf-it4-protocol-20261014.md)) is untouched: its Colab sessions, branch `claude/axial-pf-it4-20261014`, datasets and arms `ax-crop`/`ax-full` are read-only for this lane.

Decision trail: [plan of record](../research/plan-update-20261014.md) (own-model lane closes after iteration 4 unless a component can reach A′/B1′ by the Oct 17 rehearsal), [iteration-2 E1 plane ablation](raptor64-original-methods-it2-result-20261014.md), [iteration-3 result](raptor64-original-methods-it3-result-20261014.md), [domain research](../research/knee-domain-research-20261014.md) sections 2, 3 and 6 (items 3 and 4). Labels: **[measured]** = number in a receipt or repo file; **[literature]** = other participants or clinical sources; **[estimate]** = my judgement.

## Why this slot group and not the others

| Candidate on the 336 cache | Verdict | Reason |
|---|---|---|
| **Sagittal joint-line crop (slots 0–31), this lane** | run | Sagittal carries the two targets with the largest remaining gap that are measurable on fold 0: Lateral Meniscus 0.8776 (109 positives, SE about 0.022) and ACL 0.9456 (109 positives). E1: dropping sagittal costs LM −0.103 and ACL −0.122 [measured]. A symmetric central crop needs no orientation audit, because the 140 mm field is a centre crop in mm (`RAPTOR_RECIPE`, `knee/raptor_input.py`). |
| Coronal medial MCL crop (slots 32–51) | not run | MCL has 20 fold-0 positives (SE 0.048): no pre-registered gate can resolve it. A medial crop needs the side, which is absent from the Colab cache (`Laterality` is on 49% of DICOM series, not in the cache). The distal MCL insertion lies 50–70 mm below the plateau and leaves any central crop [literature]. |
| Coronal Lateral OA specialist | not run | E1: dropping coronal did not lower Lateral OA; dropping sagittal raised it +0.007, within noise (SE 0.028) [measured]. No mechanism the global model lacks. |
| Side-split sagittal (lateral third for LM) | not run | Needs laterality; iteration 3's flip result says the model uses absolute side as a cue, so an unsided split is unsafe. |

## Hypothesis

A dedicated model on a central crop of the sagittal slots adds Lateral Meniscus signal that the 64-slot global model leaves unused. The global model already reads sagittal for LM, so the lever is not plane access but magnification (about 1.76× at 384 input) and dedicated capacity for 2–5 mm structures (menisci, cruciates) that occupy 5–12 px at the cache's 0.42 mm/px. The crop is centred on the joint, which the 140 mm centre crop already places near the image centre.

Prior: lane 1's lever is identical and is itself untested; the domain research puts LM headroom at +0.01 to +0.02 with localisation [estimate]. I put P(primary passes) at about 0.3 [estimate].

## Arms (single factor: the crop)

| Arm | Input | Everything else |
|---|---|---|
| `sag-crop` | Slots 0–31 (18 sagittal fluid-sensitive + 14 sagittal non-FS) of the existing raptor64 336 cache. Central 65% × 65% of each slot (218 × 218 px, about 91 mm), bilinear upsample to 384. | Same as in-distil (below) |
| `sag-full` (control) | Slots 0–31, the full 336 slot resized to 384 | Same as in-distil (below) |

Shared recipe, matching in-distil and the lane-1 arms:
- CoAtNet-384, ImageNet init, per-finding attention MIL heads over all 12 targets;
- 3-slice windows over the valid sagittal range; up to 30 windows per study (16 when one of the two sagittal series is missing). **Training samples K = 10 windows per study**, every centre once, shuffled (the lane-1 `_centers` rule), so the per-step cost equals lane 1's. **Selection and export use all valid windows ("K-all", nominal K30).**
- report-label loss plus 0.5 × BCE toward the pinned Dread v4_full K42 teacher (`dab0621a…`), masked targets included as in `DistilBCE`;
- grouped fold 0, the same seed policy, 12 epochs, bs 4 with accumulation 2, the frozen learning rates, top-3 SWA by the per-epoch K-all report-masked macro on the 870.
- **No flips.** An AP flip inverts anterior/posterior anatomy (the ACL runs posterior-superior to anterior-inferior); a left/right in-plane mirror of a sagittal slice is the AP flip. Iteration 3's design rule (every geometric augmentation states its invariant) is satisfied by adding none.

Studies with no sagittal content have an all-masked sagittal block. They are excluded from arm training and fall back to in-distil alone at evaluation. The count is recorded; I expect close to zero, since sagittal is the base plane [estimate].

Crop geometry justification [literature]: the tibial plateau is about 45–55 mm anterior–posterior and the femoral condyle about 60–70 mm; a 91 mm window centred on the image centre contains both menisci horns and the whole notch with at least 10 mm margin whenever the joint centre is within about 15 mm of the field centre. 65% rather than 60% buys margin at the cost of magnification (1.76× instead of 1.9×). The box is symmetric, so in-plane orientation does not matter. No per-study localisation is used.

## Baseline

in-distil K42 on the fixed 870 (fold 0, masked weak labels) [measured, `outputs/analysis/raptor64-20261013/it3/endpoint-it3-v1.json`]: Lateral Meniscus 0.8776 (109/713), ACL 0.9456 (109/716), Medial Meniscus 0.9423 (330/496), macro 0.91490. The pinned Dread v4_full K42 fold-0 predictions (exposed, 0.9596 on these 870) serve the transfer gate only.

## Primary metric and pre-registered kill rule

- **Primary metric:** Lateral Meniscus AUC of a fixed 0.7 × in-distil + 0.3 × `sag-crop` rank blend against in-distil K42 alone, on the 870, paired whole-component bootstrap (843 report components, 2,000 draws, seed 42), the same evaluator family as lane 1 (`scripts/evaluate_raptor64_axial_blend.py`, parameterised by target and arm).
- **Keep `sag-crop` only if all of these hold:**
  1. LM Δ ≥ +0.010 with CI lower bound > 0;
  2. arm-alone LM ≥ 0.80 on sagittal-present studies;
  3. the sign of Δ agrees across the SWA export and the best single epoch;
  4. `sag-crop` beats `sag-full` on the same blended LM, which shows the crop is the active factor;
  5. transfer gate before any submission: the same 0.3 blend into the pinned Dread v4_full K42 fold-0 predictions must not lower LM.
- **Secondary entry targets, ACL and Medial Meniscus:** each enters the entry (same fixed 0.3, its own target only) only if the primary passes **and** that target's own blended Δ has CI lower bound > 0 **and** passes gate 5 for that target. If the primary fails, nothing enters even if a secondary looks good: the crop idea is then closed for this slot group. Targets that do not enter are untouched in the entry, so no do-no-harm test is needed for them.
- **No weight tuning on fold 0.** 0.3 is fixed here, as in lane 1.
- **Training-health stop (not a scientific gate):** epoch-1 K-all macro < 0.75 stops that arm. The bound is lower than iteration 3's 0.80 because a sagittal-only arm is expected to score lower on axial-dependent targets (PF OA, Effusion, Baker's).
- **Schedule kill:** if the fold-0 read-out is not available by **Oct 15 18:00 UTC**, the lane closes; a full-data retrain, Codex integration and the Oct 17 rehearsal cannot all fit afterwards.
- **Infrastructure kill:** as lane 1: per-minute RAM and uptime logs; a VM that dies in under 40 min twice stops the lane, and the agent reports instead of chaining further.

## Data and compute

- **Data:** the existing raptor64 v4 336 cache (`raptor64-v4-336-sharded-v1`, content `0bdc189c…`), staged in full (about 29 min). The unchanged loader reads 64-slot volumes; the arm slices 0–31. A derived sagittal-only cache (the lane-1 `scripts/raptor64_axial_cache.py` derive step with a slot-range parameter, Kaggle CPU, no GPU quota) is built **only** if lane 1's investigation attributes its VM deaths to a lifetime cap that makes the 29 min staging decisive.
- **GPU:** **H100 80 GB preferred**; the code is device-agnostic (fp16 autocast, `timm==1.0.26`, the controller probe records `get_device_name`). Both arms concurrent on one VM as in iteration 3 (8 GB each at K8; about 10 GB each at K10 [estimate]). Fall back to an A100 40 GB if the H100 probe fails or none allocates within 15 min. Record host RAM at the probe; if lane 1's root cause turns out to be host RAM, run the two arms sequentially.
- **Cost [estimate]:** A100: about 25 min/epoch for both arms (iteration 3 ran 21 min/epoch at K8/K16; K10 training and K-all selection add about 20%), so about 5 h training plus 29 min staging plus about 15 min K-all export, about 30 units at 5.3 units/h. H100: about 3–3.5 h in total at 1.5–1.8× A100 throughput; the H100 unit rate is unverified, so budget 35–45 units. Kaggle GPU quota: **zero** for this lane until the rehearsal.
- **Timeline [estimate]:** implementation and tests 2–3 h; launch by about 18:30 UTC today; read-out about 22:00–01:00 UTC; decision recorded in a result note before Oct 15 09:00 UTC.

## Final-entry path if it passes

- **Full-data retrain** on all 4,345 studies with the fold-0 run's three SWA epochs pre-registered as fixed epochs (the existing full-data mode), about 4–5 h A100 or about 3 h H100; then gate 5.
- **Blending:** Codex adds the arm as a per-target component of the frozen R7 wrapper: Lateral Meniscus, plus ACL and Medial Meniscus if they passed, at a fixed 0.3 rank weight each; weights parameterised, no tuning; one regression test that the other targets are byte-identical to A/B1.
- **Runtime [estimate]:** about 30 windows × 1,300 studies, about 39k forwards at 384 fp16 on one T4, 10–15 min; to be measured in the rehearsal. **Priority in A′:** lane 1's PF OA arm first. If the conservative projection of A′ with both arms exceeds the 8.3 h gate, this arm drops from A′ and enters B1′ only (B1 runs one worker and has the larger margin).
- **Rehearsal and submissions:** rides the plan's step-2 rehearsal after the Oct 17 quota refresh (about 0.5 h T4 shared with lane 1's arm), then A′ on Oct 17 and B1′ on Oct 18. No separate hidden run is spent on this arm alone.
- **Expected entry gain [estimate]:** LM +0.003 to +0.010 at target level, so macro +0.0003 to +0.0008; with ACL and MM at most +0.0010 in total. Public arms' LM is unknown, so the lower end is as likely as the upper. In rank terms, lane 3's model prices each +0.001.

## Implementation notes (files the Colab training agent must touch)

Branch from `claude/axial-pf-it4-20261014` (the crop-arm infrastructure lives there and is unmerged) into a new branch such as `claude/sagittal-joint-crop-20261014`; do not edit the `ax-*` arms, their specs or their tests.

- `knee/raptor64_training.py`: add `sag-crop` and `sag-full` to `ARMS`/`INIT`/`VARIANTS`/`DISTIL_WEIGHTS`; generalise `AXIAL_SLOTS`/`AXIAL_SPECS`/`axial_variant` into a plane-group spec (slot range `(0, 32)`, central crop box `{'central_h': 0.65, 'central_w': 0.65}`, train K 10 with every-centre-once sampling, selection/export K-all, no flip, exclusion of studies without sagittal content); the policy guard must pin the spec as `AXIAL_SPECS` is pinned.
- `scripts/build_raptor64_training.py`: `ARMS`/`INIT`, `arm_windows()` returning (10, all, all) for the sagittal arms, `comparison_endpoint()` for K-all.
- `scripts/colab_raptor64_run.py`: the variant branch now keyed `'axial'` becomes a plane-group variant; probe K environment values; export K-all.
- `scripts/build_raptor64_colab_bundle.py`: build the two packages with the pinned teacher (the `--iteration4-teacher-sha256` pattern).
- `scripts/evaluate_raptor64_axial_blend.py`: parameterise arm, control, primary target and secondary targets (or add a sibling evaluator that imports its functions); gates 1–5 computed for LM, ACL and MM.
- Tests: `tests/test_raptor64_variants.py`, `tests/test_evaluate_raptor64_axial_blend.py`, `tests/test_raptor64_e2e.py`, `tests/test_colab_raptor64_run.py`, `tests/test_build_raptor64_training.py` if present: crop-box geometry, byte-identity of the uncropped `sag-full` slices with the cache, K-all window counts for 32- and 18-slot valid ranges, exclusion of sagittal-absent studies, and the evaluator's fallback for them.
- Optional: `scripts/raptor64_axial_cache.py` slot-range parameter, only under the derived-cache condition above.
- Reporting: a result note `docs/experiments/lane2-sagittal-joint-crop-result-20261014.md` with the gate table, per-target Δ for all 12, run identities, and private dataset names; evidence under `outputs/analysis/raptor64-20261013/lane2/`; a handoff log entry.

## Not run in this lane

| Item | Reason |
|---|---|
| FS-only sagittal arm (slots 0–17) | A second factor; the two-series block keeps the non-FS views radiologists use for menisci and ACL [literature]. |
| Native-resolution rebuild | Codex's M4 lane with its Oct 16 kill date; this lane's lever is magnification on the existing cache, with no new DICOM path. |
| Two-slice-consistent pooling head (item 5) | No per-window logits are exported; lowest-confidence item. |
| Any weight above 0.3 or per-target weight search | Pre-registered fixed weight; lane 3 explains why fold-0 weight fitting is invalid for the entries. |
