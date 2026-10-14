# Lane 2: sagittal joint-crop arm — result (October 14–15, 2026)

**Verdict: CLOSED.** Kill rule 5 fails (the arm lowers the Dread teacher's Lateral Meniscus AUC), and so does the A′ Raptor-column gate. The arm enters no entry.

- **Protocol:** [lane2-sagittal-joint-crop-protocol-20261014.md](lane2-sagittal-joint-crop-protocol-20261014.md), with the Raptor-gate addendum in commit f0c42fc.
- **Readout:** `outputs/analysis/raptor64-20261013/lane2/lane2-readout-v1.json` (ignored). Evaluator command: `outputs/lane2/evaluate.sh` in worktree `agent-a72620776aac37fa2`.

## Run

- **Arms:** `sag-crop` and `sag-full`, 12 epochs each on one Colab A100 (`raptor64-sagcrop-l2-20261014`), fold 0, using the chain driver `scripts/colab_raptor64_chain.py`.
- **Timeline:** launched 16:36 UTC; terminal and stopped 23:43 UTC.
- **Checkpoints:** f8de77da… (crop) and 89495259… (full).
- **Terminal phase `export_failed` for both arms:** the Kaggle backup datasets (`ozgung/knee-raptor64-sag-{crop,full}-fold0-final-20261014`) were not confirmed, because the metadata was inaccessible. The OOF CSVs, best-epoch NPZs, run receipts and logs were all collected locally under the worktree's `outputs/lane2/final/` (27 files). The checkpoints were not preserved off the VM; that is acceptable because the lane is closed.
- **Sagittal coverage:** sagittal present for all 870 development studies. The Lateral Meniscus support is 109 positive, 713 negative and 48 unobserved.

## Gates (primary target: Lateral Meniscus, K42 paired bootstrap, 2,000 draws)

| Rule | Result | Pass |
|---|---|---|
| 1. Blend vs in-distil Δ ≥ +0.010, CI lower > 0 | 0.8776 → 0.8930, Δ **+0.0154** [+0.0075, +0.0234] | yes |
| 2. Arm alone LM ≥ 0.80 (sagittal present) | 0.905 | yes |
| 3. SWA and best-epoch signs agree | agree | yes |
| 4. Crop beats full (blended) | Δ +0.0057 [+0.0006, +0.0106] | yes |
| 5. Dread gate not lowered | 0.9578 → 0.9502, Δ **−0.0076** [−0.0127, −0.0030] | **no** |
| A′ Raptor gate: Raptor-column blend LM CI lower ≥ +0.0017 | Raptor alone 0.9639; Δ **−0.0115** [−0.0178, −0.0062] | **no** |

- **Secondary entries (ACL, Medial Meniscus):** both have a positive CI against in-distil, but both fail the Dread gate, so neither enters.
- **Arm-alone macro:** 0.915 (crop) and 0.914 (full).

## Reading

The crop helps our weak in-distil model; it is the best own sagittal arm so far, with LM 0.905 alone. It still sits well below the teacher-grade columns (Dread at 0.958, Raptor at 0.964 LM), and blending it in dilutes them. Lane 1 (axial PF) failed the same way. Own fold-0 arms at about 0.915 macro cannot add to OAI-pretrained Raptor columns at a 0.3 weight. That matches Fable's no-new-lane and no-external-data verdicts (`docs/research/external-data-lane-20261015.md`).

**Limits:**
- Report-selected development: the top-3 SWA was chosen on the same 870.
- The labels are masked weak report labels.
- Dread saw these studies (second-order teacher exposure).
- This is not a public-score forecast.
