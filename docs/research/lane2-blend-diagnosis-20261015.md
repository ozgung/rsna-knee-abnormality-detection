# Lane 2 blend diagnosis: why the sagittal crop arm is worse when mixed in (2026-10-15)

Read-only diagnosis of the lane-2 read-out ([result](../experiments/lane2-sagittal-joint-crop-result-20261014.md), [protocol](../experiments/lane2-sagittal-joint-crop-protocol-20261014.md)). Fold 0, 870 development studies, Lateral Meniscus (LM) primary: 109 positive, 713 negative, 48 unobserved. All numbers below were recomputed independently of the evaluator (own rank, AUC, bootstrap calls) from the same pinned inputs. Author: Claude (Fable). Scratch scripts: `~/.claude/jobs/74614526/tmp/l2_analysis.py`, `l2_followup.py` (outputs `l2_analysis.json`, `l2_followup.log` there; not in the repo).

## Verdict

- **(a) No bug in the mix-in.** The evaluator's rank blend into in-distil, Dread and Raptor reproduces to 1e-6 (0.892977 / 0.950249 / 0.952411); IDs, order, target columns, masks, presence, rank direction and dtypes all check out (table below). The 2e-5 difference on the Dread blend between `(1-w)*rb+w*ra` and `0.7*rb+0.3*ra` is a 1-ulp floating-point tie artefact (29 vs 22 exact ties in the blended column, both averaged), not an error.
- **(b) It is worse because it is a weaker scorer that is highly correlated with the stronger one, so averaging adds noise, not information.** Within-class Spearman arm–Raptor 0.86–0.89, arm–Dread 0.79–0.86. Under a binormal model the arm helps only if d'(arm) > ρ·d'(base); here d'(arm) = 1.72 vs ρ·d'(Raptor) = 1.77 (LDA optimal arm weight −0.08) and ρ·d'(Dread) = 1.62 (optimal 0.12, but the unequal-variance curve still peaks at w = 0). Empirically **no weight w > 0 raises LM or macro AUC against Dread or Raptor for rank, logit or probability averaging**, for `sag-crop`, `sag-full` or the best single epoch. The residual of the arm given Raptor has AUC 0.489; the arm never beats Raptor by more than 0.1 on any positive and does so on 2 of 713 negatives, while Raptor beats the arm by more than 0.1 on 18 positives and 180 negatives. In the region where the strong models are uncertain (Dread rank 0.5–0.8, 228 studies, 13 positives) the arm is at chance (AUC 0.46 vs Raptor 0.85). There is no subgroup (manufacturer, field strength, site, slice thickness, pixel spacing, matrix, number of sagittal series) where the arm or the 0.3 blend beats Raptor. Positive control: Dread into Raptor at 0.3 gives macro +0.0071 [+0.0055, +0.0088], so blending a strong, different model does help; blending in-distil (0.878) into Raptor costs −0.0178, the arm (0.905) costs −0.0115. Lane 1 failed the same way for the same reason.
- **(c) No alternative mixing method is worth testing for A′.** Stacking (nested 5-fold by report component, logit features) gives LM +0.0036 [−0.0029, +0.0104] against Raptor and +0.0008 [−0.0029, +0.0047] against Dread, both CI-inclusive of 0 and below the A′ threshold (CI lower ≥ +0.0017); the in-sample stacker uses the arm with a **negative** coefficient (−1.9 vs +4.6 for Raptor, exposed), i.e. the only "signal" is "when the arm disagrees upward, trust it less", which is not a component worth shipping. Gating by disagreement: 0 of 90 configurations beat Raptor; nested −0.0029. Per-target weights, nested: macro −0.0002 [−0.0007, +0.0002] (Raptor), +0.0003 [−0.0005, +0.0011] (Dread). Max/min rank: 0.952 / 0.928 vs 0.964. Rank-of-logit equals rank-of-probability exactly (monotone), so it is the same method.

## Evidence

### 1. Mix-in audit (`scripts/evaluate_raptor64_axial_blend.py`, worktree copy that produced the read-out)

| Check | Result |
|---|---|
| ID order: in-distil CSV, arm CSVs, best-epoch NPZ, Raptor NPZ vs endpoint | identical, 870 unique, verified in code (`load_csv`/`load_npz` raise) and recomputed |
| Target columns | CSV header equals `TARGETS` order; LM index 3; no side swap (arm LM vs LM truth 0.905, arm LM vs MM truth 0.743, arm MM vs LM truth 0.635, arm MM vs MM truth 0.945) |
| Raptor / Dread arrays are probabilities, not ranks | yes (`control` float32 in [0.023, 0.997]; its rank reproduces the file's `control_rank`; Dread `probs` float32) |
| Rank direction / normalisation | ascending average ranks / n in (0, 1]; higher = more positive for every component; blend = 0.7·rank(base) + 0.3·rank(arm) |
| Presence / fallback | all 870 sagittal-present, 0 fallbacks; `lateral_meniscus_observed_absent` 0 |
| Masked LM rows (48) | included in the rank population, excluded from the AUC; ranking over the 822 observed instead changes the Raptor blend by +3e-5 |
| Ties / NaN / dtype | 1 tie in the arm LM column, 0 in the others; all finite; float64 |
| Fold / leakage | arm OOF = fold-0 dev; Raptor column = fold-0 public control; Dread = exposed teacher (known); labels file identical for in-distil and both lane-2 arms (`d2cdb8a3…`) |
| Independent recomputation of the three read-out blends | 0.892977 / 0.950249 (0.950230 with `0.7*rb+0.3*ra`, tie artefact) / 0.952411; bootstrap CIs reproduced |

### 2. AUC vs weight (rank blend; LM, macro over 12 targets)

| w | arm→Raptor LM | arm→Raptor macro | arm→Dread LM | arm→Dread macro | arm→in-distil LM | Dread→Raptor LM | Dread→Raptor macro |
|---|---|---|---|---|---|---|---|
| 0 | 0.9639 | 0.9513 | 0.9578 | 0.9596 | 0.8776 | 0.9639 | 0.9513 |
| 0.1 | 0.9607 | 0.9499 | 0.9560 | 0.9584 | 0.8831 | 0.9650 | 0.9541 |
| 0.2 | 0.9571 | 0.9479 | 0.9536 | 0.9562 | 0.8886 | 0.9662 | 0.9566 |
| 0.3 | 0.9524 | 0.9455 | 0.9502 | 0.9533 | 0.8930 | 0.9665 | 0.9584 |
| 0.5 | 0.9414 | 0.9390 | 0.9415 | 0.9455 | 0.9004 | 0.9666 | 0.9611 |
| 0.7 | 0.9282 | 0.9304 | 0.9296 | 0.9351 | 0.9048 | 0.9645 | 0.9618 |
| 1.0 | 0.9052 | 0.9151 | 0.9052 | 0.9151 | 0.9052 | 0.9578 | 0.9596 |

Logit and probability averaging are slightly less harmful (arm→Raptor LM at 0.3: 0.9551 / 0.9552 vs 0.9524 rank) but still monotone decreasing. `sag-full` and the best single epoch behave the same. Paired whole-component bootstrap (843 components, 2,000 draws, seed 42) at w = 0.1 / 0.2 / 0.3: arm→Raptor rank −0.0032 [−0.0053, −0.0015] / −0.0068 [−0.0108, −0.0034] / −0.0115 [−0.0178, −0.0062]; arm→Dread rank −0.0018 [−0.0036, −0.0002] / −0.0043 [−0.0076, −0.0011] / −0.0076 [−0.0127, −0.0030]. Against in-distil the best weight is 0.8 (0.9060, +0.028), i.e. the arm is simply the better of two weak models. Any weight chosen on the 870 is exposed; a nested 5-fold choice picks w = 0 in every fold against both Dread and Raptor.

### 3. Correlation and expectation

| Pair (LM, 822 observed) | Spearman all | positives | negatives |
|---|---|---|---|
| arm ~ Raptor | 0.890 | 0.889 | 0.858 |
| arm ~ Dread | 0.844 | 0.865 | 0.790 |
| arm ~ in-distil | 0.875 | 0.847 | 0.846 |
| Dread ~ Raptor | 0.904 | 0.852 | 0.865 |
| arm ~ sag-full | 0.906 | 0.925 | 0.876 |

The arm is as correlated with Raptor as Dread is, at 0.06 lower AUC. Across all 12 targets arm~Raptor Spearman is 0.76–0.95. Binormal (rank-to-normal scores, class-conditional covariances): d'(arm) 1.72, d'(Raptor) 2.08, ρ 0.85; d'(Dread) 2.05, ρ 0.79. Predicted LM drop at w = 0.3: −0.0069 (Raptor), −0.0045 (Dread); observed −0.0115 / −0.0076 (the model under-predicts absolute levels because the scores are not normal, but the sign and shape match). The arm's 0.5 × BCE distillation toward Dread explains part of the correlation (arm~Dread 0.68–0.93 per target on all 870) but the arm is even closer to Raptor, which it never saw.

### 4. Alternatives against the strong columns (LM; nested = 5-fold grouped by report component, choice made on 4/5, scored on 1/5; still fold-0-exposed inputs)

| Method | vs Raptor (0.9639) | vs Dread (0.9578) |
|---|---|---|
| Rank average, best w (exposed) | w = 0 | w = 0 |
| Logit / probability average, best w | w = 0 | w = 0 |
| Stacking logit[base, arm], nested OOF | +0.0036 [−0.0029, +0.0104] | +0.0008 [−0.0029, +0.0047] |
| Stacking logit[base, in-distil], nested | +0.0036 [−0.0024, +0.0091] | +0.0012 [−0.0015, +0.0040] |
| Stacking logit[base] alone, nested (fold-calibration noise floor) | −0.0017 [−0.0034, −0.0003] | −0.0005 [−0.0015, +0.0006] |
| In-sample stacker (exposed): standardised coefficients | base +4.60, arm **−1.90** (LRT χ² 34.5) | base +3.04, arm **−0.60** (χ² 6.2) |
| Gating by disagreement (90 configs), exposed best / nested | none beats base / −0.0029 [−0.0097, +0.0003] | +0.0008 / +0.0008 [+0.0001, +0.0019] |
| Per-target weights, nested, macro | −0.0002 [−0.0007, +0.0002] | +0.0003 [−0.0005, +0.0011] |
| max rank / min rank | 0.9523 / 0.9281 | 0.9416 / 0.9345 |

Nothing reaches the A′ gate (CI lower ≥ +0.0017). The stacker's positive point estimate rests on a negative arm coefficient and is within the fold-calibration noise of the base-only stacker.

### 5. Error analysis (LM, arm vs Raptor unless stated)

- Pair flips at w = 0.3: of 77,717 positive–negative pairs the blend loses 1,075 correctly ordered pairs and gains 180 (net −895 = −0.0115). 74 of 109 positives and 266 of 713 negatives lose pairs; one positive alone loses 137 (Raptor ranks it high, the arm low).
- Per-study contributions: positives where the arm is better by > 0.1: 0 (Raptor better: 18); negatives: 2 vs 180. Positives ranked below 80 % of negatives: arm 18, Raptor 2, Dread 4. Negatives ranked above half the positives: arm 24, Raptor 8. Top-20 % of the ranking: arm holds 83 positives and 87 negatives, Raptor 101 and 70; 59 of the arm's false alarms are shared with Raptor.
- Residual of the arm's normal score given Raptor's: AUC 0.489 (given Dread: 0.528); partial Spearman with the label −0.025 / +0.044.
- Hard region: among studies Dread ranks 0.5–0.8 (228, 13 positives) the arm scores 0.464, Raptor 0.850, Dread 0.719; among the top 20 % by Raptor (172, 101 positives) arm 0.618, Raptor 0.766, Dread 0.800. The arm separates easy cases only, which the strong models already do.
- Subgroups from DICOM headers (870/870 covered; groups with ≥ 8 positives): arm − Raptor positive contribution is negative in every group (−0.03 to −0.09): GE (181 studies) arm 0.886 / Raptor 0.975; Philips Medical Systems (134) 0.921 / 0.973; Siemens (169) 0.893 / 0.946; Siemens Healthineers (209) 0.889 / 0.955; Philips (50, 16 pos) 0.921 / 0.950; 1.5 T 0.905 / 0.961; 3 T 0.916 / 0.971; largest site (86) 0.857 / 0.931, all other sites pooled (527) 0.916 / 0.975; sagittal pixel spacing ≤ 0.4 mm 0.915 / 0.970 vs > 0.4 mm 0.881 / 0.949; matrix ≥ 512 0.914 / 0.972 vs < 512 0.892 / 0.954. The 0.3 blend into Raptor is below Raptor in every group. Only 24 development studies (2 positives) lack the non-FS sagittal series, too few to score.

### 6. Data path

- Crop code (`knee/raptor64_training.py` `sagittal_block`/`central_crop_box`): symmetric 65 % box on slots 0–31, no flip, masks sliced consistently; `sag-full` is the uncropped block. Crop vs full Spearman 0.905 on LM (0.86–0.97 per target), so the crop is a real factor, and it raised LM 0.892 → 0.905 alone.
- Labels: same `labels_v1_blend` file as in-distil; its LM soft label scores AUC 1.000 against the endpoint truth (10/822 cells differ at 0.5; masked rows carry a mean 0.10 soft label, a shared design, not lane-2 specific). The NPZ `truth` array is this soft label, not the endpoint truth; the evaluator correctly uses the endpoint.
- Training health: masked macro rose monotonically 0.766 → 0.915 over 12 epochs, SWA of epochs 9–11, K-all export (k_eval 30), 870 rows; best epoch and SWA Spearman 0.998.
- Montage (16 random slot-9 FS slices, private): the box is centred on the image, not the joint; in roughly a quarter of the samples the posterior femoral condyle, where the posterior meniscal horn sits, touches or crosses the box edge. This is a plausible reason the crop gains less than the magnification argument predicts, but it cannot make the arm look worse than `sag-full`, which it beats; it is not a defect in the blend.
- Calibration: arm mean LM probability 0.77 (positives) / 0.31 (negatives) vs Raptor 0.86 / 0.31; no degeneracy, range 0.07–0.997.

## Limits

- All inputs are fold-0 only; the 870 were used for SWA selection (exposed) and Dread saw them (second-order exposure). Bootstrap CIs are over report components, not over training seeds; 109 LM positives give SE about 0.02 on any single AUC.
- "Nested" estimates learn only the mixing choice; they cannot remove the exposure of the base and arm predictions themselves, and the base-only stacker shows a −0.0017 fold-calibration artefact, so stacking point estimates of a few 1e-3 are within noise.
- The binormal expectation is a sign/shape check only; absolute predicted AUCs are 0.01 low.
- Subgroup reads are descriptive with 3–27 positives each; no multiplicity correction. Site keys follow `knee/site_prior.py` (98 sites in the 870, largest 87).
- Crop geometry was inspected on 16 sample slices, not measured on the cache (not staged locally).
