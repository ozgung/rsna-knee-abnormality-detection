# Brief: fine-tuning recipe (cloud session, Oct 15, 2026)

Written by Claude for a cloud Claude session that the user starts from the web. The local session coordinates. Read this brief, `STATUS.md`, and the end of `docs/claude-astra-handoff.md` first.

## Setup
- **Repo:** `ozgung/rsna-knee-abnormality-detection`, branch `codex/token-efficient-instructions`.
- **Not available:** only committed files are present. The ignored `data/` and `outputs/` folders are absent, and there is no Kaggle or Colab access. Don't try to get any.
- **Use:** `docs/`, `STATUS.md`, `knee/`, `scripts/`, `tests/`.
- **Deliverable:** one new doc, `docs/research/finetune-recipe-20261015.md`. Commit only that doc, to a **new branch `claude/finetune-recipe-20261015`**, and push it. Don't touch other files or branches.
- **Privacy:** no patient IDs and no credentials.
- **Effort:** use maximum reasoning effort (the user's "Fable xhigh").

## Goal
Design a fine-tuning recipe that is genuinely promising, and that we can test at small scale first. The user will dedicate the time and GPU it needs: Colab A100/H100, where units are not binding; Kaggle 2×T4 quota of about 8.9 h, refreshing on Oct 17.

**Competition:**
- RSNA Knee Abnormality Detection: 12 knee-MRI findings, scored by study-level macro AUC.
- About 1,300 hidden studies, run offline on 2×T4 with a 9 h limit.
- 5 submissions per day; a hidden score takes about 4–7 h.
- Last hidden submission about Oct 20, 16:00 UTC; final selection Oct 21; deadline Oct 22, 23:59 UTC.

**The user's position:**
- Blending public models alone is not competitive, since about 1,100 teams sit on the same blend.
- The leaderboard is an acceptable test, even for leaky models.
- A fine-tuned model should beat its base and replace it in the ensemble. **Critique this.**
- Rather than continuing to train on the same raw inputs, which risks overfitting or washing out what the base learned, the user wants a strong **knee-pretrained init adapted to a different, processed view of the data, or given extra input tokens**:
  1. Raptor or S6 init fine-tuned on knee-centred or compartment crops at higher local resolution, using SKM-TEA orthofoundation anchors.
  2. Extra tokens or channels: sequence, plane or fat-suppression tokens, slice-position tokens, an anatomy-mask channel, laterality.
  3. Plane or sequence subsets, or multi-scale input.
- Weak labels can be improved. A local lane is building labels v2 in `docs/experiments/labels-v2-20261015.md`; it may not exist yet.

## Facts (verify in the docs)
- **Entries:**
  - **A** = nartaa R384 CoAtNet (`raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt`, with the two-view anatomical mirror) + heliosli R7 (four-fold ConvNeXt-small), blended 0.5/0.5 on percentile ranks. Public 0.956.
  - **A′** = A with R224 taking 0.35 of the Raptor half (R384 keeps 0.65). Submitted as 57229227.
- **Runtime:** there is no headroom. A′ projects about 8.6 h conservative. R7 takes about 65% of the time, R384 16%, R224 10%. The R7 prefetch saved only about 11%. See `docs/research/runtime-headroom-plan-20261015.md`.
- **Lineage:**
  - nartaa's R384 (0.949 solo) was fine-tuned from dreaddevelopment's Raptor, which is CC0 and was trained on this competition. It used all data, blended JEV labels, 2,399 OAI knees as masked supervision, 96-slice depth, 384 px and SWA; gold-58 was used for selection. The R224 sibling scores 0.945. See `docs/claude-astra-handoff.md`, roughly lines 85–110.
  - heliosli's R7 and S6 are ConvNeXt-small with per-finding spatial attention, window MIL, SKM-TEA knee-centred local views and fastMRI aux.
  - nartaa's S6fall is the S6 architecture trained on all data with JEV-blend soft labels: 0.950 solo.
- **Our own training** (`docs/experiments/raptor64-*.md`, `docs/experiments/own-base-gap-audit-20261012.md`):
  - Own224 from ImageNet: 0.914 public.
  - Raptor64 arms from ImageNet: K42 0.907–0.916 on fold 0.
  - Recipe: `train_knee.py` defaults (OneCycle, 10–12 epochs, bb_lr 3e-5, head_lr 1e-3, K8 windows, top-3 SWA).
  - Distillation from the Dread teacher gave +0.008.
- **The crop arms failed, mainly because they started from ImageNet**; arms alone reached only about 0.86–0.915:
  - axial PF crop: `docs/experiments/raptor64-axial-pf-it4-result-20261014.md`;
  - sagittal crop, whose box was centred on the image, not the joint: `docs/experiments/lane2-sagittal-joint-crop-result-20261014.md` and `docs/research/lane2-blend-diagnosis-20261015.md`;
  - the ROI reader qualification left anatomy UNVERIFIED and fell back to the global view: `docs/experiments/same-series-extent-roi-qualification-20261014.md`.
  - The rank-77 team's anatomy-localised arm reaches 0.959 on top of a public base: `docs/research/public-scan-20261015b.md`.
- **Dread-init fine-tunes:**
  - `dr-orig` (Dread v4 init plus our recipe) scored K42 0.9471 on fold 0, exposed. Dread itself scores 0.9596 there, also exposed.
  - `dr-full` was trained on all data but has never been scored on the leaderboard. A swap probe, B0′, is being built: dr-full replaces Dread in B0, which is submission 57119585 at 0.940.
- **Evaluation:**
  - There is no clean held-out set, because the public models trained on everything.
  - The public paired noise SD is 0.0004–0.0009, and each true +0.001 is worth about 356 places.
  - Leaderboard: 10th place 0.964; we are 909th at 0.956.

## Deliverables (all in the doc: verdict first, then tables, plan, timeline and limits)
1. A critique of "a fine-tune beats its base", using the dr-orig evidence.
2. Two to four concrete recipes, including the processed-input ones, ranked by P(true gain ≥ +0.001 in the final ensemble) and expected Δ. For each recipe give:
   - base checkpoint;
   - input processing;
   - labels;
   - resolution and depth;
   - LR schedule (layer-wise decay, warmup, EMA/SWA);
   - augmentation;
   - epochs and A100-hours;
   - inference cost, and which ensemble member it replaces or how much headroom it needs.
3. A small-scale validation plan that discriminates despite the leakage. Options include leaderboard swap probes, equal-exposure paired arms whose *difference* is informative, and gold-58. Include kill rules and a timeline that fits the dates above.
4. Your view on the dr-full B0′ probe.
5. The first experiment to launch now: exact data, compute, duration and acceptance criteria.

When you finish, also append a short dated entry, signed "Claude (cloud) → Codex", to the end of `docs/claude-astra-handoff.md`, on the same new branch, saying where the doc is.
