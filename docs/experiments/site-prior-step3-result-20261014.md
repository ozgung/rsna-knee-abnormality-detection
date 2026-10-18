# Step 3, DICOM site/protocol prior: result, 2026-10-14

**Verdict: closed under the pre-registered rules.** The transfer gate failed, and an ACL drop exceeded the 0.005 limit. Pre-registration: [plan update, step 3](../research/plan-update-20261014.md).

## Runs

**Header kernel.**
- Kernel: `ozgung/knee-header-meta-20261014` v1, private, CPU, offline. It ran for 754 s.
- Coverage: 24,371 train series and 15 test series, with 0 read errors.
- Test tags are a subset of train tags: there are no test-only tags, and 19 train-only tags come from diffusion and parallel-imaging extras.
- `PatientID` is present in the headers but is not written to the output.
- Code: `outputs/kaggle/header-meta-20261014/header_meta.py`. Output: `outputs/analysis/site-prior-20261014/kernel-v1/` (ignored).

**Fit.**
- Code: `knee/site_prior.py` and `scripts/site_prior_fold0.py`, commit 9c2c6c8, with 6 tests in `tests/test_site_prior.py`.
- Training: 3,475 studies, all with headers; 43 site buckets with at least 20 studies each.
- C chosen per target by inner 5-fold CV: [0.3, 0.01, 0.3, 0.1, 0.1, 0.1, 0.3, 1, 1, 1, 0.3, 1].
- Truth: `train_labels_v1` y ≥ 0.5 under the m mask. On the image npz this reproduces the it2 per-target AUCs within 0.01.

## Results (fold 0, 870 studies, paired whole-study bootstrap with 2,000 draws)

| Read | Macro Δ | 95% CI | Notes |
|---|---|---|---|
| Primary: in-distil + 0.5 × meta shift | +0.0030 | [+0.00003, +0.0062] | ACL **−0.008** (kill: drop > 0.005); PF OA −0.003; Effusion +0.012, Fracture +0.010, Synovitis +0.008, Lateral OA +0.007 |
| Transfer gate: Dread v4_full K42 + the same shift | **−0.0016** | [−0.0039, +0.0006] | Must be ≥ 0; it fails |

- **Image predictions used.** The primary read used the local in-distil SWA export (`raptor_oof_fold0_swa_fold0.npz`, macro 0.9130 under this evaluation), not the K42 evaluation (0.9149), which is not available locally.
- **Why the verdict is still final.** The transfer gate depends only on the meta model and the pinned Dread predictions, so a K42 substitution cannot change it.
- **Meta-only AUCs:** ACL 0.656, MCL 0.538, MM 0.564, LM 0.621, MOA 0.632, LOA 0.666, PF 0.646, Effusion 0.683, **Synovitis 0.776**, Baker's 0.705, Contusion 0.601, Fracture 0.663.

## Reading

- Site and protocol carry real label signal, strongest for Synovitis. A strong image model already absorbs most of it, which is why Dread got worse.
- The weak own model gains about +0.003, but it pays on ACL, so this is not a component for A or B1.

## By-product

`Laterality` is populated on 49.3% of train series, and `ImageLaterality` is empty. This gives a free partial side oracle for plan step 5 (side canonicalisation), if iteration 3's flip arm wins.
