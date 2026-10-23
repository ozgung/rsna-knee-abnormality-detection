# Public scan, October 19 (bounded)

Sources: `kaggle kernels list` sorted by score (top 40) and by last run (60), plus one discussion search. Raw listings and three pulled notebooks are in `outputs/research/public-scan-20261019/` (not executed).

- **No public notebook or discussion claims 0.959.** The top notebook by score is still goodpjw2008 "CoAtNet + R6 + 2.5D MIL" (0.958, last run Oct 16). The 64 teams at 0.959 on the Oct 19 board have no published recipe. They are most likely private variants, or 0.958-family entries landing high within public noise (SD about 0.003).
- **New since Oct 18:**
  - evgendvorkin Versia-7 (re-run Oct 19 15:51) now reproduces the goodpjw2008 v27 recipe:
    - Raptor 384 and 224, both with anatomical mirror TTA, with 224 taking 0.35 of the Raptor slot;
    - R6 `E_lrprior` with flip TTA and an 8.3 h deadline;
    - per-finding Raptor weights at v1 (Medial OA 0.5, Lateral OA 0.2, PF OA 0.6, Synovitis 0.8, Baker's 0.35);
    - the goodpjw2008 ConvNeXt 2.5D MIL reader, rank-blended at the end with per-target weights: Baker's 0.035, Contusion 0.03, Medial OA 0.025, ACL 0.025, Medial Meniscus 0.02, MCL 0.02, Lateral Meniscus / Fracture / Lateral OA / PF OA 0.008, Effusion 0.004, Synovitis 0.003.
  - seanmoch tri-reader: a reproduction of Rabari's 0.954 notebook (CoAtNet 384/224 + three-fold ConvNeXt).
  - basith007 model-ensembling: a CoAtNet + DINOv2 feature notebook with no score claim.
- **Differences from our P2 (57340072):** the R224 mirror and the end-stage MIL leg. Both were already known from the Oct 18 scan. Together they are the remaining difference from the public 0.958 recipe, and the expected displayed effect is within noise.
