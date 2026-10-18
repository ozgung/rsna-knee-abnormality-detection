# Source96 fold-0 secondary result (K94 plain, header-gated mirror TTA) — 2026-10-12

Run by Claude after taking over from Codex. One `colab exec` of `outputs/analysis/source96-training-20261012/vm-run-secondary-841fcfe-fold0gate.py`, which is Codex's wrapper with the export-receipt gate narrowed to fold 0. A100 session `source96-fold0-r2-20261012`, started about 20:38 UTC, return code 0 at 20:58 UTC.

- **Receipt:** PASS, mode `header_gated_approx_lr_tta`. K16 parity against the primary export was enforced first: max absolute error 2.98e-8, tolerance 1e-4. `mirror_orientation_verified` and `universal_geometry_verified` are both false.
- **Evidence:** `outputs/analysis/source96-training-20261012/claude-takeover/secondary/`: receipt, CSVs, gated npz and `scores.json`.
- **Scorer:** `claude-takeover/score_secondary.py`. It scores on the pinned own224 endpoint (truth, masks and report-component groups from `own224-window-samerun-20261012-result-v1/.../endpoint-private.npz`). That is the same basis as the earlier 0.9018 and 0.9059, and both reproduce exactly here; source96 K16 reproduces 0.8945295.
- **Bootstrap:** paired, 2,000 report-component draws, seed 42.

| Arm (870 development studies) | Macro AUC | Eligible subset (805) |
| --- | ---: | ---: |
| own44 K16 (historical control) | 0.901842 | 0.900658 |
| own44 K42 | 0.905877 | 0.904813 |
| 96-wide K16 (frozen primary) | 0.894530 | 0.894262 |
| 96-wide K94 plain | 0.899102 | 0.898394 |
| 96-wide K94 header-gated approximate-L/R TTA | 0.899450 | 0.898805 |

| Paired comparison | Delta | 95% CI |
| --- | ---: | --- |
| 96-wide K94 plain − own44 K42 (dense against dense) | −0.006775 | [−0.013126, −0.000304] |
| 96-wide K94 plain − 96-wide K16 | +0.004573 | [+0.001793, +0.007449] |
| Gated TTA − plain, full 870 | +0.000348 | [−0.000968, +0.001567] |
| Gated TTA − plain, eligible 805 | +0.000411 | [−0.000999, +0.001801] |

## Reading

1. **The 96-wide representation is closed.** Even under dense inference it trails the 44-cache recipe by 0.0068, and the CI excludes zero. The K16 under-sampling explanation accounts for only part of the K16 gap.
2. **Dense inference helps.** K94 adds +0.0046 over K16 on the same checkpoint, consistent with the own44 K16 to K42 gain of +0.0040. Any own-model entry should use dense K.
3. **Header-gated mirror TTA shows no measurable benefit.** The delta is about +0.0004 with a CI spanning zero. Do not add it to an entry for accuracy.

All of this is exposed development evidence on report-derived labels, a recipe-level comparison (different initialization bytes), and not a public-LB forecast. Fold 1 (0.8989 at K16) has no matched own44 comparison computed here.
