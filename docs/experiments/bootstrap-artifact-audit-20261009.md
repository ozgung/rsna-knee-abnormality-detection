# Bootstrap v1 artifact audit

The exact fixed bootstrap table is accepted for the predeclared round-1 experiment. This is an artifact-specific acceptance; the builder's blanket `--allow-partial` option remains insufficient for future tables.

Independent CPU audit passed134 checks with no failures: exact4,345 eligible studies (gold58 and four related studies excluded); allfive OOF memberships; original/copy prediction byte equality; notebook/source/fold/label provenance; ten completed epochs with generic timm initialization and report-only top3SWA selection; recomputed per-target validation AUC/support; all52,140 generated cells reconstruct fixed addressed/silent alpha0.5/0.8 and clipping within1.11e-16. The current fold01 entry script had changed after fitting; the immutable notebook's embedded original entry exactly matches the recorded run hash.

Accepted original parquet SHA: `b260412732573fa9601907501b753f73d7d4725c3d0942cb463428510c1d48f4`. CSV SHA: `ef320553602d056da8f4947984b41d331e1f0a778d80e4ba874d1c4d174d7d8b`. Runtime must pin the exact parquet file. No gold outcome was used in this acceptance. The separate synovitis-prior variant is not selected.

Limits: run receipts lack OOF-file hashes, so binding is retrospective consistency of original outputs, saved summary hashes and recomputed metrics. Selected checkpoint bytes0/1/4 passed;2/3 downloads remain pending for inference packaging. The verified predictions suffice to reconstruct this label table; missing local weight copies do not block this fixed training comparison. Global row-wise OOF teachers indirectly expose outer validation folds to training-label construction: round1 is a biased experiment, not independent CV evidence. Public upstream labels also have gold-informed provenance.

Receipt and reproducible audit script: `outputs/analysis/bootstrap-v1-parent-audit-20261009/`. Parent accepted this result and notified Fable through the shared handoff; no training launch by the audit worker.
