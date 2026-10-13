# Raptor64 targeted acceptance and initializer repair

Codex reviewed Claude's worktree `agent-a3334099cc20effac` on October 13. Claude retains live Colab launch ownership. This review does not establish real CUDA training, concurrency throughput or complete real-corpus parity.

The Dread initializer still rejected the single-file Kaggle ZIP whenever its internal filename differed from the requested checkpoint name. Because VM setup prepares both initializers, this blocked ImageNet setup as well. Codex repaired `scripts/prepare_raptor64_initializer.py` and its tests in that worktree: the sole member must be a regular file with the pinned size; extraction streams into the fixed destination, never into a path supplied by the archive. Final SHA verification remains mandatory. Tests cover renamed and path-like member names, wrong bytes, multiple members, directories and symbolic links.

Independent package verification: **121 passed, 3 skipped** (124 discovered). Missing local timm/safetensors prevent repeating the real ImageNet schema and Dread checkpoint checks and the complete synthetic trainer test. Cache-reader parity on synthetic DICOM, metadata alignment, package/AST checks, controller and persistence tests passed. Log: `outputs/analysis/raptor64-20261013/codex-package-tests.log`. Receipt: `outputs/analysis/raptor64-20261013/codex-targeted-acceptance.json`.

The runtime audit confirms fixed 12-epoch recipe/config/cache/initializer identity, preservation of optimizer and OneCycle state on resume, and retained epoch backups. Persistent-worker RNG is not restored. The VM launch helper deliberately rejects existing run artifacts; recovery requires a separately pinned resume config and direct controller invocation.

Two remaining items belong to Claude: remove the evaluator's arbitrary `.92 = broken` decision, and document or implement verified per-file terminal recovery if the bulk dataset archive again returns 404. The latter exporter preserves failed staging and never reports false success. The repaired initializer must enter a newly hashed bundle; prior frozen bundles are unchanged. Whole-package integration remains pending these decisions and exact launch provenance.

Read-only `colab sessions` confirms active A100 session `raptor64-fold0-r2-20261013`, endpoint `gpu-a100-s-kkb-usc1c2-3eu5ppbx4wl4k`. No training controller was independently confirmed and no runtime was allocated by Codex.
