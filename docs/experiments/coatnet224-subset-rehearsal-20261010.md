# CoAtNet224 three-fold CUDA rehearsal — October 10, 2026

`ozgung/knee-own-coatnet224-rehearsal-fold014-20261010` v1 completed successfully. Parent validated its exact contract, report and visible CSV hash against the authoritative three test IDs. Final-submission qualification correctly rejects this three-fold subset; folds2/3 and a complete five-fold rehearsal are still required.

The run tested selected checkpoints0/1/4 with strict state loading, zero probability difference against the training implementation, zero repeat difference, finite missing-plane predictions, and exact raw/corpus tensor, mask and window parity across five studies/all25 slots. All16 synthetic reader stress cases passed. Inference read no report labels.

Measured runtime was101.99s inside the rehearsal,126.7s for the full notebook. Peak host RSS was2,448,891,904bytes (2.28GiB); peak per-model CUDA allocation was796,068,864bytes (0.74GiB). These bounded measurements do not establish hidden-test runtime or simultaneous five-model memory.

Environment: TeslaT4, Python3.12.13, torch2.10.0+cu128, timm1.0.26. ContractSHA `1ffaf9753272873da56fa5237c07e68edd27adde12db33b33a9faa4e51b73eae`; visibleCSV SHA `031f11b5bc9be15ef8130d55d2a7a8e81392d5d2df5851098060212baae7ba4e`. Build/result folders: `outputs/kaggle/own-coatnet224-rehearsal-fold014-20261010{,-result}/`. No submission was made.

Before launch, parent stopped the accidental seed42 repeat named `knee-tracka-coatnet224-s43-fold01-20261010` v1 through its Active Events control. Kaggle confirmed cancellation after555.4seconds. Its notebook/logs are preserved; do not restart. The intended seed43 had not reached the resolved training command. Any future seed-diversity run must prove its actual seed before launch and wait until qualification has capacity.
