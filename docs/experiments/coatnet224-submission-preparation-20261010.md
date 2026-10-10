# CoAtNet224 final-submission preparation

The separately pinned final runtime and builder are implemented and tested. **No real final notebook was built, no CUDA run launched, and no CoAtNet entry submitted.** The family still needs folds2/3 and a complete five-fold offline rehearsal.

New files: `knee/coatnet224_submission.py`, `scripts/build_coatnet224_submission.py`, `tests/test_coatnet224_submission.py`. They adapt the frozen ConvNeXt final wrapper without modifying it or the separately pinned224 rehearsal sources. The latter's hashes still match the existing subset build.

The builder requires a complete five-fold CoAtNet224 contract and PASS report, reconstructs the exact archived executable, verifies source and report hashes, and checks the visible CSV. It rejects the actual folds0/1/4 preparation archive before constructing an entry. Synthetic full-five test fixtures exercise these gates only; they are not real qualification evidence and are not retained as competition artifacts.

Runtime retains image-only streaming from authoritative `test.csv` IDs, official target order and finite probabilities, exact visible CSV parity, five strict checkpoint loads with finite tensor checks, offline dependencies, no CPU fallback, and the nine-hour platform alarm. No report labels or training corpus are attached to the final notebook. The reader and model mathematics are unchanged.

Parent verification: **16 combined rehearsal/submission tests PASS, no skips**. They include pinned224 window calculations, sixteen reader stress cases, hidden-ID and duplicate/schema handling, subset/foreign-family rejection, archive tampering, dependency closure and the exact runtime limit. Synthetic smoke passed (`torch_forward=false`). Logs: `outputs/analysis/reconcile-20261010-0435/coatnet224-tests.log`. Full repository discovery was not repeated: it ran in the previous checkpoint, and these changes add only isolated wrapper files. Actual model execution, GPU memory/runtime and competitive performance remain unverified.

Next: collect and audit original-label folds2/3, verify their checkpoint bytes, build/run full-five rehearsal, inspect real PASS evidence, then invoke this builder and run final-visible inference. Require exact CSV parity and current daily allowance before a single authorized entry. Preserve the0.950 best; do not use the closed ConvNeXt bootstrap branch as a default blend member.
