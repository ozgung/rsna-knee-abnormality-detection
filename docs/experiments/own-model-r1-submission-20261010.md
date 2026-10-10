# Own round1 entry — October 10

Entry **57028725** was accepted at02:45:28UTC and completed with public score **0.874**, verified October10 at04:33UTC. Four entries remain today. Best public score remains0.950. Kernel: `ozgung/knee-own-convnext-r1-submission-20261010` v1.

The fixed comparison changes bootstrap training labels relative to round0 entry57016778 (0.878); inference remains five ConvNeXt-T256 folds, selected top3 SWA, 44-slice reader, k16, CUDA FP16, equal probability mean, no TTA. Global OOF bootstrap exposure and heldout epoch selection bias the local validation; no gold outcome selected this entry.

Five-fold rehearsal `ozgung/knee-own-convnext-r1-rehearsal-fivefold-20261009` v1 completed in62.333 seconds: all25 raw/corpus pixel and mask slots matched, windows and repeated probabilities matched exactly, allfive checkpoints strictly loaded, all16 reader stresses passed. No labels were read. Host peak1.941GB; serial GPU peak0.312GB.

Final-visible inference completed in55.198 seconds, host peak1.577GB and GPU peak0.764GB. Three visible studies had exact ordered coverage, twelve finite probabilities each, and byte-identical output to rehearsal. The hidden rerun completed with public score0.874; its runtime was not exposed.

- Final notebook SHA: `6823fcf91d9f02563b3c7b7c7af4cee6dfce4636118e77a443b947b1171cb31a`
- Contract SHA: `363270e91f65b5b17d28e18ee71f4f36820bcc4c6163b15b49d2e625b1b8ffeb`
- CSV SHA: `d062b8a194d38d24d11051a504943c02b0c876f8d393492fc796a91078ca5f38`
- Final report SHA: `712f8f5fb1cdc67595fd2d1caa8b116687c60991b616b57add99f21128717917`

Evidence: `outputs/analysis/own-model-r1-submission-20261010/parent-final-review.json` and reproducible `verify_final.py`; build/result directories under `outputs/kaggle/own-convnext-r1-submission-20261010*`. Builder ran from revision357309e. Initial systemPython build lacked numpy and created no bundle; rerunning with the installed ML interpreter succeeded. No runtime-source changes or duplicate accepted pushes.

Result: round1 is0.004 below round0 (0.878). Public uncertainty cannot be quantified from these two scores, so this is evidence of no observed gain, not a statistical noise estimate. Local raw-report AUC improved0.011513 while public AUC declined; validation gains did not transfer in this fixed comparison. Close this ConvNeXt bootstrap branch and do not extend or blend it by default. Finish the original-label CoAtNet224 family and its distinct inference qualification. Existing CoAtNet bootstrap fits may finish as paired diagnostics; do not expand them from the biased report metric or select new label schemes on gold58.
