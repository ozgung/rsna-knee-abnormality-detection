# Colab384 artifact recovery — October 10

The completed fold0 model was recovered locally from private dataset `ozgung/knee-colab-resume-c384-fold0`, reported version14/epoch13. **No training or GPU allocation was needed.** This is artifact recovery, not a qualified inference model or proof of exact optimizer continuation.

Downloaded resume size2,049,836,511bytes matches the server file listing. Local SHA256: `0a60dbbfb1c653de8db95337d29e64d2970e9d15bc16b19941e113a82ee8d888`. No original remote cryptographic hash was supplied; this records the downloaded content, not independent source attestation. Before download, local disk had19GiB free and physical memory48GiB.

The checkpoint loaded on CPU with `weights_only=True` and an explicit allowlist for known NumPy serialization types. All14 history entries match the saved JSON, seed42 and epoch13/14 agree, and the saved log records fold0,384pixels,12/24windows,batch8,3,475training and870heldout studies. The checkpoint includes optimizer, scheduler, scaler and RNG state, but those were not resumed in a training process.

Selected epochs **[13,12,11]** match the three highest fullfold report AUCs. Each selected model has486 finite tensors and matching architecture, target order, tensor shapes and dtypes. Final and best model tensors equal selectedepoch13. The original trainer's fixed FP64 sum/divide/cast formula reconstructed the three-model SWA; integer buffers were copied from the first selected state. Saved weights reload exactly. The506 passing checks include486 per-tensor schema checks; they are not506 independent model evaluations.

Recovered artifact: `outputs/colab/coat384-fold0-recovery-20261010/raptor_ft_fold0_swa.pt`,292,826,563bytes.

- File SHA256: `2c9c349768d0eab83a81debb69408eaa3fb9b3e98e2a27548db97e439c2aa841`
- Canonical model-tensor SHA256: `af33707bb08e4fc2bcdf76745e4cb208c89cbec029868e9bbff7c781972a2141`
- Fingerprint format: sort tensor names; append compact JSON `[name,str(dtype),shape]` plus newline, then contiguous CPU NumPy bytes, and hash the concatenation. This permits tensor-level comparison with a separate recovery without depending on pickle metadata or archive filenames.

Detailed checks, source metadata hashes and limitations: `outputs/colab/coat384-fold0-recovery-20261010/recovery-review.json`. Reproduction script: `verify_and_recover.py` beside it; it refuses to overwrite an existing recovered output. Raw resume/log/JSON are preserved there. No identifiers, labels, weights or generated predictions are committed.

Remaining limits: the saved state omits explicit fold/source/config/label/split/exclusion identity hashes. Matching logs and history do not repair that missing binding. The original VM SWA file was unavailable for direct comparison; recovered tensors follow the saved top3 formula. Actual strict model loading/forward, selected-SWA evaluation and export remain pending. The best-epoch reportAUC0.918432 is not a measured recovered-SWA score. Gold outcomes did not select this recovery.

Fable owns export-only recovery and accurate training-versus-export provenance. Preserve the fixed384/12/24/14epoch experiment; do not present current export defaults8/10 as its training configuration. The ConvNeXt-teacher bootstrap branch remains closed for expansion; this already-trained CoAtNet checkpoint is retained for the paired diagnostic, not automatically promoted to an ensemble.
