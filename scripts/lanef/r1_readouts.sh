#!/bin/bash
# Lane F round-1 fold-bundle read-outs on the A100 (gap work before the round-2 fits).  usage: r1_readouts.sh "<folds on this VM>" <all4 0|1>
# Per fold model present on this VM: the A100 fp16 probe-36 reference on the pinned DICOM -> compact_infer path
# (vm_seed_readout.py, eval stride 1), joined to the fold's export folder (READOUT_SHA256SUMS) and published as a new version of
# the fold's Kaggle Model variation.  With all4=1: the other fold models are downloaded from their export variations and the
# four together get ONE gold-58 bf16 read (4-fold mean = predictions.csv; per-model logits in logits.json) and one fp16 probe-36
# decode; published as variation r1-bundle.  Prints no study IDs.
set -uo pipefail
cd /content; export KAGGLE_CONFIG_DIR=/content
FOLDS=$1; ALL4=${2:-0}; SEED=42
if ! grep -q "staged studies\|already staged" /content/stage_readout.log 2>/dev/null; then
  /content/py/bin/python -I /content/vm_seed_readout.py stage /content/gold_probes.json > /content/stage_readout.log 2>&1
fi
echo "$(date -u +%T) stage: $(tail -n 1 /content/stage_readout.log)"
for k in $FOLDS; do
  M=/content/runs/r1_f$k/models/E_prior_s${SEED}_f$k.pt; [ -s $M ] || { echo "no model for fold $k"; continue; }
  D=/content/models_r1_f$k; mkdir -p $D; cp $M $D/
  /content/py/bin/python -I /content/vm_seed_readout.py score $D /content/ro/probe_r1_f$k /content/probes36.json fp16 > /content/ro_probe_r1_f$k.log 2>&1
  E=/content/export_r1_f$k
  for f in predictions.csv logits.json receipt.json; do cp /content/ro/probe_r1_f$k/$f $E/probe36_fp16_$f; done
  (cd $E && sha256sum probe36_fp16_predictions.csv probe36_fp16_logits.json probe36_fp16_receipt.json > READOUT_SHA256SUMS && cat READOUT_SHA256SUMS)
  for i in 1 2 3; do
    v=$(bash /content/vm_kmodel.sh knee-s6-lanef-20261020 r1-f$k $E E_prior_s${SEED}_f$k.pt "export + probe36 fp16 reference") && { echo "PROBE_EXPORT_DONE r1-f$k version $v"; break; }
    sleep 90
  done
done
if [ "$ALL4" = 1 ]; then
  D=/content/models_r1_all4; mkdir -p $D
  for k in 0 1 2 3; do
    M=/content/runs/r1_f$k/models/E_prior_s${SEED}_f$k.pt
    if [ -s $M ]; then cp $M $D/; else
      mkdir -p /content/dl_r1_f$k
      /usr/local/bin/kaggle models instances versions download ozgung/knee-s6-lanef-20261020/pyTorch/r1-f$k/1 -p /content/dl_r1_f$k --untar -q
      cp /content/dl_r1_f$k/E_prior_s${SEED}_f$k.pt $D/
      (cd /content/dl_r1_f$k && sha256sum -c SHA256SUMS 2>/dev/null | grep -c OK | xargs -I{} echo "fold $k download: {} files verified")
    fi
  done
  ls $D
  /content/py/bin/python -I /content/vm_seed_readout.py score $D /content/ro/gold_r1_all4 /content/gold58.json bf16 > /content/ro_gold_r1_all4.log 2>&1
  /content/py/bin/python -I /content/vm_seed_readout.py score $D /content/ro/probe_r1_all4 /content/probes36.json fp16 > /content/ro_probe_r1_all4.log 2>&1
  B=/content/export_r1_bundle; mkdir -p $B
  for f in predictions.csv logits.json receipt.json; do cp /content/ro/gold_r1_all4/$f $B/gold58_4fold_$f; cp /content/ro/probe_r1_all4/$f $B/probe36_fp16_4fold_$f; done
  (cd $D && sha256sum *.pt > $B/MODELS_SHA256SUMS)
  (cd $B && sha256sum gold58_4fold_* probe36_fp16_4fold_* MODELS_SHA256SUMS > SHA256SUMS && cat SHA256SUMS)
  for i in 1 2 3; do
    v=$(bash /content/vm_kmodel.sh knee-s6-lanef-20261020 r1-bundle $B SHA256SUMS "4-fold gold-58 read + probe36 fp16 4-model decode") && { echo "BUNDLE_EXPORT_DONE version $v"; break; }
    sleep 90
  done
fi
echo "$(date -u +%T) R1_READOUTS_DONE"
