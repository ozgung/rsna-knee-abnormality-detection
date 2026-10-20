#!/bin/bash
# Lane F round-1 fold bundle on the A100: with the four fold checkpoints in /content/models_r1_all4 (verified against
# /content/bundle_SHA256SUMS; a missing one is fetched from its private dataset copy knee-s6-r1-f<k>-20261020), ONE gold-58
# bf16 read of the 4-fold mean (predictions.csv; per-model logits in logits.json) and one fp16 probe-36 decode of the four
# models; published as variation r1-bundle of knee-s6-lanef-20261020.  usage: r1_bundle.sh   Prints no study IDs.
set -uo pipefail
cd /content; export KAGGLE_CONFIG_DIR=/content
D=/content/models_r1_all4; mkdir -p $D
for k in 0 1 2 3; do
  f=$D/E_prior_s42_f$k.pt; want=$(grep " E_prior_s42_f$k.pt$" /content/bundle_SHA256SUMS | cut -d' ' -f1)
  if [ -s $f ] && [ "$(sha256sum $f | cut -d' ' -f1)" = "$want" ]; then echo "fold $k present, sha ok"; continue; fi
  rm -f $f; /usr/local/bin/kaggle datasets download ozgung/knee-s6-r1-f$k-20261020 -f E_prior_s42_f$k.pt -p $D -q
  [ -f $f.zip ] && (cd $D && unzip -o -q E_prior_s42_f$k.pt.zip && rm -f E_prior_s42_f$k.pt.zip)
  [ -s $f ] && [ "$(sha256sum $f | cut -d' ' -f1)" = "$want" ] && echo "fold $k fetched from its dataset, sha ok" || { echo "BUNDLE_FAIL fold $k checkpoint"; exit 1; }
done
[ "$(ls $D/*.pt | wc -l)" = 4 ] || { echo "BUNDLE_FAIL expected exactly 4 models"; exit 1; }
rm -rf /content/ro/gold_r1_all4 /content/ro/probe_r1_all4        # the aborted 3-model attempt is discarded (logged as an exposed read)
/content/py/bin/python -I /content/vm_seed_readout.py score $D /content/ro/gold_r1_all4 /content/gold58.json bf16 > /content/ro_gold_r1_all4.log 2>&1
/content/py/bin/python -I /content/vm_seed_readout.py score $D /content/ro/probe_r1_all4 /content/probes36.json fp16 > /content/ro_probe_r1_all4.log 2>&1
B=/content/export_r1_bundle; rm -rf $B; mkdir -p $B
for f in predictions.csv logits.json receipt.json; do cp /content/ro/gold_r1_all4/$f $B/gold58_4fold_$f && cp /content/ro/probe_r1_all4/$f $B/probe36_fp16_4fold_$f || { echo "BUNDLE_FAIL read-out files"; exit 1; }; done
cp /content/bundle_SHA256SUMS $B/MODELS_SHA256SUMS
(cd $B && sha256sum gold58_4fold_* probe36_fp16_4fold_* MODELS_SHA256SUMS > SHA256SUMS && cat SHA256SUMS)
for i in 1 2 3; do
  v=$(bash /content/vm_kmodel.sh knee-s6-lanef-20261020 r1-bundle $B SHA256SUMS "4-fold gold-58 read + probe36 fp16 4-model decode") && { echo "BUNDLE_EXPORT_DONE version $v"; break; }
  sleep 90
done
echo "$(date -u +%T) R1_BUNDLE_DONE"
