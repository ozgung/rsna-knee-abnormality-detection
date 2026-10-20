#!/bin/bash
# Lane F round 2/3: after the all-data model of <tag> <seed> exists, stage gold-58 + probe-36 once, then read out the member on
# the pinned DICOM -> compact_infer path (vm_seed_readout.py): gold-58 in bf16 (ONE read, exposed) and the probe-36 A100 fp16
# reference that Lane P9's S6C_REFERENCE gate needs.  The four files join the export folder (READOUT_SHA256SUMS) and are
# published as a new version of the member's Kaggle Model variation.  usage: lanef_after_fit.sh <tag> <seed>
TAG=$1; SEED=$2
M=/content/runs/${TAG}_s${SEED}/models/E_prior_s${SEED}_f0.pt
until [ -s $M ] || grep -q "KILL\|FIT_FAIL" /content/fit_${TAG}_s${SEED}.log 2>/dev/null; do sleep 120; done
echo "$(date -u +%T) fit ${TAG} s${SEED} ended: $(tail -1 /content/fit_${TAG}_s${SEED}.log)"
[ -s $M ] || exit 1
cd /content
if ! grep -q "staged studies\|already staged" /content/stage_readout.log 2>/dev/null; then
  /content/py/bin/python -I /content/vm_seed_readout.py stage /content/gold_probes.json > /content/stage_readout.log 2>&1
fi
D=/content/models_${TAG}_s${SEED}; mkdir -p $D; cp $M $D/
/content/py/bin/python -I /content/vm_seed_readout.py score $D /content/ro/gold_${TAG}_s${SEED} /content/gold58.json bf16 > /content/ro_gold_${TAG}_s${SEED}.log 2>&1
/content/py/bin/python -I /content/vm_seed_readout.py score $D /content/ro/probe_${TAG}_s${SEED} /content/probes36.json fp16 > /content/ro_probe_${TAG}_s${SEED}.log 2>&1
E=/content/export_${TAG}_s${SEED}
until [ -f $E/SHA256SUMS ]; do sleep 30; done            # vm_seed_fit_lanef.sh writes the export folder after the fit
cp /content/ro/gold_${TAG}_s${SEED}/predictions.csv $E/gold58_predictions.csv
cp /content/ro/gold_${TAG}_s${SEED}/logits.json $E/gold58_logits.json
cp /content/ro/probe_${TAG}_s${SEED}/predictions.csv $E/probe36_fp16_predictions.csv
cp /content/ro/probe_${TAG}_s${SEED}/logits.json $E/probe36_fp16_logits.json
(cd $E && sha256sum gold58_predictions.csv gold58_logits.json probe36_fp16_predictions.csv probe36_fp16_logits.json > READOUT_SHA256SUMS && cat READOUT_SHA256SUMS)
until grep -q "EXPORT_DONE\|EXPORT_FAIL" /content/fit_${TAG}_s${SEED}.log 2>/dev/null; do sleep 60; done
for i in 1 2 3; do
  v=$(bash /content/vm_kmodel.sh knee-s6-lanef-20261020 ${TAG}-s${SEED} $E E_prior_s${SEED}_${TAG}all.pt "export + readouts") && { echo "READOUT_EXPORT_DONE ozgung/knee-s6-lanef-20261020/pyTorch/${TAG}-s${SEED}/$v"; break; }
  sleep 120
done
echo "$(date -u +%T) READOUT_DONE ${TAG} s${SEED}"
