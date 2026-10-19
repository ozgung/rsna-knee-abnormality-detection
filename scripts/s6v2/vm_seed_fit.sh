#!/bin/bash
# Lane C (Oct 19): one all-data S6 fit (V2-all recipe: 16 epochs, every pool study) for <tag> <seed> <targets csv>, then a
# private Kaggle export of the model with SHA256SUMS + run_config + membership + partial preds.  The preflight on the
# targets must already have passed (vm_seed_boot.sh runs it once).  Seed overrides vm_train.sh's --seed-base 42 (argparse:
# last value wins).  Kill rule: epoch-0 loss > 0.55 (V2-all 0.5106) stops the fit.  Resumable (rerun = --resume).
set -euo pipefail
TAG=$1; SEED=$2; TG=$3
OUT=/content/runs/${TAG}_s${SEED}; mkdir -p $OUT
LOG=/content/runs/${TAG}_s${SEED}.log
S6V2_ALL_DATA=1 S6V2_LOADER_WORKERS=${S6V2_LOADER_WORKERS:-5} bash /content/vm_train.sh V2 "$TG" $OUT --only-folds 0 --seed-base $SEED \
  >> $LOG 2>&1 &
TP=$!
# epoch-0 kill rule watcher
while kill -0 $TP 2>/dev/null; do
  l=$(grep -oE "seed $SEED fold 0 epoch 0 loss [0-9.]+" $LOG | tail -1 | awk '{print $NF}' || true)
  if [ -n "$l" ]; then
    if python3 -c "import sys; sys.exit(0 if float('$l') > 0.55 else 1)"; then echo "KILL epoch0 loss $l > 0.55"; kill $TP; exit 3; fi
    echo "epoch0 loss $l ok"; break
  fi
  sleep 60
done
wait $TP
M=$OUT/models/E_prior_s${SEED}_f0.pt
[ -s $M ] || { echo "FIT_FAIL no model"; exit 1; }
ID=ozgung/knee-s6-${TAG}all-s${SEED}-20261019
D=/content/export_${TAG}_s${SEED}; mkdir -p $D
cp $M $D/E_prior_s${SEED}_${TAG}all.pt
cp $OUT/run_config.json $OUT/membership_E_prior_s${SEED}_f0.json $OUT/partial_preds_E_prior_s${SEED}.json $D/
(cd $D && sha256sum E_prior_s${SEED}_${TAG}all.pt run_config.json membership_E_prior_s${SEED}_f0.json partial_preds_E_prior_s${SEED}.json > SHA256SUMS)
echo "{\"title\": \"knee-s6-${TAG}all-s${SEED}-20261019\", \"id\": \"$ID\", \"licenses\": [{\"name\": \"other\"}]}" > $D/dataset-metadata.json
for i in 1 2 3 4 5; do
  KAGGLE_CONFIG_DIR=/content /usr/local/bin/kaggle datasets create -p $D --dir-mode skip -q && break
  KAGGLE_CONFIG_DIR=/content /usr/local/bin/kaggle datasets status $ID 2>/dev/null | grep -qi ready && break
  sleep 120
done
echo "EXPORT_DONE $ID"
