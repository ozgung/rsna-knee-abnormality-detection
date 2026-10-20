#!/bin/bash
# Lane D: joint all-data S6 fit (V2-all recipe: vm_train.sh, 16 epochs, seed 42) over the competition pool + fastMRI exams
# with teacher soft targets.  Arm E_fm = weighted BCE on 'prior' with per-cell weight 'pw' (competition 1, fastMRI solved so the
# effective weight is 0.5); every other flag is vm_train.sh's.  Competition rows come from /content/pxm (= /content/px files),
# fastMRI rows from --px-override /content/pxfm.  Kill rule: epoch-0 loss > 0.55.  Resumable.  Then a private Kaggle export.
# usage: vm_fm_fit.sh <targets csv> <dataset slug>
set -euo pipefail
TG=$1; SLUG=$2; SEED=42
OUT=/content/runs/FM_all; mkdir -p $OUT; LOG=/content/runs/FM_all.log
sed 's#--px /content/px #--px /content/pxm #' /content/vm_train.sh > /content/vm_train_fm.sh
grep -q -- "--px /content/pxm " /content/vm_train_fm.sh
S6V2_ALL_DATA=1 S6V2_LOADER_WORKERS=${S6V2_LOADER_WORKERS:-8} bash /content/vm_train_fm.sh FM "$TG" $OUT --only-folds 0 \
  --arms E_fm=bce:prior:pw --px-override /content/pxfm >> $LOG 2>&1 &
TP=$!
while kill -0 $TP 2>/dev/null; do
  l=$(grep -oE "seed $SEED fold 0 epoch 0 loss [0-9.]+" $LOG | tail -1 | awk '{print $NF}' || true)
  if [ -n "$l" ]; then
    if python3 -c "import sys; sys.exit(0 if float('$l') > 0.55 else 1)"; then echo "KILL epoch0 loss $l > 0.55"; kill $TP; exit 3; fi
    echo "epoch0 loss $l ok"; break
  fi
  sleep 60
done
wait $TP
M=$OUT/models/E_fm_s${SEED}_f0.pt
[ -s $M ] || { echo "FIT_FAIL no model"; exit 1; }
D=/content/export_fm; mkdir -p $D
cp $M $D/E_prior_s${SEED}_fmall.pt
cp $OUT/run_config.json $OUT/membership_E_fm_s${SEED}_f0.json $D/
cp $TG.receipt.json $D/targets_receipt.json 2>/dev/null || true
(cd $D && sha256sum * > SHA256SUMS)
echo "{\"title\": \"$SLUG\", \"id\": \"ozgung/$SLUG\", \"licenses\": [{\"name\": \"other\"}]}" > $D/dataset-metadata.json
for i in 1 2 3 4 5; do
  KAGGLE_CONFIG_DIR=/content /usr/local/bin/kaggle datasets create -p $D --dir-mode skip -q && break
  KAGGLE_CONFIG_DIR=/content /usr/local/bin/kaggle datasets status ozgung/$SLUG 2>/dev/null | grep -qi ready && break
  sleep 120
done
echo "EXPORT_DONE ozgung/$SLUG"
