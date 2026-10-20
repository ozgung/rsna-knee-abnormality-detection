#!/bin/bash
# Lane F (Oct 20): one hash-fold S6 fit for out-of-fold teacher columns.  usage: vm_fold_fit.sh <tag> <fold k> <targets csv> [seed]
# The targets' fold column must be the R6/R7 hash rule sha1(uid)[:8] % 4 (make_hash4_targets.py).  V2-all recipe (vm_train.sh)
# with --folds 4 --only-folds k --epochs 12 --eval-stride 1: trains on the three other hash folds from the ImageNet init, then
# predicts the held-out fold k (partial_oof_*.npy: only fold-k rows are filled) and gold-58 (partial_preds_*.json: gold logits
# divided by K = 4 by the trainer).  Kill rule: epoch-0 loss > 0.55.  Resumable (rerun = --resume).  Then a private Kaggle
# Model export (variation <tag>-f<k>) of the fold model + preds + oof + pool_uids + membership + run_config with SHA256SUMS.
set -euo pipefail
TAG=$1; K=$2; TG=$3; SEED=${4:-42}
OUT=/content/runs/${TAG}_f${K}; mkdir -p $OUT
LOG=/content/runs/${TAG}_f${K}.log
S6V2_LOADER_WORKERS=${S6V2_LOADER_WORKERS:-8} bash /content/vm_train.sh V2 "$TG" $OUT --folds 4 --only-folds $K --epochs 12 \
  --eval-stride 1 --seed-base $SEED >> $LOG 2>&1 &
TP=$!
while kill -0 $TP 2>/dev/null; do
  l=$(grep -oE "seed $SEED fold $K epoch 0 loss [0-9.]+" $LOG | tail -1 | awk '{print $NF}' || true)
  if [ -n "$l" ]; then
    if python3 -c "import sys; sys.exit(0 if float('$l') > 0.55 else 1)"; then echo "KILL epoch0 loss $l > 0.55"; kill $TP; exit 3; fi
    echo "epoch0 loss $l ok"; break
  fi
  sleep 60
done
wait $TP
N=E_prior_s${SEED}_f${K}
M=$OUT/models/$N.pt
[ -s $M ] || { echo "FIT_FAIL no model"; exit 1; }
[ -s $OUT/partial_oof_E_prior_s${SEED}.npy ] || { echo "FIT_FAIL no oof"; exit 1; }
D=/content/export_${TAG}_f${K}; mkdir -p $D
cp $M $D/$N.pt
cp $OUT/run_config.json $OUT/membership_$N.json $OUT/partial_preds_E_prior_s${SEED}.json $OUT/partial_oof_E_prior_s${SEED}.npy $OUT/pool_uids.json $D/
cp "$TG" $D/targets.csv
(cd $D && sha256sum $N.pt run_config.json membership_$N.json partial_preds_E_prior_s${SEED}.json partial_oof_E_prior_s${SEED}.npy pool_uids.json targets.csv > SHA256SUMS)
echo "FIT_DONE $(date -u +%FT%TZ)"
for i in 1 2 3; do
  v=$(bash /content/vm_kmodel.sh knee-s6-lanef-20261020 ${TAG}-f${K} $D $N.pt "export") && { echo "EXPORT_DONE ozgung/knee-s6-lanef-20261020/pyTorch/${TAG}-f${K}/$v"; exit 0; }
  sleep 120
done
echo "EXPORT_FAIL"; exit 4
