#!/bin/bash
# Lane F (Oct 20) VM boot: setup (pinned env), data root, code manifest, trainer, cache restore from the private backups
# (sha256-bound to the original preflight), one fail-closed preflight on the targets, then ONE hash-fold fit with its per-epoch
# checkpoint backup loop.  usage: vm_fold_boot.sh <tag> <targets csv> <fold k> [seed]
set -euo pipefail
TAG=$1; TG=$2; K=$3; SEED=${4:-42}
if [ ! -f /content/setup.done ]; then
  bash /content/setup.sh > /content/setup.log 2>&1
  /usr/local/bin/python /content/mkdata.py
  /usr/local/bin/python /content/verify_code.py
  mkdir -p /content/code /content/targets /content/runs
  cp /content/compact_train_s6v2.py /content/code/compact_train_s6v2.py
  touch /content/setup.done
fi
[ -f /content/px/FINALIZED ] || /usr/local/bin/python -I /content/vm_restore_px.py > /content/restore.log 2>&1
S6V2_PREFLIGHT_DECODE=1 /content/py/bin/python -I /content/vm_preflight.py "$TG" > /content/preflight_${TAG}.log 2>&1 \
  || { echo "PREFLIGHT FAIL $TAG"; exit 1; }
cp /content/px/preflight.json /content/px/preflight_${TAG}.json
(bash /content/vm_bk_fold_ckpt.sh $TAG $K $SEED > /content/bk_${TAG}_f${K}.log 2>&1 &)
(bash /content/vm_fold_fit.sh $TAG $K "$TG" $SEED > /content/fit_${TAG}_f${K}.log 2>&1 &)
echo BOOT_LAUNCHED
