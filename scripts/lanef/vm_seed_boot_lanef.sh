#!/bin/bash
# Lane F (Oct 20; lane C boot with the lane F fit/backup scripts) VM boot: setup (pinned env), data root, code manifest, trainer, cache restore from the private backups
# (sha256-bound to the original preflight), one fail-closed preflight on the targets, then two concurrent all-data fits
# (seeds $SEEDS, tag $TAG) each with a per-epoch checkpoint backup loop.  usage: vm_seed_boot.sh <tag> <targets csv> <seed> [seed]
set -euo pipefail
TAG=$1; TG=$2; shift 2
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
for S in "$@"; do
  (bash /content/vm_bk_seed_ckpt_lanef.sh $TAG $S > /content/bk_${TAG}_s${S}.log 2>&1 &)
  (bash /content/vm_seed_fit_lanef.sh $TAG $S "$TG" > /content/fit_${TAG}_s${S}.log 2>&1 &)
  sleep 90
done
echo BOOT_LAUNCHED
