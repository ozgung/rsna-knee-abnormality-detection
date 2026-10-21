#!/bin/bash
# Lane F round 3 (VM side): run all-data 16-epoch S6 fits for <tag> on the given seeds one after another on this VM, each with
# the per-epoch Model backup, the read-outs (gold-58 once, probe-36 fp16 reference) and the export (vm_seed_boot_lanef.sh +
# lanef_after_fit.sh).  A seed starts only after the previous job on this VM has exported and read out, and only if it can
# finish before the deadline and the local unit guard has not written /content/STOP_NEW_SEEDS; it must (default Oct 22 06:00 UTC; a fit + read-outs takes about 4.0 h).
# usage: v5_seed_queue.sh <tag> <targets csv> <wait-for "tag:seed" or none> <seed> [seed...]     log: /content/<tag>_queue.log
set -uo pipefail
TAG=$1; TG=$2; WAIT=$3; shift 3
DEADLINE=${DEADLINE_EPOCH:-$(date -u -d "2026-10-22 06:00:00" +%s)}; NEED=$((4 * 3600 + 10 * 60))
done_job() { grep -q "EXPORT_DONE\|EXPORT_FAIL\|KILL\|FIT_FAIL" /content/fit_$1_s$2.log 2>/dev/null && grep -q "READOUT_DONE\|READOUT_FAIL" /content/after_$1_s$2.log 2>/dev/null; }
if [ "$WAIT" != none ]; then
  WT=${WAIT%%:*}; WS=${WAIT##*:}
  until done_job $WT $WS; do sleep 120; done
  echo "$(date -u +%FT%TZ) previous job $WAIT finished: $(tail -n 1 /content/fit_${WT}_s${WS}.log)"
fi
for SEED in "$@"; do
  if [ -f /content/STOP_NEW_SEEDS ]; then echo "$(date -u +%FT%TZ) seed $SEED NOT started: unit guard ($(cat /content/STOP_NEW_SEEDS))"; continue; fi
  now=$(date +%s)
  if [ $((now + NEED)) -gt $DEADLINE ]; then echo "$(date -u +%FT%TZ) seed $SEED NOT started: would end after the deadline"; continue; fi
  echo "$(date -u +%FT%TZ) starting $TAG seed $SEED"
  S6V2_LOADER_WORKERS=8 bash /content/vm_seed_boot_lanef.sh $TAG "$TG" $SEED > /content/boot_${TAG}_s${SEED}.log 2>&1 || { echo "$(date -u +%FT%TZ) BOOT/PREFLIGHT FAIL for seed $SEED: $(tail -n 1 /content/boot_${TAG}_s${SEED}.log)"; continue; }
  (bash /content/lanef_after_fit.sh $TAG $SEED > /content/after_${TAG}_s${SEED}.log 2>&1 &)
  until done_job $TAG $SEED; do sleep 120; done
  echo "$(date -u +%FT%TZ) $TAG seed $SEED finished: $(tail -n 1 /content/fit_${TAG}_s${SEED}.log) | $(grep -m1 READOUT_EXPORT_DONE /content/after_${TAG}_s${SEED}.log)"
done
echo "$(date -u +%FT%TZ) QUEUE_DONE $TAG"
