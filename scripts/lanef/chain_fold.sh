#!/bin/bash
# Lane F: when fold <k> of <tag> has FIT_DONE (or failed), start fold <next> on this VM (preflight already passed at boot).
# usage: chain_fold.sh <tag> <targets> <k> <next> [seed]
TAG=$1; TG=$2; K=$3; NEXT=$4; SEED=${5:-42}
until grep -q "FIT_DONE\|FIT_FAIL\|KILL" /content/fit_${TAG}_f${K}.log 2>/dev/null; do sleep 120; done
echo "$(date -u +%T) fold $K ended: $(grep -m1 'FIT_DONE\|FIT_FAIL\|KILL' /content/fit_${TAG}_f${K}.log)"
(bash /content/vm_bk_fold_ckpt.sh $TAG $NEXT $SEED > /content/bk_${TAG}_f${NEXT}.log 2>&1 &)
(S6V2_LOADER_WORKERS=8 bash /content/vm_fold_fit.sh $TAG $NEXT "$TG" $SEED > /content/fit_${TAG}_f${NEXT}.log 2>&1 &)
echo "$(date -u +%T) fold $NEXT launched"
