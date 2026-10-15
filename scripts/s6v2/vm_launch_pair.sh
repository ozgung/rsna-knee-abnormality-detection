#!/bin/bash
# After the cache build: finalize, then launch J0 and V2 (fold 0 of folds_v0, seed 42) concurrently on the one A100.
set -euo pipefail
cd /content
/content/py/bin/python -I /content/vm_build_cache_v2.py finalize
mkdir -p /content/runs
for ARM in J0 V2; do
  if [ ! -f /content/runs/$ARM/partial_preds_E_prior_s42.json ]; then
    nohup setsid bash /content/vm_train.sh $ARM /content/targets/targets_${ARM}_f0.csv /content/runs/$ARM --only-folds 0 \
      >> /content/runs/$ARM.log 2>&1 < /dev/null &
    echo "$ARM pid $!"
  fi
done
