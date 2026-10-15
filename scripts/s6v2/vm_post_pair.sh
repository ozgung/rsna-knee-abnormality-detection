#!/bin/bash
# After both fold-0 fits: (1) trainer re-evaluation at eval stride 1 (= nartaa's compact_infer read-out stride) of gold-58 and
# the fold-0 holdout from the cache, both arms concurrently; (2) gold-58 from raw DICOM through the pinned compact_infer
# (parity check of the cache path).  Outputs under /content/runs/<ARM>_s1 and /content/runs/<ARM>_ci.
set -uo pipefail
cd /content
for ARM in J0 V2; do
  [ -f /content/runs/${ARM}_s1/partial_preds_E_prior_s42.json ] || \
  (bash /content/vm_train.sh $ARM /content/targets/targets_${ARM}_f0.csv /content/runs/${ARM}_s1 --only-folds 0 \
     --reeval-run /content/runs/$ARM --eval-stride 1 > /content/runs/${ARM}_s1.log 2>&1 &)
done
wait
for ARM in J0 V2; do
  until [ -f /content/runs/${ARM}_s1/partial_preds_E_prior_s42.json ] || ! pgrep -f "runs/${ARM}_s1 " > /dev/null; do sleep 20; done
done
for ARM in J0 V2; do
  mkdir -p /content/runs/${ARM}_ci
  S6V2_DATA=/content/data_gold /content/py/bin/python -I /content/vm_gold_infer.py score /content/runs/$ARM/models /content/runs/${ARM}_ci bf16 \
    > /content/runs/${ARM}_ci.log 2>&1
done
echo POST_PAIR_DONE
