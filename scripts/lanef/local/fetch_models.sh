#!/bin/bash
# Lane F: fetch round-1 fold checkpoints + receipts from the VMs (Kaggle's Model download API returns 404 for this account).
cd "$(dirname "$0")"; D=../members/dl
fetch() { S=$1; k=$2; mkdir -p $D/r1-f$k; for f in E_prior_s42_f$k.pt SHA256SUMS run_config.json membership_E_prior_s42_f$k.json partial_preds_E_prior_s42.json pool_uids.json targets.csv; do
  for i in 1 2 3; do ./c.sh download /content/export_r1_f$k/$f $D/r1-f$k/$f -s $S | tail -1 | grep -qi "error\|fail\|not found" || break; sleep 15; done; done
  (cd $D/r1-f$k && grep -v partial_oof SHA256SUMS | shasum -a 256 -c - 2>&1 | grep -c OK | xargs -I{} echo "$(date -u +%T) r1-f$k: {} of 7 files verified"); }
fetch lanef8 0; fetch lanef9 1; fetch lanef8 2; fetch lanef9 3; echo FETCH_DONE
