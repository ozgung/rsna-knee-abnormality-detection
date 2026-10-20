#!/bin/bash
# Lane F: download a fold's export folder (no model weights) from a session into oof_<tag>/raw/f<k>/ and verify SHA256SUMS lines.
# usage: fetch_fold.sh <session> <tag> <k> [seed]
cd "$(dirname "$0")"; SES=$1; TAG=$2; K=$3; SEED=${4:-42}
D=../oof_${TAG}/raw/f${K}; mkdir -p $D
for f in SHA256SUMS run_config.json membership_E_prior_s${SEED}_f${K}.json partial_preds_E_prior_s${SEED}.json partial_oof_E_prior_s${SEED}.npy pool_uids.json targets.csv; do
  for i in 1 2 3; do ./c.sh download /content/export_${TAG}_f${K}/$f $D/$f -s $SES | tail -1 | grep -qi "error\|fail\|not found" || break; sleep 10; done
done
(cd $D && grep -v "E_prior_s${SEED}_f${K}.pt" SHA256SUMS | shasum -a 256 -c - 2>&1 | grep -c OK) | xargs -I{} echo "$(date -u +%T) fetched $SES $TAG f$K: {} of 6 receipts verified"
