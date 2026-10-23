#!/bin/bash
# Lane F: fetch the averaged-model reeval outputs of fold <k> from <session> into oof_r1/avg_raw/f<k>/ and verify SHA256SUMS.
# usage: fetch_avg.sh <session> <k>
cd "$(dirname "$0")"; S=$1; k=$2; D=../oof_r1/avg_raw/f$k; mkdir -p $D
for f in SHA256SUMS partial_oof_E_prior_s42.npy partial_preds_E_prior_s42.json avg_receipt.json; do
  for i in 1 2 3; do ./c.sh download /content/avg_r1_f$k/out/$f $D/$f -s $S | tail -1 | grep -qi "error\|fail\|not found" || break; sleep 10; done
done
echo "$(date -u +%T) avg f$k: $(cd $D && shasum -a 256 -c SHA256SUMS 2>&1 | grep -c OK) of 2 verified; $(python3 -c "import json;r=json.load(open('$D/avg_receipt.json'));print(r['epochs_averaged'], r['tensors_averaged'], 'avg', r['averaged_sha256'][:12])")"
