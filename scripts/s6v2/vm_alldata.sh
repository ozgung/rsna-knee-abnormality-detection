#!/bin/bash
# Rule-2 follow-up: nartaa's all-data config (16 epochs, every pool study, no OOF prediction) on the V2 targets, then a private
# Kaggle backup of the model.  Preflight (fails closed) on the _all targets first.
set -euo pipefail
cd /content
S6V2_PREFLIGHT_DECODE=1 /content/py/bin/python -I /content/vm_preflight.py /content/targets/targets_V2_all.csv > /content/preflight_all.log 2>&1 \
  || { echo "PREFLIGHT_ALL FAIL"; exit 1; }
cp /content/px/preflight.json /content/px/preflight_all.json
S6V2_ALL_DATA=1 S6V2_LOADER_WORKERS=8 bash /content/vm_train.sh V2 /content/targets/targets_V2_all.csv /content/runs/V2_all --only-folds 0 \
  > /content/runs/V2_all.log 2>&1
[ -s /content/runs/V2_all/models/E_prior_s42_f0.pt ] || { echo "ALLDATA_FAIL no model"; exit 1; }
D=/content/bk_v2all; mkdir -p $D
cp /content/runs/V2_all/models/E_prior_s42_f0.pt $D/E_prior_s42_v2all.pt
cp /content/runs/V2_all/run_config.json /content/runs/V2_all/membership_E_prior_s42_f0.json /content/runs/V2_all/partial_preds_E_prior_s42.json $D/
sha256sum $D/E_prior_s42_v2all.pt > $D/SHA256SUMS
echo '{"title": "knee-s6v2-v2all-20261016", "id": "ozgung/knee-s6v2-v2all-20261016", "licenses": [{"name": "other"}]}' > $D/dataset-metadata.json
KAGGLE_CONFIG_DIR=/content /usr/local/bin/kaggle datasets create -p $D --dir-mode skip -q
echo ALLDATA_DONE
