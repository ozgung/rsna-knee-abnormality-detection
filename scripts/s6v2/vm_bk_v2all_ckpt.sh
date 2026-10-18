#!/bin/bash
# Private Kaggle backup of the all-data V2 resumable checkpoint after every new epoch checkpoint (polls every 60 s).
# State advances only after a successful publish (exit 0); a failed publish is retried on the next poll.  create vs
# version is decided by a marker written only after a create succeeded (or the dataset is confirmed to exist remotely).
# Required files (checkpoint, run_config.json, membership) must all exist before anything is published.
export KAGGLE_CONFIG_DIR=/content
K=/usr/local/bin/kaggle
ID=ozgung/knee-s6v2-v2all-ckpt-20261016
R=/content/runs/V2_all
D=/content/bk_v2all_ckpt; mkdir -p $D; last=""
while true; do
  f=$R/ckpt/E_prior_s42_f0.pt
  if [ -f $R/models/E_prior_s42_f0.pt ]; then echo "$(date -u +%T) final model present; ckpt loop ends"; break; fi
  if [ -f $f ] && [ -f $R/run_config.json ] && [ -f $R/membership_E_prior_s42_f0.json ]; then
    m=$(stat -c %Y $f)
    if [ "$m" != "$last" ] && [ $(( $(date +%s) - m )) -gt 20 ]; then
      cp $f $D/V2_all__ckpt__E_prior_s42_f0.pt.tmp && mv $D/V2_all__ckpt__E_prior_s42_f0.pt.tmp $D/V2_all__ckpt__E_prior_s42_f0.pt
      cp $R/run_config.json $R/membership_E_prior_s42_f0.json $D/
      ep=$(/content/py/bin/python -c "import torch;print(torch.load('$D/V2_all__ckpt__E_prior_s42_f0.pt',map_location='cpu',weights_only=False)['epoch'])")
      sha256sum $D/V2_all__ckpt__E_prior_s42_f0.pt > $D/SHA256SUMS 2>/dev/null; echo "epoch $ep mtime $m" >> $D/SHA256SUMS
      [ -f $D/dataset-metadata.json ] || echo "{\"title\": \"knee-s6v2-v2all-ckpt-20261016\", \"id\": \"$ID\", \"licenses\": [{\"name\": \"other\"}]}" > $D/dataset-metadata.json
      if [ ! -f $D/.remote_created ] && $K datasets status $ID 2>/dev/null | grep -q -i ready; then touch $D/.remote_created; fi
      if [ -f $D/.remote_created ]; then
        $K datasets version -p $D -m "ckpt epoch $ep" --dir-mode skip -q && r=ok || r=fail
      else
        $K datasets create -p $D --dir-mode skip -q && { r=ok; touch $D/.remote_created; } || r=fail
      fi
      if [ "$r" = ok ]; then last=$m; fi
      echo "$(date -u +%T) ckpt epoch $ep backup $r"
    fi
  fi
  sleep 60
done
