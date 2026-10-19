#!/bin/bash
# Lane C (Oct 19): vm_bk_v2all_ckpt.sh generalised to <tag> <seed>.  Private Kaggle backup of the resumable checkpoint after
# every new epoch (polls every 60 s); state advances only after a successful publish; create vs version decided by a marker
# written after a confirmed create (or remote status ready).  Ends when the final model exists.
TAG=$1; SEED=$2
export KAGGLE_CONFIG_DIR=/content
K=/usr/local/bin/kaggle
ID=ozgung/knee-s6-${TAG}all-s${SEED}-ckpt-20261019
R=/content/runs/${TAG}_s${SEED}
N=E_prior_s${SEED}_f0
D=/content/bk_${TAG}_s${SEED}_ckpt; mkdir -p $D; last=""
while true; do
  f=$R/ckpt/$N.pt
  if [ -f $R/models/$N.pt ]; then echo "$(date -u +%T) final model present; ckpt loop ends"; break; fi
  if [ -f $f ] && [ -f $R/run_config.json ] && [ -f $R/membership_$N.json ]; then
    m=$(stat -c %Y $f)
    if [ "$m" != "$last" ] && [ $(( $(date +%s) - m )) -gt 20 ]; then
      cp $f $D/ckpt__$N.pt.tmp && mv $D/ckpt__$N.pt.tmp $D/ckpt__$N.pt
      cp $R/run_config.json $R/membership_$N.json $D/
      ep=$(/content/py/bin/python -c "import torch;print(torch.load('$D/ckpt__$N.pt',map_location='cpu',weights_only=False)['epoch'])")
      (cd $D && sha256sum ckpt__$N.pt > SHA256SUMS 2>/dev/null); echo "epoch $ep mtime $m" >> $D/SHA256SUMS
      [ -f $D/dataset-metadata.json ] || echo "{\"title\": \"knee-s6-${TAG}all-s${SEED}-ckpt-20261019\", \"id\": \"$ID\", \"licenses\": [{\"name\": \"other\"}]}" > $D/dataset-metadata.json
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
