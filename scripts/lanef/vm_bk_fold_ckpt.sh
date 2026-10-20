#!/bin/bash
# Lane F: vm_bk_seed_ckpt.sh generalised to a fold fit.  usage: vm_bk_fold_ckpt.sh <tag> <fold k> [seed]
# Private Kaggle MODEL backup (ozgung/knee-s6-lanef-ckpt-20261020, variation <tag>-f<k>) of the resumable checkpoint after every
# new epoch (polls every 60 s); state advances only after a publish confirmed by the remote file listing.
TAG=$1; K=$2; SEED=${3:-42}
export KAGGLE_CONFIG_DIR=/content
R=/content/runs/${TAG}_f${K}
N=E_prior_s${SEED}_f${K}
D=/content/bk_${TAG}_f${K}_ckpt; mkdir -p $D; last=""
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
      bash /content/vm_kmodel.sh knee-s6-lanef-ckpt-20261020 ${TAG}-f${K} $D ckpt__$N.pt "ckpt epoch $ep" > /dev/null && r=ok || r=fail
      if [ "$r" = ok ]; then last=$m; fi
      echo "$(date -u +%T) ckpt epoch $ep backup $r"
    fi
  fi
  sleep 60
done
