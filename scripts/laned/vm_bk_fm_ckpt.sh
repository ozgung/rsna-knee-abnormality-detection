#!/bin/bash
# Lane D (Oct 19): Lane C's vm_bk_seed_ckpt.sh for the joint fastMRI fit (runs/FM_all, arm E_fm, seed 42).  Private Kaggle MODEL backup (dataset quota is tight) of the resumable checkpoint after
# every new epoch (polls every 60 s); state advances only after a successful publish; create vs version decided by a marker
# written after a confirmed create (or remote status ready).  Ends when the final model exists.
SEED=42
export KAGGLE_CONFIG_DIR=/content
K=/usr/local/bin/kaggle
R=/content/runs/FM_all
N=E_fm_s${SEED}_f0
D=/content/bk_fm_ckpt; mkdir -p $D; last=""
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
      bash /content/vm_kmodel.sh knee-laned-fm-ckpt-20261020 s42 $D ckpt__$N.pt "ckpt epoch $ep" > /dev/null && r=ok || r=fail
      if [ "$r" = ok ]; then last=$m; fi
      echo "$(date -u +%T) ckpt epoch $ep backup $r"
    fi
  fi
  sleep 60
done
