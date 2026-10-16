#!/bin/bash
# usage: vm_backup.sh cache | ckpt.  Private Kaggle datasets (kaggle CLI default is private); system Python client.
set -euo pipefail
export KAGGLE_CONFIG_DIR=/content
K=/usr/local/bin/kaggle
if [ "$1" = cache ]; then
  cat > /content/px/dataset-metadata.json <<J
{"title": "knee-s6-px-v4-320-20261015", "id": "ozgung/knee-s6-px-v4-320-20261015", "licenses": [{"name": "other"}]}
J
  nice -n 15 ionice -c3 $K datasets create -p /content/px --dir-mode skip -q
else
  D=/content/bk_ckpt; mkdir -p $D
  for ARM in J0 V2; do
    for f in /content/runs/$ARM/ckpt/*.pt /content/runs/$ARM/models/*.pt; do
      [ -f "$f" ] && cp "$f" $D/${ARM}__$(basename $(dirname $f))__$(basename $f)
    done
    cp /content/runs/$ARM/*.json $D/ 2>/dev/null || true
  done
  if [ ! -f $D/dataset-metadata.json ]; then
    echo '{"title": "knee-s6v2-ckpt-20261015", "id": "ozgung/knee-s6v2-ckpt-20261015", "licenses": [{"name": "other"}]}' > $D/dataset-metadata.json
    $K datasets create -p $D --dir-mode skip -q
  else
    $K datasets version -p $D -m "checkpoint $(date -u +%H%MZ)" --dir-mode skip -q
  fi
fi
echo BACKUP_DONE $1
