#!/bin/bash
# usage: vm_train.sh <ARM J0|V2> <targets csv> <out dir> [extra args...]; nartaa's S6fall run_config, fold 0 of folds_v0, seed 42.
# Set S6V2_ALL_DATA=1 for the all-data fit (then pass the _all targets).  Resumable (--resume, checkpoint every epoch).
set -euo pipefail
ARM=$1; TG=$2; OUT=$3; shift 3
mkdir -p "$OUT"
export PYTHONDONTWRITEBYTECODE=1
exec /content/py/bin/python -I /content/code/compact_train_s6v2.py --px /content/px --targets "$TG" --out "$OUT" \
  --arms E_prior --backbone convnext_small --init /content/model.safetensors --seeds 1 --folds 5 --epochs 16 --batch 2 \
  --train-windows 64 --crop 288 --slices 3 --slice-jitter 0.0 --ema 0.0 --grad-ckpt --phys-mm 3.5 --lr 1e-4 --freeze-blocks 16 \
  --llrd 1.0 --head-lr 1e-3 --wd 0.05 --workers ${S6V2_LOADER_WORKERS:-5} --eval-workers 3 --resume --ckpt-every 1 \
  --eval-stride 3 --train-sampling uniform --cover-stride 3 --spatial-attn --seed-base 42 --aug crop --aug-res 0.0 \
  --i-have-user-authorisation-to-finetune "$@"
