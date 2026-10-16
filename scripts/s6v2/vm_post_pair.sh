#!/bin/bash
# After both fits, re-evaluate the cache at stride 1, then score raw gold DICOMs.
# S6V2_ROOT overrides /content only for isolated shell acceptance fixtures.
set -euo pipefail
ROOT=${S6V2_ROOT:-/content}
cd "$ROOT"

require_output() {
  [ -s "$1" ] || { echo "POST_PAIR_FAIL missing output: $1" >&2; return 1; }
}

for ARM in J0 V2; do
  require_output "$ROOT/runs/$ARM/models/E_prior_s42_f0.pt"
done

PIDS=""
for ARM in J0 V2; do
  OUT="$ROOT/runs/${ARM}_s1"
  if [ ! -f "$OUT/partial_preds_E_prior_s42.json" ]; then
    bash "$ROOT/vm_train.sh" "$ARM" "$ROOT/targets/targets_${ARM}_f0.csv" "$OUT" --only-folds 0 \
      --reeval-run "$ROOT/runs/$ARM" --eval-stride 1 > "$ROOT/runs/${ARM}_s1.log" 2>&1 &
    PIDS="$PIDS $!"
  fi
done
FAILED=0
# PIDS contains only numeric child IDs; the empty list also works with Bash 3.2 + nounset.
for PID in $PIDS; do
  wait "$PID" || FAILED=1
done
if [ "$FAILED" -ne 0 ]; then
  echo POST_PAIR_FAIL_REEVALUATION >&2
  exit 1
fi
for ARM in J0 V2; do
  for FILE in partial_preds_E_prior_s42.json partial_oof_E_prior_s42.npy pool_uids.json; do
    require_output "$ROOT/runs/${ARM}_s1/$FILE"
  done
done

for ARM in J0 V2; do
  OUT="$ROOT/runs/${ARM}_ci"
  mkdir -p "$OUT"
  S6V2_DATA="$ROOT/data_gold" "$ROOT/py/bin/python" -I "$ROOT/vm_gold_infer.py" score "$ROOT/runs/$ARM/models" "$OUT" bf16 \
    > "$ROOT/runs/${ARM}_ci.log" 2>&1
  require_output "$OUT/predictions.csv"
  require_output "$OUT/logits.json"
done
echo POST_PAIR_DONE
