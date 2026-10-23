#!/bin/bash
# Lane F checkpoint-averaging test for one round-1 fold model.  usage: r1_avg.sh <fold k> <targets csv> [seed]
# Averages the weights of the last three epochs (trainer epochs 9 and 10 from the per-epoch backups, epoch 11 = the final
# deploy state), writes the averaged model into a reeval run directory and recomputes the held-out hash-fold column with the
# trainer's --reeval-run (same eval: pixel cache, stride 1).  Output: /content/avg_r1_f<k>/out/partial_oof_E_prior_s<seed>.npy
# and partial_preds (gold logits of the averaged model, divided by K as usual).  Epoch identities are read from the backups'
# SHA256SUMS "epoch N" lines, never assumed from version numbers.  Prints no study IDs.
set -uo pipefail
cd /content; export KAGGLE_CONFIG_DIR=/content
K=$1; TG=$2; SEED=${3:-42}; N=E_prior_s${SEED}_f${K}
R=/content/runs/r1_f$K; A=/content/avg_r1_f$K; mkdir -p $A/models $A/ckpts
[ -s $R/models/$N.pt ] || { echo "AVG_FAIL no final model"; exit 1; }
# epoch 10 is the last local backup (verify), epoch 9 comes from the Kaggle ckpt Model (walk versions downwards until the receipt says epoch 9)
have() { grep -q "^epoch $1 " $2/SHA256SUMS 2>/dev/null; }
if have 10 /content/bk_r1_f${K}_ckpt; then cp /content/bk_r1_f${K}_ckpt/ckpt__$N.pt $A/ckpts/e10.pt; fi
REF=ozgung/knee-s6-lanef-ckpt-20261020/pyTorch/r1-f$K
# Kaggle's whole-version download endpoints answer 404 for this account; the per-file route works:
#   https://www.kaggle.com/api/v1/models/<owner>/<model>/pyTorch/<variation>/<version>/download/<file>  (302 to signed storage)
AUTH=$(/content/py/bin/python -c "import json;c=json.load(open('/content/kaggle.json'));print(c['username']+':'+c['key'])")
kfile() { curl -sL -u "$AUTH" -o "$3" "https://www.kaggle.com/api/v1/models/$REF/$1/download/$2"; }
vers=$(/usr/local/bin/kaggle models instances versions list $REF --page-size 50 2>/dev/null | awk -v v="r1-f$K" '$2==v {print $1}' | sort -rn)
for v in $vers; do
  [ -s $A/ckpts/e9.pt ] && [ -s $A/ckpts/e10.pt ] && break
  d=$A/dl_v$v; mkdir -p $d
  kfile $v SHA256SUMS $d/SHA256SUMS || continue
  ep=$(grep -oE "^epoch [0-9]+" $d/SHA256SUMS | awk '{print $2}')
  echo "version $v = epoch ${ep:-?}"
  for want in 9 10; do
    if [ "$ep" = "$want" ] && [ ! -s $A/ckpts/e$want.pt ]; then
      kfile $v ckpt__$N.pt $d/ckpt__$N.pt
      (cd $d && grep "ckpt__$N.pt" SHA256SUMS | sha256sum -c - 2>/dev/null | grep -q OK) && mv $d/ckpt__$N.pt $A/ckpts/e$want.pt && echo "epoch $want checkpoint fetched and verified" || echo "epoch $want checkpoint sha mismatch"
    fi
  done
  rm -rf $d
  [ -n "$ep" ] && [ "$ep" -lt 9 ] 2>/dev/null && break
done
[ -s $A/ckpts/e9.pt ] && [ -s $A/ckpts/e10.pt ] || { echo "AVG_FAIL missing epoch 9/10 backups"; exit 1; }
/content/py/bin/python - "$A" "$R/models/$N.pt" "$N" <<'PY'
import sys, torch, hashlib, json
A, final_path, N = sys.argv[1], sys.argv[2], sys.argv[3]
final = torch.load(final_path, map_location="cpu")
c9 = torch.load(A + "/ckpts/e9.pt", map_location="cpu", weights_only=False); c10 = torch.load(A + "/ckpts/e10.pt", map_location="cpu", weights_only=False)
assert c9["epoch"] == 9 and c10["epoch"] == 10, (c9["epoch"], c10["epoch"])
s9, s10 = c9["model"], c10["model"]
assert set(final) <= set(s9) and set(final) <= set(s10), "key mismatch between the final deploy state and the checkpoints"
out, n_avg, n_copy = {}, 0, 0
for k, v in final.items():
    if v.is_floating_point():
        out[k] = ((s9[k].float() + s10[k].float() + v.float()) / 3).to(v.dtype); n_avg += 1
    else:
        out[k] = v.clone(); n_copy += 1          # integer buffers (e.g. num_batches_tracked) from the final model
torch.save(out, A + "/models/" + N + ".pt")
h = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
json.dump({"fold": N, "epochs_averaged": [9, 10, 11], "tensors_averaged": n_avg, "tensors_copied": n_copy, "final_sha256": h(final_path),
           "e9_ckpt_sha256": h(A + "/ckpts/e9.pt"), "e10_ckpt_sha256": h(A + "/ckpts/e10.pt"), "averaged_sha256": h(A + "/models/" + N + ".pt")},
          open(A + "/avg_receipt.json", "w"), indent=1)
print("averaged", n_avg, "tensors, copied", n_copy)
PY
cp $R/pool_uids.json $A/pool_uids.json
rm -rf $A/out; mkdir -p $A/out
S6V2_LOADER_WORKERS=4 bash /content/vm_train.sh V2 "$TG" $A/out --folds 4 --only-folds $K --eval-stride 1 --seed-base $SEED --reeval-run $A > $A/reeval.log 2>&1
[ -s $A/out/partial_oof_E_prior_s${SEED}.npy ] && { (cd $A/out && sha256sum partial_oof_E_prior_s${SEED}.npy partial_preds_E_prior_s${SEED}.json > SHA256SUMS); cp $A/avg_receipt.json $A/out/; echo "AVG_DONE fold $K"; } || { echo "AVG_FAIL reeval"; tail -n 3 $A/reeval.log; exit 1; }
