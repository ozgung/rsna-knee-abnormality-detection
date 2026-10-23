#!/bin/bash
# Lane F round 2/3 member hand-off (local): fetch the export folder of <tag> s<seed> from <session>, verify both SHA256SUMS,
# create the private dataset copy for the solo fixture, run the strict CPU acceptance (build_s6_solo_seed.py accept with the
# lane F slug), score gold-58 from the VM read-out (one read, exposed), then write members/<name>/READY.json and append EXPORTS.md.
# usage: member_export.sh <session> <tag> <seed> <model version>
set -uo pipefail
cd "$(dirname "$0")"; SES=$1; TAG=$2; SEED=$3; VER=$4
ROOT=~/Dev/rsna-knee-abnormality-detection; WT=$ROOT/.claude/worktrees/lanef-20261020; PY=$ROOT/.venv/bin/python
NAME=${TAG}-s${SEED}; CK=E_prior_s${SEED}_${TAG}all.pt
LANE=$ROOT/outputs/analysis/lanef-20261020; DL=$LANE/$NAME/dl; mkdir -p $DL
M=$ROOT/outputs/lanef-20261020/members/$NAME; mkdir -p $M
E=/content/export_${TAG}_s${SEED}
for f in SHA256SUMS READOUT_SHA256SUMS run_config.json membership_E_prior_s${SEED}_f0.json partial_preds_E_prior_s${SEED}.json targets.csv \
         gold58_predictions.csv gold58_logits.json gold58_receipt.json probe36_fp16_predictions.csv probe36_fp16_logits.json probe36_fp16_receipt.json $CK; do
  for i in 1 2 3; do ./c.sh download $E/$f $DL/$f -s $SES | tail -1 | grep -qi "error\|fail\|not found" || break; sleep 15; done
done
[ "$(cd $DL && shasum -a 256 -c SHA256SUMS 2>&1 | grep -vc OK)" = 0 ] || { echo "SHA256SUMS MISMATCH"; exit 1; }
[ "$(cd $DL && shasum -a 256 -c READOUT_SHA256SUMS 2>&1 | grep -vc OK)" = 0 ] || { echo "READOUT_SHA256SUMS MISMATCH"; exit 1; }
echo "$(date -u +%T) fetched and verified $NAME"
# private dataset copy (same files as the Model export + read-outs)
DS=knee-s6-${TAG}all-s${SEED}-20261020; DSD=$LANE/$NAME/ds; mkdir -p $DSD
cp $DL/{$CK,SHA256SUMS,run_config.json,membership_E_prior_s${SEED}_f0.json,partial_preds_E_prior_s${SEED}.json,targets.csv,READOUT_SHA256SUMS,gold58_predictions.csv,gold58_logits.json,gold58_receipt.json,probe36_fp16_predictions.csv,probe36_fp16_logits.json,probe36_fp16_receipt.json} $DSD/
printf '{"title": "%s", "id": "ozgung/%s", "licenses": [{"name": "other"}]}\n' $DS $DS > $DSD/dataset-metadata.json
if ! kaggle datasets status ozgung/$DS > /dev/null 2>&1; then
  kaggle datasets create -p $DSD -q -r skip 2>&1 | tail -1
fi
for i in $(seq 1 20); do st=$(kaggle datasets status ozgung/$DS 2>&1 | tail -1); echo "$st" | grep -q ready && break; sleep 30; done
echo "$(date -u +%T) dataset ozgung/$DS: $st"
# strict CPU acceptance (files already in dl/, so no download); lane F slug, lane F dir
(cd $WT && $PY scripts/build_s6_solo_seed.py accept --tag $TAG --seed $SEED --version $VER --model-slug knee-s6-lanef-20261020 --lane-dir $LANE --python ${QPY:-$PY} > $M/accept.log 2>&1) && ACC=PASS || ACC=FAIL
echo "$(date -u +%T) accept $ACC ($(grep -o '"state_sha256": "[0-9a-f]\{16\}' $LANE/$NAME/seed-acceptance.json 2>/dev/null))"
# gold-58 macro from the VM read-out (exposed; the predictions.csv came from one bf16 compact_infer pass)
mkdir -p $M/ro_gold && cp $DL/gold58_predictions.csv $M/ro_gold/predictions.csv && cp $DL/gold58_receipt.json $M/ro_gold/receipt.json && cp $DL/gold58_logits.json $M/ro_gold/logits.json
$PY $ROOT/scripts/s6v2/score_gold_seeds.py $ROOT $M/gold58_score.json $NAME=$M/ro_gold > $M/gold58_score.log 2>&1
G=$($PY -c "import json;print(round(json.load(open('$M/gold58_score.json'))['runs']['$NAME']['macro'],4))" 2>/dev/null || echo unscored)
$PY - <<PY
import json, hashlib, datetime, os
dl="$DL"; sums={l.split()[1]: l.split()[0] for l in open(dl+"/SHA256SUMS").read().splitlines()}
ro={l.split()[1]: l.split()[0] for l in open(dl+"/READOUT_SHA256SUMS").read().splitlines()}
rc=json.load(open(dl+"/run_config.json")); mem=json.load(open(dl+"/membership_E_prior_s${SEED}_f0.json"))
acc=json.load(open("$LANE/$NAME/seed-acceptance.json")) if os.path.exists("$LANE/$NAME/seed-acceptance.json") else None
r={"name":"$NAME","lane":"F","tag":"$TAG","seed":$SEED,"written_utc":datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
   "model":{"ref":"ozgung/knee-s6-lanef-20261020/pyTorch/$NAME","version":$VER,"mount":"models/ozgung/knee-s6-lanef-20261020/pytorch/$NAME/$VER"},
   "dataset":{"ref":"ozgung/$DS","version":1,"status":"""$st"""},
   "checkpoint":{"file":"$CK","sha256":sums["$CK"],"bytes":os.path.getsize(dl+"/$CK"),"runtime_alias":"E_prior_s${SEED}_f0.pt"},
   "receipts_sha256":{k:v for k,v in sums.items() if k!="$CK"},"targets_sha256":sums.get("targets.csv"),
   "run_config":{k:rc.get(k) for k in ("epochs","seed_base","pool_studies","gold_studies","only_folds","eval_stride")},
   "membership":{k:mem.get(k) for k in ("train","train_sha256","holdout","gold","train_gold_overlap","all_data")},
   "probe36_fp16_reference":{"path":dl+"/probe36_fp16_predictions.csv","sha256":ro["probe36_fp16_predictions.csv"],"logits_sha256":ro["probe36_fp16_logits.json"],"amp":"fp16","device":"A100","eval_stride":1},
   "gold58":{"macro":"$G","exposed":True,"reads":1,"path":dl+"/gold58_predictions.csv","sha256":ro["gold58_predictions.csv"],"amp":"bf16"},
   "cpu_acceptance":{"status":"$ACC","path":"$LANE/$NAME/seed-acceptance.json","state_sha256":acc and acc.get("state_sha256")},
   "local_copy":dl}
json.dump(r,open("$M/READY.json","w"),indent=1)
line="| %s | %s | %s | %s/%d | ozgung/%s v1 | %s | %s | %s | %s |" % (r["written_utc"], "$NAME", "$TAG seed $SEED, 16 epochs, all-data", r["model"]["ref"], $VER, "$DS", sums["$CK"][:16], "$G", ro["probe36_fp16_predictions.csv"][:16], "$ACC")
p="$ROOT/outputs/lanef-20261020/EXPORTS.md"
if not os.path.exists(p): open(p,"w").write("# Lane F exports\n\n| written UTC | member | recipe | Model ref/version | dataset | checkpoint sha256 | gold-58 (exposed) | probe-36 fp16 ref sha256 | CPU accept |\n|---|---|---|---|---|---|---|---|---|\n")
open(p,"a").write(line+"\n"); print("READY", "$M/READY.json")
PY
