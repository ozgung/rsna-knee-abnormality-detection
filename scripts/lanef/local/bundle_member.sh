#!/bin/bash
# Lane F: fetch the 4-fold bundle read-outs from lanef8 (export_r1_bundle), verify, score gold-58 (one compact_infer bf16 read of the
# 4-fold mean; per-model logits in the same file), write members/r1-bundle/READY.json and an EXPORTS.md line.  usage: bundle_member.sh <session> <model version>
set -u; cd "$(dirname "$0")"; S=$1; VER=$2; ROOT=/Users/ozgun/Dev/rsna-knee-abnormality-detection; PY=$ROOT/.venv/bin/python
L=$ROOT/outputs/lanef-20261020; M=$L/members/r1-bundle; mkdir -p $M/dl $M/ro_gold
for f in SHA256SUMS MODELS_SHA256SUMS gold58_4fold_predictions.csv gold58_4fold_logits.json gold58_4fold_receipt.json probe36_fp16_4fold_predictions.csv probe36_fp16_4fold_logits.json probe36_fp16_4fold_receipt.json; do
  for i in 1 2 3; do ./c.sh download /content/export_r1_bundle/$f $M/dl/$f -s $S | tail -1 | grep -qi "error\|fail\|not found" || break; sleep 10; done
done
[ "$(cd $M/dl && shasum -a 256 -c SHA256SUMS 2>&1 | grep -vc OK)" = 0 ] || { echo "bundle SHA256SUMS mismatch"; exit 1; }
diff <(sort $M/dl/MODELS_SHA256SUMS) <(sort stage/bundle_SHA256SUMS) > /dev/null && echo "model manifest matches the local checkpoints" || { echo "MODEL MANIFEST MISMATCH"; exit 1; }
cp $M/dl/gold58_4fold_predictions.csv $M/ro_gold/predictions.csv; cp $M/dl/gold58_4fold_logits.json $M/ro_gold/logits.json; cp $M/dl/gold58_4fold_receipt.json $M/ro_gold/receipt.json
$PY $ROOT/scripts/s6v2/score_gold_seeds.py $ROOT $M/gold58_score.json r1-bundle=$M/ro_gold > $M/gold58_score.log 2>&1 || { echo "gold scoring failed"; tail -3 $M/gold58_score.log; }
$PY - <<PY
import json, datetime
sc=json.load(open("$M/gold58_score.json")); sums={l.split()[1]: l.split()[0] for l in open("$M/dl/SHA256SUMS").read().splitlines()}
models={l.split()[1]: l.split()[0] for l in open("$M/dl/MODELS_SHA256SUMS").read().splitlines()}
runs={k: round(v["macro"],4) for k,v in sc["runs"].items()}
r={"name":"r1-bundle","lane":"F","kind":"4-fold bundle of the round-1 hash-fold models (members r1-f0..3)","written_utc":datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
   "members":["r1-f0","r1-f1","r1-f2","r1-f3"],"checkpoints_sha256":models,
   "model":{"ref":"ozgung/knee-s6-lanef-20261020/pyTorch/r1-bundle","version":$VER,"content":"gold-58 4-fold read-outs, probe-36 fp16 4-model decode, MODELS_SHA256SUMS (no weights; the weights are in r1-f<k>)"},
   "gold58":{"macro_4fold_mean":runs.get("r1-bundle"),"per_model_from_the_same_read":{k:v for k,v in runs.items() if k!="r1-bundle"},"exposed":True,"reads":"one compact_infer bf16 pass over the 4 models (one decode); plus the discarded 3-model attempt earlier, also exposed",
             "path":"$M/dl/gold58_4fold_predictions.csv","sha256":sums["gold58_4fold_predictions.csv"],"receipt":sc.get("runs",{}).get("r1-bundle",{}).get("models")},
   "probe36_fp16_4model":{"path":"$M/dl/probe36_fp16_4fold_predictions.csv","sha256":sums["probe36_fp16_4fold_predictions.csv"],"logits_sha256":sums["probe36_fp16_4fold_logits.json"],"note":"per-member references are in members/r1-f<k>"},
   "receipts_sha256":sums}
json.dump(r,open("$M/READY.json","w"),indent=1)
line="| %s | r1-bundle | 4-fold mean of r1-f0..3 | ozgung/knee-s6-lanef-20261020/pyTorch/r1-bundle/%d | (weights in r1-f<k> datasets) | %s | %s (4-fold mean, compact_infer bf16) | %s | see members |" % (r["written_utc"], $VER, ",".join(v[:8] for v in models.values()), runs.get("r1-bundle"), sums["probe36_fp16_4fold_predictions.csv"][:16])
open("$L/EXPORTS.md","a").write(line+"\n"); print("READY r1-bundle gold-58 4-fold:", runs)
PY
