#!/bin/bash
# Lane F round-1 fold-bundle members (local): for each fold k with its checkpoint in members/dl/r1-f<k>/ — verify SHA256SUMS,
# strict CPU qualification (scripts/qualify_s6solo_cpu.py: strict load, repeat and masked-perturbation deltas, state fingerprint),
# private dataset copy knee-s6-r1-f<k>-20261020 (checkpoint + receipts + probe-36 fp16 reference when fetched), then
# members/r1-f<k>/READY.json and an EXPORTS.md line.  usage: fold_members.sh [folds...]   (default 0 1 2 3)
set -uo pipefail
cd "$(dirname "$0")"; ROOT=/Users/ozgun/Dev/rsna-knee-abnormality-detection; PY=${QPY:-$ROOT/.venv/bin/python}
L=$ROOT/outputs/lanef-20261020; EX=$ROOT/outputs/s6v2/dl/extras/heliosli; CTRL=$ROOT/outputs/analysis/s6v2-final-acceptance-20261016/control/E_prior_s42_fall.pt
TAG=${TAG:-r1}; FOLDS=${@:-0 1 2 3}
for k in $FOLDS; do
  D=$L/members/dl/$TAG-f$k; CK=E_prior_s42_f$k.pt; M=$L/members/$TAG-f$k; mkdir -p $M
  [ -s $D/$CK ] || { echo "$TAG-f$k: checkpoint not fetched yet"; continue; }
  [ "$(cd $D && grep -v partial_oof SHA256SUMS | shasum -a 256 -c - 2>&1 | grep -vc OK)" = 0 ] || { echo "$TAG-f$k: SHA256SUMS mismatch"; continue; }
  SHA=$(grep " $CK$" $D/SHA256SUMS | cut -d' ' -f1); BYTES=$(stat -f %z $D/$CK)
  if [ ! -s $M/cpu.json ]; then
    (cd $ROOT && $PY scripts/qualify_s6solo_cpu.py --train $EX/compact_train.py --infer $EX/compact_infer.py --control $CTRL --v2 $D/$CK --v2-sha $SHA --v2-bytes $BYTES --out $M/cpu.json > $M/cpu.log 2>&1) || echo "$TAG-f$k: CPU qualification FAILED (see $M/cpu.log)"
  fi
  CPU=$($PY -c "import json;c=json.load(open('$M/cpu.json'));v=c['checkpoints']['v2'];print('PASS' if c['status']=='PASS_SYNTHETIC_CPU_ONLY' and v['strict_load'] is True and v['checkpoint_sha256']=='$SHA' else 'FAIL', v.get('state_sha256','')[:16], v.get('repeat_max_abs'), v.get('masked_perturbation_max_abs'))" 2>/dev/null || echo "FAIL - - -")
  # probe-36 fp16 reference (fetched from the VM's export folder after r1_readouts.sh) — optional at first pass
  PR=$L/oof_$TAG/raw/f$k; PREF=""; PSHA=""
  if [ -s $PR/probe36_fp16_predictions.csv ] && [ -s $PR/READOUT_SHA256SUMS ]; then
    [ "$(cd $PR && shasum -a 256 -c READOUT_SHA256SUMS 2>&1 | grep -vc OK)" = 0 ] && { PREF=$PR/probe36_fp16_predictions.csv; PSHA=$(grep " probe36_fp16_predictions.csv$" $PR/READOUT_SHA256SUMS | cut -d' ' -f1); }
  fi
  # private dataset copy
  DS=knee-s6-$TAG-f$k-20261020; DSD=$M/ds; mkdir -p $DSD
  cp $D/{$CK,SHA256SUMS,run_config.json,membership_E_prior_s42_f$k.json,partial_preds_E_prior_s42.json,targets.csv} $DSD/
  [ -n "$PREF" ] && cp $PR/{probe36_fp16_predictions.csv,probe36_fp16_logits.json,probe36_fp16_receipt.json,READOUT_SHA256SUMS} $DSD/
  printf '{"title": "%s", "id": "ozgung/%s", "licenses": [{"name": "other"}]}\n' $DS $DS > $DSD/dataset-metadata.json
  if kaggle datasets status ozgung/$DS > /dev/null 2>&1; then
    [ -n "$PREF" ] && ! grep -q "\"probe_ref_in_dataset\": true" $M/READY.json 2>/dev/null && kaggle datasets version -p $DSD -m "probe-36 fp16 reference added" -q -r skip 2>&1 | tail -1
  else
    kaggle datasets create -p $DSD -q -r skip 2>&1 | tail -1
  fi
  for i in $(seq 1 12); do st=$(kaggle datasets status ozgung/$DS 2>&1 | tail -1); echo "$st" | grep -q ready && break; sleep 20; done
  DSV=$(kaggle datasets list -m --search $DS 2>/dev/null | grep -c "ozgung/$DS")
  G=$($PY -c "import json;s=json.load(open('$L/oof_$TAG/summary.json'));print(round(s['gold58']['per_fold_model']['$k'],4))")
  $PY - <<PY
import json, datetime, os
d="$D"; sums={l.split()[1]: l.split()[0] for l in open(d+"/SHA256SUMS").read().splitlines()}
rc=json.load(open(d+"/run_config.json")); mem=json.load(open(d+"/membership_E_prior_s42_f$k.json"))
cpu="$CPU".split()
r={"name":"$TAG-f$k","lane":"F","kind":"round-1 hash-fold model (OOF teacher bundle member)","written_utc":datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
   "model":{"ref":"ozgung/knee-s6-lanef-20261020/pyTorch/$TAG-f$k","version_with_checkpoint":1,"mount":"models/ozgung/knee-s6-lanef-20261020/pytorch/$TAG-f$k/1","note":"a later version adds the probe-36 reference files"},
   "dataset":{"ref":"ozgung/$DS","status":"""$st""","probe_ref_in_dataset":bool("$PREF")},
   "checkpoint":{"file":"$CK","sha256":"$SHA","bytes":$BYTES,"runtime_alias":"E_prior_s42_f$k.pt"},
   "receipts_sha256":{k:v for k,v in sums.items() if k!="$CK"},"targets_sha256":sums.get("targets.csv"),
   "run_config":{k:rc.get(k) for k in ("epochs","seed_base","folds","only_folds","eval_stride","pool_studies","gold_studies")},
   "membership":{k:mem.get(k) for k in ("fold","train","train_sha256","holdout","holdout_sha256","gold","train_gold_overlap","all_data")},
   "training":"S6 V2-all recipe, $TAG targets with hash-4 folds (see targets_sha256), ImageNet init, 12 epochs, trained on the three other hash folds; held-out fold $k = the study set this model never saw",
   "probe36_fp16_reference":({"path":"$PREF","sha256":"$PSHA","amp":"fp16","device":"A100","eval_stride":1} if "$PREF" else "pending: r1_readouts.sh on the VM"),
   "gold58":{"macro_this_fold_model":$G,"exposed":True,"read":"trainer end-of-fit prediction (pixel cache, stride 1); the 4-fold mean-logit compact_infer read is in members/$TAG-bundle/"},
   "cpu_qualification":{"status":cpu[0],"state_sha256_prefix":cpu[1] if len(cpu)>1 else None,"repeat_max_abs":cpu[2] if len(cpu)>2 else None,"masked_perturbation_max_abs":cpu[3] if len(cpu)>3 else None,"receipt":"$M/cpu.json"},
   "local_copy":d}
json.dump(r,open("$M/READY.json","w"),indent=1)
line="| %s | $TAG-f$k | fold-%s model, $TAG hash-fold fit, 12 ep | ozgung/knee-s6-lanef-20261020/pyTorch/$TAG-f$k/1 | ozgung/$DS (%s) | %s | %s (fold model, trainer read) | %s | %s |" % (r["written_utc"], "$k", """$st""".split()[-1] if """$st""" else "?", "$SHA"[:16], "$G", ("$PSHA"[:16] or "pending"), cpu[0])
p="$L/EXPORTS.md"
if not os.path.exists(p): open(p,"w").write("# Lane F exports\n\n| written UTC | member | recipe | Model ref/version | dataset | checkpoint sha256 | gold-58 (exposed) | probe-36 fp16 ref sha256 | CPU accept |\n|---|---|---|---|---|---|---|---|---|\n")
open(p,"a").write(line+"\n"); print("READY", "$TAG-f$k", cpu[0], "dataset:", """$st""".strip()[-40:], "probe:", "yes" if "$PREF" else "pending")
PY
done
