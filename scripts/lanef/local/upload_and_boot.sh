#!/bin/bash
# Lane F: usage: upload_and_boot.sh <session> <tag> <targets file name> <fold k> [seed]
cd "$(dirname "$0")"; SES=$1; TAG=$2; TGN=$3; K=$4; SEED=${5:-42}
for f in kaggle.json setup.sh mkdata.py verify_code.py train_uid_only.csv train_series.csv order.json compact_train_s6v2.py preflight_ref.json \
  vm_train.sh vm_preflight.py vm_restore_px.py vm_fold_boot.sh vm_fold_fit.sh vm_bk_fold_ckpt.sh vm_kmodel.sh vm_signed_url.py vm_gold_infer.py \
  vm_seed_readout.py vm_build_cache.py gold_probes.json gold58.json probes36.json $TGN; do
  for i in 1 2 3; do ./c.sh upload stage/$f /content/$f -s $SES | tail -1 | grep -q -i "error\|fail" || break; sleep 10; done
  echo "$(date -u +%T) up $f"
done
cat > /tmp/lanef_launch_$SES.py <<PY
import subprocess, hashlib
subprocess.run("mkdir -p /content/targets && cp /content/$TGN /content/targets/$TGN", shell=True, check=True)
for f in ['targets/$TGN','compact_train_s6v2.py','vm_fold_fit.sh','vm_fold_boot.sh','vm_bk_fold_ckpt.sh','vm_train.sh','preflight_ref.json']:
    print(f, hashlib.sha256(open('/content/'+f,'rb').read()).hexdigest()[:16])
subprocess.Popen(['bash','-c','S6V2_LOADER_WORKERS=8 bash /content/vm_fold_boot.sh $TAG /content/targets/$TGN $K $SEED'], stdout=open('/content/boot.log','wb'), stderr=subprocess.STDOUT, start_new_session=True)
print('launched')
PY
./c.sh exec -s $SES -f /tmp/lanef_launch_$SES.py --timeout 120
echo "$(date -u +%T) LAUNCHED $SES $TAG fold $K"
