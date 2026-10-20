#!/bin/bash
# Lane F round 2/3: on a session whose cache is already restored, upload the targets table and start one all-data 16-epoch fit
# (vm_seed_boot_lanef.sh: preflight on the targets, per-epoch ckpt backup, fit, export) plus lanef_after_fit.sh (gold-58 once,
# probe-36 fp16 reference, read-out export).  usage: launch_r2.sh <session> <tag> <targets file in stage/> <seed>
cd "$(dirname "$0")"; SES=$1; TAG=$2; TGN=$3; SEED=$4
for i in 1 2 3; do ./c.sh upload stage/$TGN /content/$TGN -s $SES | tail -1 | grep -qi "uploaded" && break; sleep 10; done
cat > /tmp/lanef_r2_$SES.py <<PY
import subprocess, hashlib
subprocess.run("mkdir -p /content/targets && cp /content/$TGN /content/targets/$TGN", shell=True, check=True)
for f in ['targets/$TGN','vm_seed_boot_lanef.sh','vm_seed_fit_lanef.sh','vm_bk_seed_ckpt_lanef.sh','lanef_after_fit.sh','vm_train.sh','compact_train_s6v2.py']:
    print(f, hashlib.sha256(open('/content/'+f,'rb').read()).hexdigest()[:16])
subprocess.Popen(['bash','-c','S6V2_LOADER_WORKERS=8 bash /content/vm_seed_boot_lanef.sh $TAG /content/targets/$TGN $SEED'], stdout=open('/content/boot_${TAG}_s${SEED}.log','wb'), stderr=subprocess.STDOUT, start_new_session=True)
subprocess.Popen(['bash','-c','bash /content/lanef_after_fit.sh $TAG $SEED'], stdout=open('/content/after_${TAG}_s${SEED}.log','wb'), stderr=subprocess.STDOUT, start_new_session=True)
print('launched')
PY
./c.sh exec -s $SES -f /tmp/lanef_r2_$SES.py --timeout 120
echo "$(date -u +%T) LAUNCHED $SES $TAG seed $SEED"
