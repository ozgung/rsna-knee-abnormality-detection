#!/bin/bash
# Lane F round 3: upload the stage set to a fresh session and start one seed queue.  usage: upload_and_queue.sh <session> <queue args...>
#   e.g. upload_and_queue.sh lanef10 v5 /content/targets/targets_V5_all.csv none 42
#   resume variant: RESUME45=1 upload_and_queue.sh lanef12 v4b /content/targets/targets_V4B_all.csv none 45   (then chain v5 44 separately)
cd "$(dirname "$0")"; SES=$1; shift
for f in kaggle.json setup.sh mkdata.py verify_code.py train_uid_only.csv train_series.csv order.json compact_train_s6v2.py preflight_ref.json \
  vm_train.sh vm_preflight.py vm_restore_px.py vm_seed_boot_lanef.sh vm_seed_fit_lanef.sh vm_bk_seed_ckpt_lanef.sh lanef_after_fit.sh v5_seed_queue.sh \
  vm_kmodel.sh vm_signed_url.py vm_gold_infer.py vm_seed_readout.py vm_build_cache.py gold_probes.json gold58.json probes36.json resume_v4b45.sh \
  targets_V5_all.csv targets_V4B_all.csv; do
  for i in 1 2 3; do ./c.sh upload stage/$f /content/$f -s $SES | tail -1 | grep -qi "uploaded" && break; sleep 10; done
done
ARGS="$*"
cat > /tmp/lanef_uq_$SES.py <<PY
import subprocess, hashlib
subprocess.run("mkdir -p /content/targets && cp /content/targets_V5_all.csv /content/targets_V4B_all.csv /content/targets/", shell=True, check=True)
for f in ['targets/targets_V5_all.csv','targets/targets_V4B_all.csv','compact_train_s6v2.py','vm_train.sh','vm_seed_fit_lanef.sh','lanef_after_fit.sh','v5_seed_queue.sh']:
    print(f, hashlib.sha256(open('/content/'+f,'rb').read()).hexdigest()[:16])
pre = "bash /content/resume_v4b45.sh > /content/resume45.log 2>&1 && " if "${RESUME45:-0}" == "1" else ""
subprocess.Popen(['bash','-c', pre + 'DEADLINE_EPOCH=1792634400 bash /content/v5_seed_queue.sh $ARGS'], stdout=open('/content/queue_main.log','wb'), stderr=subprocess.STDOUT, start_new_session=True)
print('queue launched: $ARGS')
PY
./c.sh exec -s $SES -f /tmp/lanef_uq_$SES.py --timeout 120
echo "$(date -u +%T) LAUNCHED $SES: $ARGS"
