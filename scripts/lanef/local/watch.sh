#!/bin/bash
# Lane F watcher: every 10 min probe the lane's sessions and the account's assignment count. Exits (so the agent is notified) on a
# new event: epoch-0 loss, fold FIT_DONE/EXPORT_DONE, KILL/FAIL, a lost session, or when a slot frees and lanef9 is created
# (then fold 1 is launched there with a 1 -> 3 chain). Seen events are kept in watch.seen. usage: watch.sh <session>...
cd "$(dirname "$0")"; touch watch.seen; MAXMIN=${MAXMIN:-400}; t0=$(date +%s)
PYC=/Users/ozgun/.local/share/uv/tools/google-colab-cli/bin/python
cat > /tmp/lanef_probe2.py <<'PY'
import subprocess, os, glob, re
def sh(c): return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
out = {"finalized": os.path.exists("/content/px/FINALIZED")}
for f in glob.glob("/content/fit_*.log") + glob.glob("/content/chain_*.log") + glob.glob("/content/after_*.log") + glob.glob("/content/boot*.log") + glob.glob("/content/r1_*.log") + glob.glob("/content/avg_*.log") + glob.glob("/content/bk_v*.log"):
    out[os.path.basename(f)] = sh("tail -n 1 %s" % f)[-160:]
for f in glob.glob("/content/runs/*.log"):
    ep = re.findall(r"(seed \d+ fold \d+ epoch \d+ loss [0-9.]+ +\d+s)", open(f, errors="replace").read())
    out[os.path.basename(f)] = ep[-1] if ep else sh("tail -n 1 %s" % f)[-160:]
print(out)
PY
while true; do
  ev=""
  for S in "$@"; do
    r=$(./c.sh exec -s $S -f /tmp/lanef_probe2.py --timeout 90 2>&1 | tail -1)
    echo "$(date -u +%FT%TZ) $S $r" >> watch.log
    for key in $(echo "$r" | grep -oE "fold [0-9] epoch 0 loss [0-9.]+|'[a-z0-9_]+\.log': '(FIT_DONE|EXPORT_DONE|EXPORT_FAIL|FIT_FAIL|KILL|READOUT_DONE|READOUT_EXPORT_DONE|R1_READOUTS_DONE|PROBE_EXPORT_DONE|BUNDLE_EXPORT_DONE|R1_BUNDLE_DONE|BUNDLE_FAIL|AVG_DONE|AVG_FAIL|PREFLIGHT FAIL|BOOT_LAUNCHED)[^']*|appears to be lost|not found" | sed "s/'//g; s/: /=/" | tr ' ' '_' | sort -u); do
      grep -qx "$S:$key" watch.seen || { echo "$S:$key" >> watch.seen; ev="$ev $S:$key"; }
    done
  done
  n=$($PYC - <<'PY' 2>/dev/null | tail -1
import os
from colab_cli.common import state
state.config_path = os.path.expanduser("~/.config/colab-cli/sessions-lanef-20261020.json")
A = state.client.list_assignments(); print(len(A), sum(1 for a in A if a.accelerator and a.accelerator.value == "A100"))
PY
)
  echo "$(date -u +%FT%TZ) assignments(total,A100)=$n" >> watch.log
  total=${n%% *}
  if [ -n "$total" ] && [ "$total" -le 2 ] && ! grep -q "^lanef9" <(./c.sh sessions 2>&1 | tr -d '[]' ) && ! grep -qx "lanef9:created" watch.seen; then
    ./c.sh new -s lanef9 --gpu A100 > new_lanef9.log 2>&1
    ep=$(./c.sh sessions 2>&1 | grep "^\[lanef9\]" | awk '{print $2}')
    if [ -n "$ep" ]; then
      echo "lanef9:created" >> watch.seen; echo "$(date -u +%FT%TZ) lanef9 created $ep" >> watch.log
      (nohup ./refresher.sh lanef9 $ep > /dev/null 2>&1 &)
      ./upload_and_boot.sh lanef9 r1 targets_V3_h4_all.csv 1 > upload_lanef9.log 2>&1
      ./c.sh upload stage/chain_fold.sh /content/chain_fold.sh -s lanef9 > /dev/null 2>&1
      printf 'import subprocess\nsubprocess.Popen(["bash","-c","bash /content/chain_fold.sh r1 /content/targets/targets_V3_h4_all.csv 1 3"], stdout=open("/content/chain_f1_f3.log","wb"), stderr=subprocess.STDOUT, start_new_session=True)\nprint("chain")\n' > /tmp/lanef_chain9.py
      ./c.sh exec -s lanef9 -f /tmp/lanef_chain9.py --timeout 60 >> watch.log 2>&1
      ev="$ev lanef9:created_fold1_launched"
    else
      echo "$(date -u +%FT%TZ) lanef9 creation failed: $(tail -c 200 new_lanef9.log | tr '\n' ' ')" >> watch.log
    fi
  fi
  for mk in labels_v4b labels_v5; do [ -f ../../lanel4-20261020/$mk/READY.json ] && ! grep -qx "L4:$mk" watch.seen && { echo "L4:$mk" >> watch.seen; ev="$ev L4:${mk}_READY"; }; done
  [ -n "$ev" ] && { echo "$(date -u +%FT%TZ) EVENT$ev" >> watch.log; echo "EVENT$ev"; exit 0; }
  [ $(( ($(date +%s) - t0) / 60 )) -ge $MAXMIN ] && { echo "WATCH_TIMEOUT"; exit 2; }
  sleep 600
done
