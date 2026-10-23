#!/bin/bash
# Lane F: every 10 min, read boot/fit progress of the given sessions; exit when every session has an epoch-0 loss line or a failure, or after MAXMIN minutes.
cd "$(dirname "$0")"; MAXMIN=${MAXMIN:-180}; t0=$(date +%s)
cat > /tmp/lanef_probe.py <<'PY'
import subprocess, os
def tail(p, n=1):
    try: return subprocess.run(["tail", "-n", str(n), p], capture_output=True, text=True).stdout.strip().replace("\n", " | ")[-300:]
    except Exception as e: return str(e)
out = {"boot": tail("/content/boot.log", 2), "restore": tail("/content/restore.log"), "setup_done": os.path.exists("/content/setup.done"),
       "finalized": os.path.exists("/content/px/FINALIZED")}
for f in sorted(os.listdir("/content")):
    if f.startswith(("fit_r", "preflight_r", "bk_r")) and f.endswith(".log"): out[f] = tail("/content/" + f)
import glob
for f in glob.glob("/content/runs/*.log"):
    ep = subprocess.run(["grep", "-oE", "epoch [0-9]+ loss [0-9.]+ +[0-9]+s", f], capture_output=True, text=True).stdout.strip().split("\n")
    out[os.path.basename(f)] = [e for e in ep if e][-2:]
print(out)
PY
while true; do
  done_n=0
  for S in "$@"; do
    r=$(./c.sh exec -s $S -f /tmp/lanef_probe.py --timeout 90 2>&1 | tail -1)
    echo "$(date -u +%FT%TZ) $S $r" >> monitor_boot.log
    echo "$r" | grep -q "epoch 0 loss\|KILL\|FAIL\|Error\|not found" && done_n=$((done_n+1))
  done
  [ $done_n -ge $# ] && { echo "$(date -u +%FT%TZ) MONITOR_DONE" >> monitor_boot.log; exit 0; }
  [ $(( ($(date +%s) - t0) / 60 )) -ge $MAXMIN ] && { echo "$(date -u +%FT%TZ) MONITOR_TIMEOUT" >> monitor_boot.log; exit 2; }
  sleep 600
done
