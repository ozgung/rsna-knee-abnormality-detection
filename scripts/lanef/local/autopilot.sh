#!/bin/bash
# Lane F round-3 autopilot (local, 10-minute cadence).  Each cycle: (1) when L4's labels_v5/READY.json appears, verify the v5
# targets (sha from READY.json, check_targets, 4,345-row pool), stage + upload them, start the v5 seed queues (lanef9: 42 44 46 at
# once; lanef8: 43 45 47 after v4b-s45); (2) for every finished job (READOUT_EXPORT_DONE in /content/after_<tag>_s<seed>.log)
# without a local members/<tag>-s<seed>/READY.json, run member_export.sh; (3) surface failures; (4) at the deadline (Oct 22
# 06:30 UTC; Oct 22 00:00 + 23400 s) regardless; each VM is stopped as soon as its queues are done and its members handed off, stop both VMs and the refreshers.  Exits on every event so the
# agent is notified; rerun to continue.  Log: autopilot.log; state: autopilot.state (v5 handled flag).
cd "$(dirname "$0")"; SESSIONS=${SESSIONS:-"lanef10 lanef11 lanef12"}; ROOT=/Users/ozgun/Dev/rsna-knee-abnormality-detection; L=$ROOT/outputs/lanef-20261020
export QPY=/Users/ozgun/miniconda3/envs/fastgen/bin/python
V5=$ROOT/outputs/lanel4-20261020/labels_v5; DEADLINE=$(date -u -j -f "%Y-%m-%d %H:%M:%S" "2026-10-22 00:00:00" +%s 2>/dev/null || date -u -d "2026-10-22 00:00:00" +%s)
touch autopilot.state; log() { echo "$(date -u +%FT%TZ) $*" >> autopilot.log; }
probe() { ./c.sh exec -s $1 -f /tmp/lanef_ap_probe.py --timeout 90 2>&1 | tail -1; }
cat > /tmp/lanef_ap_probe.py <<'PY'
import glob, re, subprocess, json
sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout
out = {"done": [], "fail": [], "running": [], "queue": sh("tail -n 1 /content/v5_queue.log 2>/dev/null").strip()[-160:]}
for f in sorted(glob.glob("/content/after_*_s*.log")):
    m = re.search(r"after_(\w+)_s(\d+)\.log", f); tag, seed = m.group(1), m.group(2)
    t = open(f, errors="replace").read()
    v = re.findall(r"READOUT_EXPORT_DONE \S+/(\d+)", t)
    if v: out["done"].append([tag, int(seed), int(v[-1])])
for f in sorted(glob.glob("/content/fit_*_s*.log")) + sorted(glob.glob("/content/boot_*_s*.log")):
    t = open(f, errors="replace").read()
    if re.search(r"KILL|FIT_FAIL|EXPORT_FAIL|PREFLIGHT FAIL", t): out["fail"].append(f.split("/")[-1] + ": " + t.strip().splitlines()[-1][:120])
for f in glob.glob("/content/runs/*_s*.log"):
    ep = re.findall(r"(seed \d+ fold \d+ epoch \d+ loss [0-9.]+ +\d+s)", open(f, errors="replace").read())
    out["running"].append(f.split("/")[-1] + ": " + (ep[-1] if ep else "no epoch yet"))
qs = sorted(glob.glob("/content/queue_*.log")); out["queues_done"] = bool(qs) and all("QUEUE_DONE" in open(q, errors="replace").read() for q in qs)
out["gpu_procs"] = int(sh("nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l").strip() or 0)
print(json.dumps(out))
PY
while true; do
  ev=""
  # (2) member hand-offs, (3) failures
  running=0
  for S in $SESSIONS; do
    grep -qx "stopped:$S" autopilot.state && continue
    r=$(probe $S); log "$S $r"; echo "$r" | grep -q '^{' && echo "$r" > /tmp/lanef_ap_last_$S.json || rm -f /tmp/lanef_ap_last_$S.json
    echo "$r" | grep -q "appears to be lost\|not found\|Timeout" && { ev="$ev $S:unreachable"; continue; }
    python3 - "$S" "$r" <<'PY' > /tmp/lanef_ap_actions.txt 2>/dev/null
import json, sys, os
S, r = sys.argv[1], json.loads(sys.argv[2])
for tag, seed, ver in r["done"]:
    if not os.path.exists("/Users/ozgun/Dev/rsna-knee-abnormality-detection/outputs/lanef-20261020/members/%s-s%d/READY.json" % (tag, seed)):
        print("handoff", S, tag, seed, ver)
for f in r["fail"]: print("fail", S, f)
print("gpu", S, r["gpu_procs"])
pending = any(not os.path.exists("/Users/ozgun/Dev/rsna-knee-abnormality-detection/outputs/lanef-20261020/members/%s-s%d/READY.json" % (t, sd)) for t, sd, v in r["done"])
if r.get("queues_done") and r["gpu_procs"] == 0 and not pending: print("idle", S, 0)
PY
    while read kind a b c d; do
      case $kind in
        handoff) log "member hand-off $b-s$c from $a (version $d)"; ./member_export.sh $a $b $c $d > member_${b}_s$c.log 2>&1; log "hand-off result: $(tail -n 1 member_${b}_s$c.log | cut -c1-160)"; ev="$ev $b-s$c:handoff";;
        fail) grep -qx "$a:$b $c $d" autopilot.state || { echo "$a:$b $c $d" >> autopilot.state; ev="$ev $a:FAILURE"; };;
        gpu) [ "${b:-0}" -gt 0 ] && running=$((running + 1));;
        idle) ./c.sh stop -s $a > /dev/null 2>&1; pkill -f "refresher.sh $a"; echo "stopped:$a" >> autopilot.state; log "$a idle after its last export: stopped"; ev="$ev $a:stopped";;
      esac
    done < /tmp/lanef_ap_actions.txt
  done
  # (3b) unit guard (parent 11:35): a new seed costs about 20 units; keep >= 15 after all running fits finish
  if ! grep -q "^units_stop" autopilot.state; then
    bal=$(LB_COLAB_CONFIG=~/.config/colab-cli/sessions-lanef-20261020.json /Users/ozgun/.local/share/uv/tools/google-colab-cli/bin/python $ROOT/scripts/laneb/ccu_info.py 2>/dev/null | grep -o '"currentBalance": [0-9.]*' | awk '{print $2}')
    need=$(grep -h "running" /tmp/lanef_ap_last_*.json 2>/dev/null | python3 -c "
import sys, json, re
tot = 0.0
for line in sys.stdin:
    r = json.loads(line)
    es = [max([int(m) for m in re.findall(r'epoch (\d+) loss', x)] or [-1]) for x in r.get('running', [])]
    live = [e for e in es if e < 15]
    if r.get('gpu_procs', 0) > 0:
        e = min(live) if live else 15
        tot += 5.3 * ((15 - e) * 0.22 + 0.35)
print(round(tot, 1))" 2>/dev/null)
    log "unit guard: balance ${bal:-?} units, running fits still need about ${need:-?}, new seed 20, floor 15"
    if [ -n "$bal" ] && [ -n "$need" ] && python3 -c "import sys; sys.exit(0 if $bal - $need - 20 < 15 else 1)"; then
      for S in $SESSIONS; do grep -qx "stopped:$S" autopilot.state || printf 'open("/content/STOP_NEW_SEEDS","w").write("balance %s, running need %s")\nprint("guard set")\n' "$bal" "$need" > /tmp/lanef_guard.py && ./c.sh exec -s $S -f /tmp/lanef_guard.py --timeout 60 > /dev/null 2>&1; done
      echo "units_stop $(date -u +%FT%TZ) balance $bal need $need" >> autopilot.state; log "UNIT GUARD SET on all VMs"; ev="$ev units_guard_set"
    fi
  fi
  # (4) deadline
  now=$(date +%s)
  alive=0; for S in $SESSIONS; do grep -qx "stopped:$S" autopilot.state || alive=$((alive+1)); done
  [ $alive -eq 0 ] && { log "ALL_STOPPED"; echo "EVENT all_stopped"; exit 0; }
  if [ $now -ge $((DEADLINE + 23400)) ]; then
    for S in $SESSIONS; do ./c.sh stop -s $S 2>&1 | tail -1 | xargs -I{} echo "$(date -u +%FT%TZ) stop $S: {}" >> autopilot.log; pkill -f "refresher.sh $S"; done
    log "DEADLINE: VMs stopped (running fits at deadline: $running)"; echo "EVENT deadline_stop"; exit 0
  fi
  [ -n "$ev" ] && { log "EVENT$ev"; echo "EVENT$ev"; exit 0; }
  sleep ${CADENCE:-1500}
done
