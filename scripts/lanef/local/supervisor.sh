#!/bin/bash
# Lane F unattended supervisor (Oct 21): keeps three slots alive until the round-3 seeds are exported, then stops; hard stop 06:30.
# Slot A: v5 42 45 48; slot B: v5 43 46 49; slot C: v4b 45, then v5 44 47 50.  Every 10 min: (1) a single autopilot pass
# (hand-offs, failures, idle stop); (2) for each slot without a live assignment, if seeds remain and a fit can still finish by
# 06:00 (no new session after 01:50), request a session (rotating A100, G4, H100; any region), resume its first unfinished seed
# from the latest backup and queue the rest.  Log: supervisor.log; slot state: slots.state (slot session endpoint).
cd "$(dirname "$0")"; ROOT=/Users/ozgun/Dev/rsna-knee-abnormality-detection; M=$ROOT/outputs/lanef-20261020/members
P=/Users/ozgun/.local/share/uv/tools/google-colab-cli/bin/python
HARD=$(date -u -j -f "%Y-%m-%d %H:%M" "2026-10-22 06:30" +%s); LASTNEW=$(date -u -j -f "%Y-%m-%d %H:%M" "2026-10-22 01:50" +%s)
log() { echo "$(date -u +%FT%TZ) $*" >> supervisor.log; }
touch slots.state; n=30; gi=0; GPUS=(A100 G4 H100)
assigned() { $P - <<'PY' 2>/dev/null
import os
from colab_cli.common import state
state.config_path = os.path.expanduser("~/.config/colab-cli/sessions-lanef-20261020.json")
print(" ".join(a.endpoint for a in state.client.list_assignments()))
PY
}
remaining() { for x in "$@"; do [ -f $M/$x/READY.json ] || echo -n "$x "; done; }
while true; do
  now=$(date +%s)
  if [ $now -ge $HARD ]; then
    for S in $(awk '{print $2}' slots.state); do ./c.sh stop -s $S > /dev/null 2>&1; pkill -f "refresher.sh $S"; done
    log "HARD STOP 06:30: all sessions stopped"; exit 0
  fi
  awk '{print $2}' slots.state | sort -u > sessions.txt
  [ -s sessions.txt ] && ONESHOT=1 ./autopilot.sh >> autopilot_run.log 2>&1
  live=$(assigned); [ -z "$live" ] && live="(none)"
  for SLOT in A B C; do
    case $SLOT in A) R=$(remaining v5-s42 v5-s45 v5-s48);; B) R=$(remaining v5-s43 v5-s46 v5-s49);; C) R=$(remaining v4b-s45 v5-s44 v5-s47 v5-s50);; esac
    cur=$(grep "^$SLOT " slots.state | tail -1); ep=$(echo "$cur" | awk '{print $3}'); S=$(echo "$cur" | awk '{print $2}')
    if [ -n "$ep" ] && echo "$live" | grep -q "$ep"; then continue; fi
    if [ -n "$S" ] && grep -qx "stopped:$S" autopilot.state; then [ -z "$R" ] && continue; fi
    [ -z "$R" ] && { grep -q "^$SLOT done" slots.state || { echo "$SLOT done" >> slots.state; log "slot $SLOT: all seeds handed off"; }; continue; }
    [ -n "$S" ] && { pkill -f "refresher.sh $S"; sed -i '' "/^$SLOT /d" slots.state; log "slot $SLOT: session $S gone"; }
    [ $now -ge $LASTNEW ] && { log "slot $SLOT: past 01:50, no new session (remaining: $R)"; continue; }
    n=$((n+1)); N=lanef$n; G=${GPUS[$((gi % 3))]}; gi=$((gi+1))
    ./c.sh new -s $N --gpu $G > new_$N.log 2>&1
    ep=$(./c.sh sessions 2>&1 | grep "^\[$N\]" | awk '{print $2}')
    [ -z "$ep" ] && { log "slot $SLOT: $G request failed ($(grep -o 'TooMany[A-Za-z]*\|Service Unavailable\|Precondition Failed' new_$N.log | head -1))"; continue; }
    echo "$SLOT $N $ep $G" >> slots.state; (nohup ./refresher.sh $N $ep > /dev/null 2>&1 &); log "slot $SLOT: $N $G $ep, remaining: $R"
    first=$(echo $R | awk '{print $1}'); tag=${first%%-s*}; seed=${first##*-s}
    if [ "$SLOT" = C ] && [ "$first" = v4b-s45 ]; then
      RESUME_SPEC=v4b:45 ./upload_and_queue.sh $N v4b /content/targets/targets_V4B_all.csv none 45 > uq_$N.log 2>&1
      rest=$(echo $R | cut -d' ' -f2- | sed 's/v5-s//g')
      printf 'import subprocess\nsubprocess.Popen(["bash","-c","DEADLINE_EPOCH=1792648800 bash /content/v5_seed_queue.sh v5 /content/targets/targets_V5_all.csv v4b:45 %s"], stdout=open("/content/queue_v5.log","wb"), stderr=subprocess.STDOUT, start_new_session=True)\nprint("queued")\n' "$rest" > /tmp/lanef_sq_$N.py
      ./c.sh exec -s $N -f /tmp/lanef_sq_$N.py --timeout 60 >> uq_$N.log 2>&1
    else
      seeds=$(echo $R | sed 's/v5-s//g')
      RESUME_SPEC=v5:$seed ./upload_and_queue.sh $N v5 /content/targets/targets_V5_all.csv none $seeds > uq_$N.log 2>&1
    fi
    log "slot $SLOT: launched on $N: $(grep LAUNCHED uq_$N.log | tail -1)"
  done
  sleep 600
done
