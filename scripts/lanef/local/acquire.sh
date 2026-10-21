#!/bin/bash
# Lane F: obtain the remaining A100 slots and launch their jobs.  Tries every 5 min; us-central1 only for the first 30 min, then any
# region; gives up at 21:00 UTC.  Slot jobs: lanef14 = v5-s43 (resumed) + 46 49; lanef15 = v4b-s45 (resumed), then v5 44 47 50.
cd "$(dirname "$0")"; t0=$(date +%s); END=$(date -u -j -f "%Y-%m-%d %H:%M" "2026-10-22 01:30" +%s)
launch() {
  case $1 in
    lanef16|lanef19|lanef21) RESUME_SPEC=v5:42 ./upload_and_queue.sh $1 v5 /content/targets/targets_V5_all.csv none 42 45 48 > uq_$1.log 2>&1;;
    lanef17|lanef20|lanef23) RESUME_SPEC=v5:43 ./upload_and_queue.sh $1 v5 /content/targets/targets_V5_all.csv none 43 46 49 > uq_$1.log 2>&1;;
    lanef18|lanef22|lanef24) RESUME_SPEC=v4b:45 ./upload_and_queue.sh $1 v4b /content/targets/targets_V4B_all.csv none 45 > uq_$1.log 2>&1
             printf 'import subprocess\nsubprocess.Popen(["bash","-c","DEADLINE_EPOCH=1792648800 bash /content/v5_seed_queue.sh v5 /content/targets/targets_V5_all.csv v4b:45 44 47 50"], stdout=open("/content/queue_v5.log","wb"), stderr=subprocess.STDOUT, start_new_session=True)\nprint("v5 44 47 50 queued after v4b-s45")\n' > /tmp/lanef_q18.py
             ./c.sh exec -s $1 -f /tmp/lanef_q18.py --timeout 60 >> uq_$1.log 2>&1;;
    lanef14) RESUME_SPEC=v5:43 ./upload_and_queue.sh lanef14 v5 /content/targets/targets_V5_all.csv none 43 46 49 > uq_lanef14.log 2>&1;;
    lanef15) RESUME_SPEC=v4b:45 ./upload_and_queue.sh lanef15 v4b /content/targets/targets_V4B_all.csv none 45 > uq_lanef15.log 2>&1
             printf 'import subprocess\nsubprocess.Popen(["bash","-c","DEADLINE_EPOCH=1792648800 bash /content/v5_seed_queue.sh v5 /content/targets/targets_V5_all.csv v4b:45 44 47 50"], stdout=open("/content/queue_v5.log","wb"), stderr=subprocess.STDOUT, start_new_session=True)\nprint("v5 44 47 50 queued after v4b-s45")\n' > /tmp/lanef_q15.py
             ./c.sh exec -s lanef15 -f /tmp/lanef_q15.py --timeout 60 >> uq_lanef15.log 2>&1;;
  esac
  echo $1 >> sessions.txt; echo "$(date -u +%T) launched $1: $(grep LAUNCHED uq_$1.log | tail -1)"
}
for N in ${@:-lanef14 lanef15}; do
  while true; do
    [ $(date +%s) -ge $END ] && { echo "$(date -u +%T) GIVE_UP $N"; exit 1; }
    G=$(echo "${GPUS:-}" | tr ',' '\n' | grep "^$N=" | cut -d= -f2); ./c.sh new -s $N --gpu ${G:-A100} > new_$N.log 2>&1
    ep=$(./c.sh sessions 2>&1 | grep "^\[$N\]" | awk '{print $2}')
    if [ -n "$ep" ]; then
      if [[ $ep == *-usc1* ]] || [ $(( $(date +%s) - t0 )) -ge ${USC_ONLY_S:-1800} ]; then
        echo "$(date -u +%T) $N OK $ep"; (nohup ./refresher.sh $N $ep > /dev/null 2>&1 &); launch $N; break
      fi
      echo "$(date -u +%T) $N got $ep (not usc1, first 30 min): stopping"; ./c.sh stop -s $N > /dev/null 2>&1
    else
      echo "$(date -u +%T) $N none: $(grep -o 'Service Unavailable\|TooMany[A-Za-z]*\|Precondition Failed' new_$N.log | head -1)"
    fi
    sleep 300
  done
done
echo "$(date -u +%T) ACQUIRE_DONE"
