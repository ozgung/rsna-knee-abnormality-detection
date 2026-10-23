#!/bin/bash
# Lane F: create one A100 session named <name> in us-central1 (re-roll up to 6 times; non-usc1 runtimes are stopped at once).
cd "$(dirname "$0")"; N=$1
for i in 1 2 3 4 5 6; do
  ./c.sh new -s $N --gpu A100 > new_$N.log 2>&1
  ep=$(./c.sh sessions 2>&1 | grep "^\[$N\]" | awk '{print $2}')
  [ -z "$ep" ] && { echo "$(date -u +%T) $N attempt $i: no session ($(grep -o 'TooMany[A-Za-z]*\|Precondition Failed' new_$N.log | head -1))"; sleep 20; continue; }
  case $ep in *-usc1*) echo "$(date -u +%T) $N OK $ep"; (nohup ./refresher.sh $N $ep > /dev/null 2>&1 &); exit 0;; esac
  echo "$(date -u +%T) $N attempt $i: $ep not us-central1, stopping"; ./c.sh stop -s $N > /dev/null 2>&1; sleep 10
done
echo "$(date -u +%T) $N FAILED to get a us-central1 A100"; exit 1
