#!/bin/bash
# Every 30 min: refresh the runtime proxy token + keep-alive of the existing Lane B runtime (no allocation).
# usage: refresher_laneb.sh <session> <endpoint> <log>
D="$(cd "$(dirname "$0")" && pwd)"
while true; do
  echo "$(date -u +%FT%TZ) $(~/.local/share/uv/tools/google-colab-cli/bin/python "$D/reconnect_laneb.py" "$1" "$2" 2>&1 | tail -1 | cut -c1-200)" >> "$3"
  sleep 1800
done
