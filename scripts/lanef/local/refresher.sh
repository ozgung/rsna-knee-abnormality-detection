#!/bin/bash
# Every 30 min: refresh the runtime proxy token + keep-alive of an existing lane F runtime (no allocation). usage: refresher.sh <session> <endpoint>; log refresher_<session>.log
cd "$(dirname "$0")"
while true; do
  echo "$(date -u +%FT%TZ) $(~/.local/share/uv/tools/google-colab-cli/bin/python tools/reconnect.py $1 $2 2>&1 | tail -1 | cut -c1-200)" >> refresher_$1.log
  sleep 1800
done
