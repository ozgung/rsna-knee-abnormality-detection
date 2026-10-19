#!/bin/bash
# Lane D: stream the fastMRI knee DICOM batch-2 archive from its signed link straight into an extracted tree
# (curl -> sha256 tee -> xz -> tar), so the 30.6 GB archive never sits on disk.  The signed links live only in
# /content/.fm_links (mode 600, uploaded from the user's private config); no link is ever echoed or logged.
# Usage: bash vm_fm_fetch.sh <line number of the archive in the links file> <line number of the SHA256 file> <out dir>
set -euo pipefail
LN=$1; LS=$2; OUT=$3
mkdir -p "$OUT" /content/fm_meta
umask 077
url_cfg() {  # write a curl config holding the URL of line $1 (never on argv)
  python3 -I - "$1" "$2" <<'PY'
import re, sys
lines = [l for l in open("/content/.fm_links") if l.strip()]
u = re.search(r'https?://[^"\s]+', lines[int(sys.argv[1]) - 1]).group(0)
open(sys.argv[2], "w").write('url = "%s"\n' % u)
PY
}
url_cfg "$LS" /content/.fm_sha.cfg
curl -sS --fail --retry 5 -K /content/.fm_sha.cfg -o /content/fm_meta/SHA256
url_cfg "$LN" /content/.fm_arc.cfg
echo "$(date -u +%FT%TZ) FETCH_START"
curl -sS --fail --retry 5 -K /content/.fm_arc.cfg \
  | tee >(sha256sum | cut -d' ' -f1 > /content/fm_meta/archive.sha256) \
  | tee >(wc -c > /content/fm_meta/archive.bytes) \
  | xz -dc -T0 | tar -x -C "$OUT"
sleep 5
rm -f /content/.fm_arc.cfg /content/.fm_sha.cfg
echo "$(date -u +%FT%TZ) FETCH_DONE bytes $(cat /content/fm_meta/archive.bytes) sha $(cat /content/fm_meta/archive.sha256)"
