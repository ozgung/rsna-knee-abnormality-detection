#!/bin/bash
# Lane D2 (VM): after the batch-2 fetch: hash check against the official list, headers, the 2,141-exam layouts
# (/content/fmdata for the cache builder, /content/fmteach/part0 = teacher root), then the fastMRI pixel cache build.
set -euo pipefail
until grep -q FETCH_DONE /content/fetch.log; do sleep 20; done
h=$(cat /content/fm_meta/archive.sha256); grep -q "$h" /content/fm_meta/SHA256 && echo "ARCHIVE_SHA_MATCH" || { echo ARCHIVE_SHA_MISMATCH; exit 1; }
/content/py/bin/python -I /content/vm_fm_headers.py /content/fmraw
cp /content/fm_train_series_all.csv /content/fm_meta/fm_train_series.csv
/content/py/bin/python -I /content/vm_fm_layout.py /content/fm_meta/fm_train_series.csv 1 knee-laned-d2-unused
/content/py/bin/python -I -c "import csv,json; json.dump([r['StudyInstanceUID'] for r in csv.DictReader(open('/content/fmdata/train.csv'))], open('/content/fm_order.json','w'))"
echo PREP_DONE
E="S6V2_DATA=/content/fmdata S6V2_OUT=/content/pxfm S6V2_ORDER=/content/fm_order.json S6V2_TMP=/content/tmp_pxfm S6V2_WORKERS=6 S6V2_STAGERS=2 S6V2_AHEAD=40"
cd /content && env $E /content/py/bin/python -I /content/vm_build_cache_local.py > /content/buildfm.log 2>&1 && env $E /content/py/bin/python -I /content/vm_build_cache.py finalize >> /content/buildfm.log 2>&1
echo BUILD_EXIT $?
