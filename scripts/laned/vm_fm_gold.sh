#!/bin/bash
# Lane D: gold-58 read-out of the exported student through the pinned DICOM -> compact_infer path (vm_gold_infer.py:
# v4, 320/288, 3 slices, 3.5 mm, eval stride 1, bf16), the same path V2-all / S6fall / Lane C seeds were read with.
# The checkpoint is staged under the loader alias E_prior_s42_f0.pt.  Read once; recorded as exposed.
set -euo pipefail
cd /content
[ -d /content/data_gold/train_series ] || /content/py/bin/python -I /content/vm_gold_infer.py stage > /content/gold_stage2.log 2>&1
M=/content/gold_models_fm; rm -rf $M; mkdir -p $M
cp /content/export_fm/E_prior_s42_fmall.pt $M/E_prior_s42_f0.pt
O=/content/gold_fm; rm -rf $O; mkdir -p $O
/content/py/bin/python -I /content/vm_gold_infer.py score $M $O bf16 > /content/gold_fm.log 2>&1
ls $O
echo GOLD_DONE
