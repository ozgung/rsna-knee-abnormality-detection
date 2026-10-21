#!/bin/bash
# Lane F: put the epoch-3 backup of v4b seed 45 where the trainer resumes from (per-file Kaggle Model route; sha-verified).
set -uo pipefail
R=ozgung/knee-s6-lanef-ckpt-20261020/pyTorch/v4b-s45/4; N=E_prior_s45_f0
AUTH=$(/usr/bin/python3 -c "import json;c=json.load(open('/content/kaggle.json'));print(c['username']+':'+c['key'])")
D=/content/runs/v4b_s45/ckpt; mkdir -p $D /content/rs45
curl -sL -u "$AUTH" -o /content/rs45/SHA256SUMS https://www.kaggle.com/api/v1/models/$R/download/SHA256SUMS
curl -sL -u "$AUTH" -o /content/rs45/ckpt__$N.pt https://www.kaggle.com/api/v1/models/$R/download/ckpt__$N.pt
(cd /content/rs45 && grep "ckpt__$N.pt" SHA256SUMS | sha256sum -c -) && mv /content/rs45/ckpt__$N.pt $D/$N.pt && echo "RESUME_CKPT_OK $(grep -o 'epoch [0-9]*' /content/rs45/SHA256SUMS)" || { echo "RESUME_CKPT_FAIL"; exit 1; }
