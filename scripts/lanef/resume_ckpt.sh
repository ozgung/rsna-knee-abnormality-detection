#!/bin/bash
# Lane F: place the latest per-epoch backup of <tag> s<seed> where the trainer resumes (per-file Kaggle Model route, sha-verified).
# No backup -> exit 0 with NO_BACKUP (the fit starts fresh).  usage: resume_ckpt.sh <tag> <seed>
set -uo pipefail
TAG=$1; SEED=$2; N=E_prior_s${SEED}_f0; VAR=$TAG-s$SEED; REF=ozgung/knee-s6-lanef-ckpt-20261020/pyTorch/$VAR
export KAGGLE_CONFIG_DIR=/content
v=$(/usr/local/bin/kaggle models instances versions list $REF --page-size 100 2>/dev/null | awk -v x="$VAR" '$2==x {print $1}' | sort -n | tail -1)
[ -z "$v" ] && { echo "NO_BACKUP $VAR"; exit 0; }
AUTH=$(/usr/bin/python3 -c "import json;c=json.load(open('/content/kaggle.json'));print(c['username']+':'+c['key'])")
D=/content/runs/${TAG}_s${SEED}/ckpt; T=/content/rs_${VAR}; mkdir -p $D $T
curl -sL -u "$AUTH" -o $T/SHA256SUMS https://www.kaggle.com/api/v1/models/$REF/$v/download/SHA256SUMS
curl -sL -u "$AUTH" -o $T/ckpt__$N.pt https://www.kaggle.com/api/v1/models/$REF/$v/download/ckpt__$N.pt
(cd $T && grep "ckpt__$N.pt" SHA256SUMS | sha256sum -c - > /dev/null) && mv $T/ckpt__$N.pt $D/$N.pt && echo "RESUME_CKPT_OK $VAR version $v $(grep -o 'epoch [0-9]*' $T/SHA256SUMS)" || { echo "RESUME_CKPT_FAIL $VAR"; exit 1; }
