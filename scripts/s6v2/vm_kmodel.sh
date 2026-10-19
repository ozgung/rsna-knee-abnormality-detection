#!/bin/bash
# Lane C: publish a folder as a private Kaggle Model variation (instance) or a new version of it, then CONFIRM by listing the
# remote files (the kaggle CLI can exit 0 on quota/permission errors).  usage: vm_kmodel.sh <model slug> <variation> <dir> <must-have file> [notes]
# Prints the remote version number on success (exit 0); exit 1 otherwise.  Private dataset quota is exhausted; Models are not.
export KAGGLE_CONFIG_DIR=${KAGGLE_CONFIG_DIR:-/content}
K=${KAGGLE_BIN:-/usr/local/bin/kaggle}
MODEL=$1; VAR=$2; D=$3; MUST=$4; NOTES=${5:-update}
REF=ozgung/$MODEL/pyTorch/$VAR
size=$(stat -c %s "$D/$MUST" 2>/dev/null || stat -f %z "$D/$MUST")
before=$($K models instances versions list $REF 2>/dev/null | awk -v var="$VAR" '$1 ~ /^[0-9]+$/ && $2 == var {print $1}' | sort -n | tail -1)
if [ -z "$before" ]; then
  printf '{"ownerSlug": "ozgung", "modelSlug": "%s", "instanceSlug": "%s", "framework": "pyTorch", "overview": "private lane C artifact", "usage": "", "licenseName": "Apache 2.0", "fineTunable": false, "trainingData": [], "modelInstanceType": "Unspecified", "baseModelInstanceId": 0, "externalBaseModelUrl": ""}\n' "$MODEL" "$VAR" > "$D/model-instance-metadata.json"
  $K models instances create -p "$D" -q -r skip > "/tmp/kmodel_$VAR.log" 2>&1
else
  $K models instances versions create $REF -p "$D" -n "$NOTES" -q -r skip > "/tmp/kmodel_$VAR.log" 2>&1
fi
for i in 1 2 3 4 5 6; do
  v=$($K models instances versions list $REF 2>/dev/null | awk -v var="$VAR" '$1 ~ /^[0-9]+$/ && $2 == var {print $1}' | sort -n | tail -1)
  if [ -n "$v" ] && [ "$v" != "$before" ]; then
    got=$($K models instances versions files $REF/$v 2>/dev/null | awk -v f="$MUST" '$1 == f {print $2}')
    if [ "$got" = "$size" ]; then echo $v; exit 0; fi
  fi
  sleep 30
done
echo "PUBLISH_UNCONFIRMED $(tail -c 300 /tmp/kmodel_$VAR.log | tr '\n' ' ')" >&2; exit 1
