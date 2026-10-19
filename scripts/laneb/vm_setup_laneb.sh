#!/bin/bash
# Lane B VM setup: the s6v2 environment (python 3.12 venv, torch 2.8.0+cu128, heliosli code + anatomy, nartaa extras overlay),
# the pinned R6 sources in /content/r6src, the R6 and R7 bundles from their public datasets (sha-checked by the driver),
# and the data root (train.csv UID column only, train_series.csv).  Log: /content/setup_laneb.log
set -euxo pipefail
cd /content
export KAGGLE_CONFIG_DIR=/content
chmod 600 /content/kaggle.json
/usr/local/bin/python -m pip install -q uv
uv python install 3.12
uv venv --seed --python 3.12 /content/py
P=/content/py/bin/python
$P -m pip install -q --upgrade pip
$P -m pip install -q --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0+cu128 torchvision==0.23.0+cu128
$P -m pip install -q numpy==2.0.2 pandas==2.3.3 pydicom==3.0.1 scipy==1.16.3 setuptools==80.9.0 opencv-python-headless==4.12.0.88 safetensors requests kaggle==1.7.4.5 scikit-learn huggingface_hub
$P -m pip install -q --no-deps timm==1.0.20
$P -m pip install -q zstandard || true
mkdir -p /content/helios /content/code /content/r6 /content/r7 /content/r6src /content/data
/usr/local/bin/kaggle datasets download heliosli/rsna-orthofoundation-fold0-epoch7-20261004 -p /content/helios --unzip -q
/usr/local/bin/kaggle datasets download nartaa/rsna-knee-s6fall-weights -f s6_extras.bin -p /content/helios -q
/usr/local/bin/kaggle datasets download nartaa/rsna-knee-s6fall-weights -f code_manifest.json -p /content/helios -q
/usr/local/bin/kaggle datasets download heliosli/rsna-knee-r6-frozen-fe055406-w1 -p /content/r6 --unzip -q
/usr/local/bin/kaggle datasets download heliosli/rsna-knee-compact-r7cnxfsss -p /content/r7 --unzip -q
$P - <<'PY'
import tarfile, io, json, hashlib, shutil
from pathlib import Path
H = Path('/content/helios'); C = Path('/content/code')
b = (H / 's6_extras.bin').read_bytes(); assert b[:11] == b'K1PAYLOAD1\n'
tf = tarfile.open(fileobj=io.BytesIO(b[11:]))
shutil.copytree(H / 'cloud_code', C / 'cloud_code', dirs_exist_ok=True)
for m in tf.getmembers():
    if m.isfile():
        dst = C / m.name; dst.parent.mkdir(parents=True, exist_ok=True); dst.write_bytes(tf.extractfile(m).read())
man = json.load(open(H / 'code_manifest.json'))['members']
bad, ok = [], 0
for grp, files in man.items():
    for name, rec in files.items():
        p = C / grp / name if grp == 'heliosli' else C / 'cloud_code' / name if grp == 'cloud_code' else C / grp / name
        if not p.exists():
            bad.append(('missing', grp, name)); continue
        if hashlib.sha256(p.read_bytes()).hexdigest() != rec['sha256']:
            bad.append(('sha', grp, name))
        else: ok += 1
print('code manifest ok', ok, 'bad', len(bad), bad[:10])
print('anatomy sha', hashlib.sha256((H / 'anatomy.pt').read_bytes()).hexdigest())
PY
tar -xf /content/r6src.tar -C /content/r6src
cp /content/train_uid_only.csv /content/data/train.csv
cp /content/train_series.csv /content/data/train_series.csv
ls -la /content/r6 /content/r7 /content/r6src
$P -m pip freeze > /content/pip-freeze.txt
echo SETUP_DONE
