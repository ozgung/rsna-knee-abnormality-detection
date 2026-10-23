#!/bin/bash
# Lane D2: P2's (57340072) teacher workers on a Colab A100, with P2's frozen payload byte-identical.
# 1. Python 3.12 venv /content/p2py with the package versions P2's workers pin (torch 2.10.0+cu128, timm 1.0.26, numpy 2.0.2,
#    pandas 2.3.3, pyarrow 24.0.0, pydicom 3.0.2, scipy 1.16.3, setuptools 81.0.0) + P2's three offline codec wheels.
# 2. P2's attached datasets at their Kaggle mount paths (/kaggle/input/datasets/<owner>/<slug>), so P2's configs apply unchanged.
# 3. The payload sources written exactly as P2's bootstrap writes them (/kaggle/working/r6flip-project/{knee,source}).
set -euxo pipefail
export KAGGLE_CONFIG_DIR=/content
K=/usr/local/bin/kaggle
mountpoint -q /kaggle/input && umount /kaggle/input || true      # Colab mounts an empty read-only /kaggle/input
[ -x /content/p2py/bin/python ] || uv venv --seed --python 3.12 /content/p2py
P=/content/p2py/bin/python
$P -m pip install -q --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0+cu128 torchvision==0.25.0+cu128
$P -m pip install -q numpy==2.0.2 pandas==2.3.3 pyarrow==24.0.0 pydicom==3.0.2 scipy==1.16.3 setuptools==81.0.0 \
   opencv-python-headless==4.13.0.92 pillow safetensors tqdm requests huggingface_hub pyyaml
$P -m pip install -q --no-deps timm==1.0.26
I=/kaggle/input/datasets
mkdir -p $I/heliosli/rsna-knee-r6-frozen-fe055406-w1 $I/nartaa/rsna-knee-publication-swa-weights-20261007 \
         $I/goodpjw2008/rsna-knee-2-5d-convnext-reader $I/heliosli/rsna-orthofoundation-fold0-epoch7-20261004 /kaggle/working
$K datasets download heliosli/rsna-knee-r6-frozen-fe055406-w1 -p $I/heliosli/rsna-knee-r6-frozen-fe055406-w1 --unzip -q
for f in raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt; do
  $K datasets download nartaa/rsna-knee-publication-swa-weights-20261007 -f $f -p $I/nartaa/rsna-knee-publication-swa-weights-20261007 -q
done
for f in pylibjpeg-2.1.0-py3-none-any.whl pylibjpeg_libjpeg-2.4.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl \
         pylibjpeg_openjpeg-2.6.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl; do
  $K datasets download goodpjw2008/rsna-knee-2-5d-convnext-reader -f $f -p $I/goodpjw2008/rsna-knee-2-5d-convnext-reader -q
done
cp /content/helios/anatomy.pt $I/heliosli/rsna-orthofoundation-fold0-epoch7-20261004/anatomy.pt
cd $I && find . -name "*.zip" -exec sh -c 'cd "$(dirname "$1")" && unzip -o -q "$(basename "$1")" && rm "$(basename "$1")"' _ {} \;
$P -I /content/p2_bootstrap.py
$P -m pip install --no-index --no-deps $I/goodpjw2008/rsna-knee-2-5d-convnext-reader/*.whl
$P -m pip freeze > /content/p2-pip-freeze.txt
echo P2_SETUP_DONE
