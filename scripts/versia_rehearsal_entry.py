"""Private offline entry appended by the Versia component builder."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from knee.fresh_reader_setup import dataset_root, exact_root
from knee.reference import canonical_bytes, sha256
from knee.reference_blend import CONVNEXT_DATASET, select_codec_wheels
from knee.versia import RECIPE, SOURCE_HASHES, SPEC_SHA256, WEIGHTS, policy_contract, rehearse

if (sys.version_info[:2]!=(3,12) or importlib.metadata.version('torch')!='2.10.0+cu128' or
        importlib.metadata.version('timm')!='1.0.26'):
    raise ValueError('Versia requires the pinned Python3.12/Torch/timm environment')
if (VERSIA_CONTRACT['recipe']!=RECIPE or VERSIA_CONTRACT['policy']!=policy_contract() or
        VERSIA_CONTRACT['sources']!=SOURCE_HASHES or VERSIA_CONTRACT['weights']!=WEIGHTS or
        VERSIA_CONTRACT['spec_sha256']!=SPEC_SHA256):
    raise ValueError('Versia frozen component contract differs')
for name,digest in VERSIA_CONTRACT['runtime_sources'].items():
    if sha256((Path('/kaggle/working/project')/name).read_bytes())!=digest:
        raise ValueError('Versia runtime source changed')
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
assets=dataset_root('/kaggle/input',CONVNEXT_DATASET,list(WEIGHTS))
wheels=select_codec_wheels(assets);hashes={}
for path in wheels:
    with path.open('rb') as handle:hashes[path.name]=hashlib.file_digest(handle,'sha256').hexdigest()
if hashes!=VERSIA_CONTRACT['codecs']:raise ValueError('Versia offline codecs changed')
subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps',*map(str,wheels)],check=True)
root=exact_root([Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),Path('/kaggle/input/rsna-knee-abnormality-detection')],
                ['test.csv','test_series.csv','train_series.csv','sample_submission.csv'])
report=rehearse(root,assets,VERSIA_SOURCES,VERSIA_CONTRACT,'/kaggle/working/versia-rehearsal',
                started=VERSIA_STARTED,bundle_sha256=os.environ['KNEE_SOURCE_SHA256'])
signal.setitimer(signal.ITIMER_REAL,0)
print(json.dumps({'event':'versia_rehearsal_done','status':report['status'],'studies':report['studies'],
                  'elapsed_seconds':report['elapsed_seconds']}),flush=True)
