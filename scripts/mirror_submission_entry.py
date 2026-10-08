"""Appended to the archived scientific bundle by build_mirror_submission."""
import time
import os
import signal
from pathlib import Path
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys

from knee.fresh_reader_setup import dataset_root, exact_root
from knee.reference import canonical_bytes, sha256
from knee.reference_blend import CONVNEXT_DATASET, select_codec_wheels
from knee.mirror_submission import MAX_SECONDS, run


def expired(signum, frame):
    raise TimeoutError('Actual nine-hour mirror runtime exhausted')

signal.signal(signal.SIGALRM, expired)
remaining = MAX_SECONDS-(time.monotonic()-MIRROR_STARTED)
if remaining <= 0: raise TimeoutError('Mirror bootstrap exhausted runtime')
signal.setitimer(signal.ITIMER_REAL, remaining)
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
q = MIRROR_CONTRACT['qualification']
if sha256(canonical_bytes(q)) != MIRROR_CONTRACT['qualification_sha256']:
    raise ValueError('Mirror qualification fingerprint differs')
if sys.version_info[:2] != (3,12) or any(importlib.metadata.version(p) != q['environment'][p] for p in ('torch','timm')):
    raise ValueError('Mirror qualified Python/Torch/timm environment changed')
project = Path('/kaggle/working/project')
for name, digest in q['runtime_sources'].items():
    if sha256((project/name).read_bytes()) != digest:
        raise ValueError('Mirror scientific/runtime source integrity differs')
weights = dataset_root('/kaggle/input', q['manifest']['weight_dataset'], [q['manifest']['checkpoint_file']])
publisher = q['manifest']['publisher_manifest']
if sha256((weights/publisher['filename']).read_bytes()) != publisher['sha256']:
    raise ValueError('Mirror publisher manifest differs')
codecs = dataset_root('/kaggle/input', CONVNEXT_DATASET, ['pylibjpeg-2.1.0-py3-none-any.whl'])
wheels = select_codec_wheels(codecs)
with_hash = {}
for path in wheels:
    with path.open('rb') as handle:
        with_hash[path.name] = hashlib.file_digest(handle, 'sha256').hexdigest()
if with_hash != q['codec_hashes']:
    raise ValueError('Mirror offline codec hash differs')
subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps',*map(str,wheels)], check=True)
root = exact_root([Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),
                   Path('/kaggle/input/rsna-knee-abnormality-detection')],
                  ['test.csv','test_series.csv','sample_submission.csv'])
out = Path('/kaggle/working/mirror-submission')
report = run(root, weights/q['manifest']['checkpoint_file'], PUBLIC_SOURCE, DECODER_SOURCE,
             MIRROR_CONTRACT, out, started=MIRROR_STARTED)
if not MIRROR_CONTRACT['visible']:
    target = Path('/kaggle/working/submission.csv')
    with (out/'submission.csv').open('rb') as src, target.open('xb') as dst:
        shutil.copyfileobj(src,dst)
report['codec_hashes'] = with_hash
report['elapsed_seconds'] = time.monotonic()-MIRROR_STARTED
(out/'mirror-audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
signal.setitimer(signal.ITIMER_REAL,0)
print(json.dumps({'event':'mirror_entry_done','status':report['status'],'visible':report['visible'],
                  'test_studies':report['test_studies'],'elapsed_seconds':report['elapsed_seconds']}),flush=True)
