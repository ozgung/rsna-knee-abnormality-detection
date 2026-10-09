"""Offline parent: verified codec setup and one fresh bounded MaViT CUDA process."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from knee.fresh_reader_setup import dataset_root,exact_root
from knee.mavit_checkpoint import DATASET,CHECKPOINT,SOURCE_HASHES,digest
from knee.reference_blend import CONVNEXT_DATASET,select_codec_wheels


def run_process(command,deadline):
    if deadline<=time.monotonic():raise TimeoutError('MaViT whole-run deadline exhausted')
    process=subprocess.Popen(command,start_new_session=True)
    try:
        if process.wait(timeout=max(.001,deadline-time.monotonic())):raise RuntimeError('MaViT child process failed')
    finally:
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        process.wait()


if __name__=='__main__':
    deadline=MAVIT_STARTED+32400
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',PYTHONPATH=str(project),PYTHONUNBUFFERED='1')
    try:
        for name,sha in MAVIT_CONTRACT['runtime_sources'].items():
            if digest((project/name).read_bytes())!=sha:raise ValueError('MaViT runtime source changed')
        assets=dataset_root('/kaggle/input',DATASET,[CHECKPOINT,*SOURCE_HASHES])
        for name,sha in SOURCE_HASHES.items():
            if digest((assets/name).read_bytes())!=sha:raise ValueError('MaViT mounted public source changed')
        codecs=dataset_root('/kaggle/input',CONVNEXT_DATASET,['pylibjpeg-2.1.0-py3-none-any.whl']);wheels=select_codec_wheels(codecs);hashes={}
        for path in wheels:
            with path.open('rb') as h:hashes[path.name]=hashlib.file_digest(h,'sha256').hexdigest()
        if hashes!=MAVIT_CONTRACT['codecs']:raise ValueError('Offline codec hashes changed')
        run_process([sys.executable,'-m','pip','install','--no-index','--no-deps',*map(str,wheels)],deadline)
        root=exact_root([Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),Path('/kaggle/input/rsna-knee-abnormality-detection')],
                        ['test.csv','test_series.csv','train_series.csv','sample_submission.csv'])
        payload={'contract':MAVIT_CONTRACT,'project':str(project),'root':str(root),'assets':str(assets),'output':'/kaggle/working/mavit-rehearsal'}
        path=Path('/kaggle/working/mavit-runtime-input.json');path.write_text(json.dumps(payload,allow_nan=False))
        run_process([sys.executable,'-m','knee.mavit_rehearsal','--payload',str(path),'--deadline',str(deadline)],deadline)
        out=Path(payload['output'])/'mavit-audit.json';r=json.loads(out.read_text());r['elapsed_seconds']=time.monotonic()-MAVIT_STARTED;r['parent_complete']=True
        from knee.mavit_rehearsal import require_rehearsal
        require_rehearsal(MAVIT_CONTRACT,r);out.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'event':'mavit_rehearsal_done','status':'PASS','studies':67,'elapsed_seconds':r['elapsed_seconds']}),flush=True)
    except BaseException as error:
        Path('/kaggle/working/mavit-parent-failure.json').write_text(json.dumps({'status':'FAIL','failure_type':type(error).__name__,
                                                                           'elapsed_seconds':time.monotonic()-MAVIT_STARTED})+'\n')
        raise RuntimeError('MaViT rehearsal failed; aggregate receipt only') from None
    finally:signal.setitimer(signal.ITIMER_REAL,0)
