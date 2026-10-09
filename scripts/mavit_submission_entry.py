"""Offline parent setup/cleanup for the unchanged batch1 MaViT inference process."""
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
from knee.fresh_reader_setup import dataset_root,exact_root
from knee import mavit_checkpoint as cp
from knee.mavit_submission import qualification,finish,MAX_SECONDS
from knee.reference_blend import CONVNEXT_DATASET,select_codec_wheels
from scripts.rehearse_mavit_entry import run_process


if __name__=='__main__':
    deadline=MAVIT_STARTED+MAX_SECONDS;owned_outputs=set()
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',PYTHONPATH=str(project),PYTHONUNBUFFERED='1')
    try:
        q=qualification(MAVIT_CONTRACT)
        for name,sha in q['runtime_sources'].items():
            if cp.digest((project/name).read_bytes())!=sha:raise ValueError('MaViT qualified runtime changed')
        assets=dataset_root('/kaggle/input',cp.DATASET,[cp.CHECKPOINT,*cp.SOURCE_HASHES])
        for name,sha in cp.SOURCE_HASHES.items():
            if cp.digest((assets/name).read_bytes())!=sha:raise ValueError('MaViT mounted source changed')
        codecs=dataset_root('/kaggle/input',CONVNEXT_DATASET,['pylibjpeg-2.1.0-py3-none-any.whl']);wheels=select_codec_wheels(codecs);hashes={}
        for path in wheels:
            with path.open('rb') as handle:hashes[path.name]=hashlib.file_digest(handle,'sha256').hexdigest()
        if hashes!=q['rehearsal_contract']['codecs']:raise ValueError('Qualified codec hashes differ')
        run_process([sys.executable,'-m','pip','install','--no-index','--no-deps',*map(str,wheels)],deadline)
        root=exact_root([Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),Path('/kaggle/input/rsna-knee-abnormality-detection')],
                        ['test.csv','test_series.csv','sample_submission.csv'])
        out=Path('/kaggle/working/mavit-submission');final=Path('/kaggle/working/submission.csv')
        if out.exists() or final.exists():raise ValueError('New immutable MaViT output directory required')
        payload={'contract':MAVIT_CONTRACT,'project':str(project),'root':str(root),'assets':str(assets),'output':str(out)}
        path=Path('/kaggle/working/mavit-submission-input.json');path.write_text(json.dumps(payload,allow_nan=False))
        run_process([sys.executable,'-m','knee.mavit_submission','--payload',str(path),'--deadline',str(deadline)],deadline)
        report=finish(out,MAVIT_CONTRACT,started=MAVIT_STARTED,target=None if MAVIT_CONTRACT['visible'] else final,owned_outputs=owned_outputs)
        print(json.dumps({'event':'mavit_production_done','status':'PASS','visible':MAVIT_CONTRACT['visible'],'test_studies':report['test_studies']}),flush=True)
    except BaseException as error:
        for output in owned_outputs:output.unlink(missing_ok=True)
        Path('/kaggle/working/mavit-submission-parent-failure.json').write_text(json.dumps({'status':'FAIL','failure_type':type(error).__name__,
                                                                                       'elapsed_seconds':time.monotonic()-MAVIT_STARTED})+'\n')
        raise RuntimeError('MaViT production failed; aggregate receipt only') from None
    finally:signal.setitimer(signal.ITIMER_REAL,0)
