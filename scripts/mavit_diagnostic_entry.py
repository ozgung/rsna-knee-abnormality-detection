"""Offline aggregate-only diagnostic parent with inherited9h process-group cleanup."""
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
from knee.fresh_reader_setup import dataset_root,exact_root,notebook_root
from knee import mavit_checkpoint as cp,mavit_diagnostic as diagnostic
from knee.reference_blend import CONVNEXT_DATASET,select_codec_wheels
from scripts.rehearse_mavit_entry import run_process


if __name__=='__main__':
    deadline=MAVIT_STARTED+diagnostic.MAX_SECONDS;parent_phase='setup'
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',PYTHONPATH=str(project),PYTHONUNBUFFERED='1')
    try:
        q=diagnostic.qualification(MAVIT_CONTRACT)
        for name,sha in q['runtime_sources'].items():
            if cp.digest((project/name).read_bytes())!=sha:raise ValueError('Qualified diagnostic runtime changed')
        assets=dataset_root('/kaggle/input',cp.DATASET,[cp.CHECKPOINT,*cp.SOURCE_HASHES])
        for name,sha in cp.SOURCE_HASHES.items():
            if cp.digest((assets/name).read_bytes())!=sha:raise ValueError('Pinned public source changed')
        cache=notebook_root('/kaggle/input',diagnostic.KERNEL,'coverage_cache',['plan-private.json'])
        _,cohort=diagnostic.membership(cache/'plan-private.json')
        if cohort!=MAVIT_CONTRACT['cohort']:raise ValueError('Mounted original membership differs')
        codecs=dataset_root('/kaggle/input',CONVNEXT_DATASET,['pylibjpeg-2.1.0-py3-none-any.whl']);wheels=select_codec_wheels(codecs);hashes={}
        for path in wheels:
            with path.open('rb') as handle:hashes[path.name]=hashlib.file_digest(handle,'sha256').hexdigest()
        if hashes!=q['rehearsal_contract']['codecs']:raise ValueError('Qualified offline codec hashes differ')
        run_process([sys.executable,'-m','pip','install','--no-index','--no-deps',*map(str,wheels)],deadline)
        root=exact_root([Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),Path('/kaggle/input/rsna-knee-abnormality-detection')],
                        ['train_series.csv','sample_submission.csv'])
        out=Path('/kaggle/working/mavit-diagnostic')
        if out.exists():raise ValueError('New immutable diagnostic output required')
        payload={'contract':MAVIT_CONTRACT,'project':str(project),'root':str(root),'assets':str(assets),'plan':str(cache/'plan-private.json'),'output':str(out)}
        path=Path('/kaggle/working/mavit-diagnostic-input.json');path.write_text(json.dumps(payload,allow_nan=False))
        parent_phase='child'
        run_process([sys.executable,'-m','knee.mavit_diagnostic','--payload',str(path),'--deadline',str(deadline)],deadline)
        parent_phase='completion';audit=out/'diagnostic.json';report=json.loads(audit.read_text())
        if {p.name for p in out.iterdir()}!={'diagnostic.json'}:raise ValueError('Unexpected diagnostic output')
        report['elapsed_seconds']=time.monotonic()-MAVIT_STARTED
        diagnostic.require_result(report,MAVIT_CONTRACT,parent_complete=False)
        report['parent_complete']=True;diagnostic.require_result(report,MAVIT_CONTRACT)
        audit.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'event':'mavit_train_diagnostic_done','status':'PASS','completed_studies':report['completed_studies'],
                          'elapsed_seconds':report['elapsed_seconds']}),flush=True)
    except BaseException as error:
        Path('/kaggle/working/mavit-diagnostic-parent-failure.json').write_text(json.dumps({'status':'FAIL','phase':parent_phase,
            **diagnostic.safe_failure(error),'elapsed_seconds':time.monotonic()-MAVIT_STARTED})+'\n')
        raise RuntimeError('MaViT diagnostic failed; aggregate receipts only') from None
    finally:signal.setitimer(signal.ITIMER_REAL,0)
