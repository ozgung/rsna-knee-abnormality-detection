"""CPU qualification of the private dr-full raptor64 checkpoint for the B3 second old arm.

Mirrors scripts/qualify_b0_dread_cpu.py (unchanged, reused): SHA/bytes, the audited Dread raptor
source and K42 manifest (dr-full's export K), strict load into the audited public RaptorClassifier,
label order/arch/res, (42,3,384,384) windows from the same synthetic 64x336x336 volume, finite
repeatable CPU outputs. It records (does not assert) the state and constructor-grid pins; the B3
worker then requires them on CUDA, as B0 does for Dread. It also re-derives the B0 Dread pins with
the same method as a check. Synthetic inputs only; CPU fp32 is not a CUDA reference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

DRFULL_CHECKPOINT_SHA256='35b7fc2cc95155b13271d3bc43ebd4931a915eaa490c572d2fd02d3cac2a061c'
DRFULL_CHECKPOINT_BYTES=292826627
DRFULL_CHECKPOINT_FILE='raptor_ft_fold0_swa.pt'
DRFULL_DATASET='ozgung/knee-raptor64-dr-full-fold0-final-20261013'
DRFULL_WINDOWS=42


def drfull_manifest(source_text):
    from knee.reference import audit_source
    module,decoder_source,manifest=audit_source(source_text,profile='raptor',checkpoint_sha256=DRFULL_CHECKPOINT_SHA256,raptor_windows=DRFULL_WINDOWS)
    return module,decoder_source,{**manifest,'checkpoint_file':DRFULL_CHECKPOINT_FILE,'weight_dataset':DRFULL_DATASET,
                                  'attribution':'own raptor64 dr-full (Dread v4_full init, all 4,345 studies, SWA epochs 9-11) on Dread Development\'s audited inference definitions'}


def qualify(source_path,checkpoint_path,dread_checkpoint=None,forward_windows=DRFULL_WINDOWS):
    import numpy as np
    import torch
    from knee.reference import load_definitions,load_model
    from knee.public_fullfield import snapshot
    from knee.public_adaptation_pilot import state_fingerprint
    from knee.public_s6_qualification import LABELS,require
    from scripts.qualify_b0_dread_cpu import synthetic_volume,grid_fingerprint
    checkpoint=Path(checkpoint_path);started=time.monotonic()
    with checkpoint.open('rb') as h:sha=hashlib.file_digest(h,'sha256').hexdigest()
    require(sha==DRFULL_CHECKPOINT_SHA256 and checkpoint.stat().st_size==DRFULL_CHECKPOINT_BYTES,'DRFULL_CHECKPOINT_IDENTITY')
    module,decoder_source,manifest=drfull_manifest(Path(source_path).read_text())
    public,decoder=load_definitions(module,decoder_source,manifest);require(decoder is None and public.K_EVAL==DRFULL_WINDOWS,'RAPTOR_K42_DEFINITIONS')
    raw=torch.load(checkpoint,map_location='cpu',weights_only=True)
    require(list(raw.get('lab',[]))==LABELS==list(public.LAB) and raw.get('arch')==manifest['recipe']['arch'] and int(raw.get('res',0))==384,'CHECKPOINT_LABEL_ORDER_ARCH')
    meta={k:raw[k] for k in ('epoch','swa_over') if k in raw};del raw
    model=load_model(public,checkpoint,'cpu',manifest)  # strict=True state load
    state=snapshot(model);grid=grid_fingerprint(model)
    volume,mask=synthetic_volume();windows=public.eval_windows(volume,mask,DRFULL_WINDOWS,384)
    require(tuple(windows.shape)==(DRFULL_WINDOWS,3,384,384) and windows.dtype==torch.float32 and bool(torch.isfinite(windows).all()),'K42_WINDOW_SCHEMA')
    receipt={'checkpoint_sha256':sha,'checkpoint_bytes':checkpoint.stat().st_size,'manifest':manifest,'extracted_module_sha256':manifest['extracted_module_sha256'],
             'labels':LABELS,'strict_state_load':True,'state_sha256':state['state'],'buffers_sha256':state['buffers'],'constructor_grid_sha256':grid,'checkpoint_meta':meta,
             'windows_shape':list(windows.shape),'windows_fingerprint':state_fingerprint(windows),'torch':torch.__version__,'device':'cpu','uses_raw_data':False,'uses_labels':False}
    from knee.reference import infer_checked
    sub=windows[:forward_windows];t=time.monotonic()
    p=np.asarray(infer_checked(public,model,[sub],'cpu',mirror=False));q=np.asarray(infer_checked(public,model,[sub],'cpu',mirror=False))
    require(p.shape==(12,) and np.isfinite(p).all() and float(np.max(np.abs(p-q)))<=1e-6,'FINITE_REPEATABLE_CPU_OUTPUTS')
    receipt.update(forward_windows=int(sub.shape[0]),cpu_probabilities=p.tolist(),cpu_repeat_max_abs=float(np.max(np.abs(p-q))),forward_seconds=(time.monotonic()-t)/2)
    require(snapshot(model)==state,'CPU_STATE_UNCHANGED_AFTER_FORWARD')
    if dread_checkpoint is not None:
        # Method check: the same CPU procedure reproduces B0's CUDA-verified Dread pins.
        import knee.public_s6_recovery_b0 as B0
        from knee.reference import audit_source
        m2,d2,man2=audit_source(Path(source_path).read_text(),profile='raptor',checkpoint_sha256=B0.DREAD_CHECKPOINT_SHA256,raptor_windows=B0.RAPTOR_WINDOWS)
        p2,_=load_definitions(m2,d2,man2);dread=load_model(p2,dread_checkpoint,'cpu',man2)
        receipt['dread_pin_method_check']=snapshot(dread)['state']==B0.DREAD_STATE_SHA256 and grid_fingerprint(dread)==B0.DREAD_GRID_SHA256
        require(receipt['dread_pin_method_check'],'CPU_PIN_METHOD_REPRODUCES_B0_DREAD')
    receipt['seconds']=time.monotonic()-started
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--dread-checkpoint');p.add_argument('--out',required=True)
    a=p.parse_args()
    from knee.public_s6_qualification import require
    out=Path(a.out);require(not out.exists(),'FRESH_RECEIPT')
    r=qualify(a.source,a.checkpoint,a.dread_checkpoint)
    out.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:r[k] for k in ('state_sha256','constructor_grid_sha256','windows_shape','forward_windows','cpu_repeat_max_abs','forward_seconds','dread_pin_method_check','checkpoint_meta') if k in r}))
