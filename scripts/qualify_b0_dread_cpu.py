"""CPU qualification of the local Dread v4 checkpoint for the B0 raptor old arm.

Checks SHA/bytes, the audited raptor62 manifest, strict load, label order, the
B0 state and constructor-grid pins, (62,3,384,384) windows from a synthetic
64x336x336 volume and finite repeatable CPU outputs. Synthetic inputs only;
CPU fp32 is not CUDA fp16, so no probability is a CUDA reference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

NARTAA_GRID_SHA256='ce369a0e8f83c9eff93053fc9053e9dd3f958a72ceeeb328f2416bdc23179523'


def synthetic_volume(seed=20261013):
    import numpy as np
    rng=np.random.default_rng(seed);volume=rng.integers(1,255,(64,336,336),dtype=np.uint8)
    mask=np.ones(64,dtype=np.uint8);volume[40:44]=0;mask[40:44]=0  # one absent-style gap
    return volume,mask


def grid_fingerprint(model):
    from knee.public_fullfield import GRID_NAMES
    from knee.public_adaptation_pilot import state_fingerprint
    grid={k:v.detach().cpu().clone() for k,v in model.named_buffers() if k in GRID_NAMES}
    if len(grid)!=len(GRID_NAMES):raise ValueError('CONSTRUCTOR_GRID_INVENTORY')
    return state_fingerprint(grid)


def qualify(source_path,checkpoint_path,*,raptor62_contract=None,forward_windows=62,evaluation_build=None):
    import numpy as np
    import torch
    from knee.reference import audit_source,load_definitions,load_model
    from knee.public_fullfield import snapshot
    from knee.public_adaptation_pilot import state_fingerprint
    from knee.public_s6_qualification import LABELS,require
    import knee.public_s6_recovery_b0 as B0
    checkpoint=Path(checkpoint_path);started=time.monotonic()
    with checkpoint.open('rb') as h:sha=hashlib.file_digest(h,'sha256').hexdigest()
    require(sha==B0.DREAD_CHECKPOINT_SHA256 and checkpoint.stat().st_size==B0.DREAD_CHECKPOINT_BYTES,'DREAD_CHECKPOINT_IDENTITY')
    module,decoder_source,manifest=audit_source(Path(source_path).read_text(),profile='raptor',checkpoint_sha256=sha,raptor_windows=B0.RAPTOR_WINDOWS)
    if raptor62_contract is not None:
        require(json.loads(Path(raptor62_contract).read_bytes())['source']==manifest,'RAPTOR62_QUALIFIED_MANIFEST')
    public,decoder=load_definitions(module,decoder_source,manifest);require(decoder is None and decoder_source=='','RAPTOR_HAS_NO_PINNED_DECODER')
    B0.raptor_profile_gate(public,manifest)
    raw=torch.load(checkpoint,map_location='cpu',weights_only=True)
    require(list(raw.get('lab',[]))==LABELS==list(public.LAB) and raw.get('arch')==manifest['recipe']['arch'] and int(raw.get('res',0))==384,'CHECKPOINT_LABEL_ORDER_ARCH')
    del raw
    model=load_model(public,checkpoint,'cpu',manifest)  # strict=True state load
    state=snapshot(model);grid=grid_fingerprint(model)
    require(state['state']==B0.DREAD_STATE_SHA256,'B0_STATE_PIN');require(grid==B0.DREAD_GRID_SHA256,'B0_GRID_PIN')
    volume,mask=synthetic_volume();windows=B0.raptor_windows(public,volume,mask)
    receipt={'checkpoint_sha256':sha,'checkpoint_bytes':checkpoint.stat().st_size,'manifest':manifest,'extracted_module_sha256':manifest['extracted_module_sha256'],
             'profile':B0.OLD_PROFILE,'labels':LABELS,'strict_state_load':True,'state_sha256':state['state'],'buffers_sha256':state['buffers'],'constructor_grid_sha256':grid,
             'windows_shape':list(windows.shape),'windows_fingerprint':state_fingerprint(windows),
             'raptor62_manifest_equal':raptor62_contract is not None,'torch':torch.__version__,'device':'cpu','uses_raw_data':False,'uses_labels':False}
    if forward_windows:
        sub=windows[:forward_windows];t=time.monotonic()
        p=B0.raptor_forward(public,model,sub,'cpu');q=B0.raptor_forward(public,model,sub,'cpu')
        require(p.shape==(12,) and np.isfinite(p).all() and float(np.max(np.abs(p-q)))<=1e-6,'FINITE_REPEATABLE_CPU_OUTPUTS')
        receipt.update(forward_windows=int(sub.shape[0]),cpu_probabilities=p.tolist(),cpu_repeat_max_abs=float(np.max(np.abs(p-q))),forward_seconds=(time.monotonic()-t)/2)
    require(snapshot(model)==state,'CPU_STATE_UNCHANGED_AFTER_FORWARD')
    if evaluation_build is not None:
        # Method check: the same CPU constructor-grid fingerprint reproduces the
        # default nartaa pin (different input geometry, hence a different grid).
        import base64
        import lzma
        import timm
        from scripts.build_public_s6_recovery_entry import literals,EVALUATION_NOTEBOOK_SHA
        ev=literals(Path(evaluation_build)/'notebook.ipynb',EVALUATION_NOTEBOOK_SHA)
        cfg=json.loads(lzma.decompress(base64.b64decode(ev['encoded'],validate=True)))['cfg']
        mirror,_=load_definitions(cfg['public_source'],cfg['decoder_source'],cfg['public_manifest']);mirror.timm=timm
        backbone=mirror.build_backbone(cfg['public_manifest']['recipe']['arch'],pretrained=False)
        receipt['nartaa_grid_method_check']=grid_fingerprint(mirror.RaptorClassifier(backbone,F_dim=backbone.num_features))==cfg['control_grid_sha256']==NARTAA_GRID_SHA256
        require(receipt['nartaa_grid_method_check'],'CPU_GRID_METHOD_REPRODUCES_DEFAULT_PIN')
    receipt['seconds']=time.monotonic()-started
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--raptor62-contract')
    p.add_argument('--forward-windows',type=int,default=62);p.add_argument('--evaluation-build');p.add_argument('--out')
    a=p.parse_args()
    r=qualify(a.source,a.checkpoint,raptor62_contract=a.raptor62_contract,forward_windows=a.forward_windows,evaluation_build=a.evaluation_build)
    text=json.dumps(r,indent=2,sort_keys=True)+'\n'
    if a.out:
        from knee.public_s6_qualification import require
        out=Path(a.out);require(not out.exists(),'FRESH_RECEIPT');out.write_text(text)
    print(json.dumps({k:r[k] for k in ('state_sha256','constructor_grid_sha256','windows_shape','forward_windows','cpu_repeat_max_abs','forward_seconds','nartaa_grid_method_check') if k in r}))
