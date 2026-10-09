"""CPU-only, source-pinned MaViT checkpoint inspection; no image/model forward."""
import ast
from collections import Counter, OrderedDict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import stat
import sys
import time
import zipfile

VERSION='mavit-checkpoint-audit-v1'
DATASET='hengck23/hengck23-rnsa-knee'
CHECKPOINT='00000008.pth'
CHECKPOINT_BYTES=79601008  # Publisher inventory; SHA is a first-observation output.
SOURCE_HASHES={
 'configure.py':'3a46d4e6f22171e28909574d98d006154a519363a36f40aba31ea799226f4cee',
 'mavit.py':'e4a48c8ca5fcc0417d5c051b0c0410a0dc0a34c74f51f3e8bc58285ee363936e',
 'model.py':'082c8c8baf86c3c25f878d2bb5c5e0ba1645600e055810f293bfc3888233e8d3',
 'my_fastread_knee.py':'37140c1f96f3afa186c75820e445619511f349d41a18a3ef75b2cffa7d57efa7'}
CONFIG={'NUM_VIEW':6,'NUM_SLICE':8,'NUM_INJURY':12,'PRE_SIZE':480,'SIZE':288}
BACKBONE_NAMES=('SwishImplementation','MemoryEfficientSwish','rotate_every_two','theta_shift','RoPE','LayerNorm2d',
 'AddLinearAttention','VanillaSelfAttention','FeedForwardNetwork','Block','PatchMerging','BasicLayer','PatchEmbed','MALA','MALA_T')
MODEL_NAMES=('do_augment','KneeModel')
PACKAGES={'torch':'2.10.0+cu128','timm':'1.0.26'}
SAFE_METADATA_KEYS={'model_state_dict','optimizer_state_dict','scheduler_state_dict','scaler_state_dict',
                    'epoch','iteration','step','best_score','config','args','state_dict','optimizer','scheduler','scaler'}
PREFIX='''from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, trunc_normal_
from timm.models.vision_transformer import _cfg
from einops import rearrange
'''


def digest(data):return hashlib.sha256(data).hexdigest()

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def source_contract(directory):
    raw={n:(Path(directory)/n).read_bytes() for n in SOURCE_HASHES}
    if any(digest(v)!=SOURCE_HASHES[n] for n,v in raw.items()):raise ValueError('Pinned source integrity differs')
    # The decoder is hashed for package identity but never parsed/imported/executed.
    selected=[];proof={}
    config=ast.parse(raw['configure.py'].decode());found={}
    for n in config.body:
        if not isinstance(n,ast.Assign) or len(n.targets)!=1 or not isinstance(n.targets[0],ast.Name):raise ValueError('Configuration syntax changed')
        found[n.targets[0].id]=ast.literal_eval(n.value)
    if found!=CONFIG:raise ValueError('Published model configuration changed')
    selected.extend(config.body)
    for name,names in [('mavit.py',BACKBONE_NAMES),('model.py',MODEL_NAMES)]:
        nodes=[n for n in ast.parse(raw[name].decode()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
        if tuple(n.name for n in nodes)!=names:raise ValueError('Architecture definition inventory changed')
        for n in nodes:
            # Unmodified definitions, including their decorators and method bodies.
            proof[n.name]=digest(ast.dump(n,include_attributes=False).encode())
        selected.extend(nodes)
    code=PREFIX+ast.unparse(ast.Module(body=selected,type_ignores=[]))+'\n'
    return code,{'source_sha256':SOURCE_HASHES,'config':CONFIG,'definition_ast_sha256':proof,
                 'extracted_source_sha256':digest(code.encode()),'imports':PREFIX.splitlines(),
                 'adaptations':['no-op factory registration','pretrain loader raises if called',
                                'omit unused factories, reader and executable notebook/main blocks']}


def instantiate(source):
    """No source import path exposure; the two hooks are explicitly non-scientific."""
    def forbidden(*args,**kwargs):raise RuntimeError('Pretrained loading is forbidden in checkpoint audit')
    env={'__name__':'audited_mavit_checkpoint','register_model':lambda fn:fn,'load_mala_pretrain':forbidden}
    exec(compile(source,'<pinned-mavit-model-definitions>','exec'),env)
    # Never call forward/infer/head, and never move the public model to CUDA.
    return env['KneeModel'](pretrained=False).cpu().eval()


def archive_audit(handle):
    """Inspect member metadata only; no extraction or pickle execution."""
    with zipfile.ZipFile(handle) as z:
        members=z.infolist()
        if not 1<=len(members)<=10000:raise ValueError('Checkpoint archive member count invalid')
        names=[m.filename for m in members]
        if len(names)!=len(set(names)):raise ValueError('Duplicate checkpoint archive member')
        for m in members:
            parts=m.filename.split('/')
            if (m.filename.startswith('/') or '\\' in m.filename or '..' in parts or ':' in m.filename or
                    stat.S_ISLNK(m.external_attr>>16) or m.flag_bits&1):raise ValueError('Unsafe checkpoint archive member')
        total=sum(m.file_size for m in members)
        if total>1024**3:raise ValueError('Checkpoint archive expands beyond one GiB safety bound')
        if sum(n.endswith('/data.pkl') for n in names)!=1:raise ValueError('Expected one standard torch checkpoint pickle')
        return {'format':'torch_zip','members':len(members),'uncompressed_bytes':total,'extracted':False}


def checkpoint_schema(payload,report):
    """Inspect aggregate types/finiteness; never expose scalar values or private keys."""
    import torch
    if type(payload) not in (dict,OrderedDict):raise ValueError('Checkpoint payload must be a plain mapping')
    safe=[];hidden=[]
    for key,value in payload.items():
        if type(key) is not str:raise ValueError('Top-level checkpoint keys must be strings')
        kind=type(value).__name__ if type(value) in (dict,OrderedDict,list,tuple,int,float,bool,str,bytes,type(None)) or torch.is_tensor(value) else 'unsupported'
        row={'type':kind}
        if key in SAFE_METADATA_KEYS:row['key']=key;safe.append(row)
        else:row['key_sha256']=digest(key.encode());hidden.append(row)
    report['payload_schema']={'known_keys':sorted(safe,key=lambda r:r['key']),
        'other_key_count':len(hidden),'other_keys':sorted(hidden,key=lambda r:r['key_sha256'])[:64],
        'other_key_list_sha256':digest(canonical(sorted(hidden,key=lambda r:r['key_sha256']))),
        'other_key_listing_limit':64,'metadata_values_exported':False}
    counts=Counter();dtypes=Counter();tensor_count=numel=nonfinite=0;seen=set();todo=[payload]
    while todo:
        x=todo.pop()
        if torch.is_tensor(x):
            if id(x) in seen:continue
            seen.add(id(x))
            if vars(x):todo.append(vars(x))
            tensor_count+=1;numel+=x.numel();dtypes[str(x.dtype)]+=1
            if x.device.type!='cpu' or x.layout!=torch.strided:raise ValueError('Only dense CPU checkpoint tensors are supported')
            nonfinite+=int(not bool(torch.isfinite(x).all()))
        elif type(x) in (dict,OrderedDict,list,tuple):
            if id(x) in seen:continue
            seen.add(id(x));counts[type(x).__name__]+=1
            if type(x) is OrderedDict and vars(x):todo.append(vars(x))
            if type(x) in (dict,OrderedDict):
                todo.extend(x.keys());todo.extend(x.values())
            else:todo.extend(x)
        elif type(x) in (str,bytes,int,float,bool,type(None)):counts[type(x).__name__]+=1
        else:raise ValueError('Unsupported safe-loaded metadata object')
    report['all_checkpoint_tensors']={'count':tensor_count,'numel':numel,'dtype_counts':dict(dtypes),'nonfinite_tensors':nonfinite,
                                     'metadata_type_counts':dict(counts),'counts_unique_tensor_objects':True}
    if nonfinite:raise ValueError('Checkpoint includes nonfinite tensors')
    state=payload.get('model_state_dict')
    if type(state) not in (dict,OrderedDict) or not state or any(type(k) is not str or not torch.is_tensor(v) for k,v in state.items()):
        raise ValueError('Expected nonempty model_state_dict containing tensors only')
    return state


def match_state(model,state,report):
    import torch
    expected=model.state_dict();missing=set(expected)-set(state);extra=set(state)-set(expected);common=set(expected)&set(state)
    shape=[k for k in common if tuple(expected[k].shape)!=tuple(state[k].shape)]
    dtype=[k for k in common if expected[k].dtype!=state[k].dtype]
    def schema(mapping):return [[k,list(v.shape),str(v.dtype)] for k,v in sorted(mapping.items())]
    report['state_compatibility']={'expected_keys':len(expected),'actual_keys':len(state),'missing_keys':len(missing),'unexpected_keys':len(extra),
        'shape_mismatches':len(shape),'dtype_mismatches':len(dtype),'expected_schema_sha256':digest(canonical(schema(expected))),
        'actual_schema_sha256':digest(canonical(schema(state))),'strict_load':False,
        'parameter_numel':sum(p.numel() for p in model.parameters()),'buffer_numel':sum(b.numel() for b in model.buffers())}
    if missing or extra or shape or dtype:raise ValueError('Checkpoint does not exactly match the full public architecture')
    loaded=model.load_state_dict(state,strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:raise ValueError('Strict state load returned mismatches')
    if not all(torch.equal(v,state[k]) for k,v in model.state_dict().items()):raise ValueError('Loaded tensors differ from checkpoint')
    if not all(torch.isfinite(v).all() for v in model.state_dict().values()):raise ValueError('Loaded model has nonfinite tensors')
    report['state_compatibility'].update(strict_load=True,loaded_tensors_equal=True,all_loaded_tensors_finite=True)


def inspect_checkpoint(path,model_factory,report,*,expected_bytes=CHECKPOINT_BYTES):
    """Safe load is mandatory, with no retry, allowlist or alternative pickle loader."""
    import torch
    report['stage']='checkpoint_hash'
    with Path(path).open('rb') as handle:
        actual=os.fstat(handle.fileno()).st_size
        report['checkpoint']={'bytes':actual,'sha256':hashlib.file_digest(handle,'sha256').hexdigest(),
                              'sha256_status':'first_observation; not previously verified','weights_only':True,'map_location':'cpu'}
        if actual!=expected_bytes:raise ValueError('Checkpoint size differs from publisher inventory')
        report['stage']='archive_metadata';handle.seek(0);report['archive']=archive_audit(handle)
        report['stage']='safe_deserialization';handle.seek(0)
        payload=torch.load(handle,map_location='cpu',weights_only=True)
        handle.seek(0)
        if hashlib.file_digest(handle,'sha256').hexdigest()!=report['checkpoint']['sha256']:raise ValueError('Checkpoint changed while being inspected')
    report['stage']='checkpoint_schema';state=checkpoint_schema(payload,report)
    report['stage']='architecture_construction'
    # Even an accidental public constructor branch must not perform another load.
    from unittest.mock import patch
    with patch.object(torch,'load',side_effect=RuntimeError('Nested checkpoint/pretrain load forbidden')):
        model=model_factory()
    if any(p.device.type!='cpu' for p in [*model.parameters(),*model.buffers()]):raise ValueError('Audit model must stay on CPU')
    report['stage']='strict_compatibility';match_state(model,state,report)
    del model,state,payload


def run(directory,output,contract,*,started=None):
    started=time.monotonic() if started is None else started
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    report={'status':'RUNNING','version':VERSION,'contract_sha256':digest(canonical(contract)),
        'training_lineage':'unknown; no training cohort/labels/provenance inferred from checkpoint or filename',
        'license':'MIT declared by dataset publisher; upstream rights not independently verified',
        'device':'cpu','image_forward_calls':0,'training_steps':0,'metadata_values_exported':False,'stage':'environment',
        'source_revision':contract.get('source_revision'),'runtime_sha256':contract.get('runtime_sha256')}
    def save():
        report['elapsed_seconds']=time.monotonic()-started
        report['peak_host_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
        raw=json.dumps(report,indent=2,allow_nan=False)+'\n'
        if len(raw.encode())>65536:raise ValueError('Aggregate audit output exceeds64KiB')
        (out/'checkpoint-audit.json').write_text(raw)
    try:
        if sys.version_info[:2]!=(3,12):raise ValueError('Pinned audit Python differs')
        versions={p:importlib.metadata.version(p) for p in ('torch','timm','einops','numpy','torchvision')}
        report['environment']={'python_major_minor':list(sys.version_info[:2]),'packages':versions,'threads':2,
                               'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES','')}
        if any(versions[p]!=v for p,v in PACKAGES.items()):raise ValueError('Pinned audit torch/timm differs')
        import torch
        torch.set_num_threads(2)
        report['stage']='source_integrity';source,proof=source_contract(directory)
        if contract.get('version')!=VERSION or contract.get('source_contract')!=proof:raise ValueError('Reviewed source contract differs')
        report['source_contract']=proof;save()
        inspect_checkpoint(Path(directory)/CHECKPOINT,lambda:instantiate(source),report)
        report.update(status='PASS',stage='complete');save()
    except Exception as error:
        report.update(status='FAIL',failure_type=type(error).__name__)
        save()
        raise RuntimeError('MaViT checkpoint audit failed; see aggregate receipt') from None
    print(json.dumps({'event':'mavit_checkpoint_audit','status':report['status'],'elapsed_seconds':report['elapsed_seconds'],
                      'checkpoint_bytes':report['checkpoint']['bytes']}),flush=True)
    return report
