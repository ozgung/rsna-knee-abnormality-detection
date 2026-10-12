"""VM-only pinned generic ImageNet initializer preparation; no timm downloads."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import urllib.parse
import urllib.request

ARCH='coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k'
REPO='timm/'+ARCH
REVISION='50b0fc810164980f0e494198ed3da32f5abf6dbe'
SHA256='8558fd38c0c67be0425f3ca19752a7baa4356314940fa4954633a6395a97d973'
SIZE=295695280
TIMM_VERSION='1.0.26'
PIN={'repo':REPO,'revision':REVISION,'file':'model.safetensors','sha256':SHA256,'size':SIZE}
CHUNK=1024*1024


def file_sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(CHUNK),b''):h.update(block)
    return h.hexdigest()


def state_sha(state):
    h=hashlib.sha256()
    for name,t in sorted(state.items()):
        a=t.detach().cpu().contiguous().numpy()
        h.update(json.dumps([name,str(a.dtype),list(a.shape)],sort_keys=True,separators=(',',':')).encode());h.update(a.tobytes())
    return h.hexdigest()


def atomic_json(path,value):
    tmp=path.with_name(path.name+'.part')
    with open(tmp,'w') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


@contextmanager
def preserved_rng():
    import numpy as np,torch
    py=random.getstate();npstate=np.random.get_state();cpu=torch.get_rng_state()
    cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:yield
    finally:
        random.setstate(py);np.random.set_state(npstate);torch.set_rng_state(cpu)
        if cuda is not None:torch.cuda.set_rng_state_all(cuda)


def model_factory(num_classes):
    import timm
    if timm.__version__!=TIMM_VERSION:raise RuntimeError('Requires historical timm '+TIMM_VERSION+'; do not replace VM torch')
    return timm.create_model(ARCH,pretrained=False,num_classes=num_classes,in_chans=3,global_pool='avg')


def safe_load(path):
    from safetensors.torch import load_file
    return load_file(str(path),device='cpu')


def download(url):
    # Fixed public endpoint, no credentials. Redirect URLs are neither printed nor stored.
    response=urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'source96-initializer/1'}),timeout=60)
    if urllib.parse.urlsplit(response.geturl()).scheme!='https':
        response.close();raise ValueError('Non-HTTPS artifact redirect rejected')
    return response


def normalize(state,full,backbone):
    import torch
    classifier=full.get_classifier()
    paths=[name for name,module in full.named_modules() if module is classifier]
    if paths!=['head.fc'] or not isinstance(classifier,torch.nn.Linear) or classifier.out_features!=1000:
        raise ValueError('Unexpected ImageNet classifier schema; conversion requires review')
    omitted={'head.fc.weight','head.fc.bias'}
    fs=full.state_dict();bs=backbone.state_dict()
    if set(fs)-set(bs)!=omitted or set(bs)-set(fs) or set(classifier.state_dict())!={'weight','bias'}:
        raise ValueError('Only reviewed ImageNet classifier tensors may be omitted')
    if not isinstance(state,dict) or not state or set(state)!=set(fs):raise ValueError('Origin must exactly cover full ImageNet model')
    for key,v in state.items():
        expected=fs[key]
        if not torch.is_tensor(v) or v.device.type!='cpu' or v.shape!=expected.shape or v.dtype!=expected.dtype or not bool(torch.isfinite(v).all()):
            raise ValueError('Origin tensor schema/finiteness differs')
        if key in bs and (v.shape!=bs[key].shape or v.dtype!=bs[key].dtype):raise ValueError('Historical backbone compatibility differs')
    result={k:state[k].detach().contiguous() for k in bs}
    backbone.load_state_dict(result,strict=True)
    return result,{'omitted_keys':sorted(omitted),'origin_keys':len(state),'backbone_keys':len(result),
                   'origin_key_sha256':hashlib.sha256(json.dumps(sorted(state),separators=(',',':')).encode()).hexdigest(),
                   'backbone_key_sha256':hashlib.sha256(json.dumps(sorted(result),separators=(',',':')).encode()).hexdigest(),
                   'origin_tensor_sha256':state_sha(state),'backbone_tensor_sha256':state_sha(result)}


def _prepare(root,pin,*,fetch=download,factory=model_factory,load=safe_load,disk_usage=shutil.disk_usage):
    """Injectable I/O for tests; production CLI always uses PIN and real safetensors."""
    import torch
    root=Path(root)
    if root.is_symlink():raise ValueError('Symlink output root rejected')
    root.mkdir(parents=True,exist_ok=True)
    origin=root/'model.safetensors';weights=root/'backbone.pt';descriptor=root/'initializer.json'
    if any(p.is_symlink() for p in (origin,weights,descriptor,origin.with_name(origin.name+'.part'),weights.with_name(weights.name+'.part'),descriptor.with_name(descriptor.name+'.part'))):raise ValueError('Symlink artifact rejected')
    if descriptor.exists() or weights.exists():raise ValueError('Completed/partial normalized output preserved; use a fresh directory')
    needed=(0 if origin.exists() else pin['size'])+pin['size']+512*1024*1024
    if disk_usage(root).free<needed:raise OSError('Insufficient disk for origin plus atomic normalized output and 512MiB reserve')
    if not origin.exists():
        url='https://huggingface.co/'+pin['repo']+'/resolve/'+pin['revision']+'/'+pin['file']
        temp=origin.with_name(origin.name+'.part');h=hashlib.sha256();count=0
        try:
            with fetch(url) as response,open(temp,'wb') as out:
                while True:
                    block=response.read(CHUNK)
                    if not block:break
                    count+=len(block)
                    if count>pin['size']:raise ValueError('Origin exceeds pinned size')
                    h.update(block);out.write(block)
                out.flush();os.fsync(out.fileno())
            if count!=pin['size'] or h.hexdigest()!=pin['sha256']:raise ValueError('Origin size/checksum differs')
            os.replace(temp,origin)
        finally:
            if temp.exists():temp.unlink()
    if origin.stat().st_size!=pin['size'] or file_sha(origin)!=pin['sha256']:raise ValueError('Existing origin size/checksum differs')
    with preserved_rng():
        full=factory(1000);backbone=factory(0)
        normalized,audit=normalize(load(origin),full,backbone)
        temp=weights.with_name(weights.name+'.part')
        try:
            with open(temp,'wb') as f:torch.save(normalized,f);f.flush();os.fsync(f.fileno())
            reloaded=torch.load(temp,map_location='cpu',weights_only=True)
            if state_sha(reloaded)!=audit['backbone_tensor_sha256']:raise ValueError('Normalized reload tensor identity differs')
            backbone.load_state_dict(reloaded,strict=True);os.replace(temp,weights)
        finally:
            if temp.exists():temp.unlink()
    d={'kind':'imagenet_backbone','arch':ARCH,'pretrained_tag':'sw_in12k_ft_in1k','hf_repo':pin['repo'],
       'hf_revision':pin['revision'],'origin_file_sha256':pin['sha256'],'origin_file_bytes':pin['size'],
       'weights_file':str(weights.resolve()),'weights_sha256':file_sha(weights),'weights_bytes':weights.stat().st_size,
       'timm':TIMM_VERSION,'torch':torch.__version__,'audit':audit,
       'historical_initializer_identity':'Unresolved; fixed revision is a recipe-level comparator, not proof of original initialization',
       'rng':'Python/NumPy/torch main RNG restored; already initialized CUDA RNG restored',
       'source':'Official public timm ImageNet12k-to-1k artifact; no knee-trained weights'}
    atomic_json(descriptor,d)
    return {'descriptor':str(descriptor.resolve()),'descriptor_sha256':file_sha(descriptor),'weights_sha256':d['weights_sha256'],
            'origin_sha256':pin['sha256'],'backbone_tensor_sha256':audit['backbone_tensor_sha256'],'timm':TIMM_VERSION,'training':False}


def prepare(root):
    import timm
    from safetensors.torch import load_file
    if timm.__version__!=TIMM_VERSION:raise RuntimeError('Requires historical timm '+TIMM_VERSION+'; do not replace VM torch')
    return _prepare(root,PIN)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    print(json.dumps(prepare(**vars(parser.parse_args())),indent=2))
