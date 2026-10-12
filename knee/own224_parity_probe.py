"""Bounded numerical investigation; cannot qualify K42 or a submission."""
import os
import platform


def compare(a,b):
    import numpy as np
    delta=np.abs(np.asarray(a,dtype=np.float64)-np.asarray(b,dtype=np.float64))
    if delta.shape!=(32,12) or not np.isfinite(delta).all():
        raise ValueError('Probe comparison schema')
    return {'max_abs':float(delta.max()),'median_abs':float(np.median(delta)),
            'above_1e4':int((delta>1e-4).sum()),'exact':bool((delta==0).all())}


def probe(dataset,volumes,masks,dev,id2row,labels,model,saved):
    import hashlib
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    # First32 preserve eight complete historical batches; no error-based selection.
    ids=dev[:32]
    def make():
        ds=dataset('',ids,id2row,labels,224,16,False,norm='imagenet')
        ds.vols=volumes;ds.masks=masks;return ds
    def digest(x):return hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()
    def infer(x,amp=True):
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16,enabled=amp):
            return torch.sigmoid(model(x.to('cuda')).float()).cpu().numpy()
    threads=torch.get_num_threads()
    result={'purpose':'Numerical diagnosis only; no K42, fitting, or qualification',
            'selection':'first32 in frozen historical order, eight unchanged batch4 groups',
            'environment':{'cpu':platform.processor(),'torch_threads':threads,
                'interop_threads':torch.get_num_interop_threads(),'cuda':torch.version.cuda,
                'cudnn':torch.backends.cudnn.version(),'cudnn_benchmark':torch.backends.cudnn.benchmark,
                'cudnn_deterministic':torch.backends.cudnn.deterministic,
                'cudnn_tf32':torch.backends.cudnn.allow_tf32,
                'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,
                'matmul_precision':torch.get_float32_matmul_precision(),
                'fused_attention_modules':sum(bool(getattr(m,'fused_attn',False)) for m in model.modules()),
                'env':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','TIMM_FUSED_ATTN','CUBLAS_WORKSPACE_CONFIG')}},
            'input_hashes':{},'comparisons':{}}
    base=[];repeat=[];hashes=[];fp32=[]
    ds=make()
    for start in range(0,32,4):
        x=torch.stack([ds[i][0] for i in range(start,start+4)])
        hashes.append(digest(x));base.extend(infer(x));repeat.extend(infer(x));fp32.extend(infer(x,False))
    result['input_hashes']['direct_default']=hashes
    variants={'direct_default':np.asarray(base),'direct_repeat':np.asarray(repeat),'direct_fp32':np.asarray(fp32)}
    # Default DataLoader collation stacks the same tensors as historical tk.collate.
    values=[];hashes=[]
    for x,_ in DataLoader(make(),batch_size=4,shuffle=False,num_workers=2):
        hashes.append(digest(x));values.extend(infer(x))
    result['input_hashes']['workers2']=hashes;variants['workers2']=np.asarray(values)
    try:
        torch.set_num_threads(1);ds=make();values=[];hashes=[]
        for start in range(0,32,4):
            x=torch.stack([ds[i][0] for i in range(start,start+4)])
            hashes.append(digest(x));values.extend(infer(x))
        result['input_hashes']['direct_thread1']=hashes;variants['direct_thread1']=np.asarray(values)
    finally:torch.set_num_threads(threads)
    for name,p in variants.items():
        result['comparisons'][name]={'vs_saved':compare(p,saved[:32]),'vs_direct':compare(p,variants['direct_default'])}
    result['inputs_exact_workers']=result['input_hashes']['workers2']==result['input_hashes']['direct_default']
    result['inputs_exact_thread1']=result['input_hashes']['direct_thread1']==result['input_hashes']['direct_default']
    return result
