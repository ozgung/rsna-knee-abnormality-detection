"""Bounded no-update CUDA timing on the already qualified TRAIN04 CPU pilot."""
import json
import os
from pathlib import Path
import time

STARTED=time.monotonic()
from knee.fresh_reader_setup import notebook_root,time_limit
from knee.fresh_reader import load_source,tensor_state_sha256,write_json,environment
from knee.coverage_cache import verify_index,validate_plan
from knee.coverage_training import CONFIG,VolumeCache,sample_windows,weighted_terms,macro_loss,timing_projection
out=Path('/kaggle/working/coverage_probe');out.mkdir()
report={'status':'RUNNING','optimizer_updates':0,'fitted_weights_saved':False,'environment':environment(),
        'probe_entry_sha256':PROBE_SOURCE_SHA256}
try:
    with time_limit(PROBE_SECONDS-20-(time.monotonic()-STARTED),'coverage CUDA timing probe'):
        import numpy as np
        import torch
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        cache=notebook_root(Path('/kaggle/input'),'ozgung/knee-train04-cache-pilot-20261008',
                            'coverage_cache',['plan-private.json','index-private.json','cache-report.json'])
        plan=json.loads((cache/'plan-private.json').read_text());validate_plan(plan)
        if plan['fingerprint']!='3f4f0230aff05c8ade9b78ff6e898571839349e3cca41a86c6ded491c7408a32':
            raise ValueError('Pilot plan changed')
        audit=json.loads((cache/'cache-report.json').read_text())
        if audit['status']!='PASS' or audit['cached_studies']!=64:raise ValueError('CPU pilot failed')
        index=json.loads((cache/'index-private.json').read_text());verify_index(cache,plan,index,complete=False)
        rows=[r for r in plan['records'] if r['uid'] in index and r['split']=='train']
        if len(rows)<32:raise ValueError('Insufficient cached training rows')
        np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(2)
        torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
        os.environ['HF_HUB_DOWNLOAD_TIMEOUT']='120';os.environ['HF_HUB_ETAG_TIMEOUT']='30'
        code=load_source(SOURCE_DIR,'knee')
        with time_limit(300,'generic ImageNet initialization'):
            model=code.KneeNet(CONFIG['backbone'],pretrained=True,img_size=256).cuda().to(memory_format=torch.channels_last)
        report['initial_encoder_sha256']=tensor_state_sha256(model.enc.state_dict())
        report['pretrained_cfg']=model.enc.pretrained_cfg
        memory=VolumeCache(cache,index)
        torch.cuda.reset_peak_memory_stats()
        def batch(i,training):
            chosen=[rows[(2*i+j)%len(rows)] for j in range(2)]
            arrays=[sample_windows(memory.get(r['uid']),8 if training else 12,train=training,rng=np.random) for r in chosen]
            x,m,p=[torch.from_numpy(np.stack([a[j] for a in arrays])).cuda() for j in range(3)]
            y=torch.tensor([r['labels'] for r in chosen],device='cuda');return x,m,p,y,2*(y-.5).abs()
        model.train();tick=time.monotonic()
        for i in range(64):
            if i==16:torch.cuda.synchronize();tick=time.monotonic()
            model.zero_grad(set_to_none=True);x,m,p,y,w=batch(i,True)
            with torch.autocast('cuda',dtype=torch.float16):n,d=weighted_terms(model(x,m,p,res=256,train=True),y,w)
            loss=macro_loss(n,d)
            if not torch.isfinite(loss):raise ValueError('Nonfinite probe loss')
            (loss*65536).backward()  # No optimizer, scheduler or EMA update.
        torch.cuda.synchronize();train_seconds=(time.monotonic()-tick)/48
        model.zero_grad(set_to_none=True);model.eval();eval_passes=[]
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            for repeat in range(2):
                timings=[]
                for i in range(8):
                    torch.cuda.synchronize();tick=time.monotonic()
                    x,m,p,y,w=batch(i,False);z=model(x,m,p,res=256,train=False)
                    if not torch.isfinite(z).all():raise ValueError('Nonfinite probe predictions')
                    torch.cuda.synchronize();timings.append(time.monotonic()-tick)
                eval_passes.append(timings)
        eval_seconds=sum(eval_passes[1])/8
        report.update(status='PASS',training_rows=len(rows),warmup_batches=16,timed_batches=48,
            train_batch_seconds=train_seconds,eval_batch_seconds=eval_seconds,eval_batch_timing_passes=eval_passes,
            projection=timing_projection(train_seconds,eval_seconds,time.monotonic()-STARTED),
            peak_cuda_bytes=torch.cuda.max_memory_allocated(),
            limitation='Repeated pilot training rows, cached I/O and generic model only; full2000 timing gate remains mandatory')
except BaseException as exc:
    report.update(status='FAIL',failure_type=type(exc).__name__)
    raise RuntimeError('No-update coverage probe failed; inspect receipt') from None
finally:
    report['elapsed_seconds']=time.monotonic()-STARTED
    write_json(out/'probe-report.json',report)
    print(json.dumps(report),flush=True)
