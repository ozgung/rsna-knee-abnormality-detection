"""One fixed own224 fold0 K16/K42 diagnostic; no fitting or production changes."""
import csv
import hashlib
import json
from pathlib import Path
import time

ARCH='coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k'
TARGETS=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA',
         'Effusion','Synovitis',"Baker's",'Contusion','Fracture']
CHECKPOINT_SHA='6b2bd74cb2ececbe65e61c2c95d5b43cc1029c5743ef445d276b8b48dc3e95b1'
OOF_SHA='3928c002f00a0aec69a78effcd496141153e51810d43dc1d5f6a8b55b5c9072b'
PATCHED_SHA='eb3d34ffc6aa6f7c500f8ef2039dba0f313b42e371e075e58d4b7a47dde0bd3c'
TRUTH_SHA='a134cfce2f751f90f4219d791a563ee677b6223ac28f32e456bb3759c1d3fba9'
TRAIN_SHA='8ca2203c0e9d61c080c7a314c7cdb51c1b03a1d9eb4770819f7f34af53ef4e33'
PARITY_ATOL=1e-4
SLOTS=(12,10,8,6,8)


def require(ok,message):
    if not ok:raise ValueError(message)


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def checked(path,pin):
    require(file_sha(path)==pin,'Pinned artifact bytes differ');return Path(path)


def save(path,value):
    p=Path(path);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(p)


def probabilities(values,rows):
    import numpy as np
    a=np.asarray(values)
    require(a.shape==(rows,12) and np.isfinite(a).all() and ((a>=0)&(a<=1)).all(),'Probability schema/range')
    return a


def validate_corpus(ids,masks,volumes):
    import numpy as np
    ids=list(map(str,ids))
    require(len(ids)==len(set(ids)) and all(ids),'Unique cache IDs required')
    require(masks.shape==(len(ids),44) and np.isin(masks,[0,1]).all(),'Binary44 masks required')
    require(volumes.shape==(len(ids),44,336,336) and volumes.dtype==np.uint8,'Legacy44 uint8 volume schema')
    return ids


def population(cache_ids,folds,excluded,saved_ids,count=870):
    ids=[s for s in cache_ids if folds.get(s)==0 and s not in excluded]
    require(len(ids)==count and len(saved_ids)==count and len(set(saved_ids))==count and ids==list(saved_ids),'Exact ordered held-out coverage required')
    return ids


def training_population(cache_ids,folds,excluded,rows):
    """Match the historical exporter: gold studies are never training members."""
    indexed={r['StudyInstanceUID']:r for r in rows}
    require(len(indexed)==len(rows) and set(cache_ids)<=indexed.keys(),'Official membership coverage')
    gold={sid for sid,r in indexed.items() if all(str(r.get(t,'')).strip() for t in TARGETS)}
    train=[sid for sid in cache_ids if sid in folds and folds[sid]!=0 and sid not in excluded and sid not in gold]
    return train,gold


def center_plan(mask,legacy,k):
    """Exact source min/max policy and fixed K, including duplicate/fallback centers."""
    import numpy as np
    m=np.asarray(mask);require(m.shape==(44,) and np.isin(m,[0,1]).all(),'Binary44 mask required')
    require(k in (16,42),'Frozen K16/K42 only')
    valid=np.flatnonzero(m>0);fallback=len(valid)<3
    if fallback:valid=np.arange(3)
    legacy._D=44;centers=legacy._centers(valid,k);slot=np.repeat(np.arange(5),SLOTS)
    require(len(centers)==k,'Fixed K centers required')
    return centers,{'windows':k,'unique_centers':len(set(centers)),'duplicate_centers':k-len(set(centers)),
        'masked_triplets':sum(not bool(m[c-1:c+2].all()) for c in centers),
        'boundary_triplets':sum(len(set(slot[c-1:c+2]))>1 for c in centers),'fallback_studies':int(fallback)}


def endpoint(ids,y,masks,control,candidate):
    import numpy as np
    require(len(ids)==len(set(ids)) and len(ids)>0,'Endpoint unique IDs')
    require(y.shape==masks.shape==(len(ids),12) and np.isin(masks,[0,1]).all() and np.isin(y[masks>0],[0,1]).all(),'Endpoint truth/mask schema')
    return probabilities(control,len(ids)),probabilities(candidate,len(ids))


def parity(current,saved,*,enforce=True):
    import numpy as np
    a,b=probabilities(current,len(saved)),probabilities(saved,len(saved));error=float(np.max(np.abs(a-b)))
    if enforce:require(error<=PARITY_ATOL,'Historical K16 FP16 parity failed')
    return {'max_absolute_error':error,'absolute_tolerance':PARITY_ATOL,'relative_tolerance':0,
            'within_historical_tolerance':error<=PARITY_ATOL,'enforced':enforce,
            'reason':'Same Tesla T4, torch2.10/cu128, timm1.0.26, FP16, chunk0 and batch4; allow small nondeterministic FP16 reduction differences. No tolerance tuning.'}


def state_digest(model):
    h=hashlib.sha256()
    for name,t in sorted(model.state_dict().items()):
        a=t.detach().cpu().contiguous().numpy();h.update(json.dumps([name,str(a.dtype),list(a.shape)]).encode());h.update(a.tobytes())
    return h.hexdigest()


def verify_checkpoint(checkpoint):
    import torch
    require(checkpoint.get('arch')==ARCH and checkpoint.get('res')==224 and checkpoint.get('lab')==TARGETS and checkpoint.get('swa_over')==[8,9,7] and checkpoint.get('epoch')==[8,9,7],'Pinned checkpoint identity')
    require(checkpoint.get('model') and all(torch.is_tensor(v) and bool(torch.isfinite(v).all()) for v in checkpoint['model'].values()),'Finite checkpoint state')


def run(inputs,out,contract,model_code,utility_code):
    import importlib.metadata
    import random,sys,unicodedata,re,resource
    import numpy as np
    import torch,timm
    from torch import nn
    from torch.nn import functional as F
    from knee.fresh_reader_setup import dataset_root,notebook_root,exact_root
    out=Path(out);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    receipt={'status':'RUNNING','contract':contract,'optimizer_updates':0,'completed':{},'phase':'pins'}
    def write():
        receipt['elapsed_seconds']=time.monotonic()-started
        receipt['peak_host_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        save(out/'receipt.json',receipt)
    write()
    try:
        require(torch.cuda.is_available() and torch.cuda.get_device_name(0)=='Tesla T4' and torch.__version__=='2.10.0+cu128' and timm.__version__=='1.0.26','Historical T4 environment required')
        receipt['environment']={'python':sys.version,'torch':torch.__version__,'timm':timm.__version__,'device':'cuda:0','device_name':torch.cuda.get_device_name(0),
            'packages':{n:importlib.metadata.version(n) for n in ('numpy','pandas','pyarrow')},'fp16':True,'batch_studies':4,'win_chunk':0}
        require(fingerprint(model_code)==contract['model_code_sha256'] and fingerprint(utility_code)==contract['utility_code_sha256'],'Embedded definitions differ')
        ns={'torch':torch,'timm':timm,'np':np,'nn':nn,'F':F,'random':random,'Dataset':torch.utils.data.Dataset,'_roi_square_box':None}
        exec(compile(model_code,'<pinned-own224-definitions>','exec'),ns)
        metrics={'require':require,'fingerprint':fingerprint,'file_sha':file_sha,'MASK_SHA':TRUTH_SHA,'re':re,'unicodedata':unicodedata}
        exec(compile(utility_code,'<unchanged-metric-utilities>','exec'),metrics)
        mounts=[dataset_root(inputs,r,['all_ids.npy','all_masks.npy','all_vols.npy'] if i==0 else ['extra_ids.npy','extra_masks.npy','extra_vols.npy']) for i,r in enumerate(contract['corpora'])]
        arrays=[];ids=[];masks=[];receipt['cache_files']={}
        for i,root in enumerate(mounts):
            prefix='all' if i==0 else 'extra'
            for suffix in ('ids','masks','vols'):
                name=prefix+'_'+suffix+'.npy';p=root/name;pin=contract['cache_pins'][name]
                require(p.stat().st_size==pin['bytes'],'Pinned cache file size differs');digest=file_sha(p)
                if pin.get('sha256'):require(digest==pin['sha256'],'Pinned cache metadata differs')
                receipt['cache_files'][name]={'sha256':digest,'bytes':p.stat().st_size,'hash_status':'independently pinned' if pin.get('sha256') else 'initial volume hash observed; not independently pinned'}
            a=np.load(root/(prefix+'_vols.npy'),mmap_mode='r');m=np.load(root/(prefix+'_masks.npy'),allow_pickle=False);ii=np.load(root/(prefix+'_ids.npy'),allow_pickle=False).astype(str)
            validate_corpus(ii,m,a);arrays.append(a);ids.extend(ii.tolist());masks.append(m)
        require(len(ids)==4407 and len(set(ids))==4407,'Full cache coverage')
        masks=np.concatenate(masks);cut=len(arrays[0])
        class Volumes:
            shape=(4407,44,336,336)
            def __getitem__(self,index):return arrays[0][index] if index<cut else arrays[1][index-cut]
        owned=notebook_root(inputs,contract['checkpoint_source'],'trackA',['run_fold0.json','raptor_ft_fold0_swa.pt','oof_fold0.csv'])
        for name,pin in contract['owned_pins'].items():checked(owned/name,pin)
        code_root=out.parent/'own224_code'
        for name,pin in contract['private_pins'].items():checked(code_root/name,pin)
        folds=json.loads((code_root/'folds_v0.json').read_text())['folds'];excluded=set(json.loads((code_root/'excluded_ids.json').read_text())['quarantined_study_ids'])
        with (owned/'oof_fold0.csv').open(newline='') as stream:
            reader=csv.DictReader(stream);require(reader.fieldnames==['StudyInstanceUID',*TARGETS,'set'],'Historical OOF schema');saved=[r for r in reader if r['set']=='val']
        dev=population(ids,folds,excluded,[r['StudyInstanceUID'] for r in saved]);saved_p=probabilities(np.asarray([[float(r[t]) for t in TARGETS] for r in saved]),870)
        require(all(r['set']=='val' for r in saved),'Historical OOF partition')
        # Validate scoring membership before spending GPU time. Truth never enters the model.
        hpo=dataset_root(inputs,contract['hpo_source'],['train_labels_v1.parquet'])
        truth=checked(hpo/'train_labels_v1.parquet',TRUTH_SHA)
        y,m=metrics['load_binary_truth'](truth,dev,TARGETS)
        competition=exact_root([Path(inputs)/'rsna-knee-abnormality-detection',Path(inputs)/'competitions/rsna-knee-abnormality-detection'],['train.csv'])
        train_csv=checked(competition/'train.csv',TRAIN_SHA)
        with train_csv.open(newline='') as stream:rows=list(csv.DictReader(stream))
        train,gold=training_population(ids,folds,excluded,rows)
        require(len(train)==3475 and len(gold)==58 and not set(dev)&gold,'Historical train/gold coverage')
        groups,grouping=metrics['report_components'](rows,train,dev)
        receipt['membership']={'train':len(train),'development':len(dev),'gold_excluded':len(gold),'components':len(set(groups))}
        id2row={s:i for i,s in enumerate(ids)};labels={s:np.zeros(12,np.float32) for s in dev}
        checkpoint=torch.load(owned/'raptor_ft_fold0_swa.pt',map_location='cpu',weights_only=True);verify_checkpoint(checkpoint)
        random.seed(42);np.random.seed(42);torch.manual_seed(42)
        bb=ns['build_backbone'](ARCH,pretrained=False);model=ns['RaptorClassifier'](bb,F_dim=bb.num_features).to('cuda');model.win_chunk=0
        model.load_state_dict(checkpoint['model'],strict=True);model.eval();del checkpoint
        before=state_digest(model);receipt['state_before']=before;write();preds={};aggregate={}
        if contract.get('probe_only'):
            from knee.own224_parity_probe import probe
            receipt['phase']='parity_probe';write()
            result=probe(ns['StudyWindows'],Volumes(),masks,dev,id2row,labels,model,saved_p)
            require(state_digest(model)==before,'Model parameters/buffers changed')
            save(out/'probe.json',result)
            receipt.update(status='PROBE_COMPLETE',phase='complete',state_after=state_digest(model),
                           completed={'probe_studies':32},artifacts={'probe.json':file_sha(out/'probe.json')})
            write();return
        for k in (16,42):
            arm='k'+str(k);receipt['phase']=arm;arm_started=time.monotonic();stats={}
            ds=ns['StudyWindows']('',dev,id2row,labels,224,k,False,aug=False,norm='imagenet',roi=False);ds.vols=Volumes();ds.masks=masks
            plans=[]
            for sid in dev:
                centers,flags=center_plan(masks[id2row[sid]],ds,k);plans.append(centers)
                for name,value in flags.items():stats[name]=stats.get(name,0)+value
            values=[]
            with torch.no_grad():
                for position in range(0,len(dev),4):
                    xs=[ds[j][0] for j in range(position,min(position+4,len(dev)))]
                    require(all(tuple(x.shape)==(k,3,224,224) and bool(x.isfinite().all()) for x in xs),'Window schema')
                    with torch.autocast('cuda',dtype=torch.float16):p=torch.sigmoid(model(torch.stack(xs).to('cuda')).float()).cpu().numpy()
                    values.extend(p);receipt['completed'][arm]=position+len(xs)
                    if (position+len(xs))%100==0:write()
            p=probabilities(np.asarray(values),870);preds[arm]=p
            np.savez(out/(arm+'-private.npz'),ids=np.asarray(dev),probabilities=p)
            require(state_digest(model)==before,'Model parameters/buffers changed')
            aggregate[arm]={**stats,'studies':870,'seconds':time.monotonic()-arm_started,'plans_sha256':fingerprint(plans)}
            if k==16:
                receipt['saved_oof_parity']=parity(p,saved_p,enforce=contract.get('endpoint_policy','historical_parity')!='same_run_v2')
            write()
        receipt['phase']='endpoint'
        control,candidate=endpoint(dev,y,m,preds['k16'],preds['k42'])
        result={'control':metrics['metric_report'](y,m,control,TARGETS),'candidate':metrics['metric_report'](y,m,candidate,TARGETS),
            'paired_bootstrap':metrics['paired_bootstrap'](y,m,control,candidate,groups,draws=2000,seed=42,targets=TARGETS),'grouping':grouping,'window_aggregate':aggregate,
            'limits':'Exposed report-selected SWA fold; coverage-policy diagnostic, not independent generalization/public-score forecast. Legacy duplicate/masked/boundary centers preserved, not anatomical validity.'}
        result['macro_delta']=result['candidate']['macro_auc']-result['control']['macro_auc']
        result['per_target_delta']={t:result['candidate']['targets'][t]['auc']-result['control']['targets'][t]['auc'] if result['control']['targets'][t]['auc'] is not None else None for t in TARGETS}
        result['saved_oof_metrics']=metrics['metric_report'](y,m,saved_p,TARGETS)
        np.savez(out/'endpoint-private.npz',ids=np.asarray(dev),truth=y,masks=m,groups=np.asarray(groups))
        save(out/'result.json',result);receipt.update(status='PASS',phase='complete',state_after=state_digest(model),window_aggregate=aggregate,
            artifacts={p.name:file_sha(p) for p in out.iterdir() if p.name!='receipt.json'})
        receipt['cuda_peak_bytes']=torch.cuda.max_memory_allocated();write()
    except BaseException as error:
        receipt.update(status='FAIL',failure_type=type(error).__name__);write();raise
