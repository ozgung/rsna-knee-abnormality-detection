"""Frozen published fusion orchestration; qualified image modules stay unchanged."""
from collections import Counter
import csv
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import warnings

from knee.reference import canonical_bytes, sha256
from knee.reference_blend import read_test_contract, write_probabilities
from knee.mirror_submission import read_predictions

VERSION = 'versia-two-process-fusion-v1'
MAX_SECONDS = 32400
TARGETS = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA',
           'Effusion','Synovitis',"Baker's",'Contusion','Fracture']
WEIGHTS = [.25,.20,.20,.08,.25,.08,.08,.04,.03,.35,.30,.08]
POLICY = {'version':VERSION,'formula':'R((1-w)*R(float32(Q(M)))+w*R(C))',
          'Q':'clip(0,1).argsort(0).argsort(0)/max(1,N-1); NumPy default argsort',
          'R':'pandas ascending rank(method=average,pct=True)', 'targets':TARGETS,'convnext_weights':WEIGHTS,
          'execution':'mirror then ConvNeXt in distinct sequential processes; no retries',
          'cudnn_deterministic':False,'cudnn_allow_tf32':True,
          'convnext_test_batches':'official test order, batch4/workers4, final remainder; no cross-split batching',
          'mirror_cross_run_atol':1e-4,'convnext_cross_run_atol':1e-6,'cross_run_rtol':0,
          'visible_pair_rank_and_tie_parity':True,'within_run_atol':1e-6,'max_seconds':MAX_SECONDS}


def fuse(columns, ids, mirror, convnext_ids, convnext):
    """Published double-argsort and average-tie ranks, aligned to official IDs."""
    import numpy as np
    import pandas as pd
    if (columns!=['StudyInstanceUID',*TARGETS] or len(ids)<2 or len(set(ids))!=len(ids) or
            len(set(convnext_ids))!=len(convnext_ids) or set(ids)!=set(convnext_ids)):
        raise ValueError('Fusion requires both exact unique official ID/target sets')
    m,c=np.asarray(mirror,dtype=np.float64),np.asarray(convnext,dtype=np.float64)
    for a in (m,c):
        if a.shape!=(len(ids),12) or not np.isfinite(a).all() or np.any((a<0)|(a>1)):
            raise ValueError('Both finite component probabilities required')
    index={sid:i for i,sid in enumerate(convnext_ids)};c=c[[index[sid] for sid in ids]]
    if np.any([len(np.unique(c[:,j]))<=1 for j in range(12)]):
        raise ValueError('Published ConvNeXt constant-column guard failed')
    q=(m.clip(0,1).argsort(0).argsort(0)/max(1,len(ids)-1)).astype(np.float32)
    mr=pd.DataFrame(q,columns=TARGETS).rank(method='average',pct=True)
    cr=pd.DataFrame(c,columns=TARGETS).rank(method='average',pct=True)
    mixed=mr.copy()
    for target,w in zip(TARGETS,WEIGHTS):mixed[target]=(1-w)*mr[target]+w*cr[target]
    final=mixed.rank(method='average',pct=True)
    if any(final[t].nunique()<=1 for t in TARGETS):raise ValueError('Published fused constant-column guard failed')
    return final.to_numpy(dtype=np.float64)


def read_rank_csv(path):
    """Ranks can legitimately have entire rows at 1 or .5; never use raw fallback guards."""
    import numpy as np
    with Path(path).open(newline='') as handle:
        reader=csv.reader(handle);columns=next(reader);rows=list(reader)
    if columns!=['StudyInstanceUID',*TARGETS] or not rows or any(len(r)!=13 for r in rows):
        raise ValueError('Fused rank CSV schema differs')
    ids=[r[0] for r in rows];values=np.asarray([[float(v) for v in r[1:]] for r in rows])
    if len(set(ids))!=len(ids) or not np.isfinite(values).all() or np.any((values<0)|(values>1)):
        raise ValueError('Fused rank CSV coverage/range differs')
    return columns,ids,values


def compare_component(reference, actual, *, arm):
    """Cross-process absolute precision guard plus every visible rank/tie relation."""
    import numpy as np
    a,b=read_predictions(reference),read_predictions(actual)
    if arm not in ('mirror','convnext') or a[:2]!=b[:2] or a[0]!=['StudyInstanceUID',*TARGETS] or len(a[1])!=3:
        raise ValueError('Component visible official schema/order differs')
    x,y=np.asarray(a[2]),np.asarray(b[2]);error=float(np.max(np.abs(x-y)))
    if error>POLICY[arm+'_cross_run_atol']:raise ValueError('Component cross-run precision guard failed')
    if not np.array_equal(np.sign(x[:,None]-x[None,:]),np.sign(y[:,None]-y[None,:])):
        raise ValueError('Component visible rank/tie changed')
    return error


def verify_sources(project, q):
    if q.get('policy')!=POLICY:raise ValueError('Frozen fusion policy changed')
    for name,digest in q['runtime_sources'].items():
        if sha256((Path(project)/name).read_bytes())!=digest:raise ValueError('Fusion runtime source changed')


def run_process(command, *, deadline, env=None):
    """Kill/reap the entire component group on failure, including decoder workers."""
    remaining=deadline-time.monotonic()
    if remaining<=0:raise TimeoutError('Fusion whole-run deadline exhausted')
    process=subprocess.Popen(command,env=env,start_new_session=True)
    try:
        code=process.wait(timeout=max(.001,deadline-time.monotonic()))
        if code:raise RuntimeError('Fusion component process failed; no other-arm fallback')
    except BaseException:
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        process.wait()
        raise
    else:
        # A successful main process must not leave prefetched decoder workers
        # alive when the next arm starts. Their entire group belongs to this run.
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass


def _finite_positive(report, keys):
    return all(isinstance(report.get(k),(int,float)) and math.isfinite(report[k]) and report[k]>0 for k in keys)


def require_components(q, reports, paths, *, visible):
    """Fail closed before any composite predictions are produced."""
    from knee import mirror_input,versia
    schemas=[]
    for arm in ('mirror','convnext'):
        r=reports[arm];columns,ids,_=read_predictions(paths[arm]);schemas.append((columns,ids))
        if (r.get('status')!='PASS' or r.get('fusion_qualification_sha256')!=sha256(canonical_bytes(q)) or
                r.get('csv_sha256')!=sha256(Path(paths[arm]).read_bytes()) or r.get('visible') is not visible or
                r.get('completed_studies')!=len(ids) or r.get('test_studies')!=len(ids) or
                r.get('ordered_ids_sha256')!=sha256(canonical_bytes(ids)) or
                r.get('target_order_sha256')!=sha256(canonical_bytes(TARGETS)) or
                r.get('prediction_fallbacks')!=0 or r.get('codec_hashes')!=q['codecs'] or
                not _finite_positive(r,('component_elapsed_seconds','elapsed_seconds','peak_cuda_allocated_bytes',
                                          'peak_cuda_reserved_bytes','peak_host_rss_bytes','disk_free_bytes','output_disk_bytes')) or
                r['elapsed_seconds']>MAX_SECONDS):raise ValueError('Complete successful component evidence required')
        if visible and (len(ids)!=3 or not 0<=r.get('repeat_prediction_max_error',float('inf'))<=1e-6):
            raise ValueError('Both visible component repeats required')
        env=r.get('environment',{})
        if (env.get('python_major_minor')!=[3,12] or
                any(env.get('packages',{}).get(p)!=v for p,v in q['environment']['packages'].items())):
            raise ValueError('Component common pinned environment differs')
        if (r.get('backend_flags')!={'cudnn_benchmark':arm=='mirror','cuda_matmul_allow_tf32':arm=='mirror',
                'cudnn_deterministic':False,'cudnn_allow_tf32':True}):
            raise ValueError('Component backend isolation differs')
        if arm=='mirror':
            if (r.get('checkpoint_verified_sha256')!=q['mirror']['manifest']['checkpoint_sha256'] or
                    r.get('checkpoint_bytes')!=mirror_input.CHECKPOINT_BYTES or
                    r.get('input_policy')!=mirror_input.policy_contract() or
                    r.get('production_input_counts',{}).get('resource_errors')!=0):
                raise ValueError('Mirror scientific/resource evidence differs')
        else:
            expected=[min(4,len(ids)-i) for i in range(0,len(ids),4)]
            if (r.get('checkpoint_sha256')!=versia.WEIGHTS or r.get('policy')!=versia.policy_contract() or
                    r.get('input_counts',{}).get('resource_errors')!=0 or r.get('input_counts',{}).get('zero_studies')!=0 or
                    r.get('batch_sizes')!=expected or r.get('execution')!={k:versia.RECIPE[k] for k in ('batch_size','workers','prefetch_factor','pin_memory','multiprocessing')}):
                raise ValueError('ConvNeXt scientific/execution evidence differs')
            if visible and (r.get('public_tensor_max_error')!=0 or
                    sum(r.get('strict_coverage',{}).values())!=3 or
                    r.get('batch2_diagnostic',{}).get('studies')!=2 or
                    not 0<=r.get('batch2_diagnostic',{}).get('repeat_max_abs_error',float('inf'))<=1e-6 or
                    not math.isfinite(r.get('batch2_diagnostic',{}).get('batch3_max_abs_error',float('inf'))) or
                    r.get('batch2_diagnostic',{}).get('batch3_max_abs_error',-1)<0 or
                    r.get('batch2_diagnostic',{}).get('affects_production_predictions') is not False):
                raise ValueError('Visible public parity and remainder diagnostic required')
    if schemas[0]!=schemas[1] or schemas[0][0]!=['StudyInstanceUID',*TARGETS]:
        raise ValueError('Both component outputs must match exact official order')
    return schemas[0]


def combine(out, q, reports, paths, *, visible, started):
    import numpy as np
    out=Path(out);columns,ids=require_components(q,reports,paths,visible=visible)
    a,b=read_predictions(paths['mirror']),read_predictions(paths['convnext'])
    values=fuse(columns,ids,a[2],b[1],b[2])
    repeated=fuse(columns,ids,a[2],b[1],b[2])
    if not np.array_equal(values,repeated):raise ValueError('Frozen fusion repeat differs')
    if time.monotonic()-started>=MAX_SECONDS:raise TimeoutError('Whole-run fusion deadline exhausted')
    name='visible-predictions.csv' if visible else 'submission.csv'
    digest=write_probabilities(out/name,columns,ids,values)
    report={'status':'PASS','version':VERSION,'qualification_sha256':sha256(canonical_bytes(q)),
        'visible':visible,'test_studies':len(ids),'ordered_ids_sha256':sha256(canonical_bytes(ids)),
        'target_order_sha256':sha256(canonical_bytes(TARGETS)),'csv_sha256':digest,
        'components':reports,'component_csv_sha256':{k:sha256(Path(p).read_bytes()) for k,p in paths.items()},
        'repeat_fusion_max_error':0,'prediction_fallbacks':0,'elapsed_seconds':time.monotonic()-started,
        'max_seconds':MAX_SECONDS,'policy':POLICY,'runtime_estimate_policy':'informational_only',
        'peak_component_cuda_allocated_bytes':max(r['peak_cuda_allocated_bytes'] for r in reports.values()),
        'peak_component_cuda_reserved_bytes':max(r['peak_cuda_reserved_bytes'] for r in reports.values()),
        'disk_free_bytes':__import__('shutil').disk_usage(out).free,
        'output_disk_bytes':sum(p.stat().st_size for p in out.rglob('*') if p.is_file())}
    (out/'fusion-audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def require_visible(q, directory):
    """Local only: common visible output, component receipt bindings, exact reranking."""
    import numpy as np
    d=Path(directory)
    if (d/'fusion-failure.json').exists():raise ValueError('Failed common run cannot qualify')
    r=json.loads((d/'fusion-audit.json').read_text())
    if (r.get('status')!='PASS' or r.get('version')!=VERSION or r.get('visible') is not True or
            r.get('qualification_sha256')!=sha256(canonical_bytes(q)) or r.get('test_studies')!=3 or
            r.get('policy')!=POLICY or r.get('prediction_fallbacks')!=0 or r.get('repeat_fusion_max_error')!=0 or
            not _finite_positive(r,('elapsed_seconds','disk_free_bytes','output_disk_bytes')) or r['elapsed_seconds']>MAX_SECONDS):
        raise ValueError('New common visible qualification required')
    paths={'mirror':d/'mirror/visible-predictions.csv','convnext':d/'convnext/visible-predictions.csv'}
    columns,ids=require_components(q,r['components'],paths,visible=True)
    for arm in paths:
        if r.get('component_csv_sha256',{}).get(arm)!=sha256(paths[arm].read_bytes()):raise ValueError('Common component CSV changed')
        file='mirror-audit.json' if arm=='mirror' else 'convnext-audit.json'
        if json.loads((paths[arm].parent/file).read_text())!=r['components'][arm]:raise ValueError('Independent component receipt changed')
    c,i,p=read_rank_csv(d/'visible-predictions.csv')
    if (c!=columns or i!=ids or r.get('csv_sha256')!=sha256((d/'visible-predictions.csv').read_bytes()) or
            r.get('ordered_ids_sha256')!=sha256(canonical_bytes(ids)) or r.get('target_order_sha256')!=sha256(canonical_bytes(TARGETS)) or
            not np.array_equal(np.asarray(p),fuse(columns,ids,read_predictions(paths['mirror'])[2],ids,read_predictions(paths['convnext'])[2]))):
        raise ValueError('Common fused output/order changed')
    return {'report_sha256':sha256((d/'fusion-audit.json').read_bytes()),'csv_sha256':r['csv_sha256']},paths


def convnext_run(root, assets, sources, out, *, visible, started):
    """Same published16 Dataset, batch4/four workers and three-fold predictor."""
    import numpy as np
    import torch
    from knee import versia
    if not torch.cuda.is_available():raise RuntimeError('ConvNeXt fusion arm requires CUDA')
    warnings.filterwarnings('ignore',module=r'pydicom(?:\..*)?')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    protected=versia.definitions(sources);plain=versia.definitions(sources,protected=True) if visible else None
    prep,model_code,_=protected
    columns,ids,_=read_test_contract(root,rehearsal=visible)
    if columns!=['StudyInstanceUID',*model_code.LABELS]:raise ValueError('ConvNeXt official target order differs')
    table=versia.series_table(model_code,prep,root,'test',ids)
    class TestDataset(torch.utils.data.Dataset):
        def __len__(self):return len(ids)
        def __getitem__(self,index):
            arrays,counts=versia.study_arrays(protected,ids[index],table,Path(root)/'test_series')
            if visible:
                expected,_=versia.study_arrays(plain,ids[index],table,Path(root)/'test_series',strict_public=True)
                if not all(np.array_equal(a,b) for a,b in zip(arrays,expected)):
                    raise ValueError('Common visible/public ConvNeXt tensors differ')
            return (*[torch.from_numpy(a) for a in arrays],counts,versia.strict_eligible(counts))
    dl=torch.utils.data.DataLoader(TestDataset(),batch_size=4,shuffle=False,num_workers=4,pin_memory=True,
                                  prefetch_factor=2,multiprocessing_context='fork')
    iterator=iter(dl)  # CPU workers start before CUDA model construction, as qualified.
    models=versia.load_models(model_code,assets);torch.cuda.reset_peak_memory_stats()
    values=[];counts=Counter({k:0 for k in versia.COUNTERS});coverage=Counter();sizes=[];repeat=0.;diagnostic=None
    for x,mask,pos,events,eligible in iterator:
        if time.monotonic()-started>=MAX_SECONDS:raise TimeoutError('Fusion whole-run deadline exhausted')
        p=versia.predict(models,(x,mask,pos));sizes.append(len(p));values.extend(p)
        if visible:
            repeated=versia.predict(models,(x,mask,pos));repeat=max(repeat,float(np.max(np.abs(p-repeated))))
            if repeat>1e-6:raise ValueError('ConvNeXt within-run repeat differs')
            # Exercise the otherwise unseen remainder2 shape without replacing or
            # averaging the production batch3 predictions. Shape drift is diagnostic.
            small=versia.predict(models,(x[:2],mask[:2],pos[:2]))
            small_repeat=versia.predict(models,(x[:2],mask[:2],pos[:2]))
            small_error=float(np.max(np.abs(small-small_repeat)))
            if small_error>1e-6:raise ValueError('ConvNeXt remainder2 repeat differs')
            diagnostic={'studies':2,'repeat_max_abs_error':small_error,
                'batch3_max_abs_error':float(np.max(np.abs(p[:2]-small))),
                'rank_and_tie_equal':bool(np.array_equal(np.sign(p[0]-p[1]),np.sign(small[0]-small[1]))),
                'affects_production_predictions':False}
        counts.update({k:int(v.sum()) for k,v in events.items()});coverage.update('PASS' if bool(v) else 'ineligible_partial_input' for v in eligible)
        print(json.dumps({'event':'fusion_convnext_batch','completed':len(values),'total':len(ids)}),flush=True)
    del iterator,dl
    name='visible-predictions.csv' if visible else 'submission.csv'
    digest=write_probabilities(out/name,columns,ids,np.stack(values))
    return {'status':'PASS','visible':visible,'test_studies':len(ids),'completed_studies':len(values),
        'csv_sha256':digest,'ordered_ids_sha256':sha256(canonical_bytes(ids)),'target_order_sha256':sha256(canonical_bytes(model_code.LABELS)),
        'prediction_fallbacks':0,'checkpoint_sha256':versia.WEIGHTS,'policy':versia.policy_contract(),
        'execution':{k:versia.RECIPE[k] for k in ('batch_size','workers','prefetch_factor','pin_memory','multiprocessing')},
        'batch_sizes':sizes,'input_counts':dict(counts),'strict_coverage':dict(coverage) if visible else None,
        'public_tensor_max_error':0 if visible else None,'repeat_prediction_max_error':repeat if visible else None,
        'batch2_diagnostic':diagnostic,
        'elapsed_seconds':time.monotonic()-started,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
        'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),
        'peak_host_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'worker_maxrss_bytes':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024,
        'disk_free_bytes':__import__('shutil').disk_usage(out).free,'output_disk_bytes':(out/name).stat().st_size}


def component_main(payload, arm, deadline):
    """Only reached in a fresh process; no checkpoint/source download or device retry."""
    from knee.mirror_submission import run as mirror_run, VERSION as MV
    data=json.loads(Path(payload).read_text());c=data['contract'];q=c['qualification']
    if sha256(canonical_bytes(q))!=c['qualification_sha256']:raise ValueError('Fusion qualification changed')
    verify_sources(data['project'],q)
    remaining=deadline-time.monotonic()
    if remaining<=0:raise TimeoutError('Fusion component startup exhausted deadline')
    def expired(signum,frame):raise TimeoutError('Fusion whole-run deadline exhausted')
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,remaining)
    import torch
    torch.backends.cudnn.deterministic=False;torch.backends.cudnn.allow_tf32=True
    env={'python_major_minor':list(sys.version_info[:2]),'packages':{p:importlib.metadata.version(p) for p in q['environment']['packages']}}
    if env['python_major_minor']!=[3,12] or env['packages']!=q['environment']['packages']:raise ValueError('Fusion environment changed')
    out=Path(data['output'])/arm;before=time.monotonic();started=deadline-MAX_SECONDS
    if arm=='mirror':
        mq={'manifest':q['mirror']['manifest'],'input_policy':q['mirror']['input_policy'],'fusion_qualification_sha256':c['qualification_sha256']}
        mc={'version':MV,'qualification':mq,'qualification_sha256':sha256(canonical_bytes(mq)),
            'visible':c['visible'],'max_seconds':MAX_SECONDS}
        r=mirror_run(data['root'],data['checkpoint'],data['mirror_public'],data['mirror_decoder'],mc,out,started=started)
        filename='mirror-audit.json'
    elif arm=='convnext':
        r=convnext_run(data['root'],data['assets'],data['versia_sources'],out,visible=c['visible'],started=started)
        filename='convnext-audit.json'
    else:raise ValueError('Unknown fusion component')
    env.update(gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version())
    r.update(fusion_qualification_sha256=c['qualification_sha256'],component_elapsed_seconds=time.monotonic()-before,
             codec_hashes=q['codecs'],environment=env,backend_flags={'cudnn_benchmark':torch.backends.cudnn.benchmark,
             'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
             'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32})
    (out/filename).write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--component',choices=['mirror','convnext'],required=True)
    p.add_argument('--payload',required=True);p.add_argument('--deadline',type=float,required=True)
    args=p.parse_args()
    try:component_main(args.payload,args.component,args.deadline)
    except BaseException as error:
        print(json.dumps({'event':'fusion_component_failed','component':args.component,'failure_type':type(error).__name__}),flush=True)
        sys.exit(1)
