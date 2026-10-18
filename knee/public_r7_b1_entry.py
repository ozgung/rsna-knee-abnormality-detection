"""Selection-B probe B1: heliosli R7 alone (no OAI-trained weights attached or run).

The compact-slot worker is the exact v7-qualified R7 worker bytecode (knee.public_r7_recovery_entry),
rebound to this module's globals (version, launch modules, rehearsal hook). The old arm is never
launched and its weights are not attached. Recovery is the frozen capped arm-mean policy applied
to the single R7 arm: classified data-local rows (at most floor(3% of studies)) take the R7 genuine
mean, then the submission is the R7 average percentile rank over all actual rows (weight 1.0; a
monotone transform of the filled R7 probabilities, so AUC equals submitting those probabilities).
"""
import json
import math
from pathlib import Path
import time
import types
import knee.public_r7_recovery_entry as R7
import knee.public_s6_recovery_entry as base

globals().update({k:v for k,v in vars(R7).items() if not k.startswith('__')})

VERSION='r7-only-capped-recovery-b1-v1'
ENTRY_MODULE='knee.public_r7_b1_entry'
REHEARSAL_MODULE='knee.public_r7_b1_rehearsal'
RECOVERY_RECIPE_B1='r7-only-capped-arm-mean-rank-v1'
METHOD={**R7.METHOD,'old':None,'blend':[0.,1.],'raptor_per_finding':None,'variant':'B1: R7 alone; no Raptor/OAI-trained weights attached or run','recovery_recipe':RECOVERY_RECIPE_B1}
ARMS=('s6',)
rehearsal_hook=None


def recover(r7,r7_missing,*,fatal_errors=()):
    """Frozen capped arm-mean policy, one arm: fill classified rows with the R7 genuine mean
    ('%.9g'), then the R7 average percentile rank over the entire actual population."""
    import numpy as np
    import pandas as pd
    if fatal_errors:raise ValueError('FATAL_ERRORS_REQUIRE_ABORT')
    s=np.asarray(r7);sm=np.asarray(r7_missing);n=len(s)
    if n==0 or s.shape!=(n,12):raise ValueError('ARM_SCHEMA')
    if sm.shape!=(n,) or sm.dtype!=bool:raise ValueError('EXPLICIT_ROW_MASK_REQUIRED')
    cap=math.floor(3*n/100);count=int(sm.sum())
    if count>cap:raise ValueError('UNION_RECOVERY_CAP')
    genuine=np.asarray(s[~sm],dtype=np.float64)
    if not len(genuine):raise ValueError('GENUINE_SUCCESS_REQUIRED_s6')
    if not np.isfinite(genuine).all() or not ((genuine>=0)&(genuine<=1)).all():raise ValueError('GENUINE_PROBABILITIES_s6')
    genuine=stackers.quantize9(genuine);mean=stackers.quantize9(genuine.mean(axis=0,dtype=np.float64)[None])[0]
    filled=np.empty((n,12),dtype=np.float64);filled[~sm]=genuine;filled[sm]=mean
    rank=pd.DataFrame(filled).rank(method='average',pct=True).to_numpy()
    return {'recipe':RECOVERY_RECIPE_B1,'population':n,'cap':cap,'union_recovered':count,'means':{'s6':mean.tolist()},
            'statuses':{'s6':np.where(sm,'recovered','genuine').tolist()},'s6':filled,'s6_rank':rank,'blend':rank}


def projection(cfg,phases,reports,setup,merge_seconds):
    """Default projection policy with the single R7 phase."""
    windows=sum(x['windows'] for r in reports['s6'] for x in r['records'])
    d=cfg['development_windows'];require(len(d)==870 and all(w>0 for w in d),'FROZEN870_WINDOW_DISTRIBUTION');require(windows>0,'MEASURED_THROUGHPUT')
    production=max(sum(x['production_seconds'] for x in r['records']) for r in reports['s6']);require(0<production<=phases['s6']['wall_seconds'],'MEASURED_PHASE_PRODUCTION')
    overhead=phases['s6']['wall_seconds']-production;mean=sum(d)/870;scenarios=[]
    for multiplier in (1.,1.25,1.5):
        for imbalance in (1.,1.15,1.3):
            scenarios.append({'hidden_window_multiplier':multiplier,'shard_imbalance':imbalance,'seconds':setup+merge_seconds+overhead+production*(1300*mean*multiplier/windows)*imbalance})
    return {'population':1300,'window_distribution':{'count':870,'total':sum(d),'mean':mean,'max':max(d)},'scenarios':scenarios,
            'measured_production_wall_seconds':{'s6':production},'measured_fixed_phase_overhead_seconds':{'s6':overhead},
            'conservative_seconds':next(x['seconds'] for x in scenarios if x['hidden_window_multiplier']==1.25 and x['shard_imbalance']==1.15),
            'policy':'Default policy, R7 phase only: slower shard production, all phase overhead once, 25% more windows and 15% imbalance.'}


def run(root,cfg,out,*,started):
    """Default run core with only the R7 arm; the old arm is never launched."""
    import subprocess
    out=Path(out);out.mkdir();csv_created=False;r={'status':'RUNNING','version':VERSION,'mode':cfg['mode'],'binding':cfg['binding']};save(out/'receipt.json',r)
    try:
        columns,visible,records=test_contract(root,LABELS,visible=cfg['mode']=='rehearsal');paths={s:str(Path(root)/'test_series') for s in visible};ids=list(visible)
        if cfg['mode']=='rehearsal':
            probes=cfg['probes'];require(len(probes)==36 and not set(probes)&set(ids),'FROZEN32_DISTINCT_PROBES');training=read_series(Path(root)/'train_series.csv')
            for sid in probes:require(sid in training,'PROBE_METADATA');records[sid]=training[sid];paths[sid]=str(Path(root)/'train_series')
            require(len(cfg['probe_order'])==36 and set(cfg['probe_order'])==set(probes),'FROZEN36_CANONICAL_ORDER');ids=cfg['probe_order']+ids
        cfg={**cfg,'staged_ids':ids,'assignment':assignments(ids)};save(out/'assignment.json',{'ids':ids,'assignment':cfg['assignment'],'policy':METHOD['assignment']})
        staged=out/'images-only';cfg.update(staged=str(staged),stage_binding=stage(staged,ids,records,paths));r['stage_binding']=cfg['stage_binding'];save(out/'receipt.json',r)
        inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name','--format=csv,noheader'],text=True);boards=[]
        for line in inventory.splitlines():
            index,uuid,name=[x.strip() for x in line.split(',')];boards.append({'index':int(index),'uuid':uuid,'name':name})
        require([b['index'] for b in boards]==[0,1] and len({b['uuid'] for b in boards})==2 and all(b['name']=='Tesla T4' for b in boards),'EXACT_TWO_T4_INVENTORY');cfg['boards']=boards
        r.update(boards=boards,visible_ids=visible,setup_seconds=time.monotonic()-started);save(out/'receipt.json',r);phases={};reports={};arms={}
        for arm in ARMS:
            reports[arm],phases[arm]=launch(cfg,out,arm);arms[arm]=merge(ids,[(x,out/f'board{i}-{arm}'/arm/'predictions.npz') for i,x in enumerate(reports[arm])],arm)
            r.update(phases=phases,workers=reports);save(out/'receipt.json',r)
        merge_start=time.monotonic();index={s:i for i,s in enumerate(ids)};indices=[index[s] for s in visible]
        recovery=recover(arms['s6'][0][indices],arms['s6'][1][indices])
        prediction=recovery['blend'];r['recovery']={k:v for k,v in recovery.items() if k not in ('s6','s6_rank','blend')}
        csv_path=out/'visible-predictions.csv' if cfg['mode']=='rehearsal' else Path('/kaggle/working/submission.csv')
        require(time.monotonic()-started<MAX_SECONDS,'ACTUAL_NINE_HOUR_LIMIT_BEFORE_CSV')
        sha=write_csv(csv_path,columns,visible,prediction);csv_created=True
        r.update(status='PASS',completed=len(ids),completed_visible=len(visible),completed_probes=len(ids)-len(visible),completed_s6_probes=32 if cfg['mode']=='rehearsal' else 0,zero_failed=recovery['union_recovered']==0,zero_imputed=recovery['union_recovered']==0,csv_sha256=sha,csv_path=str(csv_path),
            all_required_recipe_gates=True,all_input_state_probability_gates=recovery['union_recovered']==0,sentinel_ids={'s6':cfg.get('s6_sentinels',[])},elapsed_seconds=time.monotonic()-started,
            projection=projection(cfg,phases,reports,r['setup_seconds'],time.monotonic()-merge_start) if cfg['mode']=='rehearsal' else None)
        require(r['elapsed_seconds']<MAX_SECONDS,'ACTUAL_NINE_HOUR_LIMIT')
        if cfg['mode']=='rehearsal':require_rehearsal(r,cfg['binding'],csv_path)
        save(out/'receipt.json',r);return r
    except BaseException as e:
        if csv_created and cfg['mode']=='entry' and csv_path.exists():csv_path.replace(out/'rejected-submission.csv')
        r.update(status='INTERRUPTED' if isinstance(e,(KeyboardInterrupt,InterruptedError,TimeoutError)) else 'FAIL',failure_type=type(e).__name__,elapsed_seconds=time.monotonic()-started)
        save(out/'failure-private.json',{'type':type(e).__name__,'detail':str(e)});save(out/'receipt.json',r);raise


def require_r7_workers(r,binding):
    """The v7 R7-slot gates, unchanged (shared by B1 and B2)."""
    import numpy as np
    s60=r.get('workers',{}).get('s6',[{}])[0]
    guarded_native=s60.get('guarded_native_availability_stress',{})
    require(guarded_native.get('status')=='PASS' and guarded_native.get('guarded') is True and guarded_native.get('case_count')==2 and set(guarded_native.get('cases',{}))=={'absent','empty'},'GUARDED_NATIVE_PARTIAL_TWO')
    native=s60.get('native_availability_stress',{})
    require(native.get('status')=='PASS' and native.get('case_count')==2 and native.get('uses_raw_data') is False and native.get('uses_labels') is False and set(native.get('cases',{}))=={'absent','empty'} and all(x.get('status')=='PASS' and x.get('usable_series')==1 and x.get('slices',0)>0 for x in native['cases'].values()),'REHEARSAL_NATIVE_AVAILABILITY_STRESS')
    r7=binding['r7'];fluid_doc,state_doc=r7['fluid'],r7['state']
    require('s6' in r.get('phases',{}) and r['phases']['s6']['wall_seconds']>0 and r['phases']['s6']['peaks']['host_rss_bytes']>0 and r['phases']['s6']['peaks']['output_disk_bytes']>0 and r['phases']['s6']['minimum_disk_free_bytes']>0,'CONCURRENT_RESOURCE_MEASUREMENT')
    rr=r.get('workers',{}).get('s6',[]);require(len(rr)==2,'REHEARSAL_WORKERS')
    require(all(x.get('recovery_injection',{}).get('status')=='PASS' and x['recovery_injection'].get('failure_status')=='data_local_failure' and x['recovery_injection'].get('injected_rows_genuine') is False and x['recovery_injection'].get('state_healthy') is True for x in rr),'ACTUAL_CLASSIFIED_RECOVERY_INJECTIONS')
    require(len({x['state_before'] for x in rr})==1 and rr[0]['state_before'] is not None,'CROSS_BOARD_R7_STATE')
    for board,x in enumerate(rr):
        require(x['status']=='PASS' and x['board']==board and x['state_before']==x['state_after'] and x['completed']==len(x['assigned']) and x['peak_cuda_bytes']>0 and x['peak_rss_bytes']>0 and x['zero_failed'] is True and x['zero_imputed'] is True and x['fp32_recoveries']==0,'REHEARSAL_WORKER_STATE_RESOURCE')
        require(x.get('slot')=='r7' and x.get('r7_manifest',{}).get('bundle_id')==R7_BUNDLE_ID and x.get('fluid_selfcheck',{}).get('status')=='PASS'
                and x.get('stackers',{}).get('adopted')=={'fluid':list(stackers.FLUID_ADOPTED),'state':list(stackers.STATE_ADOPTED)}
                and x['stackers'].get('fluid_sha256')==R7_ASSET_SHA['fluid_stacker.json'] and x['stackers'].get('state_sha256')==R7_ASSET_SHA['state_stacker.json'],'QUALIFIED_R7_SLOT')
        require_compiled(x.get('compile',{}));require(x.get('fluid_guard',{}).get('site')==FLUID_SITE and x['fluid_guard'].get('unknown')=='fatal','R7_FLUID_BOUNDARY_INSTALLED')
        require(all((row.get('fluid') or {}).get('status')=='ok' or (row.get('fluid') or {}).get('status') in FLUID_MISSING_STATUSES or str((row.get('fluid') or {}).get('status','')).startswith(FLUID_NO_REGION_PREFIX) for row in x['records']+x['sentinels']),'R7_FLUID_STATUSES')
        require(x['environment']['torch']==binding['environment']['torch'] and x['environment']['packages']==binding['environment']['packages'],'QUALIFIED_ENVIRONMENT_BINDING')
        require(x['environment']['physical_uuid']==r['boards'][board]['uuid'] and x['environment']['cuda_device_count']==1,'ISOLATED_BOARD_RUNTIME')
        require(len(x['sentinels'])==4 and [s['id'] for s in x['sentinels']]==r['sentinel_ids']['s6'] and all(s['strict_reload'] for s in x['sentinels']),'BOTH_BOARD_SENTINELS')
        for s in x['sentinels']:
            require(all(s['input'].get(k)==v for k,v in binding['sentinel_inputs']['s6'][s['id']].items()),'BOUND_SENTINEL_INPUTS')
            p=np.array(s['probabilities']);q=np.array(s['reloaded_probabilities'])
            require(binding['sentinel_probabilities']['s6'][s['id']] is None and float(np.max(np.abs(p-q)))<=1e-6 and s.get('pilot_input_parity') is True,'R7_SENTINEL_RELOAD_GATE')
            for field,target in [('logits',s['model_probabilities']),('reloaded_logits',s['reloaded_model_probabilities'])]:
                z=np.asarray(s[field],dtype=np.float32);require(z.shape==(4,12) and np.isfinite(z).all(),'SENTINEL_RAW_FOUR_LOGITS')
                mean=z.mean(0).astype(np.float64);sigmoid=np.exp(-np.logaddexp(0.,-mean))
                require(float(np.max(np.abs(sigmoid-np.asarray(target))))<=1e-6,'SENTINEL_MEAN_LOGIT_SIGMOID')
            recompute_stackers(s,fluid_doc,state_doc)
    for a,b in zip(rr[0]['sentinels'],rr[1]['sentinels']):
        require(a['input']==b['input'] and max(abs(x-y) for x,y in zip(a['probabilities'],b['probabilities']))<=CROSS_BOARD_TOLERANCE['s6'],'CROSS_BOARD_SENTINEL_PARITY')
    s6rows={row['id']:row for worker in rr for row in worker['records']}
    require(len(s6rows)==39,'COMPLETE39_MECHANICS')
    for sid in binding['s6_probe_ids']:require(s6rows[sid].get('pilot_input_parity') is True,'ALL32_R7_PILOT_INPUT_PARITY')
    for row in s6rows.values():recompute_stackers(row,fluid_doc,state_doc)
    for sid in r['visible_ids']:
        g=s6rows[sid].get('heliosli_reference',{})
        require(g.get('atol')==r7['visible_atol'] and g.get('within_atol') is True and 0<=g.get('max_abs',math.inf)<=r7['visible_atol'],'R7_HELIOSLI_VISIBLE_PARITY')
    return s6rows


def require_common(r,binding,csv_path,version,method):
    import csv
    from knee.public_s6_qualification import UID
    require(binding.get('version')==version and binding.get('method')==method and binding.get('input_policy')==INPUT_POLICY and binding.get('r7',{}).get('recipe')==R7_RECIPE,'EXACT_FROZEN_METHOD')
    require(r.get('status')=='PASS' and r.get('version')==version and r.get('mode')=='rehearsal' and r.get('binding')==binding,'PASS_EXACT_REHEARSAL_REQUIRED')
    require(r.get('completed')==39 and r.get('completed_visible')==3 and r.get('completed_probes')==36 and r.get('completed_s6_probes')==32 and r.get('all_input_state_probability_gates') is True and r.get('zero_failed') is True and r.get('zero_imputed') is True,'REHEARSAL_COMPLETE_GATES')
    require(len(r.get('boards',[]))==2 and len({x['uuid'] for x in r['boards']})==2 and all(x['name']=='Tesla T4' for x in r['boards']),'TWO_PHYSICAL_T4_GATES')
    require(0<r.get('projection',{}).get('conservative_seconds',math.inf)<MAX_SECONDS and 0<r.get('elapsed_seconds',math.inf)<MAX_SECONDS,'NINE_HOUR_PROJECTION_GATE')
    require(r['projection']['conservative_seconds']<=R7_PROJECTION_GATE_SECONDS,'R7_HELIOSLI_HORIZON_PROJECTION_GATE')
    require(r.get('csv_sha256')==file_sha(csv_path),'REHEARSAL_CSV_BINDING')
    with Path(csv_path).open(newline='') as h:rows=list(csv.reader(h))
    require(rows[0]==[UID]+LABELS and len(rows)==4 and [x[0] for x in rows[1:]]==r['visible_ids'],'VISIBLE_CSV_COVERAGE')
    probabilities([[float(x) for x in row[1:]] for row in rows[1:]],3)


def require_rehearsal(r,binding,csv_path):
    """B1 contract: the v7 R7 gates; no old arm launched."""
    require_common(r,binding,csv_path,VERSION,METHOD)
    require(set(r.get('workers',{}))=={'s6'} and set(r.get('phases',{}))=={'s6'} and r.get('sentinel_ids',{}).keys()=={'s6'},'R7_ONLY_WORKERS')
    require_r7_workers(r,binding)


def r7_parity(out,cfg):
    """Rehearsal diagnostic: visible R7-arm probabilities vs heliosli's ours_sub and entry v2's R7 arm."""
    import numpy as np
    out=Path(out);ids=cfg['binding']['visible_ids'];rows={}
    for b in range(2):
        with np.load(out/f'board{b}-s6'/'s6'/'predictions.npz',allow_pickle=False) as a:rows.update(zip(a['study_ids'].tolist(),a['probabilities']))
    ours=np.array([rows[s] for s in ids],dtype=np.float64);result={}
    for name,ref in (('r7_vs_heliosli_ours_sub',cfg['heliosli_reference']['r7']),('r7_vs_entry_v2_r7',cfg['r7_entry_reference'])):
        d=np.abs(ours-np.array([ref[s] for s in ids],dtype=np.float64));result[name]={'max_abs':float(d.max()),'max_abs_per_target':dict(zip(LABELS,d.max(0).tolist()))}
    save(out/'r7-parity.json',result);return result


old_worker=None
heliosli_parity=None


def _bind(function):
    clone=types.FunctionType(function.__code__,globals(),function.__name__,function.__defaults__,function.__closure__)
    clone.__kwdefaults__=function.__kwdefaults__;clone.__doc__=function.__doc__;clone.__qualname__=function.__qualname__
    return clone


r7_worker=_bind(R7.r7_worker)
launch=_bind(R7.launch)
merge=_bind(base.merge)


if __name__=='__main__':
    import argparse
    import os
    require(Path(__file__).resolve()==(Path(os.environ['KNEE_ENTRY_PROJECT'])/'knee/public_r7_b1_entry.py').resolve(),'EXACT_FRESH_SOURCE_IMPORT')
    p=argparse.ArgumentParser();p.add_argument('--worker',required=True);a=p.parse_args();cfg=json.loads(Path(a.worker).read_bytes())
    require(cfg['arm']=='s6','B1_R7_ONLY');r7_worker(a.worker)
