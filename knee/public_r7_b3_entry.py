"""Selection-B probe B3: B2 plus own dr-full at a fixed 0.2 share of the Dread (old-arm) slot.

Arms (2xT4 each, sequential phases):
  old : Dread CC0 Raptor v4 K62, the exact B0/B2 worker bytecode;
  dr  : own raptor64 dr-full (Dread v4_full init, all 4,345 studies, SWA epochs 9-11), K42. Its
        training/export contract equals the B0 Dread reader and eval windows (same prepare_study
        64-slot volume, _eval_centers, /255, bilinear 336->384, ImageNet norm), so it runs the same
        B0 old-arm worker bytecode with only its 'old' string constant replaced by 'dr', in a
        namespace holding the dr-full pins (CPU-qualified: scripts/qualify_b3_drfull_cpu.py);
  s6  : heliosli R7, the exact v7 worker bytecode.
Blend: the frozen capped arm-mean algorithm generalized to three arms (union cap floor(3%), each
arm's genuine-mean fill, whole-population average percentile ranks), then
0.5 x (0.8 rank(Dread) + 0.2 rank(dr-full)) + 0.5 rank(R7) = 0.4 / 0.1 / 0.5 per finding.
No nartaa/OAI-trained weights are attached.
"""
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import types
import knee.public_r7_recovery_entry as R7
import knee.public_s6_recovery_b0 as B0
import knee.public_s6_recovery_entry as base
import knee.public_r7_b1_entry as B1
import knee.public_r7_b2_entry as B2

globals().update({k:v for k,v in vars(B2).items() if not k.startswith('__')})

VERSION='r7-dread-drfull-capped-recovery-b3-v1'
ENTRY_MODULE='knee.public_r7_b3_entry'
REHEARSAL_MODULE='knee.public_r7_b3_rehearsal'
DRFULL_CHECKPOINT_SHA256='35b7fc2cc95155b13271d3bc43ebd4931a915eaa490c572d2fd02d3cac2a061c'
DRFULL_CHECKPOINT_BYTES=292826627
DRFULL_CHECKPOINT_FILE='raptor_ft_fold0_swa.pt'
DRFULL_DATASET='ozgung/knee-raptor64-dr-full-fold0-final-20261013'
DRFULL_DATASET_VERSION=1
# scripts/qualify_b3_drfull_cpu.py receipt (fastgen torch 2.10.0 CPU; the same method reproduces B0's Dread pins).
DRFULL_STATE_SHA256='a3dbd31bdd65f306e987808708ef7afe691f9fb61e67fb4ddd86843cd7ce7812'
DRFULL_GRID_SHA256='67ae62b6e4ba164abf4757e0948b4fa9cc410b587538dd19ce7be58186d465bd'
DRFULL_WINDOWS=42
DR_PROFILE={**B0.OLD_PROFILE,'checkpoint_sha256':DRFULL_CHECKPOINT_SHA256,'checkpoint_bytes':DRFULL_CHECKPOINT_BYTES,'windows':DRFULL_WINDOWS,
            'model':'own raptor64 dr-full: Dread v4_full init, original head, all 4,345 studies, SWA epochs 9-11','export_k':DRFULL_WINDOWS}
ARMS=('old','dr','s6')
OLD_SLOT={'old':.8,'dr':.2}
ARM_WEIGHTS={'old':.5*OLD_SLOT['old'],'dr':.5*OLD_SLOT['dr'],'s6':.5}
RECOVERY_RECIPE_B3='r7-dread-drfull-fixed40-10-50-capped-arm-mean-v1'
METHOD={**B2.METHOD,'dr':'own raptor64 dr-full K42 (B0 raptor reader and eval windows), no TTA, CUDA fp16','dr_profile':DR_PROFILE,'old_slot':OLD_SLOT,
        'arm_weights':ARM_WEIGHTS,'recovery_recipe':RECOVERY_RECIPE_B3,
        'variant':'B3: Dread K62 0.8 + dr-full K42 0.2 in the old slot, then 0.5/0.5 with R7; no nartaa/OAI-trained weights'}
CROSS_BOARD_TOLERANCE={**R7.CROSS_BOARD_TOLERANCE,'dr':1e-6}
rehearsal_hook=None


def dr_profile_gate(public,manifest):
    """dr-full profile: exactly the B0 raptor recipe at K42 with the private dr-full checkpoint."""
    recipe=manifest.get('recipe',{})
    require(manifest.get('profile')=='raptor' and manifest.get('checkpoint_sha256')==DRFULL_CHECKPOINT_SHA256 and manifest.get('checkpoint_file')==DRFULL_CHECKPOINT_FILE
            and manifest.get('weight_dataset')==DRFULL_DATASET and manifest.get('decoded_fastread_sha256') is None,'B3_DRFULL_MANIFEST')
    require(recipe.get('windows')==DRFULL_WINDOWS and recipe.get('input_res')==B0.RAPTOR_INPUT and recipe.get('image_size')==B0.RAPTOR_IMAGE and recipe.get('input_mode')=='resize'
            and recipe.get('slice_span')==B0.OLD_PROFILE['slice_span'] and [s[2] for s in recipe.get('slots',[])]==B0.OLD_PROFILE['slot_sizes'],'B3_DRFULL_K42_RECIPE')
    if public is not None:
        require(public.K_EVAL==DRFULL_WINDOWS and public.IMG==B0.RAPTOR_IMAGE and public.MAXS==B0.RAPTOR_SLOTS and [list(s) for s in public.SLOTS]==recipe['slots']
                and list(public.LAB)==LABELS and public.NORM=='imagenet','B3_DRFULL_SOURCE_CONSTANTS')
    return dict(DR_PROFILE)


def _swap_old_constant(code):
    """Replace the exact string constant 'old' with 'dr' (recursively); bytecode unchanged."""
    consts=tuple('dr' if c=='old' else (_swap_old_constant(c) if isinstance(c,types.CodeType) else c) for c in code.co_consts)
    return code.replace(co_consts=consts)


# dr namespace: B3 globals with the dr-full pins under the names the B0 worker reads.
DR=dict(globals())
DR.update(DREAD_CHECKPOINT_SHA256=DRFULL_CHECKPOINT_SHA256,DREAD_CHECKPOINT_BYTES=DRFULL_CHECKPOINT_BYTES,DREAD_STATE_SHA256=DRFULL_STATE_SHA256,
          DREAD_GRID_SHA256=DRFULL_GRID_SHA256,RAPTOR_WINDOWS=DRFULL_WINDOWS,OLD_PROFILE=DR_PROFILE,raptor_profile_gate=dr_profile_gate,
          raptor62_gate=lambda sid,p,counters,cfg:None)
for _name in ('raptor_windows','raptor_forward'):
    _f=getattr(B0,_name);DR[_name]=types.FunctionType(_f.__code__,DR,_f.__name__,_f.__defaults__,_f.__closure__)
_dr_old_worker=types.FunctionType(_swap_old_constant(B0.old_worker.__code__),DR,'dr_worker',B0.old_worker.__defaults__,B0.old_worker.__closure__)


def dr_worker(config_path):
    """The B0 old-arm worker on the dr-full arm (hook and version resolved from this module)."""
    DR.update(rehearsal_hook=rehearsal_hook,VERSION=VERSION)
    return _dr_old_worker(config_path)


def recover(control,dr,r7,control_missing,dr_missing,r7_missing,*,fatal_errors=()):
    """Frozen capped arm-mean recovery generalized to three arms (R7 at '%.9g')."""
    import numpy as np
    import pandas as pd
    if fatal_errors:raise ValueError('FATAL_ERRORS_REQUIRE_ABORT')
    arms=[('old320',np.asarray(control),np.asarray(control_missing),None),('dr',np.asarray(dr),np.asarray(dr_missing),None),('s6',np.asarray(r7),np.asarray(r7_missing),stackers.quantize9)]
    n=len(arms[0][1])
    if n==0 or any(a.shape!=(n,12) for _,a,_,_ in arms):raise ValueError('ARM_SCHEMA')
    if any(m.shape!=(n,) or m.dtype!=bool for _,_,m,_ in arms):raise ValueError('EXPLICIT_ROW_MASK_REQUIRED')
    cap=math.floor(3*n/100);union=arms[0][2]|arms[1][2]|arms[2][2];count=int(union.sum())
    if count>cap:raise ValueError('UNION_RECOVERY_CAP')
    filled={};means={};ranks={}
    for name,a,missing,q in arms:
        genuine=np.asarray(a[~missing],dtype=np.float64)
        if not len(genuine):raise ValueError('GENUINE_SUCCESS_REQUIRED_'+name)
        if not np.isfinite(genuine).all() or not ((genuine>=0)&(genuine<=1)).all():raise ValueError('GENUINE_PROBABILITIES_'+name)
        if q is not None:genuine=q(genuine)
        mean=genuine.mean(axis=0,dtype=np.float64)
        if q is not None:mean=q(mean[None])[0]
        f=np.empty((n,12),dtype=np.float64);f[~missing]=genuine;f[missing]=mean
        filled[name]=f;means[name]=mean.tolist();ranks[name]=pd.DataFrame(f).rank(method='average',pct=True).to_numpy()
    blend=ARM_WEIGHTS['old']*ranks['old320']+ARM_WEIGHTS['dr']*ranks['dr']+ARM_WEIGHTS['s6']*ranks['s6']
    return {'recipe':RECOVERY_RECIPE_B3,'population':n,'cap':cap,'union_recovered':count,'means':means,'weights':ARM_WEIGHTS,
            'statuses':{name:np.where(m,'recovered','genuine').tolist() for name,_,m,_ in arms},
            'control':filled['old320'],'dr':filled['dr'],'s6':filled['s6'],'control_rank':ranks['old320'],'dr_rank':ranks['dr'],'s6_rank':ranks['s6'],'blend':blend}


def projection(cfg,phases,reports,setup,merge_seconds):
    """Default projection policy over three phases: Raptor arms scale by studies, R7 by windows."""
    n=sum(len(r['records']) for r in reports['old']);windows=sum(x['windows'] for r in reports['s6'] for x in r['records'])
    d=cfg['development_windows'];require(len(d)==870 and all(w>0 for w in d),'FROZEN870_WINDOW_DISTRIBUTION');require(n>0 and windows>0,'MEASURED_THROUGHPUT')
    production={};overhead={}
    for arm in ARMS:
        production[arm]=max(sum(x['production_seconds'] for x in r['records']) for r in reports[arm])
        require(0<production[arm]<=phases[arm]['wall_seconds'],'MEASURED_PHASE_PRODUCTION');overhead[arm]=phases[arm]['wall_seconds']-production[arm]
    mean=sum(d)/870;scenarios=[]
    for multiplier in (1.,1.25,1.5):
        for imbalance in (1.,1.15,1.3):
            seconds=setup+merge_seconds+sum(overhead.values())+(production['old']+production['dr'])*1300/n*imbalance+production['s6']*(1300*mean*multiplier/windows)*imbalance
            scenarios.append({'hidden_window_multiplier':multiplier,'shard_imbalance':imbalance,'seconds':seconds})
    return {'population':1300,'window_distribution':{'count':870,'total':sum(d),'mean':mean,'max':max(d)},'scenarios':scenarios,
            'measured_production_wall_seconds':production,'measured_fixed_phase_overhead_seconds':overhead,
            'conservative_seconds':next(x['seconds'] for x in scenarios if x['hidden_window_multiplier']==1.25 and x['shard_imbalance']==1.15),
            'policy':'Default policy with three phases: slower shard per arm, all phase overhead once, 25% more R7 windows and 15% imbalance.'}


def launch(cfg,out,arm):
    """Default launch semantics; the dr arm's worker config overlays its own manifest/checkpoint/pins."""
    import signal
    children=[];start=time.monotonic();high={'host_rss_bytes':0,'output_disk_bytes':0};min_free=shutil.disk_usage(out).free
    try:
        with PlatformInterruption() as stop:
            for board in range(2):
                folder=out/f'board{board}-{arm}';folder.mkdir()
                c={**cfg,**cfg.get('arm_overlays',{}).get(arm,{}),'board':board,'arm':arm,'dev':cfg['assignment'][board],'physical':cfg['boards'][board],'out':str(folder)}
                path=out/f'board{board}-{arm}-config.json';save(path,c)
                env=worker_environment(board)
                log=(out/f'board{board}-{arm}.log').open('xb')
                child=subprocess.Popen([sys.executable,'-m',(REHEARSAL_MODULE if cfg['mode']=='rehearsal' else ENTRY_MODULE),'--worker',str(path)],env=env,stdout=log,stderr=subprocess.STDOUT);log.close();children.append(child)
            while any(c.poll() is None for c in children):
                s=resource_snapshot(out)
                for k in high:high[k]=max(high[k],s[k])
                min_free=min(min_free,s['disk_free_bytes'])
                if stop.requested or any(c.poll() not in (None,0) for c in children):
                    for c in children:
                        if c.poll() is None:c.send_signal(stop.signum or signal.SIGTERM)
                    if stop.requested:raise InterruptedError('Platform interruption')
                    raise RuntimeError('Worker failure; no partial merge')
                time.sleep(.5)
            require(all(c.returncode==0 for c in children),'WORKER_EXIT_FAILURE')
        reports=[json.loads((out/f'board{b}-{arm}'/arm/'receipt.json').read_bytes()) for b in range(2)]
        return reports,{'wall_seconds':time.monotonic()-start,'sample_interval_seconds':.5,'peaks':high,'minimum_disk_free_bytes':min_free}
    finally:
        for c in children:
            if c.poll() is None:c.terminate()
        for c in children:
            try:c.wait(timeout=45)
            except subprocess.TimeoutExpired:c.kill();c.wait()


def run(root,cfg,out,*,started):
    """Default run core over the three arms."""
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
        recovery=recover(arms['old'][0][indices],arms['dr'][0][indices],arms['s6'][0][indices],arms['old'][1][indices],arms['dr'][1][indices],arms['s6'][1][indices])
        prediction=recovery['blend'];r['recovery']={k:v for k,v in recovery.items() if k not in ('control','dr','s6','control_rank','dr_rank','s6_rank','blend')}
        csv_path=out/'visible-predictions.csv' if cfg['mode']=='rehearsal' else Path('/kaggle/working/submission.csv')
        require(time.monotonic()-started<MAX_SECONDS,'ACTUAL_NINE_HOUR_LIMIT_BEFORE_CSV')
        sha=write_csv(csv_path,columns,visible,prediction);csv_created=True
        r.update(status='PASS',completed=len(ids),completed_visible=len(visible),completed_probes=len(ids)-len(visible),completed_s6_probes=32 if cfg['mode']=='rehearsal' else 0,completed_old_historical=4 if cfg['mode']=='rehearsal' else 0,
            zero_failed=recovery['union_recovered']==0,zero_imputed=recovery['union_recovered']==0,csv_sha256=sha,csv_path=str(csv_path),all_required_recipe_gates=True,all_input_state_probability_gates=recovery['union_recovered']==0,
            sentinel_ids={'old':cfg.get('old_sentinels',[]),'dr':cfg.get('old_sentinels',[]),'s6':cfg.get('s6_sentinels',[])},elapsed_seconds=time.monotonic()-started,
            projection=projection(cfg,phases,reports,r['setup_seconds'],time.monotonic()-merge_start) if cfg['mode']=='rehearsal' else None)
        require(r['elapsed_seconds']<MAX_SECONDS,'ACTUAL_NINE_HOUR_LIMIT')
        if cfg['mode']=='rehearsal':require_rehearsal(r,cfg['binding'],csv_path)
        save(out/'receipt.json',r);return r
    except BaseException as e:
        if csv_created and cfg['mode']=='entry' and csv_path.exists():csv_path.replace(out/'rejected-submission.csv')
        r.update(status='INTERRUPTED' if isinstance(e,(KeyboardInterrupt,InterruptedError,TimeoutError)) else 'FAIL',failure_type=type(e).__name__,elapsed_seconds=time.monotonic()-started)
        save(out/'failure-private.json',{'type':type(e).__name__,'detail':str(e)});save(out/'receipt.json',r);raise


def raptor_arm_gates(r,binding,arm,profile,state,grid,*,raptor62):
    """B0 old-arm rehearsal gates for one Raptor-profile arm (B2's, parameterized)."""
    import numpy as np
    from knee import raptor_input as RI
    w0=r.get('workers',{}).get(arm,[{}])[0]
    stress=w0.get('partial_input_stress',{})
    require(stress.get('status')=='PASS' and stress.get('cases')==dict.fromkeys(RI.STRESS_CASES,'PASS') and stress.get('case_count')==16 and stress.get('uses_raw_data') is False and stress.get('uses_labels') is False,'REHEARSAL_RAPTOR_PARTIAL_INPUT_STRESS_'+arm)
    guarded=w0.get('guarded_partial_input_stress',{})
    require(guarded.get('status')=='PASS' and guarded.get('guarded') is True and guarded.get('case_count')==10 and guarded.get('cases')==dict.fromkeys(GUARDED_PARTIAL_CASES,'PASS'),'GUARDED_RAPTOR_PARTIAL_TEN_'+arm)
    require(arm in r.get('phases',{}) and r['phases'][arm]['wall_seconds']>0 and r['phases'][arm]['peaks']['host_rss_bytes']>0 and r['phases'][arm]['peaks']['output_disk_bytes']>0 and r['phases'][arm]['minimum_disk_free_bytes']>0,'CONCURRENT_RESOURCE_MEASUREMENT')
    rr=r.get('workers',{}).get(arm,[]);require(len(rr)==2,'REHEARSAL_WORKERS')
    require(all(x.get('recovery_injection',{}).get('status')=='PASS' and x['recovery_injection'].get('failure_status')=='data_local_failure' and x['recovery_injection'].get('injected_rows_genuine') is False and x['recovery_injection'].get('state_healthy') is True for x in rr),'ACTUAL_CLASSIFIED_RECOVERY_INJECTIONS')
    for board,x in enumerate(rr):
        require(x['status']=='PASS' and x['board']==board and x.get('arm')==arm and x.get('version')==VERSION and x['state_before']==x['state_after'] and x['completed']==len(x['assigned']) and x['peak_cuda_bytes']>0 and x['peak_rss_bytes']>0 and x['zero_failed'] is True and x['zero_imputed'] is True and x['fp32_recoveries']==0,'REHEARSAL_WORKER_STATE_RESOURCE')
        require(x['state_before']['state']==state and x.get('constructor_grid_sha256')==grid and x.get('old_profile')==profile and x.get('source_guard_receipt',{}).get('profile')=='raptor','QUALIFIED_COMPLETE_STATE_BINDING_'+arm)
        require(x['environment']['torch']==binding['environment']['torch'] and x['environment']['packages']==binding['environment']['packages'],'QUALIFIED_ENVIRONMENT_BINDING')
        require(x['environment']['physical_uuid']==r['boards'][board]['uuid'] and x['environment']['cuda_device_count']==1,'ISOLATED_BOARD_RUNTIME')
        require(len(x['sentinels'])==4 and [s['id'] for s in x['sentinels']]==r['sentinel_ids'][arm] and all(s['strict_reload'] for s in x['sentinels']),'BOTH_BOARD_SENTINELS')
        for s in x['sentinels']:
            p=np.array(s['probabilities']);q=np.array(s['reloaded_probabilities'])
            require(binding['sentinel_probabilities']['old'][s['id']] is None and float(np.max(np.abs(p-q)))<=1e-6 and s.get('strict_source_comparison') in ('PASS','ineligible_partial_input'),'OLD_SENTINEL_RELOAD_GATE_'+arm)
    for a,b in zip(rr[0]['sentinels'],rr[1]['sentinels']):
        require(a['input']==b['input'] and max(abs(x-y) for x,y in zip(a['probabilities'],b['probabilities']))<=CROSS_BOARD_TOLERANCE[arm],'CROSS_BOARD_SENTINEL_PARITY_'+arm)
    rows={row['id']:row for worker in rr for row in worker['records']};strict=[]
    require(len(rows)==39,'COMPLETE39_MECHANICS_'+arm)
    for sid in binding['s6_probe_ids']+binding['old_historical_ids']:
        row=rows[sid];strict.append(row.get('strict_source_comparison'))
        require(row.get('source_reference_parity') is True and row.get('strict_reload') is True and 0<=row.get('reload_max_abs',math.inf)<=1e-6 and row.get('strict_source_comparison') in ('PASS','ineligible_partial_input'),'ALL36_OLD_SOURCE_STRICT_REPEAT_RELOAD_'+arm)
    require('PASS' in strict,'STRICT_PYDICOM_SOURCE_COMPARISON_EXERCISED_'+arm)
    if raptor62:
        contract=binding['old_qualification']['raptor62']
        for sid in r['visible_ids']:
            g=rows[sid].get('raptor62',{})
            require(g.get('decode_counts_equal') is True and g.get('atol')==contract['visible_atol'] and g.get('within_atol') is True and 0<=g.get('max_abs',math.inf)<=contract['visible_atol'],'RAPTOR62_VISIBLE_RECEIPT_PARITY')
    return rows


def require_rehearsal(r,binding,csv_path):
    """B3 contract: B0 gates for the Dread and dr-full arms, the v7 R7 gates, three-arm completeness."""
    B1.require_common(r,binding,csv_path,VERSION,METHOD)
    require(binding.get('old_profile')==B0.OLD_PROFILE and binding.get('dr_profile')==DR_PROFILE and set(r.get('workers',{}))==set(ARMS) and r.get('sentinel_ids',{}).keys()==set(ARMS),'EXACT_B3_PROFILE')
    old=raptor_arm_gates(r,binding,'old',B0.OLD_PROFILE,B0.DREAD_STATE_SHA256,B0.DREAD_GRID_SHA256,raptor62=True)
    dr=raptor_arm_gates(r,binding,'dr',DR_PROFILE,DRFULL_STATE_SHA256,DRFULL_GRID_SHA256,raptor62=False)
    s6=B1.require_r7_workers(r,binding)
    require(set(old)==set(dr)==set(s6),'COMPLETE39_THREE_ARMS')
    require(r.get('recovery',{}).get('weights')==ARM_WEIGHTS and r['recovery'].get('recipe')==RECOVERY_RECIPE_B3,'B3_FIXED_WEIGHTS')


def sanity(out,cfg):
    """Rehearsal diagnostic: dr-full vs Dread per target (39 rows and visible 3) and R7-arm parity."""
    import numpy as np
    import pandas as pd
    out=Path(out);ids=cfg['binding']['visible_ids'];arms={}
    for arm in ARMS:
        rows={}
        for b in range(2):
            with np.load(out/f'board{b}-{arm}'/arm/'predictions.npz',allow_pickle=False) as a:rows.update(zip(a['study_ids'].tolist(),a['probabilities']))
        arms[arm]=rows
    common=sorted(arms['old']);a=np.array([arms['old'][s] for s in common]);b=np.array([arms['dr'][s] for s in common])
    result={'dr_vs_dread_spearman_39':dict(zip(LABELS,[float(pd.Series(a[:,j]).corr(pd.Series(b[:,j]),method='spearman')) for j in range(12)])),
            'visible_dread':{s:arms['old'][s].tolist() for s in ids},'visible_drfull':{s:arms['dr'][s].tolist() for s in ids},
            'visible_spearman_3':dict(zip(LABELS,[float(pd.Series([arms['old'][s][j] for s in ids]).corr(pd.Series([arms['dr'][s][j] for s in ids]),method='spearman')) for j in range(12)]))}
    for name,ref in (('r7_vs_heliosli_ours_sub',cfg['heliosli_reference']['r7']),('r7_vs_entry_v2_r7',cfg['r7_entry_reference'])):
        d=np.abs(np.array([arms['s6'][s] for s in ids])-np.array([ref[s] for s in ids]));result[name]={'max_abs':float(d.max()),'max_abs_per_target':dict(zip(LABELS,d.max(0).tolist()))}
    save(out/'b3-sanity.json',result);return result


r7_parity=sanity
old_worker=types.FunctionType(B0.old_worker.__code__,globals(),B0.old_worker.__name__,B0.old_worker.__defaults__,B0.old_worker.__closure__)
r7_worker=types.FunctionType(R7.r7_worker.__code__,globals(),'r7_worker',R7.r7_worker.__defaults__,R7.r7_worker.__closure__)
merge=types.FunctionType(base.merge.__code__,globals(),'merge',base.merge.__defaults__,base.merge.__closure__)
WORKERS={'old':old_worker,'dr':dr_worker,'s6':r7_worker}


if __name__=='__main__':
    import argparse
    import os
    require(Path(__file__).resolve()==(Path(os.environ['KNEE_ENTRY_PROJECT'])/'knee/public_r7_b3_entry.py').resolve(),'EXACT_FRESH_SOURCE_IMPORT')
    p=argparse.ArgumentParser();p.add_argument('--worker',required=True);a=p.parse_args();cfg=json.loads(Path(a.worker).read_bytes())
    WORKERS[cfg['arm']](a.worker)
