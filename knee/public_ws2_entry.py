"""A-prime R224 third arm around the immutable R7 v7 workers.

The Kaggle rehearsal first executes the frozen R7 run on the same 39 images, then
runs a native 224 worker. No fallback substitutes a baseline CSV for R224.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time

from knee import public_r7_recovery_entry as R7
from knee import public_ws2_checkpoint as C224
from knee import public_ws2_recovery as REC
from knee.public_s6_qualification import LABELS, digest, file_sha, require, save
from knee.public_s6_recovery_entry import assignments, board_environment, old_decode_gate, probabilities, probe_gate, stage, verify_staged, worker_environment, resource_snapshot
from knee.public_s6_recovery_runtime import DataLocalFailure, FatalSourceError, cleanup_study, install_old_guards, save_ledger
from knee.public_fullfield_entry import read_series, test_contract, write_csv
from knee.mirror_input import prepare_study
from knee.public_adaptation_fit import PlatformInterruption, atomic_save

VERSION = 'ws2-a-prime-r224-v1'
ENTRY_MODULE = 'knee.public_ws2_entry'
MAX_SECONDS = 9 * 3600
PROJECTION_GATE_SECONDS = 8.3 * 3600


def crop224_worker(config_path):
    import numpy as np
    import torch
    from knee.reference import load_definitions

    cfg = json.loads(Path(config_path).read_bytes())
    out = Path(cfg['out']) / 'crop224'; out.mkdir()
    start = time.monotonic()
    r = {'status':'RUNNING','version':R7.VERSION,'arm':'crop224','slot':'native224',
         'board':cfg['board'],'assigned':cfg['dev'],'completed':0,'records':[],'sentinels':[]}
    save(out/'receipt.json',r)
    try:
        r['environment'] = board_environment(torch,cfg)
        verify_staged(cfg['staged'],cfg['staged_ids'],cfg['stage_binding'])
        public, decoder = load_definitions(cfg['public_source'],cfg['decoder_source'],cfg['public_manifest'])
        r['source_guard_receipt'] = install_old_guards(public,decoder)
        C224.verify_checkpoint_file(cfg['crop224_checkpoint'])
        model, model_receipt = C224.load_model(public,cfg['crop224_checkpoint'],'cuda')
        state = model_receipt['state_sha256'];r['model']=model_receipt;r['state_before']=state
        if cfg['mode']=='entry':require(state==cfg['r224_state_pin'],'QUALIFIED_R224_STATE_PIN')
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        torch.use_deterministic_algorithms(True)
        frozen=out/'frozen-state.pt'
        atomic_save(frozen,{k:v.detach().cpu() for k,v in model.state_dict().items()})
        records=read_series(Path(cfg['staged'])/'train_series.csv')
        def infer(sid, cpu_parity=False):
            started=time.monotonic()
            v,m,counters=prepare_study(public,decoder,sid,records,Path(cfg['staged'])/'train_series')
            decoded=time.monotonic()
            old_decode_gate(counters)
            C224.validate_volume(v,m)
            volume_sha=hashlib.sha256(v.tobytes()).hexdigest();mask_sha=hashlib.sha256(m.tobytes()).hexdigest()
            prior=cfg.get('probes',{}).get(sid,{}).get('old',{}).get('input',{})
            if prior:
                require(volume_sha==prior['volume_sha256'] and mask_sha==prior['mask_sha256'],'FROZEN384_INPUT_PARITY')
            x=C224.windows(v,m,device='cuda')
            windowed=time.monotonic()
            if cpu_parity:
                y=C224.windows(v,m,device='cpu').to('cuda')
                crop_diff=float((x-y).abs().max().item())
                require(crop_diff<=1e-5,'CROP224_CPU_CUDA_PARITY')
                del y
            else:crop_diff=None
            p=C224.infer_probs(model,x,'cuda',expected_state=state)
            probabilities([p],1)
            torch.cuda.synchronize()
            forwarded=time.monotonic()
            return p,{'volume_sha256':volume_sha,'mask_sha256':mask_sha,
                      'crop224_window_sha256':hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest(),
                      'crop224_cpu_cuda_max_abs':crop_diff,'decode_counters':counters},\
                {'decode_seconds':decoded-started,'window_seconds':windowed-decoded,
                 'forward_seconds':forwarded-windowed,'production_seconds':time.monotonic()-started}
        ps=[];statuses=[]
        with PlatformInterruption() as stop:
            for sid in cfg.get('crop224_sentinels',[]):
                p,fp,_=infer(sid,cpu_parity=True)
                model.load_state_dict(torch.load(frozen,map_location='cpu',weights_only=True),strict=True)
                model.eval();C224.require_eval(model)
                require(C224.state_digest(model)==state,'R224_STRICT_RELOAD_STATE')
                q,g,_=infer(sid,cpu_parity=True)
                require(fp==g and float(np.max(np.abs(p-q)))<=1e-6,'R224_FRESH_DECODE_RELOAD')
                r['sentinels'].append({'id':sid,'input':fp,'probabilities':p.tolist(),
                    'reloaded_probabilities':q.tolist(),'strict_reload':True})
                save(out/'receipt.json',r)
                if stop.requested:raise InterruptedError('Interrupted R224 sentinel')
            for sid in cfg['dev']:
                study_start=time.monotonic()
                try:
                    r['current_id']=sid;save(out/'receipt.json',r)
                    p,fp,timing=infer(sid,cpu_parity=False)
                    ps.append(p);statuses.append('genuine')
                    row={'id':sid,'status':'genuine','input':fp,**timing}
                except DataLocalFailure as e:
                    require(cfg['mode']!='rehearsal','HEALTHY39_MUST_BE_GENUINE')
                    ps.append(np.full(12,np.nan));statuses.append('data_local_failure')
                    row={'id':sid,'status':'data_local_failure','phase':'crop224_input',
                         'failure_code':str(e),'production_seconds':time.monotonic()-study_start}
                    save(out/('failure-'+str(len(ps)-1)+'-private.json'),
                         {'id':sid,'type':type(e).__name__,'detail':e.detail,'site':e.site})
                finally:
                    clean=cleanup_study(out/'scratch',sid,{})
                    if sys.exc_info()[0] is None:require(C224.state_digest(model)==state,'R224_STATE_AFTER_STUDY')
                require(statuses.count('data_local_failure')<=math.floor(.03*len(cfg['staged_ids'])),'R224_FAILURE_CAP')
                row.update(cleanup=clean,state_healthy=True)
                r['records'].append(row);r['completed']=len(ps)
                save_ledger(out/'predictions.npz',cfg['dev'][:len(ps)],ps,statuses)
                save(out/'receipt.json',r)
                if stop.requested:raise InterruptedError('Interrupted R224 study')
        require(len(ps)==len(cfg['dev']) and C224.state_digest(model)==state,'R224_COMPLETE_STATE')
        r.update(status='PASS',state_after=state,zero_failed=all(s=='genuine' for s in statuses),
                 genuine=statuses.count('genuine'),failed=statuses.count('data_local_failure'),
                 zero_imputed=True,fp32_recoveries=0,ledger_sha256=file_sha(out/'predictions.npz'),
                 peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                 peak_cuda_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
        frozen.unlink();save(out/'receipt.json',r)
    except BaseException as e:
        actual=e.error if isinstance(e,FatalSourceError) else e
        r.update(status='INTERRUPTED' if isinstance(actual,(InterruptedError,KeyboardInterrupt,TimeoutError)) else 'FAIL',failure_type=type(actual).__name__)
        save(out/'failure-private.json',{'type':type(actual).__name__,'detail':str(actual)})
        save(out/'receipt.json',r)
        if actual is not e:raise actual from e
        raise
    finally:
        r['elapsed_seconds']=time.monotonic()-start;save(out/'receipt.json',r)


def launch_crop224(cfg,out):
    children=[];start=time.monotonic();high={'host_rss_bytes':0,'output_disk_bytes':0};minimum=shutil.disk_usage(out).free
    try:
        with PlatformInterruption() as stop:
            for board in range(2):
                folder=out/f'board{board}-crop224';folder.mkdir()
                c={**cfg,'board':board,'arm':'crop224','dev':cfg['assignment'][board],
                   'physical':cfg['boards'][board],'out':str(folder)}
                path=out/f'board{board}-crop224-config.json';save(path,c)
                log=(out/f'board{board}-crop224.log').open('xb')
                child=subprocess.Popen([sys.executable,'-m',ENTRY_MODULE,'--worker',str(path)],
                    env=worker_environment(board),stdout=log,stderr=subprocess.STDOUT)
                log.close();children.append(child)
            while any(c.poll() is None for c in children):
                s=resource_snapshot(out)
                for k in high:high[k]=max(high[k],s[k])
                minimum=min(minimum,s['disk_free_bytes'])
                if stop.requested or any(c.poll() not in (None,0) for c in children):
                    for c in children:
                        if c.poll() is None:c.send_signal(stop.signum or signal.SIGTERM)
                    if stop.requested:raise InterruptedError('R224 platform interruption')
                    raise RuntimeError('R224 worker failure; no partial merge')
                time.sleep(.5)
            require(all(c.returncode==0 for c in children),'R224_WORKER_EXIT')
        reports=[json.loads((out/f'board{b}-crop224'/'crop224'/'receipt.json').read_bytes()) for b in range(2)]
        return reports,{'wall_seconds':time.monotonic()-start,'sample_interval_seconds':.5,
                        'peaks':high,'minimum_disk_free_bytes':minimum}
    finally:
        for c in children:
            if c.poll() is None:c.terminate()
        for c in children:
            try:c.wait(timeout=45)
            except subprocess.TimeoutExpired:c.kill();c.wait()


def merged_arm(ids,reports,out,arm):
    return R7.merge(ids,[(x,out/f'board{i}-{arm}'/arm/'predictions.npz') for i,x in enumerate(reports)],arm)


def projection(base,r224_phase,r224_reports,merge_seconds):
    """Measured full-39 third-arm rate plus frozen R7 conservative projection."""
    import numpy as np
    old=base['projection']['conservative_seconds']
    require(math.isfinite(float(merge_seconds)) and merge_seconds>=0 and math.isfinite(float(old)) and old>0,
            'MEASURED_PROJECTION_INTERVALS')
    times=[[float(x['production_seconds']) for x in r['records']] for r in r224_reports]
    require(all(len(t)>0 and all(np.isfinite(t)) and min(t)>0 for t in times),'R224_MEASURED_TIMES')
    # Frozen R7 projects each board to the entire unseen test population. Keep its
    # conservative total and add the measured R224 board rate and final merge wall.
    n=base['projection'].get('population')
    require(type(n) is int and n>0,'R224_PROJECTED_POPULATION_REQUIRED')
    production=max(sum(t) for t in times)
    overhead=r224_phase['wall_seconds']-production
    require(overhead>=0,'R224_MEASURED_OVERHEAD')
    arm_seconds=production*n/sum(len(t) for t in times)*1.15
    total=float(old+arm_seconds+overhead+merge_seconds)
    require(math.isfinite(total) and total>0,'R224_PROJECTION_FINITE')
    return {'baseline_conservative_seconds':old,'crop224_conservative_seconds':float(arm_seconds),
            'conservative_seconds':total,'target_studies':n,'measured_crop224_39_wall_seconds':r224_phase['wall_seconds'],
            'measured_crop224_production_seconds':production,'measured_crop224_fixed_overhead_seconds':overhead,
            'gate_seconds':PROJECTION_GATE_SECONDS,'gate_pass':total<=PROJECTION_GATE_SECONDS}


def run(root,cfg,out,*,started):
    """Fresh 2xT4 full39 baseline and R224 rehearsal, or bound image-only entry."""
    import numpy as np
    out=Path(out);out.mkdir();r={'status':'RUNNING','version':VERSION,'mode':cfg['mode'],'binding':cfg['ws2_binding']}
    save(out/'receipt.json',r)
    csv_path=Path('/kaggle/working/submission.csv') if cfg['mode']=='entry' else out/'visible-predictions.csv'
    try:
        base_projector=R7.projection
        captured={}
        def capture_projection(project_cfg,phases,reports,setup,merge):
            captured.update(base_setup_seconds=setup,base_merge_seconds=merge,
                            development_windows=project_cfg['development_windows'])
            return base_projector(project_cfg,phases,reports,setup,merge)
        R7.projection=capture_projection
        try:base=R7.run(root,cfg,out/'base',started=started)
        finally:R7.projection=base_projector
        require(base['status']=='PASS','FRESH_BASELINE_RUN')
        if cfg['mode']=='entry':
            source=Path('/kaggle/working/submission.csv')
            source.replace(out/'baseline-candidate-private.csv')
        ids=json.loads((out/'base'/'assignment.json').read_bytes())['ids']
        cfg={**cfg,'staged_ids':ids,'assignment':assignments(ids),'boards':base['boards'],
             'staged':str(out/'base'/'images-only'),'stage_binding':base['stage_binding']}
        cfg['crop224_sentinels']=cfg.get('s6_sentinels',[]) if cfg['mode']=='rehearsal' else []
        reports,phase=launch_crop224(cfg,out)
        arms={name:merged_arm(ids,base['workers'][base_name],out/'base',base_name)
              for name,base_name in (('old320','old'),('r7','s6'))}
        arms['crop224']=merged_arm(ids,reports,out,'crop224')
        begin=time.monotonic()
        baseline_full=R7.recover(arms['old320'][0],arms['r7'][0],
                                 arms['old320'][1],arms['r7'][1])
        rec=REC.recover(*(arms[x][0] for x in REC.ARMS),*(arms[x][1] for x in REC.ARMS))
        require(np.array_equal(baseline_full['control_rank'],rec['ranks']['old320'])
                and np.array_equal(baseline_full['s6_rank'],rec['ranks']['r7']),
                'FROZEN_A_FULL_POPULATION_ENDPOINT_RANKS')
        visible=base['visible_ids'];indices=[ids.index(s) for s in visible]
        require(len(visible)==len(set(visible)) and set(visible)<=set(ids),'VISIBLE_COVERAGE')
        require(time.monotonic()-started<MAX_SECONDS,'ACTUAL_NINE_HOUR_LIMIT_BEFORE_CSV')
        columns=cfg['binding']['columns'];sha=write_csv(csv_path,columns,visible,rec['blend'][indices])
        merged_seconds=time.monotonic()-begin
        captured['final_merge_seconds']=merged_seconds
        r.update(status='PASS',base_receipt_sha256=file_sha(out/'base'/'receipt.json'),
            base_csv_sha256=base['csv_sha256'],crop224_workers=reports,
            crop224_phase=phase,projection_inputs=captured,
            completed=len(ids),completed_visible=len(visible),
            zero_failed=rec['union_recovered']==0,csv_sha256=sha,csv_path=str(csv_path),
            baseline_full_population={'old_rank_sha256':hashlib.sha256(baseline_full['control_rank'].tobytes()).hexdigest(),
                'r7_rank_sha256':hashlib.sha256(baseline_full['s6_rank'].tobytes()).hexdigest(),
                'blend_sha256':hashlib.sha256(baseline_full['blend'].tobytes()).hexdigest()},
            resources={'peak_host_rss_bytes':max(p['peaks']['host_rss_bytes'] for p in [*base['phases'].values(),phase]),
                'peak_output_disk_bytes':max(p['peaks']['output_disk_bytes'] for p in [*base['phases'].values(),phase]),
                'minimum_disk_free_bytes':min(p['minimum_disk_free_bytes'] for p in [*base['phases'].values(),phase]),
                'peak_cuda_bytes_by_arm':{arm:[x['peak_cuda_bytes'] for x in rr] for arm,rr in
                    [('old',base['workers']['old']),('s6',base['workers']['s6']),('crop224',reports)]}},
            recovery={k:v for k,v in rec.items() if k not in ('filled','ranks','source224_csv_values','raptor_slot','blend')},
            elapsed_seconds=time.monotonic()-started)
        if cfg['mode']=='rehearsal':
            r['projection']=projection(base,phase,reports,merged_seconds)
            require_rehearsal(r,cfg['ws2_binding'],out/'base'/'receipt.json',csv_path)
        require(r['elapsed_seconds']<MAX_SECONDS,'ACTUAL_NINE_HOUR_LIMIT')
        save(out/'receipt.json',r);return r
    except BaseException as e:
        if cfg['mode']=='entry' and csv_path.exists():csv_path.replace(out/'rejected-submission.csv')
        r.update(status='INTERRUPTED' if isinstance(e,(InterruptedError,KeyboardInterrupt,TimeoutError)) else 'FAIL',
                 failure_type=type(e).__name__,elapsed_seconds=time.monotonic()-started)
        save(out/'failure-private.json',{'type':type(e).__name__,'detail':str(e)})
        save(out/'receipt.json',r)
        raise


def require_rehearsal(r,binding,base_receipt_path,csv_path):
    """A fresh full rehearsal is the only route to a qualified entry binding."""
    import csv
    base=json.loads(Path(base_receipt_path).read_bytes())
    require(digest(base.get('binding'))==binding['frozen_r7_binding_sha256'],
            'EXACT_ARCHIVED_V7_BINDING_DIGEST')
    import numpy as np
    R7.require_rehearsal(base,base['binding'],Path(base_receipt_path).parent/'visible-predictions.csv')
    require(r.get('status')=='PASS' and r.get('version')==VERSION and r.get('mode')=='rehearsal'
            and r.get('binding')==binding and r.get('base_receipt_sha256')==file_sha(base_receipt_path),'FRESH_FULL_REHEARSAL_BINDING')
    require(r.get('completed')==39 and r.get('completed_visible')==3 and r.get('zero_failed') is True
            and r.get('recovery',{}).get('union_recovered')==0,'FRESH39_R224_COVERAGE')
    workers=r.get('crop224_workers',[])
    require(len(workers)==2 and len({x['environment']['physical_uuid'] for x in workers})==2,'TWO_PHYSICAL_T4_R224')
    require(all(x.get('status')=='PASS' and x.get('slot')=='native224' and x.get('completed')==len(x['assigned'])
        and x.get('environment',{}).get('physical_uuid')==base['boards'][i]['uuid']
        and x.get('environment',{}).get('torch')==binding['frozen_base_environment']['torch']
        and x.get('environment',{}).get('packages')==binding['frozen_base_environment']['packages']
        and x.get('state_before')==x.get('state_after') and x.get('model',{}).get('checkpoint_sha256')==C224.CHECKPOINT_SHA256
        and x.get('model',{}).get('arch')==C224.ARCH and x.get('model',{}).get('res')==224
        and x.get('zero_failed') is True and x.get('fp32_recoveries')==0 and x.get('peak_cuda_bytes',0)>0
        and x.get('peak_rss_bytes',0)>0 and x.get('ledger_sha256')
        and len(x.get('sentinels',[]))==4 and all(s.get('strict_reload') is True and s['input']['crop224_cpu_cuda_max_abs']<=1e-5 for s in x['sentinels'])
        for i,x in enumerate(workers)),'R224_STATE_RELOAD_CROP_RESOURCES')
    require(workers[0]['state_before']==workers[1]['state_before'],'R224_CROSS_BOARD_STATE')
    old_rows={row['id']:row for w in base['workers']['old'] for row in w['records']}
    require(len(old_rows)==39,'FROZEN_OLD39_RECORDS')
    for worker in workers:
        require([s['id'] for s in worker['sentinels']]==base['sentinel_ids']['s6'],
                'R224_SENTINEL_IDENTITY')
        for s in worker['sentinels']:
            original=old_rows[s['id']]['input']
            require(all(s['input'][key]==original[key] for key in ('volume_sha256','mask_sha256')),
                    'R224_SENTINEL_GUARDED384_SOURCE')
            p=np.asarray(s['probabilities']);q=np.asarray(s['reloaded_probabilities'])
            require(p.shape==q.shape==(12,) and np.isfinite(p).all() and np.isfinite(q).all()
                    and ((p>=0)&(p<=1)).all() and ((q>=0)&(q<=1)).all()
                    and float(np.max(np.abs(p-q)))<=1e-6,'R224_SENTINEL_PROBABILITY_RELOAD')
    for left,right in zip(workers[0]['sentinels'],workers[1]['sentinels']):
        require(left['id']==right['id'] and left['input']==right['input']
                and max(abs(a-b) for a,b in zip(left['probabilities'],right['probabilities']))<=1e-6,
                'R224_CROSS_BOARD_SENTINEL')
    inputs=r.get('projection_inputs',{})
    require(set(inputs)=={'base_setup_seconds','base_merge_seconds','development_windows','final_merge_seconds'},
            'SAVED_MEASURED_PROJECTION_INPUTS')
    require(digest(inputs['development_windows'])==binding['frozen_development_windows_sha256'],
            'EXACT_ARCHIVED_WINDOW_DISTRIBUTION')
    require(all(type(inputs[key]) in (int,float) and math.isfinite(inputs[key]) and inputs[key]>=0
                for key in ('base_setup_seconds','base_merge_seconds','final_merge_seconds'))
            and inputs['base_setup_seconds']==base['setup_seconds'],
            'ACTUAL_NONNEGATIVE_SETUP_MERGE_INTERVALS')
    frozen_projection=R7.projection({'development_windows':inputs['development_windows']},
        base['phases'],base['workers'],inputs['base_setup_seconds'],inputs['base_merge_seconds'])
    require(frozen_projection==base['projection'],'REPLAY_FROZEN_R7_PROJECTION')
    fresh_projection=projection(base,r['crop224_phase'],workers,inputs['final_merge_seconds'])
    require(fresh_projection==r.get('projection'),'REPLAY_COMBINED_PROJECTION')
    for phase in ('old','s6'):
        measured=base['phases'][phase]
        require(measured['wall_seconds']>0 and measured['peaks']['host_rss_bytes']>0
                and measured['peaks']['output_disk_bytes']>0 and measured['minimum_disk_free_bytes']>0,
                'FROZEN_ARM_RESOURCE_PHASE')
    measured=r['crop224_phase']
    require(measured['wall_seconds']>0 and measured['peaks']['host_rss_bytes']>0
            and measured['peaks']['output_disk_bytes']>0 and measured['minimum_disk_free_bytes']>0,
            'R224_RESOURCE_PHASE')
    phases=[*base['phases'].values(),measured]
    expected_resources={'peak_host_rss_bytes':max(x['peaks']['host_rss_bytes'] for x in phases),
        'peak_output_disk_bytes':max(x['peaks']['output_disk_bytes'] for x in phases),
        'minimum_disk_free_bytes':min(x['minimum_disk_free_bytes'] for x in phases),
        'peak_cuda_bytes_by_arm':{arm:[x['peak_cuda_bytes'] for x in rr] for arm,rr in
            [('old',base['workers']['old']),('s6',base['workers']['s6']),('crop224',workers)]}}
    require(r.get('resources')==expected_resources,'REPLAY_COMPLETE_RESOURCES')
    p=fresh_projection
    require(0<p['conservative_seconds']<=PROJECTION_GATE_SECONDS and 0<r['elapsed_seconds']<MAX_SECONDS,
            'COMBINED_PROJECTION_AND_ACTUAL_LIMIT')
    require(r.get('csv_sha256')==file_sha(csv_path),'R224_CSV_HASH')
    with Path(csv_path).open(newline='') as handle:rows=list(csv.reader(handle))
    require(rows[0]==[binding['uid']]+LABELS and len(rows)==4 and [x[0] for x in rows[1:]]==base['visible_ids'],
            'R224_CSV_SCHEMA_ORDER')
    probabilities([[float(x) for x in row[1:]] for row in rows[1:]],3)
    return True


if __name__=='__main__':
    import argparse
    require(Path(__file__).resolve()==(Path(os.environ['KNEE_ENTRY_PROJECT'])/'knee/public_ws2_entry.py').resolve(),
            'EXACT_FRESH_WS2_SOURCE_IMPORT')
    p=argparse.ArgumentParser();p.add_argument('--worker',required=True)
    a=p.parse_args();crop224_worker(a.worker)
