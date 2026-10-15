"""Lane 3b P3: paired serial-control vs prefetch comparison of the R7 component on one 2xT4 session.

For Codex's P3 rehearsal. Inputs are the R7 ('s6') worker folders of a serial control run (r7_prefetch False,
rehearsal only) and of the prefetch run, on the same session with the same assignment, plus the archived v7 receipt.
Nothing here relaxes an existing tolerance:
  compiled 'same'      both runs share the compiled graphs (one worker process): within-board reload contract 1e-6;
  compiled 'separate'  separately compiled worker processes: the existing R7 cross-board contract 1e-3
                       (CROSS_BOARD_TOLERANCE['s6']; v3 measured up to 2.6e-4 between separate compilations).
Exact: ids/assignment/order, the eight input fingerprints (both runs and archived v7), anchor/side/fluid/scratch/
source-diagnostic mechanics, statuses, states. Within tolerance: model and final probabilities, stacker features and
outputs, ledger probabilities and sigmoid(mean logits), sentinels. Speed and resources are reported from the closed
evaluation intervals (paired wall), not from production/forward alone; peak worker RSS must stay <= 9.5 GB.
It checks only the unchanged R7 component; A' arm additions are compared separately. Exit status 1 on any failure.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
V7_RECEIPT_SHA='e10075cf8bf53d74b5a3750e31bd81d003e7714453d4ca55047d363d96d0cc07'
TOLERANCE={'same':1e-6,'separate':1e-3}
WORKER_RSS_LIMIT=9.5e9
COMBINED_RSS_LIMIT=22e9
SLOWER_BOARD_WALL_RATIO_LIMIT=.85
EXACT=('id','status','windows','input','anchor_method','anchor_reasons','side','native_scratch_bytes','field_center_fallbacks','fluid','source_diagnostics','pilot_input_parity')


def load_worker(folder):
    import numpy as np
    folder=Path(folder);r=json.loads((folder/'receipt.json').read_bytes())
    with np.load(folder/'predictions.npz',allow_pickle=False) as a:ledger={k:a[k] for k in a.files}
    return r,ledger


def maxdiff(a,b):
    import numpy as np
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    if a.shape!=b.shape:return math.inf
    both=np.isnan(a)&np.isnan(b)
    if not np.array_equal(np.isnan(a),np.isnan(b)):return math.inf
    return float(np.max(np.abs(np.where(both,0.,a-b)))) if a.size else 0.


def compare(serial_folders,prefetch_folders,v7,compiled,expected_studies=39,expected_sentinels=4,
            serial_phase=None,prefetch_phase=None):
    import numpy as np
    from knee.public_r7_prefetch_candidate import VERSION,CROSS_BOARD_TOLERANCE,require_compiled
    require_tol=TOLERANCE[compiled];worker_rss_limit=WORKER_RSS_LIMIT;assert TOLERANCE['separate']==CROSS_BOARD_TOLERANCE['s6']
    checks=[];note=lambda name,ok,detail=None:checks.append({'check':name,'pass':bool(ok),**({'detail':detail} if detail is not None else {})})
    v7w={w['board']:w for w in v7['workers']['s6']};v7in={row['id']:row['input'] for w in v7['workers']['s6'] for row in w['records']}
    note('v7_reference_39_inputs',len(v7in)==expected_studies and all(len(x)==8 for x in v7in.values()))
    boards=[];seen=[]
    for board,(fs,fp) in enumerate(zip(serial_folders,prefetch_folders)):
        (s,ls),(p,lp)=load_worker(fs),load_worker(fp);b={'board':board}
        for tag,x,enabled in (('serial',s,False),('prefetch',p,True)):
            note(f'{tag}{board}_pass_version_slot',x.get('status')=='PASS' and x.get('version')==VERSION and x.get('slot')=='r7' and x.get('board')==board and x.get('prefetch_enabled') is enabled)
            try:ok=require_compiled(x.get('compile',{}))
            except ValueError:ok=False
            note(f'{tag}{board}_compiled_no_fallback',ok)
            note(f'{tag}{board}_assignment_equals_v7',x.get('assigned')==v7w[board]['assigned'])
            note(f'{tag}{board}_state_equals_v7',x.get('state_before')==x.get('state_after')==v7w[board]['state_before'])
            note(f'{tag}{board}_zero_failed',x.get('zero_failed') is True and x.get('fp32_recoveries')==0)
            w=x.get('evaluation_wall') or {}
            note(f'{tag}{board}_evaluation_wall_closed',bool(w) and abs(w['seconds']-w['sum_production_seconds'])<=1e-6*max(1.,w['seconds']) and w['studies']==len(x['records']))
            note(f'{tag}{board}_worker_rss_limit',0<x.get('peak_rss_bytes',math.inf)<=worker_rss_limit,x.get('peak_rss_bytes'))
        pf=p.get('prefetch',{})
        note(f'prefetch{board}_producer_bound_and_teardown',pf.get('max_concurrent_stages')==1 and pf.get('alive_at_exit') is False and pf.get('prefetched')==len(p['records'])-1 and pf.get('submitted')==len(p['records'])-1,pf)
        note(f'prefetch{board}_cuda_context_default_stream',bool(pf.get('contexts')) and all(c['main']==c['producer'] for c in pf['contexts']),pf.get('contexts'))
        note(f'serial{board}_evaluation_input_parity_v7',(s.get('evaluation_input_parity') or {}).get('checked')==len(s['records']))
        note(f'prefetch{board}_evaluation_input_parity_v7',(p.get('evaluation_input_parity') or {}).get('checked')==len(p['records']))
        note(f'serial{board}_no_producer',s.get('prefetch',{}).get('submitted')==0)
        ids_s=[x['id'] for x in s['records']];ids_p=[x['id'] for x in p['records']];seen+=ids_p
        note(f'board{board}_record_ids_order',ids_s==ids_p==v7w[board]['assigned'] and len(set(ids_p))==len(ids_p))
        worst=0.;exact_bad=[]
        for a,c in zip(s['records'],p['records']):
            exact_bad+=[(a['id'],k) for k in EXACT if a.get(k)!=c.get(k)]
            if c.get('input')!=v7in.get(c['id']):exact_bad.append((c['id'],'input_vs_v7'))
            worst=max(worst,maxdiff(a['model_probabilities'],c['model_probabilities']),maxdiff(a['probabilities'],c['probabilities']))
            if set(a['stackers'])!=set(c['stackers']):exact_bad.append((a['id'],'stacker_classes'));continue
            for k in a['stackers']:
                x,y=a['stackers'][k],c['stackers'][k]
                if x.get('volume_missing')!=y.get('volume_missing') or x.get('stacker')!=y.get('stacker') or x.get('fallback')!=y.get('fallback'):exact_bad.append((a['id'],'stacker_'+k))
                worst=max(worst,maxdiff([x.get('z_feature'),x.get('p_out')],[y.get('z_feature'),y.get('p_out')]),maxdiff(x.get('features',[]),y.get('features',[])))
        note(f'board{board}_exact_inputs_and_mechanics',not exact_bad,exact_bad[:20])
        note(f'board{board}_record_outputs_within_tolerance',worst<=require_tol,worst)
        sig=lambda z:1/(1+np.exp(-np.asarray(z,dtype=np.float64).mean(1)))
        note(f'board{board}_ledger_ids_statuses',ls['study_ids'].tolist()==lp['study_ids'].tolist()==ids_p and ls['statuses'].tolist()==lp['statuses'].tolist())
        lw=max(maxdiff(ls['probabilities'],lp['probabilities']),maxdiff(sig(ls['logits']),sig(lp['logits'])))
        note(f'board{board}_ledger_within_tolerance',lw<=require_tol,lw)
        sw=0.;sbad=[]
        for a,c in zip(s.get('sentinels',[]),p.get('sentinels',[])):
            if a['id']!=c['id'] or a['input']!=c['input']:sbad.append(a['id'])
            sw=max(sw,maxdiff(a['probabilities'],c['probabilities']),maxdiff(a['reloaded_probabilities'],c['reloaded_probabilities']))
        note(f'board{board}_sentinels',len(s.get('sentinels',[]))==len(p.get('sentinels',[]))==expected_sentinels and not sbad and sw<=require_tol,{'max_abs':sw,'mismatch':sbad})
        ws,wp=s['evaluation_wall']['seconds'] if s.get('evaluation_wall') else None,p['evaluation_wall']['seconds'] if p.get('evaluation_wall') else None
        fwd=lambda x:sum(r['four_model_seconds'] for r in x['records'])
        b.update(serial_evaluation_wall_seconds=ws,prefetch_evaluation_wall_seconds=wp,paired_wall_ratio=None if not ws or wp is None else wp/ws,
                 production_over_forward={'serial':ws/fwd(s) if ws else None,'prefetch':wp/fwd(p) if wp else None,'note':'interleaved interval: producer work inflates four_model_seconds; read with the paired wall'},
                 prefetch_wait_seconds=sum(r.get('prefetch_wait_seconds') or 0. for r in p['records']),
                 peak_rss_bytes={'serial':s.get('peak_rss_bytes'),'prefetch':p.get('peak_rss_bytes')},
                 peak_cuda_allocated_bytes={'serial':s.get('peak_cuda_bytes'),'prefetch':p.get('peak_cuda_bytes')},
                 peak_cuda_reserved_bytes={'serial':s.get('peak_cuda_reserved_bytes'),'prefetch':p.get('peak_cuda_reserved_bytes')},
                 max_record_abs=worst,max_ledger_abs=lw)
        boards.append(b)
    note('all_39_compared_once',sorted(seen)==sorted(v7in))
    slower=max(boards,key=lambda b:b['prefetch_evaluation_wall_seconds'] or 0.) if boards else {}
    ratio=slower.get('paired_wall_ratio')
    note('slower_board_paired_wall_speed',ratio is not None and ratio<=SLOWER_BOARD_WALL_RATIO_LIMIT,
         {'ratio':ratio,'limit':SLOWER_BOARD_WALL_RATIO_LIMIT})
    runtime_gates=p3_runtime_gates(ratio,serial_phase,prefetch_phase)
    note('combined_phase_rss_limit',runtime_gates['combined_rss']['pass'],runtime_gates['combined_rss'])
    return {'pass':all(c['pass'] for c in checks),'compiled':compiled,'tolerance':require_tol,'checks':checks,'boards':boards,
            'slower_board_paired_wall_ratio':ratio,
            'combined_phase_peak_host_rss_bytes':runtime_gates['combined_rss']['combined_peak_host_rss_bytes']}


def run_folders(root,arm='s6'):
    return [Path(root)/f'board{b}-{arm}'/arm for b in range(2)]


def p3_runtime_gates(slower_board_ratio,serial_phase,prefetch_phase):
    """Pure speed/RSS gates so the P3 thresholds can be exercised synthetically."""
    phases={'serial':serial_phase,'prefetch':prefetch_phase};peaks={}
    for name,phase in phases.items():
        peaks[name]=(phase or {}).get('peaks',{}).get('host_rss_bytes')
    combined=max((value for value in peaks.values() if isinstance(value,(int,float))),default=0)
    speed_ok=isinstance(slower_board_ratio,(int,float)) and slower_board_ratio<=SLOWER_BOARD_WALL_RATIO_LIMIT
    memory_ok=all(isinstance(value,(int,float)) and value>0 for value in peaks.values()) and combined<=COMBINED_RSS_LIMIT
    return {
        'speed':{'pass':speed_ok,'ratio':slower_board_ratio,'limit':SLOWER_BOARD_WALL_RATIO_LIMIT},
        'combined_rss':{'pass':memory_ok,'phase_peak_host_rss_bytes':peaks,
                        'combined_peak_host_rss_bytes':combined,'limit_bytes':COMBINED_RSS_LIMIT},
    }


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--serial-run',required=True,help='rehearsal output folder of the serial control (contains board0-s6/, board1-s6/)')
    p.add_argument('--prefetch-run',required=True);p.add_argument('--v7-receipt',required=True);p.add_argument('--compiled',choices=sorted(TOLERANCE),required=True)
    p.add_argument('--serial-phase-summary',required=True,help='JSON second return value from candidate.launch for serial control')
    p.add_argument('--prefetch-phase-summary',required=True,help='JSON second return value from candidate.launch for prefetch')
    p.add_argument('--out')
    a=p.parse_args()
    raw=Path(a.v7_receipt).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=V7_RECEIPT_SHA:raise SystemExit('EXACT_V7_RECEIPT')
    serial_phase=json.loads(Path(a.serial_phase_summary).read_bytes())
    prefetch_phase=json.loads(Path(a.prefetch_phase_summary).read_bytes())
    result=compare(run_folders(a.serial_run),run_folders(a.prefetch_run),json.loads(raw),a.compiled,
                   serial_phase=serial_phase,prefetch_phase=prefetch_phase)
    text=json.dumps(result,indent=1,default=str)
    if a.out:Path(a.out).write_text(text+'\n')
    print(text);raise SystemExit(0 if result['pass'] else 1)
