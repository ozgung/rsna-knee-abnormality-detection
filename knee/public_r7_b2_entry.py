"""Selection-B probe B2: Dread CC0 Raptor v4 K62 old arm (exactly as B0) + heliosli R7, 0.5/0.5.

No nartaa/OAI-trained weights are attached. Both workers are the exact qualified bytecode: the B0
Dread old-arm worker (knee.public_s6_recovery_b0) and the v7 R7 worker (knee.public_r7_recovery_entry),
rebound to this module's globals; merge and the complete run core are the default 0.954 bytecode.
The blend is the frozen capped-recovery algorithm at 0.5/0.5 with R7 at '%.9g' (knee.public_r7_recovery).
"""
import json
from pathlib import Path
import types
import knee.public_r7_recovery_entry as R7
import knee.public_s6_recovery_b0 as B0
import knee.public_s6_recovery_entry as base
import knee.public_r7_b1_entry as B1

globals().update({k:v for k,v in vars(R7).items() if not k.startswith('__')})
from knee.public_s6_recovery_b0 import (OLD_PROFILE,DREAD_CHECKPOINT_SHA256,DREAD_CHECKPOINT_BYTES,DREAD_STATE_SHA256,DREAD_GRID_SHA256,RAPTOR_WINDOWS,RAPTOR_INPUT,RAPTOR_IMAGE,RAPTOR_SLOTS,
    PUBLIC_CATCH_SCOPES,HEADER_DATA_LOCAL,ALL_EMPTY_SITE,GUARDED_PARTIAL_CASES,RAPTOR62_COUNT_KEYS,raptor_profile_gate,raptor_windows,raptor_forward,install_raptor_old_guards,public_guard,raptor_stress,raptor62_gate)

VERSION='r7-dread-capped-recovery-b2-v1'
ENTRY_MODULE='knee.public_r7_b2_entry'
REHEARSAL_MODULE='knee.public_r7_b2_rehearsal'
METHOD={**R7.METHOD,'old':B0.METHOD['old'],'old_profile':'raptor','variant':'B2: Dread CC0 Raptor v4 K62 + R7 0.5/0.5; no nartaa/OAI-trained weights'}
rehearsal_hook=None


def require_rehearsal(r,binding,csv_path):
    """B2 contract: the B0 Dread old-arm gates plus the v7 R7 gates and default merge gates."""
    import math
    import numpy as np
    from knee import raptor_input as RI
    B1.require_common(r,binding,csv_path,VERSION,METHOD)
    require(binding.get('old_profile')==OLD_PROFILE and r.get('sentinel_ids',{}).keys()=={'old','s6'},'EXACT_B2_PROFILE')
    old0=r.get('workers',{}).get('old',[{}])[0]
    stress=old0.get('partial_input_stress',{})
    require(stress.get('status')=='PASS' and stress.get('cases')==dict.fromkeys(RI.STRESS_CASES,'PASS') and stress.get('case_count')==16 and stress.get('uses_raw_data') is False and stress.get('uses_labels') is False,'REHEARSAL_RAPTOR_PARTIAL_INPUT_STRESS')
    guarded=old0.get('guarded_partial_input_stress',{})
    require(guarded.get('status')=='PASS' and guarded.get('guarded') is True and guarded.get('case_count')==10 and guarded.get('cases')==dict.fromkeys(GUARDED_PARTIAL_CASES,'PASS'),'GUARDED_RAPTOR_PARTIAL_TEN')
    require('old' in r.get('phases',{}) and r['phases']['old']['wall_seconds']>0 and r['phases']['old']['peaks']['host_rss_bytes']>0 and r['phases']['old']['peaks']['output_disk_bytes']>0 and r['phases']['old']['minimum_disk_free_bytes']>0,'CONCURRENT_RESOURCE_MEASUREMENT')
    rr=r.get('workers',{}).get('old',[]);require(len(rr)==2,'REHEARSAL_WORKERS')
    require(all(x.get('recovery_injection',{}).get('status')=='PASS' and x['recovery_injection'].get('failure_status')=='data_local_failure' and x['recovery_injection'].get('injected_rows_genuine') is False and x['recovery_injection'].get('state_healthy') is True for x in rr),'ACTUAL_CLASSIFIED_RECOVERY_INJECTIONS')
    for board,x in enumerate(rr):
        require(x['status']=='PASS' and x['board']==board and x['state_before']==x['state_after'] and x['completed']==len(x['assigned']) and x['peak_cuda_bytes']>0 and x['peak_rss_bytes']>0 and x['zero_failed'] is True and x['zero_imputed'] is True and x['fp32_recoveries']==0,'REHEARSAL_WORKER_STATE_RESOURCE')
        require(x['state_before']['state']==binding['qualified_state']['old_state']==DREAD_STATE_SHA256 and x.get('constructor_grid_sha256')==binding['qualified_state']['old_grid']==DREAD_GRID_SHA256 and x.get('old_profile')==OLD_PROFILE and x.get('source_guard_receipt',{}).get('profile')=='raptor','QUALIFIED_COMPLETE_STATE_BINDING')
        require(x['environment']['torch']==binding['environment']['torch'] and x['environment']['packages']==binding['environment']['packages'],'QUALIFIED_ENVIRONMENT_BINDING')
        require(x['environment']['physical_uuid']==r['boards'][board]['uuid'] and x['environment']['cuda_device_count']==1,'ISOLATED_BOARD_RUNTIME')
        require(len(x['sentinels'])==4 and [s['id'] for s in x['sentinels']]==r['sentinel_ids']['old'] and all(s['strict_reload'] for s in x['sentinels']),'BOTH_BOARD_SENTINELS')
        for s in x['sentinels']:
            p=np.array(s['probabilities']);q=np.array(s['reloaded_probabilities'])
            require(binding['sentinel_probabilities']['old'][s['id']] is None and float(np.max(np.abs(p-q)))<=1e-6 and s.get('strict_source_comparison') in ('PASS','ineligible_partial_input'),'OLD_SENTINEL_RELOAD_GATE')
    for a,b in zip(rr[0]['sentinels'],rr[1]['sentinels']):
        require(a['input']==b['input'] and max(abs(x-y) for x,y in zip(a['probabilities'],b['probabilities']))<=CROSS_BOARD_TOLERANCE['old'],'CROSS_BOARD_SENTINEL_PARITY')
    s6rows=B1.require_r7_workers(r,binding)
    oldrows={row['id']:row for worker in rr for row in worker['records']}
    require(len(oldrows)==39 and set(oldrows)==set(s6rows),'COMPLETE39_MECHANICS')
    require(set(binding['s6_probe_ids']).isdisjoint(binding['old_historical_ids']) and not set(binding['s6_probe_ids']+binding['old_historical_ids'])&set(r['visible_ids']),'DISJOINT39_MECHANICS')
    strict=[]
    for sid in binding['s6_probe_ids']+binding['old_historical_ids']:
        row=oldrows[sid];strict.append(row.get('strict_source_comparison'))
        require(row.get('source_reference_parity') is True and row.get('strict_reload') is True and 0<=row.get('reload_max_abs',math.inf)<=1e-6 and row.get('strict_source_comparison') in ('PASS','ineligible_partial_input'),'ALL36_OLD_SOURCE_STRICT_REPEAT_RELOAD')
    require('PASS' in strict,'STRICT_PYDICOM_SOURCE_COMPARISON_EXERCISED')
    contract=binding['old_qualification']['raptor62']
    for sid in r['visible_ids']:
        g=oldrows[sid].get('raptor62',{})
        require(g.get('decode_counts_equal') is True and g.get('atol')==contract['visible_atol'] and g.get('within_atol') is True and 0<=g.get('max_abs',math.inf)<=contract['visible_atol'],'RAPTOR62_VISIBLE_RECEIPT_PARITY')


heliosli_parity=None


def _bind(function):
    clone=types.FunctionType(function.__code__,globals(),function.__name__,function.__defaults__,function.__closure__)
    clone.__kwdefaults__=function.__kwdefaults__;clone.__doc__=function.__doc__;clone.__qualname__=function.__qualname__
    return clone


old_worker=_bind(B0.old_worker)
r7_worker=_bind(R7.r7_worker)
launch=_bind(R7.launch)
merge=_bind(base.merge)
run=_bind(base.run)
r7_parity=_bind(B1.r7_parity)


if __name__=='__main__':
    import argparse
    import os
    require(Path(__file__).resolve()==(Path(os.environ['KNEE_ENTRY_PROJECT'])/'knee/public_r7_b2_entry.py').resolve(),'EXACT_FRESH_SOURCE_IMPORT')
    p=argparse.ArgumentParser();p.add_argument('--worker',required=True);a=p.parse_args();cfg=json.loads(Path(a.worker).read_bytes())
    (old_worker if cfg['arm']=='old' else r7_worker)(a.worker)
