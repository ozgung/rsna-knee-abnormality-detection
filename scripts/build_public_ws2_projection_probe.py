"""Build one explicitly labelled R224 projection-waived probe; never relabel qualification.

The frozen inference modules and original builder remain unchanged. The mechanical
validator below preserves their checks, accepting only a terminal projection failure
instead of PASS and requiring actual elapsed time below the original nine-hour limit.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from scripts import build_public_ws2_entry as BASE
from knee import public_ws2_entry as WS2
from knee import public_ws2_checkpoint as C224
from knee import public_r7_recovery_entry as R7
from knee import public_ws2_recovery as REC
from knee.public_s6_qualification import LABELS, digest, file_sha, require, save
from knee.public_s6_recovery_entry import probabilities

VERSION=WS2.VERSION
PROJECTION_GATE_SECONDS=WS2.PROJECTION_GATE_SECONDS
MAX_SECONDS=WS2.MAX_SECONDS
crop_parity_ok=WS2.crop_parity_ok
projection=WS2.projection
WAIVER='projection-8.3h-20261015'
FAILED_RECEIPT_SHA='8b645d41ded79d946b57e1580586c00ca75849db20dda288d1adad5cc7319dd7'
AUTHORIZATION={'reported_by':'Claude',
    'source':'docs/claude-astra-handoff.md, 2026-10-15 11:28 UTC',
    'reported_user_choice':'Submit A-prime anyway',
    'independently_verified_direct_user_message':False,
    'scope':'one competitive probe; only the 8.3h projection gate; actual nine-hour limit unchanged'}


def require_projection_probe(r,binding,base_receipt_path,csv_path):
    """Replay frozen mechanics for this explicitly projection-waived probe only."""
    import csv
    base=json.loads(Path(base_receipt_path).read_bytes())
    require(digest(base.get('binding'))==binding['frozen_r7_binding_sha256'],
            'EXACT_ARCHIVED_V7_BINDING_DIGEST')
    import numpy as np
    R7.require_rehearsal(base,base['binding'],Path(base_receipt_path).parent/'visible-predictions.csv')
    require(r.get('status')=='FAIL' and r.get('failure_type')=='ValueError' and r.get('version')==VERSION and r.get('mode')=='rehearsal'
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
        and len(x.get('sentinels',[]))==4 and all(s.get('strict_reload') is True and crop_parity_ok(s['input'].get('crop224_cpu_cuda_max_abs'),s['input'].get('crop224_cpu_cuda_mean_abs')) for s in x['sentinels'])
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
    require(PROJECTION_GATE_SECONDS<p['conservative_seconds'] and 0<r['elapsed_seconds']<MAX_SECONDS,
            'COMBINED_PROJECTION_AND_ACTUAL_LIMIT')
    require(r.get('csv_sha256')==file_sha(csv_path),'R224_CSV_HASH')
    with Path(csv_path).open(newline='') as handle:rows=list(csv.reader(handle))
    require(rows[0]==[binding['uid']]+LABELS and len(rows)==4 and [x[0] for x in rows[1:]]==base['visible_ids'],
            'R224_CSV_SCHEMA_ORDER')
    probabilities([[float(x) for x in row[1:]] for row in rows[1:]],3)
    return True


def verify_probe_evidence(result_root,binding,user_waiver):
    """Inspect actual saved ledgers and reproduce the final CSV; PASS alone is insufficient."""
    require(user_waiver==WAIVER,'EXPLICIT_NAMED_PROJECTION_WAIVER_REQUIRED')
    root=Path(result_root)
    require(file_sha(root/'receipt.json')==FAILED_RECEIPT_SHA,'EXACT_WAIVED_FAILED_RECEIPT')
    failure=json.loads((root/'failure-private.json').read_bytes())
    require(failure=={'type':'ValueError','detail':'COMBINED_PROJECTION_AND_ACTUAL_LIMIT'},
            'ONLY_PROJECTION_FAILURE_MAY_BE_WAIVED')
    receipt_path=root/'receipt.json';base_path=root/'base'/'receipt.json';csv_path=root/'visible-predictions.csv'
    require(all(x.is_file() for x in (receipt_path,base_path,csv_path)),'FULL_REHEARSAL_FILES')
    import numpy as np
    r=json.loads(receipt_path.read_bytes());base=json.loads(base_path.read_bytes())
    require_projection_probe(r,binding,base_path,csv_path)
    old_rows={row['id']:row for worker in base['workers']['old'] for row in worker['records']}
    for worker in r['crop224_workers']:
        require(worker['model']['state_sha256']==worker['state_before']==worker['state_after'],
                'NATIVE_LOADED_STATE_PIN')
        for row in worker['records']:
            require(row['status']=='genuine' and row['state_healthy'] and all(
                row['input'][key]==old_rows[row['id']]['input'][key]
                for key in ('volume_sha256','mask_sha256')),'ALL39_NATIVE_INPUT_STATE_BINDING')
        for sentinel in worker['sentinels']:
            hi=sentinel['input']['crop224_cpu_cuda_max_abs'];mean=sentinel['input']['crop224_cpu_cuda_mean_abs']
            require(type(hi) is float and type(mean) is float and math.isfinite(hi) and math.isfinite(mean)
                    and 0<=mean<=hi<=1e-4 and mean<=1e-6,'FINITE_NONNEGATIVE_CROP_STATISTICS')
    ids=json.loads((root/'base'/'assignment.json').read_bytes())['ids']
    require(len(ids)==39 and len(set(ids))==39 and ids==base['binding']['s6_probe_ids']+base['binding']['old_historical_ids']+base['visible_ids'],
            'EXACT_FROZEN39_ORDER')
    arms={}
    for name,base_name,folder,reports in (
            ('old320','old',root/'base',base['workers']['old']),
            ('r7','s6',root/'base',base['workers']['s6']),
            ('crop224','crop224',root,r['crop224_workers'])):
        arms[name]=WS2.merged_arm(ids,reports,folder,base_name)
    computed=REC.recover(*(arms[name][0] for name in REC.ARMS),
                         *(arms[name][1] for name in REC.ARMS))
    frozen=R7.recover(arms['old320'][0],arms['r7'][0],arms['old320'][1],arms['r7'][1])
    require(np.array_equal(frozen['control_rank'],computed['ranks']['old320'])
            and np.array_equal(frozen['s6_rank'],computed['ranks']['r7']),
            'SAVED_UNCHANGED_BASELINE_RANKS')
    hashes={'old_rank_sha256':hashlib.sha256(frozen['control_rank'].tobytes()).hexdigest(),
        'r7_rank_sha256':hashlib.sha256(frozen['s6_rank'].tobytes()).hexdigest(),
        'blend_sha256':hashlib.sha256(frozen['blend'].tobytes()).hexdigest()}
    require(r.get('baseline_full_population')==hashes,'SAVED_BASELINE_OUTPUT_HASHES')
    require(computed['union_recovered']==0,'FRESH_REHEARSAL_NO_FAILURE')
    index={sid:i for i,sid in enumerate(ids)}
    with csv_path.open(newline='') as handle:rows=list(csv.reader(handle))
    actual=np.array([[float(x) for x in row[1:]] for row in rows[1:]])
    expected=computed['blend'][[index[s] for s in base['visible_ids']]]
    require(actual.shape==(3,12) and np.max(np.abs(actual-expected))<=5e-10,'RECOMPUTED_CSV_MATCH')
    return {'receipt_sha256':file_sha(receipt_path),'base_receipt_sha256':file_sha(base_path),
            'csv_sha256':file_sha(csv_path),'qualified_r224_state':r['crop224_workers'][0]['state_before'],
            'projection':r['projection'],'all_six_ledgers_recomputed':True}



def build_probe(*,rehearsal_build,rehearsal_result,destination,slug,user_waiver):
    require(user_waiver==WAIVER,'EXPLICIT_NAMED_PROJECTION_WAIVER_REQUIRED')
    project=Path(__file__).resolve().parents[1];dest=Path(destination)
    require(not dest.exists() and dest.resolve().is_relative_to(project/'outputs'),'FRESH_IGNORED_BUILD')
    require(re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*',slug) is not None and len(slug)<=50,'TITLE_ID_SLUG_CONSISTENCY')
    archive=project/'outputs/kaggle/public-r7-recovery-rehearsal-20261013-v7'
    prepared=BASE.verify_preparation_build(rehearsal_build,archive/'notebook.ipynb',archive/'build-receipt.json')
    verification=verify_probe_evidence(rehearsal_result,prepared['binding'],user_waiver)
    source=project/'outputs/kaggle/public-r7-recovery-entry-20261013-v2'
    pack,_,entry_build=BASE.archived_payload(source/'notebook.ipynb',source/'build-receipt.json','entry')
    replay,_,_=BASE.archived_payload(archive/'notebook.ipynb',archive/'build-receipt.json','rehearsal')
    shared=set(pack['sources'])&set(replay['sources'])
    require(all(pack['sources'][name]==replay['sources'][name] for name in shared
                if name!='knee/public_r7_recovery_rehearsal.py'),'ENTRY_REHEARSAL_BASE_SOURCE_IDENTITY')
    for name in BASE.NEW_FILES:pack['sources'][name]=(project/name).read_text()
    source_hashes={name:file_sha(project/name) for name in BASE.NEW_FILES}
    binding=BASE.ws2_binding(replay,source_hashes)
    require(binding==prepared['binding'],'EXACT_QUALIFIED_INFERENCE_SOURCE_BINDING')
    cfg=pack['cfg'];cfg['ws2_binding']=binding;cfg['r224_state_pin']=verification['qualified_r224_state']
    waiver_record={'id':WAIVER,'authorization':AUTHORIZATION,'failed_receipt_sha256':FAILED_RECEIPT_SHA,
        'status':'USER_REPORTED_PROJECTION_WAIVER_NOT_QUALIFICATION_PASS',
        'projection':verification['projection'],'actual_limit_seconds':MAX_SECONDS,
        'all_six_ledgers_recomputed':verification['all_six_ledgers_recomputed']}
    cfg['projection_probe_waiver']=waiver_record
    require(not any(k in cfg for k in BASE.PRIVATE_CFG) and not any(k in cfg['binding'] for k in BASE.PRIVATE_BINDING),
            'PURE_ENTRY_PRIVATE_FIXTURES_STRIPPED')
    # The new validator/builder are not part of the inference closure.
    require(set(pack['sources'])==set(entry_build['source_hashes'])|set(BASE.NEW_FILES),
            'NO_ADDITIONAL_RUNTIME_SOURCES')
    script=BASE.bootstrap(pack)
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python3','language':'python'}},
        'cells':[{'id':'a-prime-image-inference','cell_type':'code','metadata':{},'execution_count':None,
                  'outputs':[],'source':script.splitlines(True)}]}
    meta=json.loads((Path(rehearsal_build)/'kernel-metadata.json').read_bytes())
    meta.update(id='ozgung/'+slug,title=slug)
    raw=json.dumps(nb,indent=2)+'\n';require(len(raw.encode())<900000,'NOTEBOOK900K_CAP')
    dest.mkdir(parents=True);(dest/'notebook.ipynb').write_text(raw)
    (dest/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    receipt={'status':'BOUND_PROJECTION_WAIVED_PROBE_BUILD','mode':'entry','builder_version':'ws2-projection-probe-v1',
        'builder_sha256':file_sha(__file__),'frozen_builder_sha256':file_sha(BASE.__file__),
        'binding':binding,'payload_sha256':digest(pack),'waiver':waiver_record,
        'source_hashes':{name:hashlib.sha256(text.encode()).hexdigest() for name,text in pack['sources'].items()},
        'files':{name:file_sha(dest/name) for name in ('notebook.ipynb','kernel-metadata.json')},
        'rehearsal_verification':verification,'notebook_bytes':len(raw.encode()),
        'source_files':len(pack['sources']),'runtime_checks_exercised':False,
        'qualification_pass':False,'competitive_result':False}
    save(dest/'build-receipt.json',receipt)
    save(dest/'MANIFEST.json',{'status':receipt['status'],'payload_sha256':digest(pack),
        'files':{name:{'sha256':file_sha(dest/name),'bytes':(dest/name).stat().st_size}
                 for name in ('notebook.ipynb','kernel-metadata.json','build-receipt.json')}})
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('rehearsal-build','rehearsal-result','destination','slug','user-waiver'):
        p.add_argument('--'+name,required=True)
    r=build_probe(**vars(p.parse_args()))
    print(json.dumps({k:r[k] for k in ('status','payload_sha256','files','qualification_pass')}))
