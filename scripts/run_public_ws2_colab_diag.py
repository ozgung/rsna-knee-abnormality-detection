"""One-T4 actual 39-image diagnostic from an immutable WS2 preparation package.

Run on Claude's dedicated Colab T4 after placing the external inputs named in
input-requirements.json at /content/r224_inputs (or --inputs). This never writes a
submission and cannot qualify a Kaggle two-T4 entry.
"""
import argparse
import csv
import hashlib
import json
import lzma
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def sha(path):
    with open(path,'rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def require(ok,code):
    if not ok:raise ValueError(code)


def load_package(package):
    manifest=json.loads((package/'MANIFEST.json').read_bytes())
    for relative,meta in manifest['files'].items():
        path=package/relative
        require(path.is_file() and path.stat().st_size==meta['bytes'] and sha(path)==meta['sha256'],
                'PACKAGE_HASH_'+relative)
    pack=json.loads(lzma.decompress((package/'colab-payload.xz').read_bytes()))
    canonical=json.dumps(pack,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    require(hashlib.sha256(canonical).hexdigest()==manifest['payload_sha256'],'PACKAGE_PAYLOAD_HASH')
    return pack,manifest


def verify_inputs(root,requirements):
    inputs=json.loads((root/'inputs.json').read_bytes())
    expected={'competition_root','r7_root','anatomy_root','weights_root','codec_root'}
    require(set(inputs)==expected,'INPUT_PATH_SCHEMA')
    paths={name:Path(value).resolve() for name,value in inputs.items()}
    require(all(path.is_dir() for path in paths.values()),'INPUT_DIRECTORY_MISSING')
    for name,files in requirements.items():
        folder=paths[name]
        for relative,pin in files.items():
            file=folder/relative
            require(file.is_file(),'INPUT_FILE_MISSING_'+name+'_'+relative)
            if pin is not None:require(sha(file)==pin,'INPUT_HASH_'+name+'_'+relative)
    return paths


def run_worker(module,cfg,root,arm,worker_environment,resource_snapshot):
    folder=root/f'board0-{arm}';folder.mkdir()
    config=root/f'board0-{arm}-config.json'
    from knee.public_s6_qualification import save
    save(config,{**cfg,'board':0,'arm':arm,'dev':cfg['staged_ids'],
        'physical':cfg['boards'][0],'out':str(folder)})
    start=time.monotonic();high={'host_rss_bytes':0,'output_disk_bytes':0};minimum=shutil.disk_usage(root).free
    with (root/f'board0-{arm}.log').open('xb') as log:
        child=subprocess.Popen([sys.executable,'-m',module,'--worker',str(config)],
                               env=worker_environment(0),stdout=log,stderr=subprocess.STDOUT)
        try:
            while child.poll() is None:
                s=resource_snapshot(root)
                for key in high:high[key]=max(high[key],s[key])
                minimum=min(minimum,s['disk_free_bytes'])
                time.sleep(.5)
            require(child.returncode==0,'ONE_T4_WORKER_'+arm+'_FAILED')
        finally:
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    child.kill();child.wait()
    receipt=json.loads((folder/arm/'receipt.json').read_bytes())
    require(receipt.get('status')=='PASS' and receipt.get('completed')==39 and receipt.get('zero_failed') is True,
            'ONE_T4_COMPLETE_'+arm)
    return receipt,{'wall_seconds':time.monotonic()-start,'peaks':high,'minimum_disk_free_bytes':minimum}


def ledger(root,arm,ids):
    import numpy as np
    path=root/f'board0-{arm}'/arm/'predictions.npz'
    with np.load(path,allow_pickle=False) as data:
        require(data['study_ids'].tolist()==ids and data['probabilities'].shape==(39,12)
                and data['statuses'].tolist()==['genuine']*39,'ONE_T4_LEDGER_'+arm)
        return data['probabilities'].copy()


def run(package,inputs_root,out):
    start=time.monotonic()
    require(not out.exists(),'FRESH_DIAGNOSTIC_OUTPUT')
    pack,manifest=load_package(package)
    requirements=json.loads((package/'input-requirements.json').read_bytes())
    paths=verify_inputs(inputs_root,requirements['files'])
    out.mkdir(parents=True)
    project=out/'project';code=project/'source'
    for name,content in pack['sources'].items():
        p=project/name if name.startswith('knee/') else code/name
        p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(content.encode())
    sys.path.insert(0,str(project))
    os.environ.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        HF_HUB_DISABLE_TELEMETRY='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',BUILD_MONAI='0')
    sys.dont_write_bytecode=True
    from knee import public_r7_recovery_entry as R7
    from knee import public_ws2_entry as WS2
    from knee import public_ws2_recovery as REC
    from knee.public_r7_recovery import recover as base_recover
    from knee.public_s6_qualification import LABELS, file_sha, save
    from knee.public_s6_recovery_entry import stage,worker_environment,resource_snapshot,probabilities
    from knee.public_fullfield_entry import read_series,test_contract
    import numpy as np
    import torch
    require(sys.version_info[:2]==(3,12) and torch.__version__==pack['cfg']['environment']['torch'],
            'PINNED_COLAB_PYTHON_TORCH_REQUIRED')
    for filename,pin in pack['codecs'].items():
        require(file_sha(paths['codec_root']/filename)==pin,'PINNED_OFFLINE_CODEC')
    subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps',
        *[str(paths['codec_root']/n) for n in pack['codecs']]],check=True)
    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name','--format=csv,noheader'],text=True)
    boards=[]
    for line in inventory.splitlines():
        index,uuid,name=[x.strip() for x in line.split(',')]
        boards.append({'index':int(index),'uuid':uuid,'name':name})
    require(len(boards)==1 and boards[0]['index']==0 and boards[0]['name']=='Tesla T4'
            and torch.cuda.device_count()==1,'DEDICATED_ONE_T4_DIAGNOSTIC')
    cfg=pack['cfg'];cfg.update(code=str(code),compact=str(paths['r7_root']),
        anatomy=str(paths['anatomy_root']/'anatomy.pt'),
        control_checkpoint=str(paths['weights_root']/cfg['public_manifest']['checkpoint_file']),
        crop224_checkpoint=str(paths['weights_root']/cfg['ws2_binding']['crop224_checkpoint']['file']))
    root=paths['competition_root']
    columns,visible,initial_records=test_contract(root,LABELS,visible=True)
    require(columns==cfg['binding']['columns'] and visible==cfg['binding']['visible_ids'],'FROZEN_VISIBLE3_BINDING')
    training=read_series(root/'train_series.csv')
    for sid in cfg['probe_order']:
        require(sid in training,'FROZEN_PROBE_METADATA')
    ids=cfg['probe_order']+visible
    require(len(ids)==39 and len(set(ids))==39,'FROZEN39_REGISTRY')
    bootstrap_seconds=time.monotonic()-start

    def fresh_pass(label,include_crop):
        pass_start=time.monotonic()
        folder=out/label;folder.mkdir()
        records=dict(initial_records)
        paths_for_study={sid:str(root/'test_series') for sid in visible}
        for sid in cfg['probe_order']:
            records[sid]=training[sid]
            paths_for_study[sid]=str(root/'train_series')
        staged=folder/'images-only'
        staged_binding=stage(staged,ids,records,paths_for_study)
        pass_cfg={**cfg,'staged':str(staged),'staged_ids':ids,'stage_binding':staged_binding,
                  'boards':boards,'crop224_sentinels':cfg['s6_sentinels']}
        save(folder/'runtime-config-private.json',pass_cfg)
        setup_seconds=time.monotonic()-pass_start
        reports={};phases={}
        arm_modules=[('old','knee.public_r7_recovery_rehearsal'),
                     ('s6','knee.public_r7_recovery_rehearsal')]
        if include_crop:arm_modules.append(('crop224','knee.public_ws2_entry'))
        for arm,module in arm_modules:
            reports[arm],phases[arm]=run_worker(module,pass_cfg,folder,arm,worker_environment,resource_snapshot)
        merge_start=time.monotonic()
        arrays={arm:ledger(folder,arm,ids) for arm in reports}
        healthy=np.zeros(39,bool)
        baseline=base_recover(arrays['old'],arrays['s6'],healthy,healthy)
        if include_crop:
            result=REC.recover(arrays['old'],arrays['s6'],arrays['crop224'],healthy,healthy,healthy)
            require(np.array_equal(baseline['control_rank'],result['ranks']['old320'])
                    and np.array_equal(baseline['s6_rank'],result['ranks']['r7']),
                    'SAME_PASS_UNCHANGED_A_ENDPOINT_RANKS')
            blend=result['blend']
        else:
            result=baseline;blend=baseline['blend']
        # These private 39 rows remain in ignored diagnostic output.
        with (folder/'rank-output-private.csv').open('w',newline='') as handle:
            writer=csv.writer(handle);writer.writerow([cfg['ws2_binding']['uid']]+LABELS)
            writer.writerows([[sid,*row] for sid,row in zip(ids,blend)])
        merge_seconds=time.monotonic()-merge_start
        report={'status':'DIAGNOSTIC_PASS_COMPLETE','label':label,'completed':39,
            'stage_binding':staged_binding,'worker_receipt_sha256':{
                arm:file_sha(folder/f'board0-{arm}'/arm/'receipt.json') for arm in reports},
            'worker_state':{arm:{'before':r['state_before'],'after':r['state_after']} for arm,r in reports.items()},
            'rank_csv_sha256':file_sha(folder/'rank-output-private.csv'),
            'times':{'setup_seconds':setup_seconds,'phases':phases,'merge_seconds':merge_seconds,
                'full_wall_seconds':time.monotonic()-pass_start},
            'resources':{'minimum_disk_free_bytes':min(v['minimum_disk_free_bytes'] for v in phases.values()),
                'peak_host_rss_bytes':max(v['peaks']['host_rss_bytes'] for v in phases.values()),
                'peak_output_disk_bytes':max(v['peaks']['output_disk_bytes'] for v in phases.values()),
                'peak_cuda_bytes_by_arm':{arm:r['peak_cuda_bytes'] for arm,r in reports.items()}},
            'recovery_union':result['union_recovered']}
        save(folder/'receipt.json',report)
        return report,reports,arrays,baseline

    a,a_workers,a_arrays,a_baseline=fresh_pass('A',False)
    ap,ap_workers,ap_arrays,ap_baseline=fresh_pass('Aprime',True)
    require(a['stage_binding']==ap['stage_binding'],'PAIRED_EXACT_IMAGE_STAGE')
    for arm,atol in (('old',1e-6),('s6',1e-3)):
        require(a_workers[arm]['state_before']==ap_workers[arm]['state_before']
                and a_workers[arm]['state_after']==ap_workers[arm]['state_after'],
                'PAIRED_UNCHANGED_'+arm+'_STATE')
        left,right=a_workers[arm]['records'],ap_workers[arm]['records']
        require([r['id'] for r in left]==[r['id'] for r in right]==ids
                and all(x['input']==y['input'] for x,y in zip(left,right)),
                'PAIRED_UNCHANGED_'+arm+'_SOURCE_INPUT')
        require(float(np.max(np.abs(a_arrays[arm]-ap_arrays[arm])))<=atol,
                'PAIRED_UNCHANGED_'+arm+'_PROBABILITY')
    require(float(np.max(np.abs(a_baseline['blend']-ap_baseline['blend'])))<=1e-3,
            'PAIRED_A_REPLAY_OUTPUT')
    ratio=ap['times']['full_wall_seconds']/a['times']['full_wall_seconds']
    report={'status':'DIAGNOSTIC_ONLY_PENDING_KAGGLE_QUALIFICATION','mode':'one_t4_colab',
        'completed':39,'physical_t4_uuid':boards[0]['uuid'],'base_source_sha256':cfg['binding']['source_bundle_sha256'],
        'package_manifest_sha256':file_sha(package/'MANIFEST.json'),
        'input_binding':a['stage_binding'],'pass_receipt_sha256':{
            'A':file_sha(out/'A'/'receipt.json'),'Aprime':file_sha(out/'Aprime'/'receipt.json')},
        'unchanged_a_endpoints':True,'recovery':{'baseline':a['recovery_union'],
            'candidate':ap['recovery_union']},
        'times':{'shared_bootstrap_seconds':bootstrap_seconds,
            'measured_one_t4_a_seconds':a['times']['full_wall_seconds'],
            'measured_one_t4_a_prime_seconds':ap['times']['full_wall_seconds'],
            'measured_one_t4_ratio':ratio,'frozen_v7_conservative_seconds':26553.01589682397,
            'ratio_calibrated_seconds_diagnostic_only':26553.01589682397*ratio},
        'resources':{'A':a['resources'],'Aprime':ap['resources']},
        'qualification':'PENDING fresh two-T4 Kaggle 39-study run and <=8.3h combined projection'}
    save(out/'receipt.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True)
    p.add_argument('--inputs',default='/content/r224_inputs')
    a=p.parse_args()
    package=Path(__file__).resolve().parent;out=Path(a.out).resolve()
    try:
        result=run(package,Path(a.inputs).resolve(),out)
        print(json.dumps({'status':result['status'],'ratio':result['times']['measured_one_t4_ratio']}))
    except BaseException as error:
        if out.is_dir():
            (out/'failure-private.json').write_text(json.dumps({'type':type(error).__name__,'detail':str(error)}))
        raise
