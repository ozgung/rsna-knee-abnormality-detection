"""Separate B0′ checkpoint-swap packet; entry requires fresh saved acceptance.

Reuses the immutable B0 image-only packet and bootstrap. Never launches, submits
or accepts competition terms. Private references exist only in the rehearsal.
"""
import argparse
import ast
import base64
import csv
import hashlib
import json
import lzma
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from knee import public_s6_recovery_b0 as B0
from knee import public_s6_recovery_b0prime as P
from knee.public_s6_qualification import canonical,digest,file_sha,require,save
from scripts import build_public_s6_recovery_b0 as BB

PROJECT=Path(__file__).resolve().parents[1]
ARCHIVE=PROJECT/'outputs/kaggle/knee-s6-dread-b0-rehearsal-20261013'
RESULT=PROJECT/'outputs/kaggle/knee-s6-dread-b0-rehearsal-20261013-result-v1/s6-recovery-entry'
NOTEBOOK_SHA='9de1fcc114416f773f57d88d62da1b6396f5e1896737838a146d5f46bc216851'
PAYLOAD_SHA='15f5c50c8c0b406ba1f5ad3bfdd6f4148aeea3872bbf71ebd5de5de7527c40c0'
METADATA_SHA='24e8e6e65c789b3518ed0bb1714847f81f085078855c0984a48f8e9fc8df8a45'
RESULT_SHA='adab198e377bfc3ecf5abe961f306be686301e6ffab78393b018f68f4fd8388d'
FROZEN={**BB.DEFAULT_MODULE_SHA,
    'knee/public_s6_recovery_b0.py':'8b546973eba173fdca0e48584f4a95605ac63a7996fbe4e67a5d5a371d491d0c',
    'knee/public_s6_recovery_b0_rehearsal.py':'115bc2bb885b72a2ce145d8086ef1a7d5448eff251b728a7a1a23dbbe7abef01',
    'scripts/qualify_b3_drfull_cpu.py':'6d3ddf6f721c91018d67e69d410e8d08408f249ee24b88916f913e9e709439f8'}
RUNTIME='knee/public_s6_recovery_b0prime.py'
EXTENSION='knee/public_s6_recovery_b0prime_rehearsal.py'
NEW_FILES=(RUNTIME,EXTENSION,'scripts/build_public_b0prime.py','scripts/qualify_b0prime_drfull_cpu.py')
PRIVATE=(*BB.PRIVATE_BINDING,'reader_reference','s6_reference')
DATASETS=BB.DATASETS[:-1]+[f'{P.DATASET}/{P.DATASET_VERSION}']


def unpack_notebook(path):
    notebook=json.loads(Path(path).read_bytes());code=''.join(notebook['cells'][0]['source'])
    tree=ast.parse(code)
    encoded=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id=='encoded' for t in n.targets))
    return json.loads(lzma.decompress(base64.b64decode(encoded,validate=True))),code


def archived():
    require(file_sha(ARCHIVE/'notebook.ipynb')==NOTEBOOK_SHA
            and file_sha(ARCHIVE/'kernel-metadata.json')==METADATA_SHA,'EXACT_B0_ARCHIVE')
    packet,code=unpack_notebook(ARCHIVE/'notebook.ipynb')
    require(digest(packet)==PAYLOAD_SHA,'EXACT_B0_PAYLOAD')
    for name,pin in FROZEN.items():
        require(file_sha(PROJECT/name)==pin,'FROZEN_B0_DEPENDENCY_'+name)
        if name in packet['sources']:require(hashlib.sha256(packet['sources'][name].encode()).hexdigest()==pin,'FROZEN_PACKET_SOURCE_'+name)
    require(file_sha(RESULT/'receipt.json')==RESULT_SHA,'EXACT_B0_RESULT')
    r=json.loads((RESULT/'receipt.json').read_bytes())
    B0.require_rehearsal(r,packet['cfg']['binding'],RESULT/'visible-predictions.csv')
    return packet,code,r


def cpu_qualification(path,manifest):
    r=json.loads(Path(path).read_bytes())
    # Python ast.unparse formatting can differ between CPU and builder Python.
    formatting={'extracted_module_sha256','exception_policy'}
    same=lambda m:{k:v for k,v in m.items() if k not in formatting}
    require(r['checkpoint_sha256']==P.DREAD_CHECKPOINT_SHA256 and r['checkpoint_bytes']==P.DREAD_CHECKPOINT_BYTES
        and r['state_sha256']==P.DREAD_STATE_SHA256 and r['constructor_grid_sha256']==P.DREAD_GRID_SHA256
        and r['strict_state_load'] is True and r['labels']==P.LABELS
        and r['windows_shape']==[62,3,384,384] and r['forward_windows']==62
        and 0<=r['cpu_repeat_max_abs']<=1e-6 and r['dread_pin_method_check'] is True
        and r['device']=='cpu' and r['uses_raw_data'] is False and r['uses_labels'] is False
        and r['checkpoint_meta']=={'epoch':[9,10,11],'swa_over':[9,10,11]}
        and set(r['manifest'])==set(manifest) and same(r['manifest'])==same(manifest),'B0PRIME_FRESH_CPU_K62')
    return {'receipt_sha256':file_sha(path),'state_sha256':r['state_sha256'],'grid_sha256':r['constructor_grid_sha256'],
            'torch':r['torch'],'device':'cpu','windows':62,'strict_state_load':True,'manifest_formatting_fields_excluded':sorted(formatting)}


def packet(cpu_receipt):
    old,_,result=archived();pack=json.loads(canonical(old));cfg=pack['cfg'];binding=cfg['binding']
    extensions={'knee/public_s6_recovery_rehearsal.py','knee/public_s6_recovery_b0_rehearsal.py',EXTENSION}
    pack['sources'].update({n:(PROJECT/n).read_text() for n in (RUNTIME,EXTENSION)})
    manifest={**cfg['public_manifest'],'checkpoint_sha256':P.DREAD_CHECKPOINT_SHA256,
              'checkpoint_file':P.CHECKPOINT_FILE,'weight_dataset':P.DATASET,
              'attribution':'Own dr-full SWA9-11, Dread v4 init; B0 K62 single-checkpoint swap'}
    P.raptor_profile_gate(None,manifest);cpu=cpu_qualification(cpu_receipt,manifest)
    cfg.update(public_manifest=manifest,control_state_sha256=P.DREAD_STATE_SHA256,control_grid_sha256=P.DREAD_GRID_SHA256)
    cfg['raptor62_reference']={sid:{'counts':v['counts']} for sid,v in cfg['raptor62_reference'].items()}
    cfg.pop('raptor62_visible_atol')
    reader={arm:{row['id']:{'input':row['input'],**({'decode_counters':row['decode_counters']} if arm=='old' else {})}
                 for w in result['workers'][arm] for row in w['records']} for arm in ('old','s6')}
    ids=binding['s6_probe_ids']+binding['old_historical_ids']+binding['visible_ids']
    shards=[(w,RESULT/f'board{i}-s6/s6/predictions.npz') for i,w in enumerate(result['workers']['s6'])]
    probs,missing=B0.merge(ids,shards,'s6');require(not missing.any(),'B0_ALL39_S6_REFERENCE')
    s6_reference={sid:p.tolist() for sid,p in zip(ids,probs)}
    binding.update(version=P.VERSION,variant='B0prime-drfull-k62',method=P.METHOD,old_profile=P.OLD_PROFILE,
        qualified_state={**binding['qualified_state'],'old_state':P.DREAD_STATE_SHA256,'old_grid':P.DREAD_GRID_SHA256},
        public_manifest=manifest,old_qualification={'policy':'Fresh dr-full K62 CPU strict-load/pins; B0 visible reader counts only; all36 source/reload and both-board parity; no old-weight probability reference',
            'cpu':cpu,'original_b0_raptor62':binding['old_qualification']['raptor62']},
        archive={'notebook_sha256':NOTEBOOK_SHA,'payload_sha256':PAYLOAD_SHA,'result_sha256':RESULT_SHA},
        checkpoint={'dataset':P.DATASET,'dataset_version':P.DATASET_VERSION,'sha256':P.DREAD_CHECKPOINT_SHA256,
                    'bytes':P.DREAD_CHECKPOINT_BYTES,'export_windows':42,'probe_windows':62,'swa_epochs':[9,10,11]},
        scientific_scope='Exposed development: dr-full trained on all 4345 studies; leaderboard swap probe only, no clean holdout claim',
        qualification_seconds=P.QUALIFICATION_SECONDS,reader_reference=reader,s6_reference=s6_reference,
        development_windows=cfg['development_windows'],
        builder_sha256=file_sha(__file__),new_files_sha256={n:file_sha(PROJECT/n) for n in NEW_FILES},
        source_bundle_sha256=digest({n:s for n,s in pack['sources'].items() if n not in extensions}),
        rehearsal_extension_sha256=digest({n:pack['sources'][n] for n in extensions}))
    binding['sentinel_inputs']['old']={sid:reader['old'][sid]['input'] for sid in cfg['old_sentinels']}
    for name,source in pack['sources'].items():compile(source,name,'exec')
    return pack


def bootstrap(pack):
    _,template,_=archived()
    # Preserve the audited bootstrap; change only sealed payload, runtime import
    # and the one checkpoint attachment. No new network or label-reading path.
    tree=ast.parse(template);lines=template.splitlines(True);changes=[]
    for node in tree.body:
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name):
            name=node.targets[0].id
            if name in ('encoded','expected_payload'):
                value=base64.b64encode(lzma.compress(canonical(pack),preset=9)).decode() if name=='encoded' else digest(pack)
                changes.append((node.lineno-1,node.end_lineno,f'{name}={value!r}\n'))
    require(len(changes)==2,'EXACT_BOOTSTRAP_PAYLOAD_SITES')
    for start,end,value in sorted(changes,reverse=True):lines[start:end]=[value]
    code=''.join(lines)
    old='from knee.public_s6_recovery_b0 import run';require(code.count(old)==1,'EXACT_RUNTIME_IMPORT_SITE')
    code=code.replace(old,'from knee.public_s6_recovery_b0prime import run')
    old=repr('dreaddevelopment/raptor-knee-widedense');require(code.count(old)==1,'EXACT_CHECKPOINT_MOUNT_SITE')
    code=code.replace(old,repr(P.DATASET));compile(code,'<b0prime-bootstrap>','exec')
    return code


def verify_preparation(root,cpu_receipt):
    root=Path(root);r=json.loads((root/'build-receipt.json').read_bytes());m=json.loads((root/'MANIFEST.json').read_bytes())
    names={'notebook.ipynb','kernel-metadata.json','build-receipt.json'}
    require(set(m['files'])==names,'MANIFEST_COVERAGE')
    for name,pin in m['files'].items():require(pin=={'sha256':file_sha(root/name),'bytes':(root/name).stat().st_size},'IMMUTABLE_BUILD_'+name)
    expected=packet(cpu_receipt);actual,code=unpack_notebook(root/'notebook.ipynb')
    require(r['status']=='PREPARATION_ONLY' and r['mode']=='rehearsal' and actual==expected
            and code==bootstrap(expected) and r['binding']==expected['cfg']['binding']
            and r['payload_sha256']==digest(actual),'EXACT_PREPARATION')
    meta=json.loads((root/'kernel-metadata.json').read_bytes());require(meta==metadata(meta['title']),'EXACT_PRIVATE_OFFLINE_T4_METADATA')
    require(r['files']=={n:file_sha(root/n) for n in ('notebook.ipynb','kernel-metadata.json')},'BUILD_FILE_HASHES')
    require(m['mode']=='rehearsal' and m['payload_sha256']==r['payload_sha256'],'MANIFEST_IDENTITY')
    return r


def verify_result(root,binding):
    import numpy as np
    root=Path(root);r=json.loads((root/'receipt.json').read_bytes());csv_path=root/'visible-predictions.csv'
    P.require_rehearsal(r,binding,csv_path)
    ids=json.loads((root/'assignment.json').read_bytes())['ids']
    require(ids==binding['s6_probe_ids']+binding['old_historical_ids']+binding['visible_ids'],'FROZEN39_ORDER')
    arms={}
    for arm in ('old','s6'):
        for i,w in enumerate(r['workers'][arm]):
            require(json.loads((root/f'board{i}-{arm}'/arm/'receipt.json').read_bytes())==w,'SAVED_WORKER_RECEIPT_BINDING')
        arms[arm]=P.merge(ids,[(w,root/f'board{i}-{arm}'/arm/'predictions.npz') for i,w in enumerate(r['workers'][arm])],arm)
        require(not arms[arm][1].any(),'ALL39_GENUINE')
    reference=np.array([binding['s6_reference'][sid] for sid in ids])
    require(float(np.max(np.abs(arms['s6'][0]-reference)))<=1e-6,'UNCHANGED_ALL39_S6_PROBABILITIES')
    for i in range(2):
        with np.load(root/f'board{i}-s6/s6/predictions.npz',allow_pickle=False) as a:
            mean=np.asarray(a['logits'],dtype=np.float32).mean(1).astype(np.float64)
            predicted=np.exp(-np.logaddexp(0.,-mean))
            require(float(np.max(np.abs(predicted-a['probabilities'])))<=1e-6,'ALL39_S6_MEAN_LOGIT_SIGMOID')
    indices=[ids.index(sid) for sid in r['visible_ids']]
    expected=P.recover(arms['old'][0][indices],arms['s6'][0][indices],arms['old'][1][indices],arms['s6'][1][indices])['blend']
    with csv_path.open(newline='') as h:rows=list(csv.reader(h))
    require(np.max(np.abs(np.array([[float(v) for v in row[1:]] for row in rows[1:]])-expected))<=5e-10,'RECOMPUTED_VISIBLE_70_30_BLEND')
    measured=r['projection']['measured_inputs']
    require(measured['setup_seconds']==r['setup_seconds'] and measured['setup_seconds']>=0
            and measured['merge_seconds']>=0,'PROJECTION_INPUTS')
    projected=P.projection({'development_windows':binding['development_windows']},
                           r['phases'],r['workers'],measured['setup_seconds'],measured['merge_seconds'])
    require(projected==r['projection'],'RECOMPUTED_PROJECTION')
    return {'receipt_sha256':file_sha(root/'receipt.json'),'csv_sha256':file_sha(csv_path),
            'all_four_ledgers_recomputed':True,'projection':r['projection']}


def metadata(slug):
    BB.kernel_title(slug)
    m=json.loads((ARCHIVE/'kernel-metadata.json').read_bytes())
    m.update(id='ozgung/'+slug,title=slug,dataset_sources=DATASETS,machine_shape='NvidiaTeslaT4')
    return m


def build(destination,slug,cpu_receipt,*,mode='rehearsal',rehearsal_build=None,rehearsal_result=None):
    require(mode in ('rehearsal','entry'),'MODE')
    if mode=='entry':require(rehearsal_build is not None and rehearsal_result is not None,'ENTRY_REQUIRES_FRESH_REHEARSAL')
    dest=Path(destination);require(not dest.exists() and dest.resolve().is_relative_to(PROJECT/'outputs'),'FRESH_IGNORED_BUILD')
    pack=packet(cpu_receipt);cfg=pack['cfg']
    # Store measured projection inputs for independent saved-result replay.
    if mode=='entry':
        prepared=verify_preparation(rehearsal_build,cpu_receipt)
        require(prepared['binding']==cfg['binding'],'EXACT_REHEARSAL_BINDING')
        checked=verify_result(rehearsal_result,cfg['binding'])
        cfg['mode']='entry';cfg['binding']={k:v for k,v in cfg['binding'].items() if k not in (*PRIVATE,'development_windows')}
        cfg['binding']['qualified_rehearsal']=checked
        for key in ('probes','probe_order','old_sentinels','s6_sentinels','development_windows','raptor62_reference'):cfg.pop(key,None)
        for name in ('knee/public_s6_recovery_rehearsal.py','knee/public_s6_recovery_b0_rehearsal.py',EXTENSION):pack['sources'].pop(name)
    code=bootstrap(pack);nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python3','language':'python'}},
        'cells':[{'id':'image-only-b0prime','cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(True)}]}
    raw=json.dumps(nb,indent=2)+'\n';require(len(raw.encode())<900000,'NOTEBOOK900K_CAP')
    dest.mkdir(parents=True);(dest/'notebook.ipynb').write_text(raw);save(dest/'kernel-metadata.json',metadata(slug))
    changed=subprocess.check_output(['git','status','--porcelain','--',*NEW_FILES,*FROZEN],cwd=PROJECT,text=True).strip()
    r={'status':'PREPARATION_ONLY' if mode=='rehearsal' else 'QUALIFIED_ENTRY_BUILD','mode':mode,'binding':cfg['binding'],
       'payload_sha256':digest(pack),'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),
       'source_state':('uncommitted implementation' if changed else 'committed implementation')+'; exact file hashes bound',
       'files':{n:file_sha(dest/n) for n in ('notebook.ipynb','kernel-metadata.json')},'notebook_bytes':len(raw.encode()),
       'source_hashes':{n:hashlib.sha256(s.encode()).hexdigest() for n,s in pack['sources'].items()},'runtime_checks_exercised':False}
    save(dest/'build-receipt.json',r);save(dest/'MANIFEST.json',{'mode':mode,'payload_sha256':digest(pack),
        'files':{n:{'sha256':file_sha(dest/n),'bytes':(dest/n).stat().st_size} for n in (*r['files'],'build-receipt.json')}})
    return r


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('destination','slug','cpu-receipt'):p.add_argument('--'+name,required=True)
    p.add_argument('--mode',choices=['rehearsal','entry'],default='rehearsal')
    for name in ('rehearsal-build','rehearsal-result'):p.add_argument('--'+name)
    r=build(**vars(p.parse_args()));print(json.dumps({k:r[k] for k in ('status','mode','files','payload_sha256','notebook_bytes')}))
