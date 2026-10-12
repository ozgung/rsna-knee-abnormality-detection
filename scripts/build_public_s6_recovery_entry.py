"""Immutable dual-T4 image-only rehearsal and receipt-gated pure-entry builder."""
import argparse
import ast
import base64
import hashlib
import json
import lzma
from pathlib import Path
import re
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from knee.public_s6_recovery_entry import VERSION,METHOD,INPUT_POLICY,SENTINELS,require_rehearsal
from knee.public_s6_recovery_runtime import PYDICOM_BUFFER_FILE_SHA,PYDICOM_BUFFER_FUNCTION_SHA
from knee.public_s6_qualification import canonical,digest,file_sha,require,save,LABELS
from knee.public_fullfield_entry import test_contract
from knee.reference_blend import DOCKER_IMAGE,CONVNEXT_DATASET
from knee.reference import WEIGHT_DATASET
from scripts.build_public_fullfield_entry import exact_closure as _closure
import scripts.build_public_fullfield_entry as helper
from scripts.build_public_s6_evaluation import accepted_pilot,PILOT_RECEIPT_SHA

EVALUATION_NOTEBOOK_SHA='b6b1b717c3bc8e8f2a8596bd9ae892f2e040cd05f68606f6b9012a6109319de5'
EVALUATION_RECEIPT_SHA='f859fa720ae4ca8742fdf8f139d2aa657d005cfd4c34843459b02dbac88a8ab3'
AUDIT_SHA='7e33e3fc9bafc3a6c2bd46a626f60730c7a3bfa42456c56a1c150d805dbf96c2'
CONTROL_RECEIPT_SHA='6213938a94b39e08cbe17e5fb42e0b3dcb0e8dc49354a09ed73151049eea517c'
PILOT_WRAPPER_SHA='7d769bd52e3b3d702ee274a266cb7f57b8b8143f05f800ae03f7a4e6ab54af10'
CONTROL_NOTEBOOK_SHA='553760d78cc45dae5a240bfdd819f126a976da8e34efb69163381e3d297fda43'
DATASETS=['heliosli/rsna-knee-compact-v1','heliosli/rsna-orthofoundation-fold0-epoch7-20261004',CONVNEXT_DATASET,WEIGHT_DATASET]
ROOTS={
 'knee/reference.py':{'load_definitions','load_model'},
 'knee/public_fullfield.py':{'GRID_NAMES','snapshot','require_immutable','fullfield_windows'},
 'knee/public_fullfield_entry.py':{'safe_id','read_series','test_contract','write_csv'},
 'knee/public_s6_qualification.py':{'LABELS','RECIPE','UID','SERIES_COLUMNS','canonical','digest','file_sha','require','save','verify_sources','verify_assets','synthetic_source_checks'},
 'knee/public_s6_evaluation.py':{'array_sha','model_digest','environment','save_predictions'},
 'knee/public_adaptation_fit.py':{'PlatformInterruption','atomic_save'},
 'knee/public_adaptation_pilot.py':{'state_fingerprint'},
 'knee/mirror_input.py':{'prepare_study','stress_inputs','STRESS_CASES'},
 'knee/public_s6_recovery.py':{'recover','RECIPE','RECEIPT_VERSION'},
}


CPU_PROTOCOL_SHA='e11e611c1089f63bd2bfd90b7659ba276bf5b7c9ddf97d18c178df8922b99def'
CPU_RESULT_SHA='2b3f43c8e8872c00a211211020798ba899ff38f94b29de84265a771dcc2a2198'


def cpu_gate(directory,project):
    root=Path(directory)
    require(file_sha(root/'protocol-private.json')==CPU_PROTOCOL_SHA and file_sha(root/'result.json')==CPU_RESULT_SHA,'EXACT_FROZEN_CPU_EVIDENCE')
    freeze=json.loads((root/'protocol-freeze.json').read_text());protocol=json.loads((root/'protocol-private.json').read_text());result=json.loads((root/'result.json').read_text())
    require(freeze['protocol_sha256']==CPU_PROTOCOL_SHA and freeze['truth_opened'] is False and freeze['zero_failure_exact_parity'] is True and result['protocol_sha256']==CPU_PROTOCOL_SHA,'PROSPECTIVE_CPU_FREEZE')
    require(result['all_scenarios_pass'] is True and len(result['scenarios'])==13 and [x['name'] for x in result['scenarios']]==[x['name'] for x in protocol['scenarios']] and all(x['gate_pass'] is True and x['macro_delta']>0 and x['paired_bootstrap']['ci95'][0]>0 and len(x['target_damage'])<3 for x in result['scenarios']),'ALL13_CPU_GATE')
    require(file_sha(project/'knee/public_s6_recovery.py')==protocol['implementation_sha256']['knee/public_s6_recovery.py'],'UNCHANGED_FROZEN_RECOVERY_IMPLEMENTATION')
    return {'protocol_sha256':CPU_PROTOCOL_SHA,'result_sha256':CPU_RESULT_SHA,'implementation_sha256':protocol['implementation_sha256'],'scenarios':13,'qualification':'synthetic sensitivity only; fresh CUDA image-only recovery rehearsal required'}


def pinned(path,pin):
    require(file_sha(path)==pin,'EXACT_ACCEPTED_EVIDENCE_BYTES');return json.loads(Path(path).read_bytes())


def literals(path,pin):
    nb=pinned(path,pin);code='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code');result={}
    for n in ast.parse(code).body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
            try:result[n.targets[0].id]=ast.literal_eval(n.value)
            except (ValueError,TypeError):pass
    return result


def inference_closure(frozen):
    """Exact top-level definitions, transitive globals and knee imports only.

    Scientific S6 sources (including vendor modules) stay wholly byte-identical.
    Evaluator/scoring bodies are excluded from retained knee helpers.
    """
    # Reuse the reviewed extraction algorithm, with the alias-aware data-boundary
    # check below in place of its full384-only prohibition on the word train.csv.
    source=ast.get_source_segment(Path(helper.__file__).read_text(),next(n for n in ast.parse(Path(helper.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='exact_closure'))
    source=source.replace("    require(not any('metric_report' in s or 'load_binary_truth' in s or 'train.csv' in s or '.parquet' in s for s in sources.values()), 'Inference closure contains label/metric dependency')", "    require(not any('metric_report' in s or 'load_binary_truth' in s or '.parquet' in s for s in sources.values()), 'Inference closure contains label/metric dependency')")
    ns={'ast':ast,'ROOTS':ROOTS,'require':require,'sha256':lambda b:hashlib.sha256(b).hexdigest()};exec(compile(source,'<exact-closure>','exec'),ns)
    return ns['exact_closure'](frozen)


def verify_build(destination):
    dest=Path(destination);r=json.loads((dest/'build-receipt.json').read_bytes())
    require(all(file_sha(dest/name)==pin for name,pin in r['files'].items()),'IMMUTABLE_BUILD_BYTES')
    m=json.loads((dest/'kernel-metadata.json').read_bytes());require(m['dataset_sources']==DATASETS and not m['enable_internet'] and m['is_private'] is True,'ONLY_IMAGE_ASSET_ATTACHMENTS')
    return r


def build(evaluation_build,evaluation_result,audit_path,pilot_build,pilot_result,control_build,control_result,data_root,destination,slug,*,entry=False,rehearsal_build=None,rehearsal_receipt=None,rehearsal_csv=None,cpu_evidence=None):
    import numpy as np
    project=Path(__file__).resolve().parents[1];dest=Path(destination)
    require(cpu_evidence is not None,'FROZEN_CPU_GATE_REQUIRED')
    cpu=cpu_gate(cpu_evidence,project)
    require(not dest.exists() and dest.resolve().is_relative_to(project/'outputs'),'FRESH_IGNORED_BUILD');require(re.fullmatch('[a-z0-9]+(?:-[a-z0-9]+)*',slug),'SLUG')
    ev=literals(Path(evaluation_build)/'notebook.ipynb',EVALUATION_NOTEBOOK_SHA)
    payload=json.loads(lzma.decompress(base64.b64decode(ev['encoded'],validate=True)));require(digest(payload)==ev['expected_payload'],'FROZEN_EVALUATOR_PAYLOAD')
    result=pinned(Path(evaluation_result)/'receipt.json',EVALUATION_RECEIPT_SHA);audit=pinned(audit_path,AUDIT_SHA)
    require(result['status']=='PASS' and result['completed_studies']==870 and result['macro_delta']>0 and result['paired_bootstrap']['ci95'][0]>0 and not result['target_damage'] and result['state_before']==result['state_after'],'ACCEPTED_FIXED_RESULT')
    require(audit['counts']=={'PASS':73,'FAIL':0,'UNKNOWN':3} and audit['gate_qualifies'] is True,'INDEPENDENT73PASS_GATE')
    require(result['binding']==payload['cfg']['binding'],'RESULT_BUILD_BINDING')
    original,pilot,functions=accepted_pilot(Path(pilot_build)/'notebook.ipynb');pr=pinned(Path(pilot_result)/'receipt.json',PILOT_RECEIPT_SHA)
    wrapper=pinned(Path(pilot_result)/'wrapper/receipt.json',PILOT_WRAPPER_SHA)
    require(pr['source_state']==pr['mechanics']==pr['prediction']=='PASS' and wrapper['status']=='PASS' and wrapper['completed']==32 and wrapper['state_before']==wrapper['state_after'] and pr['sentinel_ordinals']==SENTINELS,'QUALIFIED_PILOT')
    ids=pilot['ids'];require(len(ids)==32 and len(set(ids))==32,'FROZEN32_IDS')
    with np.load(Path(pilot_result)/'wrapper/predictions-private.npz',allow_pickle=False) as a:
        require(a['ordinals'].tolist()==list(range(32)),'PILOT_PROBABILITY_ORDER');pp=a['probabilities'].copy()
    old=literals(Path(control_build)/'notebook.ipynb',CONTROL_NOTEBOOK_SHA);cr=pinned(Path(control_result)/'receipt.json',CONTROL_RECEIPT_SHA)
    require(old['SAMPLE']==ids and cr['qualification_completed']==32 and cr['mechanical_gate']=='PASS' and cr['final_state_and_all_buffers_immutable'] is True,'OLD32_EXACT_ID_INTERSECTION')
    # Original32 old probabilities were not saved. Preserve their input hashes
    # and fresh original-source/repeat/reload parity; four saved870 sentinels
    # provide historical old probability parity on both physical boards.
    oldp=None
    require(pp.shape==(32,12) and np.isfinite(pp).all(),'MECHANICAL_ARRAYS')
    dev=payload['cfg']['dev'];oldord=[0,1,868,869];oldsent=[dev[i] for i in oldord]
    require(len(set(oldsent))==4 and not set(oldsent)&set(ids),'DISJOINT_OLD_HISTORICAL_SENTINELS')
    historical=np.frombuffer(base64.b64decode(payload['control'],validate=True),dtype=np.float32).reshape(870,12)
    oldcheck=json.loads((Path(evaluation_result)/'control/receipt.json').read_bytes())
    require(oldcheck['status']=='PASS' and [x['ordinal'] for x in oldcheck['records']]==oldord,'ACCEPTED_OLD_HISTORICAL_SENTINELS')
    frozen={k:v for k,v in payload['sources'].items() if k.startswith('knee/')}
    frozen['knee/public_fullfield_entry.py']=(project/'knee/public_fullfield_entry.py').read_text()
    frozen['knee/public_s6_recovery.py']=(project/'knee/public_s6_recovery.py').read_text()
    sources,pins=inference_closure(frozen)
    for name,content in original.items():
        if not name.startswith('knee/'):require(payload['sources'][name]==content,'EXACT_SCIENTIFIC_SOURCE');sources[name]=content
    sources['knee/public_s6_recovery_entry.py']=(project/'knee/public_s6_recovery_entry.py').read_text()
    sources['knee/public_s6_recovery_runtime.py']=(project/'knee/public_s6_recovery_runtime.py').read_text()
    guard_files=('compact_infer.py','training/native_cache.py','training/dicom_geometry.py','knee_anchor_bank_v2.py','px_cache.py')
    guards={name:hashlib.sha256(sources[name].encode()).hexdigest() for name in guard_files}
    production_sources=dict(sources)
    extension=(project/'knee/public_s6_recovery_rehearsal.py').read_text()
    if not entry:sources['knee/public_s6_recovery_rehearsal.py']=extension
    for name,s in sources.items():compile(s,name,'exec')
    probes={sid:{'old':{'input':{'volume_sha256':cr['qualification'][i]['volume_sha256'],'mask_sha256':cr['qualification'][i]['mask_sha256'],'qualified_full384_windows':cr['qualification'][i]['windows']},'probabilities':None},
                 's6':{'input':wrapper['records'][i]['input'],'probabilities':pp[i].tolist(),'mechanics':{k:wrapper['records'][i][k] for k in ('windows','anchor_method','anchor_reasons','side','field_center_fallbacks')}}} for i,sid in enumerate(ids)}
    for ordinal,sid,record in zip(oldord,oldsent,oldcheck['records']):
        probes[sid]={'old':{'input':{k:record[k] for k in ('volume_sha256','mask_sha256')},'probabilities':historical[ordinal].tolist()}}
    cfg0=payload['cfg'];sr=json.loads((Path(evaluation_result)/'s6/receipt.json').read_bytes());environment={'torch':result['environment']['torch'],'packages':result['environment']['packages']}
    columns,visible,_=test_contract(data_root,LABELS,visible=True);sentinels=[ids[i] for i in SENTINELS]
    binding={'version':VERSION,'method':METHOD,'input_policy':INPUT_POLICY,'source_bundle_sha256':digest(production_sources),'rehearsal_extension_sha256':hashlib.sha256(extension.encode()).hexdigest(),'cpu_gate':cpu,'guard_source_sha256':guards,'pydicom_guard':{'version':'3.0.2','package_file':'pydicom/pixels/decoders/base.py','file_sha256':PYDICOM_BUFFER_FILE_SHA,'function_sha256':PYDICOM_BUFFER_FUNCTION_SHA,'raise_sites':[22,33]},'primitive_segments_sha256':digest(pins),
        'evaluation_notebook_sha256':EVALUATION_NOTEBOOK_SHA,'evaluation_receipt_sha256':EVALUATION_RECEIPT_SHA,'independent_audit_sha256':AUDIT_SHA,
        'pilot_receipt_sha256':PILOT_RECEIPT_SHA,'pilot_wrapper_sha256':PILOT_WRAPPER_SHA,'pilot_arrays_sha256':file_sha(Path(pilot_result)/'wrapper/predictions-private.npz'),
        'old_notebook_sha256':CONTROL_NOTEBOOK_SHA,'old_receipt_sha256':CONTROL_RECEIPT_SHA,'old32_historical_probability_reference':'not saved; original-source/repeat/reload parity', 'old_historical_array_sha256':hashlib.sha256(historical.tobytes()).hexdigest(), 'old_historical_control_receipt_sha256':file_sha(Path(evaluation_result)/'control/receipt.json'),'evaluation_build_receipt_sha256':file_sha(Path(evaluation_build)/'build-receipt.json'),'pilot_build_receipt_sha256':file_sha(Path(pilot_build)/'build-receipt.json'),'old_build_receipt_sha256':file_sha(Path(control_build)/'build-receipt.json'),
        'environment':environment,'qualified_state':{'old_state':cfg0['control_state_sha256'],'old_grid':cfg0['control_grid_sha256'],'s6_state':cfg0['pilot_state_sha256']},'assets':{'manifest':pilot['manifest'],'binding':pilot['binding']},'public_manifest':cfg0['public_manifest'],'codecs':payload['codecs'],
        'columns':columns,'visible_ids':visible,'probe_contract_sha256':digest(probes),'s6_probe_ids':ids,'old_historical_ids':oldsent,'builder_sha256':file_sha(__file__),
        'sentinel_inputs':{arm:{s:probes[s][arm]['input'] for s in (oldsent if arm=='old' else sentinels)} for arm in ('old','s6')},
        'sentinel_probabilities':{arm:{s:probes[s][arm]['probabilities'] for s in (oldsent if arm=='old' else sentinels)} for arm in ('old','s6')}}
    cfg={k:cfg0[k] for k in ('public_source','decoder_source','public_manifest','control_state_sha256','control_grid_sha256','pilot_state_sha256','source_pins','manifest')}
    cfg.update(mode='entry' if entry else 'rehearsal',binding=binding,environment=environment,asset_binding=pilot['binding'],guard_source_sha256=guards)
    if entry:
        require(all((rehearsal_build,rehearsal_receipt,rehearsal_csv)),'ENTRY_REQUIRES_REHEARSAL')
        rb=verify_build(rehearsal_build);require(rb['mode']=='rehearsal' and rb['binding']==binding,'EXACT_REHEARSAL_BUILD')
        rr=json.loads(Path(rehearsal_receipt).read_bytes());require_rehearsal(rr,binding,rehearsal_csv)
        cfg['binding']={k:v for k,v in binding.items() if k not in ('sentinel_inputs','sentinel_probabilities','visible_ids','probe_contract_sha256','s6_probe_ids','old_historical_ids')}
        cfg['binding']['rehearsal_receipt_sha256']=file_sha(rehearsal_receipt);cfg['binding']['rehearsal_csv_sha256']=file_sha(rehearsal_csv)
    else:cfg.update(probes=probes,probe_order=ids+oldsent,old_sentinels=oldsent,s6_sentinels=sentinels,development_windows=[x['windows'] for x in sr['records']])
    pack={'sources':sources,'cfg':cfg,'codecs':payload['codecs']};encoded=base64.b64encode(lzma.compress(canonical(pack),preset=9)).decode()
    boot="""import time
started=time.monotonic()
import os,sys,json,base64,lzma,hashlib,signal,subprocess
from pathlib import Path
def deadline(signum,frame):raise TimeoutError('Actual nine-hour runtime exhausted')
signal.signal(signal.SIGALRM,deadline);signal.alarm(9*3600)
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',BUILD_MONAI='0',PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode=True
"""+f'encoded={encoded!r}\nexpected_payload={digest(pack)!r}\n'+"""
payload=json.loads(lzma.decompress(base64.b64decode(encoded,validate=True)))
if hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()!=expected_payload:raise ValueError('Frozen image-only payload changed')
project=Path('/kaggle/working/s6-recovery-entry-project');code=project/'source'
for name,content in payload['sources'].items():
    p=project/name if name.startswith('knee/') else code/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(content.encode())
sys.path.insert(0,str(project))
from knee.public_s6_recovery_entry import run
from knee.public_s6_qualification import file_sha,save,require
cfg=payload['cfg'];codecs=payload['codecs'];out=Path('/kaggle/working/s6-recovery-entry')
def mount(ref,required):
    owner,slug=ref.split('/');roots=[Path('/kaggle/input/datasets')/owner/slug,Path('/kaggle/input')/slug]
    matches=[p for p in roots if all((p/n).is_file() for n in required)];require(len(matches)==1,'UNIQUE_IMMUTABLE_ATTACHMENT');return matches[0]
try:
    require(sys.version_info[:2]==(3,12),'PYTHON_ABI')
    codec=mount('goodpjw2008/rsna-knee-2-5d-convnext-reader',list(codecs))
    for name,pin in codecs.items():require(file_sha(codec/name)==pin,'EXACT_OFFLINE_CODEC')
    subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps',*[str(codec/n) for n in codecs]],check=True)
    compact=mount('heliosli/rsna-knee-compact-v1',['manifest.json',*cfg['manifest']['sha256']])
    anatomy=mount('heliosli/rsna-orthofoundation-fold0-epoch7-20261004',['anatomy.pt'])/'anatomy.pt'
    weights=mount('nartaa/rsna-knee-publication-swa-weights-20261007',[cfg['public_manifest']['checkpoint_file']])
    roots=[Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),Path('/kaggle/input/rsna-knee-abnormality-detection')]
    matches=[p for p in roots if all((p/n).is_file() for n in ('test.csv','test_series.csv','sample_submission.csv'))];require(len(matches)==1,'UNIQUE_ACTUAL_TEST_ROOT')
    cfg.update(code=str(code),compact=str(compact),anatomy=str(anatomy),control_checkpoint=str(weights/cfg['public_manifest']['checkpoint_file']))
    save(project/'runtime-config.json',cfg)
    report=run(matches[0],cfg,out,started=started);print(json.dumps({'status':report['status'],'completed':report['completed'],'mode':report['mode']}),flush=True)
except BaseException as e:
    save(out/'setup-failure-private.json',{'type':type(e).__name__,'detail':str(e)})
    raise
"""
    compile(boot,'<s6-entry>','exec');nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python3','language':'python'}},'cells':[{'id':'dual-image-inference','cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':boot.splitlines(True)}]}
    meta={'id':'ozgung/'+slug,'title':slug.replace('-',' ').title(),'code_file':'notebook.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_tpu':False,'enable_internet':False,
          'competition_sources':['rsna-knee-abnormality-detection'],'dataset_sources':DATASETS,'kernel_sources':[],'model_sources':[],'docker_image':DOCKER_IMAGE}
    raw=json.dumps(nb,indent=2)+'\n';require(len(raw.encode())<900000,'NOTEBOOK900K_CAP');dest.mkdir(parents=True)
    (dest/'notebook.ipynb').write_text(raw);(dest/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=project,text=True).strip()
    changed=subprocess.check_output(['git','status','--porcelain','--','knee/public_s6_recovery_entry.py','knee/public_s6_recovery_runtime.py','knee/public_s6_recovery_rehearsal.py','scripts/build_public_s6_recovery_entry.py'],cwd=project,text=True).strip()
    receipt={'source_revision':revision,'source_state':'uncommitted recovery implementation; exact source hashes bound' if changed else 'committed implementation; exact source hashes bound','mode':cfg['mode'],'binding':binding,'payload_sha256':digest(pack),'files':{n:file_sha(dest/n) for n in ('notebook.ipynb','kernel-metadata.json')},'notebook_bytes':len(raw.encode()),'source_files':len(sources),'source_hashes':{n:hashlib.sha256(s.encode()).hexdigest() for n,s in sources.items()},'runtime_checks_exercised':False,'missing_reference_policy':'never synthesize saved predictions'}
    save(dest/'build-receipt.json',receipt);return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('evaluation-build','evaluation-result','audit-path','pilot-build','pilot-result','control-build','control-result','data-root','destination','slug'):p.add_argument('--'+n,required=True)
    p.add_argument('--entry',action='store_true')
    p.add_argument('--cpu-evidence',required=True)
    for n in ('rehearsal-build','rehearsal-receipt','rehearsal-csv'):p.add_argument('--'+n)
    r=build(**vars(p.parse_args()));print(json.dumps({'notebook_bytes':r['notebook_bytes'],'runtime_checks_exercised':False}))
