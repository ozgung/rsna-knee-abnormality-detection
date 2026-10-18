"""B3 builder: B2 (Dread K62 + R7) plus own dr-full K42 at 0.2 of the old slot.

Reuses the frozen selection-B/R7/B0 builder helpers unchanged (pinned bytes). No nartaa (OAI-trained)
dataset may be attached; the private dr-full dataset is pinned at version 1. Never pushes, runs a
kernel or submits.
"""
import argparse
import base64
import hashlib
import json
import lzma
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import knee.public_r7_recovery_entry as R7
import knee.public_r7_b3_entry as B3
import knee.public_s6_recovery_b0 as B0
from knee.public_s6_recovery_entry import SENTINELS
from knee.public_s6_recovery_runtime import PYDICOM_BUFFER_FILE_SHA,PYDICOM_BUFFER_FUNCTION_SHA
from knee.public_s6_qualification import canonical,digest,file_sha,require,save,LABELS
from knee.public_fullfield_entry import test_contract
from knee.reference_blend import DOCKER_IMAGE,CONVNEXT_DATASET
from knee.reference import RAPTOR_WEIGHT_DATASET,audit_source
import scripts.build_public_s6_recovery_entry as default
import scripts.build_public_s6_recovery_b0 as B0B
import scripts.build_public_r7_recovery_entry as R7B
import scripts.build_public_r7_selection_b as SB
from scripts.qualify_b3_drfull_cpu import drfull_manifest
from scripts.build_public_s6_evaluation import accepted_pilot,PILOT_RECEIPT_SHA

# Qualified selection-B sources (B1/B2 entries 57157385/57157388), byte-identical.
SELECTION_B_SHA={'knee/public_r7_b1_entry.py':'b1e146a26bd255b5b9882931b1ec1bd84e35d32b37cf804b8829c91dd0919c39',
                 'knee/public_r7_b2_entry.py':'57e997f0a354526571ead8a2c9d7fb22d9e475c6bf681f97ae24293ee84a10e5',
                 'knee/public_r7_b2_rehearsal.py':'08061047207f31d3de956a3aa9abd54ef192bd9e151294d227d1d865187f7902',
                 'scripts/build_public_r7_selection_b.py':'791033afee3802b95bc214d82c6a80874bf7595cad4ab9bdc23548a81ac51413'}
DRFULL_CPU_RECEIPT_SHA='e2c17f1848e61da23d1e0d0d0049ad623590921404bb3ccac306b1fea4440069'
DATASETS=SB.VARIANTS['b2']['datasets']+[f'{B3.DRFULL_DATASET}/{B3.DRFULL_DATASET_VERSION}']
REHEARSAL_SLUG='knee-r7-dread-drfull-b3-rehearsal-20261014'
ENTRY_SLUG='knee-r7-dread-drfull-b3-entry-20261014'
EXTENSIONS=('knee/public_s6_recovery_rehearsal.py','knee/public_s6_recovery_b0_rehearsal.py','knee/public_r7_b3_rehearsal.py')
B3_FILES=('knee/public_r7_b3_entry.py','knee/public_r7_b3_rehearsal.py','scripts/build_public_r7_b3.py','scripts/qualify_b3_drfull_cpu.py')


def verify_build(destination):
    dest=Path(destination);r=json.loads((dest/'build-receipt.json').read_bytes())
    require(all(file_sha(dest/name)==pin for name,pin in r['files'].items()),'IMMUTABLE_BUILD_BYTES')
    m=json.loads((dest/'kernel-metadata.json').read_bytes())
    require(m['dataset_sources']==DATASETS and SB.no_oai(m['dataset_sources']) and not m['enable_internet'] and m['is_private'] is True and m['enable_gpu'] is True and m.get('machine_shape')=='NvidiaTeslaT4','PRIVATE_OFFLINE_T4_NO_OAI')
    return r


def cpu_qualification(path,manifest):
    require(file_sha(path)==DRFULL_CPU_RECEIPT_SHA,'EXACT_DRFULL_CPU_RECEIPT');r=json.loads(Path(path).read_bytes())
    # The CPU receipt ran under Python 3.12.3 (fastgen: torch 2.10.0, timm); ast.unparse there formats the
    # extracted audited definitions differently from this builder's interpreter, which changes only the
    # extracted-text hash and its derived exception-policy record. Every other manifest field must match;
    # the Kaggle worker verifies the builder's own text against the builder's manifest.
    formatting=('extracted_module_sha256','exception_policy')
    same={k:v for k,v in r['manifest'].items() if k not in formatting}=={k:v for k,v in manifest.items() if k not in formatting}
    require(r['checkpoint_sha256']==B3.DRFULL_CHECKPOINT_SHA256 and r['checkpoint_bytes']==B3.DRFULL_CHECKPOINT_BYTES and r['state_sha256']==B3.DRFULL_STATE_SHA256
            and r['constructor_grid_sha256']==B3.DRFULL_GRID_SHA256 and same and set(r['manifest'])==set(manifest) and r['windows_shape']==[42,3,384,384] and r['forward_windows']==42
            and r['cpu_repeat_max_abs']<=1e-6 and r['strict_state_load'] is True and r['labels']==LABELS and r['dread_pin_method_check'] is True
            and r['checkpoint_meta']=={'epoch':[9,10,11],'swa_over':[9,10,11]},'B3_DRFULL_CPU_QUALIFICATION')
    return {'receipt_sha256':DRFULL_CPU_RECEIPT_SHA,'state_sha256':r['state_sha256'],'grid_sha256':r['constructor_grid_sha256'],'torch':r['torch'],'device':'cpu','swa_epochs':[9,10,11],
            'manifest_formatting_fields_excluded':list(formatting),'dread_pin_method_check':True}


def build(evaluation_build,evaluation_result,audit_path,pilot_build,pilot_result,data_root,heliosli_notebook,r7_asset_dir,heliosli_output,r7_reference,dread_source,raptor62_result,dread_cpu_receipt,drfull_cpu_receipt,destination,slug,*,
          entry=False,rehearsal_build=None,rehearsal_receipt=None,rehearsal_csv=None,cpu_evidence=None,**unused):
    import numpy as np
    title=SB.kernel_title(slug);SB.no_oai(DATASETS)
    project=Path(__file__).resolve().parents[1];dest=Path(destination)
    for name,pin in {**R7B.DEFAULT_MODULE_SHA,**SB.R7_FROZEN_SHA,**SB.B0_MODULE_SHA,**SELECTION_B_SHA}.items():require(file_sha(project/name)==pin,'BYTE_IDENTICAL_QUALIFIED_MODULE_'+name)
    require(cpu_evidence is not None,'FROZEN_CPU_GATE_REQUIRED');cpu=default.cpu_gate(cpu_evidence,project)
    require(not dest.exists() and dest.resolve().is_relative_to(project/'outputs'),'FRESH_IGNORED_BUILD')
    ev=default.literals(Path(evaluation_build)/'notebook.ipynb',default.EVALUATION_NOTEBOOK_SHA)
    payload=json.loads(lzma.decompress(base64.b64decode(ev['encoded'],validate=True)));require(digest(payload)==ev['expected_payload'],'FROZEN_EVALUATOR_PAYLOAD')
    result=default.pinned(Path(evaluation_result)/'receipt.json',default.EVALUATION_RECEIPT_SHA);audit=default.pinned(audit_path,default.AUDIT_SHA)
    require(result['status']=='PASS' and result['completed_studies']==870 and result['state_before']==result['state_after'] and audit['gate_qualifies'] is True and result['binding']==payload['cfg']['binding'],'ACCEPTED_FIXED_RESULT')
    original,pilot,functions=accepted_pilot(Path(pilot_build)/'notebook.ipynb');pr=default.pinned(Path(pilot_result)/'receipt.json',PILOT_RECEIPT_SHA)
    wrapper=default.pinned(Path(pilot_result)/'wrapper/receipt.json',default.PILOT_WRAPPER_SHA)
    require(pr['source_state']==pr['mechanics']==pr['prediction']=='PASS' and wrapper['status']=='PASS' and wrapper['completed']==32 and pr['sentinel_ordinals']==SENTINELS,'QUALIFIED_PILOT')
    ids=pilot['ids'];require(len(ids)==32 and len(set(ids))==32,'FROZEN32_IDS')
    dev=payload['cfg']['dev'];oldsent=[dev[i] for i in (0,1,868,869)]
    columns,visible,_=test_contract(data_root,LABELS,visible=True)
    require(len(set(oldsent))==4 and not set(oldsent)&set(ids) and not set(oldsent+ids)&set(visible),'DISJOINT_PROBES')
    manifest,fluid,state=R7B.r7_assets(r7_asset_dir);hv=R7B.heliosli_sources(heliosli_notebook)
    reference=R7B.heliosli_reference(heliosli_output,visible);entry_ref=SB.r7_entry_reference(r7_reference,visible)
    frozen={k:v for k,v in payload['sources'].items() if k.startswith('knee/')}
    frozen['knee/public_fullfield_entry.py']=(project/'knee/public_fullfield_entry.py').read_text()
    frozen['knee/public_s6_recovery.py']=(project/'knee/public_s6_recovery.py').read_text()
    sources,pins=B0B.inference_closure(frozen)
    for name,content in original.items():
        if not name.startswith('knee/'):require(payload['sources'][name]==content,'EXACT_SCIENTIFIC_SOURCE');sources[name]=content
    swapped={}
    for name,text in hv['SOURCES'].items():
        sha=hashlib.sha256(text.encode()).hexdigest();require(sha==manifest['src_sha256'].get(name,sha),'R7_MANIFEST_SOURCE_PIN')
        if name in R7.R7_SOURCE_SHA:require(sha==R7.R7_SOURCE_SHA[name] and sources[name]!=text,'R7_SWAPPED_SOURCE');swapped[name]=text
        else:require(sources.get(name)==text,'R7_UNCHANGED_SHARED_SOURCE_'+name)
    for name,pin in hv['CODE_SHA'].items():require(hashlib.sha256(sources['training/'+name].encode()).hexdigest()==pin,'R7_MAINLINE_CODE_'+name)
    require(set(swapped)==set(R7.R7_SOURCE_SHA),'EXACTLY_THREE_SWAPPED_SOURCES');sources.update(swapped)
    source_pins=dict(payload['cfg']['source_pins']);source_pins.update(R7.R7_SOURCE_SHA)
    require(all(hashlib.sha256(sources[n].encode()).hexdigest()==p for n,p in source_pins.items()),'CODE_TREE_PINS_MATCH_BUNDLE')
    for name in ('knee/public_s6_recovery_entry.py','knee/public_s6_recovery_runtime.py','knee/public_r7_recovery_entry.py','knee/public_r7_recovery.py','knee/public_r7_stackers.py','knee/public_r7_b1_entry.py',
                 'knee/public_s6_recovery_b0.py','knee/public_r7_b2_entry.py','knee/public_r7_b3_entry.py'):sources[name]=(project/name).read_text()
    guards={name:hashlib.sha256(sources[name].encode()).hexdigest() for name in ('compact_infer.py','training/native_cache.py','training/dicom_geometry.py','knee_anchor_bank_v2.py','px_cache.py')}
    production_sources=dict(sources);extensions={name:(project/name).read_text() for name in EXTENSIONS}
    if not entry:sources.update(extensions)
    for name,s in sources.items():compile(s,name,'exec')
    probes={sid:{'old':{'input':{},'probabilities':None},'s6':{'input':wrapper['records'][i]['input'],'probabilities':None,'mechanics':{k:wrapper['records'][i][k] for k in ('windows','anchor_method','anchor_reasons','side','field_center_fallbacks')}}} for i,sid in enumerate(ids)}
    for sid in oldsent:probes[sid]={'old':{'input':{},'probabilities':None}}
    sr=json.loads((Path(evaluation_result)/'s6/receipt.json').read_bytes());environment={'torch':result['environment']['torch'],'packages':result['environment']['packages']}
    sentinels=[ids[i] for i in SENTINELS]
    # Old arm: B0 Dread exactly; dr arm: the same audited module text with the dr-full K42 manifest.
    dread_text=Path(dread_source).read_text()
    module,decoder_source,old_manifest=audit_source(dread_text,profile='raptor',checkpoint_sha256=B0.DREAD_CHECKPOINT_SHA256,raptor_windows=B0.RAPTOR_WINDOWS)
    require(decoder_source=='','RAPTOR_HAS_NO_PINNED_DECODER');B0.raptor_profile_gate(None,old_manifest)
    dr_module,dr_decoder,dr_manifest=drfull_manifest(dread_text);require(dr_module==module and dr_decoder=='','SAME_AUDITED_RAPTOR_DEFINITIONS');B3.dr_profile_gate(None,dr_manifest)
    raptor62_reference,raptor62=B0B.raptor62_evidence(raptor62_result,old_manifest,visible);cpu_pins=B0B.cpu_receipt(dread_cpu_receipt,old_manifest)
    dr_cpu=cpu_qualification(drfull_cpu_receipt,dr_manifest)
    asset_binding={'manifest_sha256':R7.R7_MANIFEST_SHA,'dataset':R7.R7_DATASET,'dataset_version':R7.R7_DATASET_VERSION,'bundle_id':R7.R7_BUNDLE_ID}
    binding={'version':B3.VERSION,'variant':B3.METHOD['variant'],'method':B3.METHOD,'input_policy':R7.INPUT_POLICY,'datasets':DATASETS,'source_bundle_sha256':digest(production_sources),'rehearsal_extension_sha256':digest(extensions),
        'cpu_gate':cpu,'guard_source_sha256':guards,'pydicom_guard':{'version':'3.0.2','package_file':'pydicom/pixels/decoders/base.py','file_sha256':PYDICOM_BUFFER_FILE_SHA,'function_sha256':PYDICOM_BUFFER_FUNCTION_SHA,'raise_sites':[22,33]},
        'primitive_segments_sha256':digest(pins),'default_module_sha256':R7B.DEFAULT_MODULE_SHA,'r7_frozen_sha256':SB.R7_FROZEN_SHA,'b0_module_sha256':SB.B0_MODULE_SHA,'selection_b_sha256':SELECTION_B_SHA,
        'evaluation_notebook_sha256':default.EVALUATION_NOTEBOOK_SHA,'evaluation_receipt_sha256':default.EVALUATION_RECEIPT_SHA,'pilot_receipt_sha256':PILOT_RECEIPT_SHA,'pilot_wrapper_sha256':default.PILOT_WRAPPER_SHA,
        'environment':environment,'qualified_state':{'old_state':B0.DREAD_STATE_SHA256,'old_grid':B0.DREAD_GRID_SHA256,'dr_state':B3.DRFULL_STATE_SHA256,'dr_grid':B3.DRFULL_GRID_SHA256,'s6_state':None},
        'assets':{'manifest':manifest,'binding':asset_binding},'codecs':payload['codecs'],'old_profile':B0.OLD_PROFILE,'dr_profile':B3.DR_PROFILE,'public_manifest':old_manifest,'dr_manifest':dr_manifest,
        'old_qualification':{'policy':'B0: Dread raptor62 T4 receipt plus all36 source-reference, strict pydicom and strict-reload repeat parity','raptor62':raptor62,'cpu':cpu_pins},
        'dr_qualification':{'policy':'CPU strict load and pins (method reproduces B0 Dread pins); all36 source-reference, strict pydicom and strict-reload repeat parity; no visible reference','cpu':dr_cpu,
                            'dataset':B3.DRFULL_DATASET,'dataset_version':B3.DRFULL_DATASET_VERSION,'checkpoint_sha256':B3.DRFULL_CHECKPOINT_SHA256},
        'r7':{'recipe':R7.R7_RECIPE,'dataset':R7.R7_DATASET,'dataset_version':R7.R7_DATASET_VERSION,'manifest_sha256':R7.R7_MANIFEST_SHA,'bundle_id':R7.R7_BUNDLE_ID,'source_sha256':R7.R7_SOURCE_SHA,
              'asset_sha256':R7.R7_ASSET_SHA,'fluid':fluid,'state':state,'heliosli_notebook_sha256':R7B.HELIOSLI_NOTEBOOK_SHA,'heliosli_output_sha256':R7B.HELIOSLI_OUTPUT_SHA,'visible_atol':R7.HELIOSLI_VISIBLE_ATOL,
              'entry_v2_reference_sha256':SB.R7_ENTRY_REFERENCE_SHA,'entry_v2_reference_atol':SB.R7_ENTRY_REFERENCE_ATOL,'state_pin':'measured on both rehearsal boards','deviations':R7B.DEVIATIONS},
        'provenance':'R7 fastMRI auxiliary training / licence permission unresolved; B probes hedge only against an OAI ruling. dr-full: own model on CC0 Dread init and competition data.',
        'columns':columns,'visible_ids':visible,'probe_contract_sha256':digest(probes),'s6_probe_ids':ids,'old_historical_ids':oldsent,'builder_sha256':file_sha(__file__),
        'sentinel_inputs':{'s6':{s:probes[s]['s6']['input'] for s in sentinels},'old':{s:{} for s in oldsent}},
        'sentinel_probabilities':{'s6':dict.fromkeys(sentinels),'old':dict.fromkeys(oldsent)}}
    cfg={'manifest':manifest,'source_pins':source_pins,'pilot_state_sha256':None,'mode':'entry' if entry else 'rehearsal','environment':environment,'asset_binding':asset_binding,'guard_source_sha256':guards,
         'public_source':module,'decoder_source':decoder_source,'public_manifest':old_manifest,'control_state_sha256':B0.DREAD_STATE_SHA256,'control_grid_sha256':B0.DREAD_GRID_SHA256,
         'arm_overlays':{'dr':{'public_manifest':dr_manifest,'control_state_sha256':B3.DRFULL_STATE_SHA256,'control_grid_sha256':B3.DRFULL_GRID_SHA256,'raptor62_reference':{}}},'binding':binding}
    rehearsal_state=None
    if entry:
        require(all((rehearsal_build,rehearsal_receipt,rehearsal_csv)),'ENTRY_REQUIRES_REHEARSAL')
        rb=verify_build(rehearsal_build);require(rb['mode']=='rehearsal' and rb['binding']==binding,'EXACT_REHEARSAL_BUILD')
        rr=json.loads(Path(rehearsal_receipt).read_bytes());B3.require_rehearsal(rr,binding,rehearsal_csv)
        require(SB.r7_visible_difference(rr,entry_ref)<=SB.R7_ENTRY_REFERENCE_ATOL,'R7_ARM_VISIBLE_PARITY_WITH_ENTRY_V2')
        rehearsal_state=rr['workers']['s6'][0]['state_before'];cfg['pilot_state_sha256']=rehearsal_state
        cfg['binding']={k:v for k,v in binding.items() if k not in SB.PRIVATE_BINDING}
        cfg['binding'].update(rehearsal_receipt_sha256=file_sha(rehearsal_receipt),rehearsal_csv_sha256=file_sha(rehearsal_csv),qualified_r7_state=rehearsal_state)
    else:
        cfg.update(probes=probes,probe_order=ids+oldsent,old_sentinels=oldsent,s6_sentinels=sentinels,development_windows=[x['windows'] for x in sr['records']],
                   heliosli_reference=reference,heliosli_visible_atol=R7.HELIOSLI_VISIBLE_ATOL,r7_entry_reference=entry_ref,raptor62_reference=raptor62_reference,raptor62_visible_atol=B0B.RAPTOR62_VISIBLE_ATOL)
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
from knee.public_r7_b3_entry import run,sanity
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
    compact=mount("""+repr(R7.R7_DATASET)+""",['manifest.json',*cfg['manifest']['sha256']])
    require(file_sha(compact/'manifest.json')==cfg['asset_binding']['manifest_sha256'],'EXACT_R7_V1_MANIFEST')
    anatomy=mount('heliosli/rsna-orthofoundation-fold0-epoch7-20261004',['anatomy.pt'])/'anatomy.pt'
    weights=mount("""+repr(RAPTOR_WEIGHT_DATASET)+""",[cfg['public_manifest']['checkpoint_file']]);cfg['control_checkpoint']=str(weights/cfg['public_manifest']['checkpoint_file'])
    drfull=mount("""+repr(B3.DRFULL_DATASET)+""",[cfg['arm_overlays']['dr']['public_manifest']['checkpoint_file']])
    cfg['arm_overlays']['dr']['control_checkpoint']=str(drfull/cfg['arm_overlays']['dr']['public_manifest']['checkpoint_file'])
    require(file_sha(cfg['arm_overlays']['dr']['control_checkpoint'])=="""+repr(B3.DRFULL_CHECKPOINT_SHA256)+""",'EXACT_DRFULL_CHECKPOINT')
    require(not any(p.exists() for p in (Path('/kaggle/input/rsna-knee-publication-swa-weights-20261007'),Path('/kaggle/input/datasets/nartaa'))),'NO_NARTAA_ATTACHMENT')
    roots=[Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),Path('/kaggle/input/rsna-knee-abnormality-detection')]
    matches=[p for p in roots if all((p/n).is_file() for n in ('test.csv','test_series.csv','sample_submission.csv'))];require(len(matches)==1,'UNIQUE_ACTUAL_TEST_ROOT')
    cfg.update(code=str(code),compact=str(compact),anatomy=str(anatomy))
    save(project/'runtime-config.json',cfg)
    try:report=run(matches[0],cfg,out,started=started)
    finally:
        if cfg['mode']=='rehearsal':
            try:print('B3_SANITY',json.dumps(sanity(out,cfg)),flush=True)
            except Exception as parity_error:print('B3_SANITY_UNAVAILABLE',repr(parity_error),flush=True)
    print(json.dumps({'status':report['status'],'completed':report['completed'],'mode':report['mode'],'elapsed_seconds':report['elapsed_seconds'],
                      'conservative_seconds':(report.get('projection') or {}).get('conservative_seconds')}),flush=True)
except BaseException as e:
    save(out/'setup-failure-private.json',{'type':type(e).__name__,'detail':str(e)})
    raise
"""
    compile(boot,'<r7-b3>','exec');nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python3','language':'python'}},'cells':[{'id':'dual-image-inference','cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':boot.splitlines(True)}]}
    meta={'id':'ozgung/'+slug,'title':title,'code_file':'notebook.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_tpu':False,'enable_internet':False,
          'competition_sources':['rsna-knee-abnormality-detection'],'dataset_sources':DATASETS,'kernel_sources':[],'model_sources':[],'docker_image':DOCKER_IMAGE,'machine_shape':'NvidiaTeslaT4'}
    raw=json.dumps(nb,indent=2)+'\n';require(len(raw.encode())<900000,'NOTEBOOK900K_CAP');dest.mkdir(parents=True)
    (dest/'notebook.ipynb').write_text(raw);(dest/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=project,text=True).strip()
    receipt={'source_revision':revision,'source_state':'uncommitted B3 implementation; exact source hashes bound','mode':cfg['mode'],'variant':'b3','binding':binding,'qualified_r7_state':rehearsal_state,
             'payload_sha256':digest(pack),'files':{n:file_sha(dest/n) for n in ('notebook.ipynb','kernel-metadata.json')},'notebook_bytes':len(raw.encode()),'source_files':len(sources),
             'source_hashes':{n:hashlib.sha256(s.encode()).hexdigest() for n,s in sources.items()},'b3_files_sha256':{n:file_sha(project/n) for n in B3_FILES},'runtime_checks_exercised':False}
    save(dest/'build-receipt.json',receipt);return receipt
