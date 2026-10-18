"""R7 dual-T4 builder: unchanged original320 old arm and recovery, heliosli R7 compact slot.

A separate entry point: the qualified default builder and modules stay byte-identical,
are pinned below and are shipped unchanged inside the R7 bundle. R7 sources are read
from heliosli's downloaded notebook as literals (ast, never executed) and must match
the R7 v1 manifest. Never pushes, runs a kernel or submits.
"""
import argparse
import ast
import base64
import csv
import hashlib
import json
import lzma
from pathlib import Path
import re
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import knee.public_r7_recovery_entry as R7
from knee import public_r7_stackers as stackers
from knee.public_s6_recovery_entry import SENTINELS
from knee.public_s6_recovery_runtime import PYDICOM_BUFFER_FILE_SHA,PYDICOM_BUFFER_FUNCTION_SHA
from knee.public_s6_qualification import canonical,digest,file_sha,require,save,LABELS,UID
from knee.public_fullfield_entry import test_contract
from knee.reference_blend import DOCKER_IMAGE,CONVNEXT_DATASET
from knee.reference import WEIGHT_DATASET
import scripts.build_public_s6_recovery_entry as default
from scripts.build_public_s6_evaluation import accepted_pilot,PILOT_RECEIPT_SHA

DATASETS=[f'{R7.R7_DATASET}/{R7.R7_DATASET_VERSION}','heliosli/rsna-orthofoundation-fold0-epoch7-20261004',CONVNEXT_DATASET,WEIGHT_DATASET]
# Byte-identical qualified default modules (0.954 entry build 82111ff, rehearsal 42d4b3d).
DEFAULT_MODULE_SHA={'knee/public_s6_recovery_entry.py':'76217a402ae3bb4c8c9517c76875691f3565b84bf0b2e548295cd9f726a90be8',
                    'knee/public_s6_recovery_runtime.py':'d67ceb738930c85b19e1866dfeb0d524e015a68ff321e039f00105c590ca6d08',
                    'knee/public_s6_recovery_rehearsal.py':'8424d87b5656ccdf449c69c5ee06c09744116f185f086861b7feee8bd5864502',
                    'knee/public_s6_recovery.py':'2fbc950cdcc54dddd9b6722ced524943b784dc0ae3a0df185aad10146f6a9ce8',
                    'scripts/build_public_s6_recovery_entry.py':'d8915ef31345f452ca8a8ea0a2cbeb12ad717363d9e7b54b3cd714ef3ed3109c'}
# heliosli "0.957 latest dataset score in one click" (kernel 138222340), pulled unexecuted 2026-10-13.
HELIOSLI_NOTEBOOK_SHA='cb2a0f57962ca9b6668661e34461aad772ed9ed10d576d32772e7c2898f2e315'
# Its published visible-3 output (kaggle kernels output), same run as the notebook source.
HELIOSLI_OUTPUT_SHA={'ours_sub.csv':'fa4d89746bef047af6b14866c3272d9818aa2af433dfe33c4fe4ed0593ac5ab3',
                     'raptor_raw.csv':'456bdf8a7bc34dfb1245bcf90435bc664d6f6241f69a27bb28be8d46c39f7d14',
                     'submission.csv':'1199c2d767b5a53219d777b6041593cc392f56eff09228bda88d68c90d5f3cc8'}
PRIVATE_BINDING=('sentinel_inputs','sentinel_probabilities','visible_ids','probe_contract_sha256','s6_probe_ids','old_historical_ids')
# Kaggle SaveKernel rejects slugs/titles over 50 characters.
MAX_KERNEL_SLUG=50
REHEARSAL_SLUG='knee-public-r7-recovery-rehearsal-20261013'
ENTRY_SLUG='knee-public-r7-recovery-entry-20261013'
PROTECTED_SLUGS=('knee-public-s6-recovery-entry-20261012','knee-s6-dread-b0-entry-20261013','knee-s6-dread-b0-rehearsal-20261013','knee-public-s6-recovery-rehearsal-20261012')
R7_FILES=('knee/public_r7_recovery_entry.py','knee/public_r7_recovery_rehearsal.py','knee/public_r7_recovery.py','knee/public_r7_stackers.py','scripts/build_public_r7_recovery_entry.py')
DEVIATIONS=['torch.compile in place (nn.Module.compile) with heliosli compile gate on the first item; no eager fallback: a failed check or compiled forward is fatal (heliosli continues eager)',
            'fluid_d2 catch-all replaced by an allowlist boundary: typed data-local errors are capped study failures, unknown errors fatal (heliosli: missing-volume path)',
            'R7 cross-board sentinel tolerance 1e-3 (separately compiled shards); within-board reload and old arm 1e-6',
            'non-finite fp16 logits and stacker exceptions are fatal (heliosli: fp32 rescue / keep model probability)',
            'no 8.3 h deadline guard: unreached-study fallback (other model rank, up to 50%) is incompatible with the frozen 3% capped recovery; 9 h alarm plus rehearsal projection gate instead',
            'recovery for classified data-local failures: capped global arm means ranked over all rows (heliosli: per-model ranks over own scored rows, other model rank for failures)',
            'chunk 192 (compact_infer default); S6 entry used 16',
            'submission written as float64 blend values (heliosli casts to float32)',
            'our pinned Docker image and environment (heliosli image sha256:2757e0c7...)']


def kernel_title(slug):
    require(isinstance(slug,str) and re.fullmatch('[a-z0-9]+(?:-[a-z0-9]+)*',slug) and 5<=len(slug)<=MAX_KERNEL_SLUG and slug not in PROTECTED_SLUGS,'SLUG')
    title=slug.replace('-',' ').title();require(5<=len(title)<=MAX_KERNEL_SLUG,'KERNEL_TITLE_LENGTH');return title


def heliosli_sources(notebook):
    """SOURCES/SOURCE_SHA/CODE_SHA/BUNDLE_ID/weights literals of the pinned notebook (ast.literal_eval only)."""
    require(file_sha(notebook)==HELIOSLI_NOTEBOOK_SHA,'EXACT_HELIOSLI_NOTEBOOK')
    nb=json.loads(Path(notebook).read_bytes());values={}
    for cell in nb['cells']:
        if cell['cell_type']!='code':continue
        lines=[l for l in ''.join(cell['source']).splitlines() if not l.lstrip().startswith(('%','!'))]
        for n in ast.parse('\n'.join(lines)).body:
            if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('SOURCES','SOURCE_SHA','CODE_SHA','BUNDLE_ID','W_RAPTOR','W_OURS','W_RAPTOR_C','C_SLUG','ARM','DEADLINE_HOURS'):
                values[n.targets[0].id]=ast.literal_eval(n.value)
            elif isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Tuple) and [t.id for t in n.targets[0].elts]==['W_RAPTOR','W_OURS']:
                values['W_RAPTOR'],values['W_OURS']=ast.literal_eval(n.value)
    require(values['BUNDLE_ID']==R7.R7_BUNDLE_ID and values['C_SLUG']==R7.R7_DATASET.split('/')[1] and values['ARM']=='E_lrprior','HELIOSLI_R7_BUNDLE')
    require((values['W_RAPTOR'],values['W_OURS'])==tuple(R7.METHOD['blend']) and values['W_RAPTOR_C']==R7.METHOD['raptor_per_finding'],'HELIOSLI_FIXED_BLEND_WEIGHTS')
    src=values['SOURCES'];require(set(src)==set(values['SOURCE_SHA']) and all(hashlib.sha256(src[n].encode()).hexdigest()==values['SOURCE_SHA'][n] for n in src),'HELIOSLI_SOURCE_PINS')
    return values


def r7_assets(directory):
    root=Path(directory);require(file_sha(root/'manifest.json')==R7.R7_MANIFEST_SHA,'EXACT_R7_V1_MANIFEST')
    manifest=json.loads((root/'manifest.json').read_bytes());R7.r7_manifest_gate(manifest)
    for name,pin in R7.R7_ASSET_SHA.items():require(file_sha(root/name)==pin==manifest['sha256'][name],'EXACT_R7_STACKER_SPEC')
    spec_raw=(root/'state_spec.json').read_bytes();spec=json.loads(spec_raw)
    fluid=stackers.load_fluid((root/'fluid_stacker.json').read_bytes(),LABELS);layout=stackers.state_layout(spec,LABELS)
    state=stackers.load_state((root/'state_stacker.json').read_bytes(),LABELS,layout,hashlib.md5(spec_raw).hexdigest())
    pair=stackers.check_pair(fluid,state);require(tuple(pair['fluid'])==stackers.FLUID_ADOPTED and tuple(pair['state'])==stackers.STATE_ADOPTED and not state['needs_volume'],'PINNED_ADOPTED_CLASSES')
    return manifest,fluid,state


def heliosli_reference(directory,visible):
    root=Path(directory);ref={}
    for key,name in (('r7','ours_sub.csv'),('raptor_raw','raptor_raw.csv'),('submission','submission.csv')):
        require(file_sha(root/name)==HELIOSLI_OUTPUT_SHA[name],'EXACT_HELIOSLI_OUTPUT_'+name)
        with (root/name).open(newline='') as h:rows=list(csv.reader(h))
        require(rows[0]==[UID]+LABELS and [r[0] for r in rows[1:]]==visible,'HELIOSLI_VISIBLE_SCHEMA_ORDER')
        ref[key]={r[0]:[float(x) for x in r[1:]] for r in rows[1:]}
        require(all(len(v)==12 and all(0<=x<=1 for x in v) for v in ref[key].values()),'HELIOSLI_PROBABILITIES')
    return ref


def visible_rank_difference(csv_path,submission,visible):
    """Max |rank difference| of the rehearsal visible blend vs heliosli's published submission.
    Visible ranks are multiples of 1/6; heliosli writes float32, so compare after rounding to 1e-6."""
    with Path(csv_path).open(newline='') as h:rows=list(csv.reader(h))
    require(rows[0]==[UID]+LABELS and [r[0] for r in rows[1:]]==visible,'VISIBLE_SCHEMA_ORDER')
    return max(abs(round(float(v),6)-round(float(w),6)) for r in rows[1:] for v,w in zip(r[1:],submission[r[0]]))


def verify_build(destination):
    dest=Path(destination);r=json.loads((dest/'build-receipt.json').read_bytes())
    require(all(file_sha(dest/name)==pin for name,pin in r['files'].items()),'IMMUTABLE_BUILD_BYTES')
    m=json.loads((dest/'kernel-metadata.json').read_bytes())
    require(m['dataset_sources']==DATASETS and not m['enable_internet'] and m['is_private'] is True and m['enable_gpu'] is True and m.get('machine_shape')=='NvidiaTeslaT4','PRIVATE_OFFLINE_T4_ATTACHMENTS')
    return r


def build(evaluation_build,evaluation_result,audit_path,pilot_build,pilot_result,control_build,control_result,data_root,heliosli_notebook,r7_asset_dir,heliosli_output,destination,slug,*,
          entry=False,rehearsal_build=None,rehearsal_receipt=None,rehearsal_csv=None,cpu_evidence=None):
    import numpy as np
    title=kernel_title(slug)
    project=Path(__file__).resolve().parents[1];dest=Path(destination)
    for name,pin in DEFAULT_MODULE_SHA.items():require(file_sha(project/name)==pin,'BYTE_IDENTICAL_DEFAULT_MODULE_'+name)
    require(cpu_evidence is not None,'FROZEN_CPU_GATE_REQUIRED');cpu=default.cpu_gate(cpu_evidence,project)
    require(not dest.exists() and dest.resolve().is_relative_to(project/'outputs'),'FRESH_IGNORED_BUILD')
    ev=default.literals(Path(evaluation_build)/'notebook.ipynb',default.EVALUATION_NOTEBOOK_SHA)
    payload=json.loads(lzma.decompress(base64.b64decode(ev['encoded'],validate=True)));require(digest(payload)==ev['expected_payload'],'FROZEN_EVALUATOR_PAYLOAD')
    result=default.pinned(Path(evaluation_result)/'receipt.json',default.EVALUATION_RECEIPT_SHA);audit=default.pinned(audit_path,default.AUDIT_SHA)
    require(result['status']=='PASS' and result['completed_studies']==870 and result['macro_delta']>0 and result['paired_bootstrap']['ci95'][0]>0 and not result['target_damage'] and result['state_before']==result['state_after'],'ACCEPTED_FIXED_RESULT')
    require(audit['counts']=={'PASS':73,'FAIL':0,'UNKNOWN':3} and audit['gate_qualifies'] is True,'INDEPENDENT73PASS_GATE')
    require(result['binding']==payload['cfg']['binding'],'RESULT_BUILD_BINDING')
    original,pilot,functions=accepted_pilot(Path(pilot_build)/'notebook.ipynb');pr=default.pinned(Path(pilot_result)/'receipt.json',PILOT_RECEIPT_SHA)
    wrapper=default.pinned(Path(pilot_result)/'wrapper/receipt.json',default.PILOT_WRAPPER_SHA)
    require(pr['source_state']==pr['mechanics']==pr['prediction']=='PASS' and wrapper['status']=='PASS' and wrapper['completed']==32 and wrapper['state_before']==wrapper['state_after'] and pr['sentinel_ordinals']==SENTINELS,'QUALIFIED_PILOT')
    ids=pilot['ids'];require(len(ids)==32 and len(set(ids))==32,'FROZEN32_IDS')
    old=default.literals(Path(control_build)/'notebook.ipynb',default.CONTROL_NOTEBOOK_SHA);cr=default.pinned(Path(control_result)/'receipt.json',default.CONTROL_RECEIPT_SHA)
    require(old['SAMPLE']==ids and cr['qualification_completed']==32 and cr['mechanical_gate']=='PASS' and cr['final_state_and_all_buffers_immutable'] is True,'OLD32_EXACT_ID_INTERSECTION')
    dev=payload['cfg']['dev'];oldord=[0,1,868,869];oldsent=[dev[i] for i in oldord]
    require(len(set(oldsent))==4 and not set(oldsent)&set(ids),'DISJOINT_OLD_HISTORICAL_SENTINELS')
    historical=np.frombuffer(base64.b64decode(payload['control'],validate=True),dtype=np.float32).reshape(870,12)
    oldcheck=json.loads((Path(evaluation_result)/'control/receipt.json').read_bytes())
    require(oldcheck['status']=='PASS' and [x['ordinal'] for x in oldcheck['records']]==oldord,'ACCEPTED_OLD_HISTORICAL_SENTINELS')
    # R7: manifest, stackers and spec of dataset version 1; sources from heliosli's notebook.
    manifest,fluid,state=r7_assets(r7_asset_dir);hv=heliosli_sources(heliosli_notebook)
    frozen={k:v for k,v in payload['sources'].items() if k.startswith('knee/')}
    frozen['knee/public_fullfield_entry.py']=(project/'knee/public_fullfield_entry.py').read_text()
    frozen['knee/public_s6_recovery.py']=(project/'knee/public_s6_recovery.py').read_text()
    sources,pins=default.inference_closure(frozen)
    for name,content in original.items():
        if not name.startswith('knee/'):require(payload['sources'][name]==content,'EXACT_SCIENTIFIC_SOURCE');sources[name]=content
    swapped={}
    for name,text in hv['SOURCES'].items():
        sha=hashlib.sha256(text.encode()).hexdigest();require(sha==manifest['src_sha256'].get(name,sha),'R7_MANIFEST_SOURCE_PIN')
        if name in R7.R7_SOURCE_SHA:require(sha==R7.R7_SOURCE_SHA[name] and name in sources and sources[name]!=text,'R7_SWAPPED_SOURCE');swapped[name]=text
        else:require(sources.get(name)==text,'R7_UNCHANGED_SHARED_SOURCE_'+name)
    for name,pin in hv['CODE_SHA'].items():require(hashlib.sha256(sources['training/'+name].encode()).hexdigest()==pin,'R7_MAINLINE_CODE_'+name)
    require(set(swapped)==set(R7.R7_SOURCE_SHA),'EXACTLY_THREE_SWAPPED_SOURCES');sources.update(swapped)
    source_pins=dict(payload['cfg']['source_pins']);require(set(R7.R7_SOURCE_SHA)<=set(source_pins),'SWAPPED_SOURCES_IN_CODE_TREE');source_pins.update(R7.R7_SOURCE_SHA)
    require(all(hashlib.sha256(sources[n].encode()).hexdigest()==p for n,p in source_pins.items()),'CODE_TREE_PINS_MATCH_BUNDLE')
    for name in ('knee/public_s6_recovery_entry.py','knee/public_s6_recovery_runtime.py','knee/public_r7_recovery_entry.py','knee/public_r7_recovery.py','knee/public_r7_stackers.py'):sources[name]=(project/name).read_text()
    guard_files=('compact_infer.py','training/native_cache.py','training/dicom_geometry.py','knee_anchor_bank_v2.py','px_cache.py')
    guards={name:hashlib.sha256(sources[name].encode()).hexdigest() for name in guard_files};require(guards['compact_infer.py']==R7.R7_SOURCE_SHA['compact_infer.py'],'R7_GUARD_SOURCE')
    production_sources=dict(sources)
    extensions={name:(project/name).read_text() for name in ('knee/public_s6_recovery_rehearsal.py','knee/public_r7_recovery_rehearsal.py')}
    if not entry:sources.update(extensions)
    for name,s in sources.items():compile(s,name,'exec')
    # Old arm: the default qualified probes exactly. R7 slot: S6 pilot render inputs and
    # mechanics (same pinned render sources), no prior probabilities.
    probes={sid:{'old':{'input':{'volume_sha256':cr['qualification'][i]['volume_sha256'],'mask_sha256':cr['qualification'][i]['mask_sha256'],'qualified_full384_windows':cr['qualification'][i]['windows']},'probabilities':None},
                 's6':{'input':wrapper['records'][i]['input'],'probabilities':None,'mechanics':{k:wrapper['records'][i][k] for k in ('windows','anchor_method','anchor_reasons','side','field_center_fallbacks')}}} for i,sid in enumerate(ids)}
    for ordinal,sid,record in zip(oldord,oldsent,oldcheck['records']):
        probes[sid]={'old':{'input':{k:record[k] for k in ('volume_sha256','mask_sha256')},'probabilities':historical[ordinal].tolist()}}
    cfg0=payload['cfg'];sr=json.loads((Path(evaluation_result)/'s6/receipt.json').read_bytes());environment={'torch':result['environment']['torch'],'packages':result['environment']['packages']}
    columns,visible,_=test_contract(data_root,LABELS,visible=True);sentinels=[ids[i] for i in SENTINELS];reference=heliosli_reference(heliosli_output,visible)
    require(not set(visible)&set(ids+oldsent),'DISJOINT_VISIBLE')
    asset_binding={'manifest_sha256':R7.R7_MANIFEST_SHA,'dataset':R7.R7_DATASET,'dataset_version':R7.R7_DATASET_VERSION,'bundle_id':R7.R7_BUNDLE_ID}
    binding={'version':R7.VERSION,'variant':'R7-swap-original320-recovery','method':R7.METHOD,'input_policy':R7.INPUT_POLICY,'source_bundle_sha256':digest(production_sources),'rehearsal_extension_sha256':digest(extensions),
        'cpu_gate':cpu,'guard_source_sha256':guards,'pydicom_guard':{'version':'3.0.2','package_file':'pydicom/pixels/decoders/base.py','file_sha256':PYDICOM_BUFFER_FILE_SHA,'function_sha256':PYDICOM_BUFFER_FUNCTION_SHA,'raise_sites':[22,33]},
        'primitive_segments_sha256':digest(pins),'default_module_sha256':DEFAULT_MODULE_SHA,
        'evaluation_notebook_sha256':default.EVALUATION_NOTEBOOK_SHA,'evaluation_receipt_sha256':default.EVALUATION_RECEIPT_SHA,'independent_audit_sha256':default.AUDIT_SHA,
        'pilot_receipt_sha256':PILOT_RECEIPT_SHA,'pilot_wrapper_sha256':default.PILOT_WRAPPER_SHA,'pilot_arrays_sha256':file_sha(Path(pilot_result)/'wrapper/predictions-private.npz'),
        'old_notebook_sha256':default.CONTROL_NOTEBOOK_SHA,'old_receipt_sha256':default.CONTROL_RECEIPT_SHA,'old_historical_array_sha256':hashlib.sha256(historical.tobytes()).hexdigest(),'old_historical_control_receipt_sha256':file_sha(Path(evaluation_result)/'control/receipt.json'),
        'evaluation_build_receipt_sha256':file_sha(Path(evaluation_build)/'build-receipt.json'),'pilot_build_receipt_sha256':file_sha(Path(pilot_build)/'build-receipt.json'),'old_build_receipt_sha256':file_sha(Path(control_build)/'build-receipt.json'),
        'environment':environment,'qualified_state':{'old_state':cfg0['control_state_sha256'],'old_grid':cfg0['control_grid_sha256'],'s6_state':None},
        'assets':{'manifest':manifest,'binding':asset_binding},'public_manifest':cfg0['public_manifest'],'codecs':payload['codecs'],
        'r7':{'recipe':R7.R7_RECIPE,'dataset':R7.R7_DATASET,'dataset_version':R7.R7_DATASET_VERSION,'manifest_sha256':R7.R7_MANIFEST_SHA,'bundle_id':R7.R7_BUNDLE_ID,'source_sha256':R7.R7_SOURCE_SHA,
              'asset_sha256':R7.R7_ASSET_SHA,'fluid':fluid,'state':state,'heliosli_notebook_sha256':HELIOSLI_NOTEBOOK_SHA,'heliosli_output_sha256':HELIOSLI_OUTPUT_SHA,'visible_atol':R7.HELIOSLI_VISIBLE_ATOL,
              'state_pin':'measured on both rehearsal boards (no CPU pin: downloaded R7 code is never executed locally)','deviations':DEVIATIONS},
        'columns':columns,'visible_ids':visible,'probe_contract_sha256':digest(probes),'s6_probe_ids':ids,'old_historical_ids':oldsent,'builder_sha256':file_sha(__file__),
        'sentinel_inputs':{arm:{s:probes[s][arm]['input'] for s in (oldsent if arm=='old' else sentinels)} for arm in ('old','s6')},
        'sentinel_probabilities':{arm:{s:probes[s][arm]['probabilities'] for s in (oldsent if arm=='old' else sentinels)} for arm in ('old','s6')}}
    cfg={k:cfg0[k] for k in ('public_source','decoder_source','public_manifest','control_state_sha256','control_grid_sha256')}
    cfg.update(manifest=manifest,source_pins=source_pins,pilot_state_sha256=None,mode='entry' if entry else 'rehearsal',binding=binding,environment=environment,asset_binding=asset_binding,guard_source_sha256=guards)
    rehearsal_state=None
    if entry:
        require(all((rehearsal_build,rehearsal_receipt,rehearsal_csv)),'ENTRY_REQUIRES_REHEARSAL')
        rb=verify_build(rehearsal_build);require(rb['mode']=='rehearsal' and rb['binding']==binding,'EXACT_REHEARSAL_BUILD')
        rr=json.loads(Path(rehearsal_receipt).read_bytes());R7.require_rehearsal(rr,binding,rehearsal_csv)
        require(visible_rank_difference(rehearsal_csv,reference['submission'],visible)==0.,'REHEARSAL_VISIBLE_BLEND_RANKS_EQUAL_HELIOSLI')
        rehearsal_state=rr['workers']['s6'][0]['state_before'];cfg['pilot_state_sha256']=rehearsal_state
        cfg['binding']={k:v for k,v in binding.items() if k not in PRIVATE_BINDING}
        cfg['binding'].update(rehearsal_receipt_sha256=file_sha(rehearsal_receipt),rehearsal_csv_sha256=file_sha(rehearsal_csv),qualified_r7_state=rehearsal_state)
    else:cfg.update(probes=probes,probe_order=ids+oldsent,old_sentinels=oldsent,s6_sentinels=sentinels,development_windows=[x['windows'] for x in sr['records']],
                    heliosli_reference=reference,heliosli_visible_atol=R7.HELIOSLI_VISIBLE_ATOL)
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
from knee.public_r7_recovery_entry import run,heliosli_parity
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
    weights=mount('nartaa/rsna-knee-publication-swa-weights-20261007',[cfg['public_manifest']['checkpoint_file']])
    roots=[Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),Path('/kaggle/input/rsna-knee-abnormality-detection')]
    matches=[p for p in roots if all((p/n).is_file() for n in ('test.csv','test_series.csv','sample_submission.csv'))];require(len(matches)==1,'UNIQUE_ACTUAL_TEST_ROOT')
    cfg.update(code=str(code),compact=str(compact),anatomy=str(anatomy),control_checkpoint=str(weights/cfg['public_manifest']['checkpoint_file']))
    save(project/'runtime-config.json',cfg)
    try:report=run(matches[0],cfg,out,started=started)
    finally:
        if cfg['mode']=='rehearsal':
            try:print('HELIOSLI_PARITY',json.dumps(heliosli_parity(out,cfg)),flush=True)
            except Exception as parity_error:print('HELIOSLI_PARITY_UNAVAILABLE',repr(parity_error),flush=True)
    print(json.dumps({'status':report['status'],'completed':report['completed'],'mode':report['mode'],'elapsed_seconds':report['elapsed_seconds'],
                      'conservative_seconds':(report.get('projection') or {}).get('conservative_seconds')}),flush=True)
except BaseException as e:
    save(out/'setup-failure-private.json',{'type':type(e).__name__,'detail':str(e)})
    raise
"""
    compile(boot,'<r7-entry>','exec');nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python3','language':'python'}},'cells':[{'id':'dual-image-inference','cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':boot.splitlines(True)}]}
    meta={'id':'ozgung/'+slug,'title':title,'code_file':'notebook.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_tpu':False,'enable_internet':False,
          'competition_sources':['rsna-knee-abnormality-detection'],'dataset_sources':DATASETS,'kernel_sources':[],'model_sources':[],'docker_image':DOCKER_IMAGE,'machine_shape':'NvidiaTeslaT4'}
    raw=json.dumps(nb,indent=2)+'\n';require(len(raw.encode())<900000,'NOTEBOOK900K_CAP');dest.mkdir(parents=True)
    (dest/'notebook.ipynb').write_text(raw);(dest/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=project,text=True).strip()
    changed=subprocess.check_output(['git','status','--porcelain','--',*R7_FILES,*DEFAULT_MODULE_SHA],cwd=project,text=True).strip()
    receipt={'source_revision':revision,'source_state':'uncommitted R7 implementation; exact source hashes bound' if changed else 'committed implementation; exact source hashes bound','mode':cfg['mode'],'variant':binding['variant'],'binding':binding,
             'qualified_r7_state':rehearsal_state,'payload_sha256':digest(pack),'files':{n:file_sha(dest/n) for n in ('notebook.ipynb','kernel-metadata.json')},'notebook_bytes':len(raw.encode()),'source_files':len(sources),
             'source_hashes':{n:hashlib.sha256(s.encode()).hexdigest() for n,s in sources.items()},'r7_files_sha256':{n:file_sha(project/n) for n in R7_FILES},'runtime_checks_exercised':False,'missing_reference_policy':'never synthesize saved predictions'}
    save(dest/'build-receipt.json',receipt);return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('evaluation-build','evaluation-result','audit-path','pilot-build','pilot-result','control-build','control-result','data-root','heliosli-notebook','r7-asset-dir','heliosli-output','destination','slug','cpu-evidence'):p.add_argument('--'+n,required=True)
    p.add_argument('--entry',action='store_true')
    for n in ('rehearsal-build','rehearsal-receipt','rehearsal-csv'):p.add_argument('--'+n)
    r=build(**vars(p.parse_args()));print(json.dumps({'notebook_bytes':r['notebook_bytes'],'source_files':r['source_files'],'notebook_sha256':r['files']['notebook.ipynb'],'payload_sha256':r['payload_sha256'],'runtime_checks_exercised':False}))
