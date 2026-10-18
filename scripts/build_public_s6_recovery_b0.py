"""B0 no-OAI dual-T4 builder: Dread CC0 Raptor v4 K62 old arm, unchanged S6 arm.

A separate entry point: the qualified default builder and its modules stay
byte-identical, are pinned below and are shipped unchanged inside the B0 bundle.
Never pushes, runs a kernel or submits.
"""
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
import knee.public_s6_recovery_b0 as B0
from knee.public_s6_recovery_entry import SENTINELS
from knee.public_s6_recovery_runtime import PYDICOM_BUFFER_FILE_SHA,PYDICOM_BUFFER_FUNCTION_SHA
from knee.public_s6_qualification import canonical,digest,file_sha,require,save,LABELS,UID
from knee.public_fullfield_entry import test_contract
from knee.reference_blend import DOCKER_IMAGE,CONVNEXT_DATASET
from knee.reference import RAPTOR_WEIGHT_DATASET,audit_source
import scripts.build_public_s6_recovery_entry as default
from scripts.build_public_s6_evaluation import accepted_pilot,PILOT_RECEIPT_SHA

DATASETS=['heliosli/rsna-knee-compact-v1','heliosli/rsna-orthofoundation-fold0-epoch7-20261004',CONVNEXT_DATASET,RAPTOR_WEIGHT_DATASET]
# Byte-identical qualified default modules (0.954 entry build 82111ff, rehearsal 42d4b3d).
DEFAULT_MODULE_SHA={'knee/public_s6_recovery_entry.py':'76217a402ae3bb4c8c9517c76875691f3565b84bf0b2e548295cd9f726a90be8',
                    'knee/public_s6_recovery_runtime.py':'d67ceb738930c85b19e1866dfeb0d524e015a68ff321e039f00105c590ca6d08',
                    'knee/public_s6_recovery_rehearsal.py':'8424d87b5656ccdf449c69c5ee06c09744116f185f086861b7feee8bd5864502',
                    'scripts/build_public_s6_recovery_entry.py':'d8915ef31345f452ca8a8ea0a2cbeb12ad717363d9e7b54b3cd714ef3ed3109c'}
# T4 raptor62 reference rehearsal (2026-10-08): same checkpoint, source and recipe.
RAPTOR62_CONTRACT_SHA='251a3c3ddba554b324a837e0d777c289b1b229e069e1dae21b9555424591cfaa'
RAPTOR62_REHEARSAL_SHA='e62ea2a5da4c3b63835b9ec5b1ffd7fe7ebc634699273efe0bfc9e12922ab4e1'
RAPTOR62_CSV_SHA='d47d249c2738c45169c47542b55ff91f322534d30acf3e5bae6ad36c92668daa'
RAPTOR_POLICY_SHA='101a7f766903bb041587dffd4b951c22ff9907f630db87498596f9df8bca4c5a'
# Declared, not measured: raptor62 ran cudnn.benchmark/TF32, B0 runs deterministic.
# Wrong-profile inputs (crop/span/slot errors) move probabilities far beyond this.
RAPTOR62_VISIBLE_ATOL=5e-3
ROOTS={k:set(v) for k,v in default.ROOTS.items() if k!='knee/mirror_input.py'}
ROOTS['knee/reference.py']={'load_definitions','load_model','infer_checked','RAPTOR_WEIGHT_DATASET','RAPTOR_WEIGHT_NAME'}
ROOTS['knee/raptor_input.py']={'prepare_study','stress_inputs','STRESS_CASES','strict_eligible','strict_compare','validate_production_volume','propagate_fatal'}
PRIVATE_BINDING=('sentinel_inputs','sentinel_probabilities','visible_ids','probe_contract_sha256','s6_probe_ids','old_historical_ids')
# Kaggle SaveKernel rejects slugs/titles over 50 characters (400 Bad Request on
# the 51-character B0 rehearsal push). Title is derived from the slug.
MAX_KERNEL_SLUG=50
REHEARSAL_SLUG='knee-s6-dread-b0-rehearsal-20261013'
ENTRY_SLUG='knee-s6-dread-b0-entry-20261013'


def kernel_title(slug):
    require(isinstance(slug,str) and re.fullmatch('[a-z0-9]+(?:-[a-z0-9]+)*',slug) and 5<=len(slug)<=MAX_KERNEL_SLUG,'SLUG')
    title=slug.replace('-',' ').title();require(5<=len(title)<=MAX_KERNEL_SLUG,'KERNEL_TITLE_LENGTH');return title


B0_FILES=('knee/public_s6_recovery_b0.py','knee/public_s6_recovery_b0_rehearsal.py','scripts/build_public_s6_recovery_b0.py','scripts/qualify_b0_dread_cpu.py')


def inference_closure(frozen):
    """The reviewed default extraction (same data-boundary rewrite) with B0 roots."""
    text=Path(default.helper.__file__).read_text()
    source=ast.get_source_segment(text,next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='exact_closure'))
    boundary="    require(not any('metric_report' in s or 'load_binary_truth' in s or 'train.csv' in s or '.parquet' in s for s in sources.values()), 'Inference closure contains label/metric dependency')"
    require(source.count(boundary)==1,'REVIEWED_CLOSURE_SOURCE')
    source=source.replace(boundary,boundary.replace(" or 'train.csv' in s",''))
    ns={'ast':ast,'ROOTS':ROOTS,'require':require,'sha256':lambda b:hashlib.sha256(b).hexdigest()};exec(compile(source,'<b0-exact-closure>','exec'),ns)
    sources,pins=ns['exact_closure'](frozen)
    for name,wanted in ROOTS.items():
        defined=set()
        for node in ast.parse(sources[name]).body:
            if isinstance(node,(ast.FunctionDef,ast.ClassDef)):defined.add(node.name)
            elif isinstance(node,ast.Assign):defined|={n.id for t in node.targets for n in ast.walk(t) if isinstance(n,ast.Name)}
        require(wanted<=defined,'B0_CLOSURE_ROOT_MISSING_'+name)
    return sources,pins


def verify_build(destination):
    dest=Path(destination);r=json.loads((dest/'build-receipt.json').read_bytes())
    require(all(file_sha(dest/name)==pin for name,pin in r['files'].items()),'IMMUTABLE_BUILD_BYTES')
    m=json.loads((dest/'kernel-metadata.json').read_bytes());require(m['dataset_sources']==DATASETS and not m['enable_internet'] and m['is_private'] is True,'ONLY_IMAGE_ASSET_ATTACHMENTS')
    return r


def raptor62_evidence(result,manifest,visible):
    """Exact raptor62 receipt: same manifest/policy; visible decode counts and probabilities."""
    import csv
    root=Path(result);contract=default.pinned(root/'reference-contract.json',RAPTOR62_CONTRACT_SHA)
    report=default.pinned(root/'reference-rehearsal/rehearsal.json',RAPTOR62_REHEARSAL_SHA);path=root/'reference-rehearsal/visible-predictions.csv'
    require(file_sha(path)==RAPTOR62_CSV_SHA==report['visible_csv_sha256'],'RAPTOR62_VISIBLE_CSV')
    require(contract['source']==manifest==report['manifest'] and contract['input_policy']==report['input_policy']=={'version':'raptor-public-partial-v2','sha256':RAPTOR_POLICY_SHA},'RAPTOR62_SAME_QUALIFIED_MANIFEST_POLICY')
    env=report['environment']
    require(report['status']=='passed' and report['prediction_fallbacks']==0 and env['gpu']=='Tesla T4' and env['packages']['torch']=='2.10.0+cu128' and env['packages']['timm']=='1.0.26'
            and report['strict_comparison_counts']=={'PASS':67} and report['sampling']['test_count']==3,'RAPTOR62_PASSED_T4_RECEIPT')
    with path.open(newline='') as h:rows=list(csv.reader(h))
    require(rows[0]==[UID]+LABELS and [r[0] for r in rows[1:]]==visible,'RAPTOR62_VISIBLE_ORDER')
    studies=report['studies'][:3];require([(s['ordinal'],s['split'],s['eval_windows'],s['strict_comparison']) for s in studies]==[(i,'test',62,'PASS') for i in range(3)],'RAPTOR62_VISIBLE_STUDIES')
    reference={}
    for sid,row,study in zip(visible,rows[1:],studies):
        p=[float(x) for x in row[1:]];require(len(p)==12 and all(0<=x<=1 for x in p),'RAPTOR62_PROBABILITIES')
        reference[sid]={'probabilities':p,'counts':{k:study[k] for k in B0.RAPTOR62_COUNT_KEYS}}
    summary={'contract_sha256':RAPTOR62_CONTRACT_SHA,'rehearsal_sha256':RAPTOR62_REHEARSAL_SHA,'csv_sha256':RAPTOR62_CSV_SHA,'visible_atol':RAPTOR62_VISIBLE_ATOL,
             'visible_ids':visible,'manifest_sha256':digest(manifest),'source_revision':report['source_revision'],'measured_seconds_mean_per_study':report['estimated_1300_seconds_mean']/1300}
    return reference,summary


def cpu_receipt(path,manifest):
    r=json.loads(Path(path).read_bytes())
    require(r['state_sha256']==B0.DREAD_STATE_SHA256 and r['constructor_grid_sha256']==B0.DREAD_GRID_SHA256 and r['checkpoint_sha256']==B0.DREAD_CHECKPOINT_SHA256
            and r['manifest']==manifest and r['windows_shape']==[62,3,384,384] and r['forward_windows']==62 and r['cpu_repeat_max_abs']<=1e-6 and r['strict_state_load'] is True
            and r['labels']==LABELS and r['nartaa_grid_method_check'] is True and r['raptor62_manifest_equal'] is True,'B0_CPU_QUALIFICATION_RECEIPT')
    return {'receipt_sha256':file_sha(path),'state_sha256':r['state_sha256'],'grid_sha256':r['constructor_grid_sha256'],'torch':r['torch'],'device':'cpu'}


def build(evaluation_build,evaluation_result,audit_path,pilot_build,pilot_result,data_root,dread_source,raptor62_result,dread_cpu_receipt,destination,slug,*,
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
    with np.load(Path(pilot_result)/'wrapper/predictions-private.npz',allow_pickle=False) as a:
        require(a['ordinals'].tolist()==list(range(32)),'PILOT_PROBABILITY_ORDER');pp=a['probabilities'].copy()
    require(pp.shape==(32,12) and np.isfinite(pp).all(),'MECHANICAL_ARRAYS')
    # Dread old arm through the existing reviewed raptor profile (K62, no TTA).
    module,decoder_source,manifest=audit_source(Path(dread_source).read_text(),profile='raptor',checkpoint_sha256=B0.DREAD_CHECKPOINT_SHA256,raptor_windows=B0.RAPTOR_WINDOWS)
    require(decoder_source=='','RAPTOR_HAS_NO_PINNED_DECODER');B0.raptor_profile_gate(None,manifest)
    columns,visible,_=test_contract(data_root,LABELS,visible=True)
    raptor62_reference,raptor62=raptor62_evidence(raptor62_result,manifest,visible);cpu_pins=cpu_receipt(dread_cpu_receipt,manifest)
    # Same four historical old-sentinel IDs as the default rehearsal; Dread has no
    # saved probabilities for them, so fresh decode/reload/cross-board parity gates.
    dev=payload['cfg']['dev'];oldsent=[dev[i] for i in (0,1,868,869)]
    require(len(set(oldsent))==4 and not set(oldsent)&set(ids) and not set(oldsent+ids)&set(visible),'DISJOINT_OLD_HISTORICAL_SENTINELS')
    frozen={k:v for k,v in payload['sources'].items() if k.startswith('knee/')}
    frozen['knee/public_fullfield_entry.py']=(project/'knee/public_fullfield_entry.py').read_text()
    frozen['knee/public_s6_recovery.py']=(project/'knee/public_s6_recovery.py').read_text()
    sources,pins=inference_closure(frozen)
    for name,content in original.items():
        if not name.startswith('knee/'):require(payload['sources'][name]==content,'EXACT_SCIENTIFIC_SOURCE');sources[name]=content
    for name in ('knee/public_s6_recovery_entry.py','knee/public_s6_recovery_runtime.py','knee/public_s6_recovery_b0.py'):sources[name]=(project/name).read_text()
    guard_files=('compact_infer.py','training/native_cache.py','training/dicom_geometry.py','knee_anchor_bank_v2.py','px_cache.py')
    guards={name:hashlib.sha256(sources[name].encode()).hexdigest() for name in guard_files}
    production_sources=dict(sources)
    extensions={name:(project/name).read_text() for name in ('knee/public_s6_recovery_rehearsal.py','knee/public_s6_recovery_b0_rehearsal.py')}
    if not entry:sources.update(extensions)
    for name,s in sources.items():compile(s,name,'exec')
    probes={sid:{'old':{'input':{},'probabilities':None},
                 's6':{'input':wrapper['records'][i]['input'],'probabilities':pp[i].tolist(),'mechanics':{k:wrapper['records'][i][k] for k in ('windows','anchor_method','anchor_reasons','side','field_center_fallbacks')}}} for i,sid in enumerate(ids)}
    for sid in oldsent:probes[sid]={'old':{'input':{},'probabilities':None}}
    cfg0=payload['cfg'];sr=json.loads((Path(evaluation_result)/'s6/receipt.json').read_bytes());environment={'torch':result['environment']['torch'],'packages':result['environment']['packages']}
    sentinels=[ids[i] for i in SENTINELS]
    binding={'version':B0.VERSION,'variant':'B0-no-OAI-dread-raptor62','method':B0.METHOD,'input_policy':B0.INPUT_POLICY,'old_profile':B0.OLD_PROFILE,
        'source_bundle_sha256':digest(production_sources),'rehearsal_extension_sha256':digest(extensions),'cpu_gate':cpu,'guard_source_sha256':guards,
        'pydicom_guard':{'version':'3.0.2','package_file':'pydicom/pixels/decoders/base.py','file_sha256':PYDICOM_BUFFER_FILE_SHA,'function_sha256':PYDICOM_BUFFER_FUNCTION_SHA,'raise_sites':[22,33]},
        'primitive_segments_sha256':digest(pins),'default_module_sha256':DEFAULT_MODULE_SHA,
        'evaluation_notebook_sha256':default.EVALUATION_NOTEBOOK_SHA,'evaluation_receipt_sha256':default.EVALUATION_RECEIPT_SHA,'independent_audit_sha256':default.AUDIT_SHA,
        'pilot_receipt_sha256':PILOT_RECEIPT_SHA,'pilot_wrapper_sha256':default.PILOT_WRAPPER_SHA,'pilot_arrays_sha256':file_sha(Path(pilot_result)/'wrapper/predictions-private.npz'),
        'old_qualification':{'policy':'Dread raptor62 T4 receipt (visible decode counts exact, probabilities within declared atol) plus all36 source-reference, strict pydicom and strict-reload repeat parity; no historical Dread probabilities exist for train probes',
                             'raptor62':raptor62,'cpu':cpu_pins},
        'evaluation_build_receipt_sha256':file_sha(Path(evaluation_build)/'build-receipt.json'),'pilot_build_receipt_sha256':file_sha(Path(pilot_build)/'build-receipt.json'),
        'environment':environment,'qualified_state':{'old_state':B0.DREAD_STATE_SHA256,'old_grid':B0.DREAD_GRID_SHA256,'s6_state':cfg0['pilot_state_sha256']},'assets':{'manifest':pilot['manifest'],'binding':pilot['binding']},
        'public_manifest':manifest,'codecs':payload['codecs'],'columns':columns,'visible_ids':visible,'probe_contract_sha256':digest(probes),'s6_probe_ids':ids,'old_historical_ids':oldsent,'builder_sha256':file_sha(__file__),
        'sentinel_inputs':{arm:{s:probes[s][arm]['input'] for s in (oldsent if arm=='old' else sentinels)} for arm in ('old','s6')},
        'sentinel_probabilities':{arm:{s:probes[s][arm]['probabilities'] for s in (oldsent if arm=='old' else sentinels)} for arm in ('old','s6')}}
    cfg={k:cfg0[k] for k in ('pilot_state_sha256','source_pins','manifest')}
    cfg.update(public_source=module,decoder_source=decoder_source,public_manifest=manifest,control_state_sha256=B0.DREAD_STATE_SHA256,control_grid_sha256=B0.DREAD_GRID_SHA256,
               mode='entry' if entry else 'rehearsal',binding=binding,environment=environment,asset_binding=pilot['binding'],guard_source_sha256=guards)
    if entry:
        require(all((rehearsal_build,rehearsal_receipt,rehearsal_csv)),'ENTRY_REQUIRES_REHEARSAL')
        rb=verify_build(rehearsal_build);require(rb['mode']=='rehearsal' and rb['binding']==binding,'EXACT_REHEARSAL_BUILD')
        rr=json.loads(Path(rehearsal_receipt).read_bytes());B0.require_rehearsal(rr,binding,rehearsal_csv)
        cfg['binding']={k:v for k,v in binding.items() if k not in PRIVATE_BINDING}
        cfg['binding']['rehearsal_receipt_sha256']=file_sha(rehearsal_receipt);cfg['binding']['rehearsal_csv_sha256']=file_sha(rehearsal_csv)
    else:cfg.update(probes=probes,probe_order=ids+oldsent,old_sentinels=oldsent,s6_sentinels=sentinels,development_windows=[x['windows'] for x in sr['records']],
                    raptor62_reference=raptor62_reference,raptor62_visible_atol=RAPTOR62_VISIBLE_ATOL)
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
from knee.public_s6_recovery_b0 import run
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
    weights=mount("""+repr(RAPTOR_WEIGHT_DATASET)+""",[cfg['public_manifest']['checkpoint_file']])
    roots=[Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),Path('/kaggle/input/rsna-knee-abnormality-detection')]
    matches=[p for p in roots if all((p/n).is_file() for n in ('test.csv','test_series.csv','sample_submission.csv'))];require(len(matches)==1,'UNIQUE_ACTUAL_TEST_ROOT')
    cfg.update(code=str(code),compact=str(compact),anatomy=str(anatomy),control_checkpoint=str(weights/cfg['public_manifest']['checkpoint_file']))
    save(project/'runtime-config.json',cfg)
    report=run(matches[0],cfg,out,started=started);print(json.dumps({'status':report['status'],'completed':report['completed'],'mode':report['mode']}),flush=True)
except BaseException as e:
    save(out/'setup-failure-private.json',{'type':type(e).__name__,'detail':str(e)})
    raise
"""
    compile(boot,'<s6-b0-entry>','exec');nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python3','language':'python'}},'cells':[{'id':'dual-image-inference','cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':boot.splitlines(True)}]}
    meta={'id':'ozgung/'+slug,'title':title,'code_file':'notebook.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_tpu':False,'enable_internet':False,
          'competition_sources':['rsna-knee-abnormality-detection'],'dataset_sources':DATASETS,'kernel_sources':[],'model_sources':[],'docker_image':DOCKER_IMAGE}
    raw=json.dumps(nb,indent=2)+'\n';require(len(raw.encode())<900000,'NOTEBOOK900K_CAP');dest.mkdir(parents=True)
    (dest/'notebook.ipynb').write_text(raw);(dest/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=project,text=True).strip()
    changed=subprocess.check_output(['git','status','--porcelain','--',*B0_FILES,*DEFAULT_MODULE_SHA],cwd=project,text=True).strip()
    receipt={'source_revision':revision,'source_state':'uncommitted B0 implementation; exact source hashes bound' if changed else 'committed implementation; exact source hashes bound','mode':cfg['mode'],'variant':binding['variant'],'binding':binding,
             'payload_sha256':digest(pack),'files':{n:file_sha(dest/n) for n in ('notebook.ipynb','kernel-metadata.json')},'notebook_bytes':len(raw.encode()),'source_files':len(sources),
             'source_hashes':{n:hashlib.sha256(s.encode()).hexdigest() for n,s in sources.items()},'runtime_checks_exercised':False,'missing_reference_policy':'never synthesize saved predictions'}
    save(dest/'build-receipt.json',receipt);return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('evaluation-build','evaluation-result','audit-path','pilot-build','pilot-result','data-root','dread-source','raptor62-result','dread-cpu-receipt','destination','slug','cpu-evidence'):p.add_argument('--'+n,required=True)
    p.add_argument('--entry',action='store_true')
    for n in ('rehearsal-build','rehearsal-receipt','rehearsal-csv'):p.add_argument('--'+n)
    r=build(**vars(p.parse_args()));print(json.dumps({'notebook_bytes':r['notebook_bytes'],'source_files':r['source_files'],'notebook_sha256':r['files']['notebook.ipynb'],'runtime_checks_exercised':False}))
