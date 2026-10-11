"""Prepare immutable image-only full384 rehearsal or receipt-gated pure entry."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from knee.reference import canonical_bytes, sha256, WEIGHT_DATASET
from knee.reference_blend import CONVNEXT_DATASET, DOCKER_IMAGE
from knee.coverage_inference import CODECS
from knee.public_fullfield_entry import VERSION, MAX_SECONDS, METHOD, require, require_rehearsal, test_contract

FOLD1_NOTEBOOK_SHA = '93643a2bb423969afab0ad7ae6c9f88e24603e9f1d017445dff774132f86c422'
FOLD1_BUNDLE_SHA = 'a052e3945ba1742d1ae286eaaad6e900ae98bac64b06c56bc26bca9ec7ca12d9'
FOLD1_RECEIPT_SHA = '482ea13858274dce4e4e3cfffd04c3c9a9ee332eb927ed680417b7c97d2057c9'
QUALIFICATION_ARRAY_SHA = 'db0271f3265c576f85a2ff80c1f73ae05f1e71ff73a6408cc3c75af7732b45ff'
# Exact inference bodies are sliced out of the hash-qualified frozen source map.
ROOTS = {
    'knee/reference.py': {'sha256','canonical_bytes','load_definitions','load_model','slot_plan','validate_predictions'},
    'knee/mirror_input.py': {'prepare_study','protect_decoder'},
    'knee/raptor_input.py': {'fatal_error','propagate_fatal','transform_exception_handlers','validate_production_volume'},
    'knee/public_fullfield.py': {'make_candidate','snapshot','require_immutable','prediction_error','fullfield_windows','infer_candidate'},
    'knee/public_adaptation_fit.py': {'require'},
    'knee/public_adaptation_pilot.py': {'state_fingerprint'},
}


def literals(path):
    raw=Path(path).read_bytes(); require(sha256(raw)==FOLD1_NOTEBOOK_SHA, 'Frozen fold1 notebook bytes')
    notebook=json.loads(raw); code='\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
    result={}
    for node in ast.parse(code).body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            name=node.targets[0].id
            if name in ('sources','expected_hashes','expected_bundle','PUBLIC_SOURCE','DECODER_SOURCE','MANIFEST','SAMPLE','BINDING'):
                require(name not in result,'Duplicate frozen literal');result[name]=ast.literal_eval(node.value)
    require(result['expected_bundle']==FOLD1_BUNDLE_SHA and sha256(canonical_bytes(result['sources']))==result['expected_bundle'], 'Frozen bundle binding')
    require({n:sha256(s.encode()) for n,s in result['sources'].items()}==result['expected_hashes'], 'Frozen source hashes')
    return result


def exact_closure(frozen):
    """Select top-level source segments without rewriting scientific functions.

    Resolve their module globals and local knee imports transitively. Imports of
    irrelevant training helpers are omitted; every retained definition is byte exact.
    """
    wanted={name:set(values) for name,values in ROOTS.items()}
    trees={name:ast.parse(source) for name,source in frozen.items()}
    def defined(node):
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)):return {node.name}
        if isinstance(node,ast.Assign):return {n.id for target in node.targets for n in ast.walk(target) if isinstance(n,ast.Name)}
        if isinstance(node,ast.AnnAssign) and isinstance(node.target,ast.Name):return {node.target.id}
        return set()
    changed=True
    while changed:
        changed=False
        for name in list(wanted):
            nodes=trees[name].body
            selected=[n for n in nodes if defined(n)&wanted[name]]
            needed={n.id for node in selected for n in ast.walk(node) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load)}
            old=len(wanted[name]);wanted[name]|=needed;changed|=len(wanted[name])!=old
            for node in [n for n in nodes if isinstance(n,ast.ImportFrom)]+[n for item in selected for n in ast.walk(item) if isinstance(n,ast.ImportFrom)]:
                if node.module and node.module.startswith('knee.'):
                    deps={a.name for a in node.names if (a.asname or a.name) in needed or node not in nodes}
                    if deps:
                        path=node.module.replace('.','/')+'.py';require(path in frozen,'Missing qualified dependency')
                        previous=set(wanted.get(path,set()));wanted.setdefault(path,set()).update(deps)
                        changed|=wanted[path]!=previous
    sources={'knee/__init__.py':''};pins={}
    for name,needed in wanted.items():
        chunks=[]
        for node in trees[name].body:
            if isinstance(node,ast.ImportFrom) and node.module and node.module.startswith('knee.'):
                aliases=[a for a in node.names if (a.asname or a.name) in needed]
                if aliases:chunks.append('from '+node.module+' import '+', '.join(a.name+(' as '+a.asname if a.asname else '') for a in aliases))
            elif isinstance(node,(ast.Import,ast.ImportFrom)) or defined(node)&needed:
                start=min([node.lineno]+[d.lineno for d in getattr(node,'decorator_list',[])])-1
                segment=''.join(frozen[name].splitlines(True)[start:node.end_lineno])
                chunks.append(segment.rstrip('\n'))
                if defined(node):pins[name+':'+','.join(sorted(defined(node)))]=sha256(segment.encode())
        sources[name]='\n\n'.join(chunks)+'\n'
        compile(sources[name],name,'exec')
    require(not any('metric_report' in s or 'load_binary_truth' in s or 'train.csv' in s or '.parquet' in s for s in sources.values()), 'Inference closure contains label/metric dependency')
    return sources,pins


def verify_build(destination):
    destination=Path(destination);receipt=json.loads((destination/'build-receipt.json').read_bytes())
    for name,pin in receipt['files'].items():require(sha256((destination/name).read_bytes())==pin,'Immutable build bytes differ')
    require(json.loads((destination/'kernel-metadata.json').read_bytes())['dataset_sources']==[WEIGHT_DATASET,CONVNEXT_DATASET], 'Only checkpoint/codec attachments permitted')
    return receipt


def build(fold1_build, fold1_result, data_root, destination, slug, *, entry=False, rehearsal_build=None, rehearsal_receipt=None, rehearsal_csv=None):
    import numpy as np
    project=Path(__file__).resolve().parents[1];dest=Path(destination)
    require(not dest.exists() and dest.resolve().is_relative_to(project/'outputs'), 'Fresh private immutable output required')
    require(re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*',slug), 'Invalid slug')
    frozen=literals(Path(fold1_build)/'notebook.ipynb')
    build_receipt=json.loads((Path(fold1_build)/'build-receipt.json').read_bytes())
    require(build_receipt['binding']==frozen['BINDING'], 'Frozen build binding')
    result=Path(fold1_result)
    require(sha256((result/'receipt.json').read_bytes()) == FOLD1_RECEIPT_SHA, 'Exact accepted fold1 receipt')
    require(sha256((result/'qualification-predictions-private.npz').read_bytes()) == QUALIFICATION_ARRAY_SHA, 'Exact independently audited qualification arrays')
    qualification=json.loads((result/'receipt.json').read_bytes())
    require(qualification['binding']==frozen['BINDING'] and qualification['status']=='PASS' and qualification['mechanical_gate']=='PASS' and qualification['qualification_completed']==32 and qualification['evaluated_studies']==866 and qualification['final_state_and_all_buffers_immutable'] is True, 'Complete unchanged replication qualification')
    require(qualification['macro_delta'] > 0 and qualification['paired_bootstrap']['ci95'][0] > 0 and len(qualification['target_damage']) < 3 and qualification['metrics']['candidate']['defined_targets'] == 12 and len(qualification['metrics']['candidate']['targets']) == 12 and all(t['auc'] is not None for t in qualification['metrics']['candidate']['targets'].values()), 'Accepted positive all-target paired comparison required')
    require(all(0<=x<=1e-6 for x in qualification['max_prediction_errors'].values()), 'Replication prediction gates')
    sources,pins=exact_closure(frozen['sources'])
    sources['knee/public_fullfield_entry.py']=(project/'knee/public_fullfield_entry.py').read_text()
    for name,source in sources.items():compile(source,name,'exec')
    bundle=sha256(canonical_bytes(sources))
    # Exact immutable mechanics arrays, ID-only sample, and per-image signatures.
    sample=frozen['SAMPLE'];require(len(sample)==32 and len(set(sample))==32,'Frozen sample')
    with np.load(result/'qualification-predictions-private.npz',allow_pickle=False) as archive:
        require(archive['study_ids'].tolist()==sample,'Frozen prediction ID order')
        probabilities=archive['candidate'].copy()
    require(probabilities.shape==(32,12) and np.isfinite(probabilities).all() and ((probabilities>=0)&(probabilities<=1)).all(),'Qualified probability schema')
    hashes=qualification['qualification'];require(len(hashes)==32 and [h['ordinal'] for h in hashes]==list(range(32)), 'Qualified hash order')
    probes=[{'id':sid,'hashes':{k:hashes[i][k] for k in ('volume_sha256','mask_sha256','windows')},'probabilities':probabilities[i].tolist()} for i,sid in enumerate(sample)]
    # Read exact schema from source LAB literal; no source execution or labels.
    public_tree=ast.parse(frozen['PUBLIC_SOURCE'])
    targets=next(ast.literal_eval(n.value) for n in public_tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='LAB' for t in n.targets))
    root=Path(data_root);columns,visible_ids,_=test_contract(root,targets,visible=True)
    env=qualification['environment']
    binding={'version':VERSION,'method':METHOD,'source_bundle_sha256':bundle,'primitives_sha256':sha256(canonical_bytes(pins)), 'qualified_grid':qualification['state_grid_gate'],
        'environment':{'python_major_minor':[3,12],'torch':env['torch'],'timm':env['timm'],'numpy':env['numpy'],'cuda':env['cuda']},
        'columns':columns,'visible_ids':visible_ids,'train_series_sha256':sha256((root/'train_series.csv').read_bytes()),
        'fold1_notebook_sha256':FOLD1_NOTEBOOK_SHA,'fold1_receipt_sha256':sha256((result/'receipt.json').read_bytes()),
        'probe_contract_sha256':sha256(canonical_bytes(probes)), 'checkpoint_sha256':frozen['MANIFEST']['checkpoint_sha256'],
        'builder_sha256':sha256(Path(__file__).read_bytes()),'docker_image':DOCKER_IMAGE,'codecs':CODECS}
    contract={'mode':'entry' if entry else 'rehearsal','binding':binding}
    if entry:
        require(all((rehearsal_build,rehearsal_receipt,rehearsal_csv)), 'Entry requires exact rehearsal evidence')
        prior=verify_build(rehearsal_build);require(prior['binding']==binding and prior['mode']=='rehearsal', 'Exact rehearsal build required')
        report=json.loads(Path(rehearsal_receipt).read_bytes());require_rehearsal(report,rehearsal_csv,binding)
        contract.update(expected_state=report['candidate_state'], rehearsal_receipt_sha256=sha256(Path(rehearsal_receipt).read_bytes()), rehearsal_csv_sha256=sha256(Path(rehearsal_csv).read_bytes()), rehearsal_build_sha256=sha256((Path(rehearsal_build)/'build-receipt.json').read_bytes()))
    else:contract['probes']=probes
    # Pure entry carries aggregate bindings only, never training IDs/prediction fixtures.
    if entry:
        contract['binding']={k:v for k,v in binding.items() if k not in ('visible_ids','train_series_sha256','probe_contract_sha256')}
    bootstrap="""
import time
started=time.monotonic()
import os,sys,json,hashlib,signal,subprocess,importlib.metadata
from pathlib import Path
def platform_deadline(signum,frame): raise TimeoutError('Actual nine-hour notebook runtime exhausted')
signal.signal(signal.SIGALRM,platform_deadline);signal.alarm(9*3600)
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
"""
    code=bootstrap+f'sources={sources!r}\nexpected_bundle={bundle!r}\nCONTRACT={contract!r}\nPUBLIC_SOURCE={frozen["PUBLIC_SOURCE"]!r}\nDECODER_SOURCE={frozen["DECODER_SOURCE"]!r}\nMANIFEST={frozen["MANIFEST"]!r}\nCODECS={CODECS!r}\nWEIGHT_DATASET={WEIGHT_DATASET!r}\nCODEC_DATASET={CONVNEXT_DATASET!r}\n'+"""
if hashlib.sha256(json.dumps(sources,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()!=expected_bundle:raise ValueError('Source binding differs')
project=Path('/kaggle/working/project')
for name,content in sources.items():
    p=project/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(content.encode())
sys.path.insert(0,str(project));os.environ['KNEE_SOURCE_SHA256']=expected_bundle
os.environ['KNEE_PRIMITIVES_SHA256']=CONTRACT['binding']['primitives_sha256']
from knee.public_fullfield_entry import run,atomic_json
out=Path('/kaggle/working/public-fullfield-entry')
def mount(reference,required):
    owner,slug=reference.split('/')
    roots=[Path('/kaggle/input/datasets')/owner/slug,Path('/kaggle/input')/slug]
    matches=[p for p in roots if all((p/n).is_file() for n in required)]
    if len(matches)!=1:raise ValueError('Expected unique exact attachment root')
    return matches[0]
try:
    env=CONTRACT['binding']['environment']
    if list(sys.version_info[:2])!=env['python_major_minor'] or any(importlib.metadata.version(p)!=env[p] for p in ('torch','timm','numpy')):raise ValueError('Pinned setup environment differs')
    codec=mount(CODEC_DATASET,list(CODECS))
    for name,pin in CODECS.items():
        if hashlib.sha256((codec/name).read_bytes()).hexdigest()!=pin:raise ValueError('Codec bytes differ')
    subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps',*[str(codec/n) for n in CODECS]],check=True)
    weights=mount(WEIGHT_DATASET,[MANIFEST['checkpoint_file']])
    roots=[Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),Path('/kaggle/input/rsna-knee-abnormality-detection')]
    matches=[p for p in roots if (p/'test_series.csv').is_file() and (p/'sample_submission.csv').is_file() and (p/'test.csv').is_file()]
    if len(matches)!=1:raise ValueError('Unique competition root required')
    report=run(matches[0],weights/MANIFEST['checkpoint_file'],PUBLIC_SOURCE,DECODER_SOURCE,MANIFEST,CONTRACT,out,started=started)
    if CONTRACT['mode']=='entry':
        import shutil
        shutil.copyfile(out/'submission.csv','/kaggle/working/submission.csv')
    print(json.dumps({'status':report['status'],'mode':CONTRACT['mode'],'elapsed_seconds':report['elapsed_seconds']}),flush=True)
except BaseException as error:
    if not (out/'receipt.json').exists():
        out.mkdir(parents=True,exist_ok=True)
        atomic_json(out/'receipt.json',{'status':'FAIL','failure_stage':'offline_setup','failure_type':type(error).__name__,'binding':CONTRACT['binding'],'elapsed_seconds':time.monotonic()-started})
    raise RuntimeError('Full384 entry failed; inspect preserved aggregate receipt') from None
finally:signal.alarm(0)
"""
    compile(code,'<full384-image-only>','exec')
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{},'cells':[{'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(True)}]}
    metadata={'id':'ozgung/'+slug,'title':slug.replace('-',' ').title(),'code_file':'notebook.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_tpu':False,'enable_internet':False,'competition_sources':['rsna-knee-abnormality-detection'],'dataset_sources':[WEIGHT_DATASET,CONVNEXT_DATASET],'kernel_sources':[],'model_sources':[],'docker_image':DOCKER_IMAGE}
    raw=(json.dumps(notebook,indent=2)+'\n').encode();require(len(raw)<900000,'Notebook payload <900000 bytes')
    dest.mkdir(parents=True);(dest/'notebook.ipynb').write_bytes(raw)
    (dest/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    receipt={'status':'BUILT_UNLAUNCHED','mode':contract['mode'],'binding':binding,'notebook_bytes':len(raw),'sources':{n:sha256(s.encode()) for n,s in sources.items()},'exact_primitive_segments':pins,'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=project,text=True).strip(),'source_state':'new implementation may be uncommitted; exact bytes bound','files':{n:sha256((dest/n).read_bytes()) for n in ('notebook.ipynb','kernel-metadata.json')}}
    (dest/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('fold1_build','fold1_result','data_root','destination','slug'):parser.add_argument(name)
    parser.add_argument('--entry',action='store_true')
    for name in ('rehearsal_build','rehearsal_receipt','rehearsal_csv'):parser.add_argument('--'+name.replace('_','-'))
    receipt=build(**vars(parser.parse_args()));print(json.dumps({'status':receipt['status'],'mode':receipt['mode'],'notebook_bytes':receipt['notebook_bytes']}))
