"""Build-only private offline fixed-checkpoint K16/K42 diagnostic. No launch."""
import argparse
import ast
import json
from pathlib import Path
import re
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from knee import own224_window_diagnostic as diagnostic
from knee import coatnet224_rehearsal as own
from knee.reference_blend import DOCKER_IMAGE


def definitions(source,names,*,remove_imports=False):
    nodes=[n for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    diagnostic.require(set(n.name for n in nodes)==set(names),'Definition inventory differs')
    if remove_imports:
        # Only local imports change: supplied symbols are the exact unchanged
        # extracted utilities from validation.py. Numerical bodies stay intact.
        for node in nodes:
            node.body=[n for n in node.body if not isinstance(n,ast.ImportFrom) or not (n.module or '').startswith('knee.')]
    return ast.unparse(ast.Module(body=nodes,type_ignores=[]))+'\n'


def build(destination,slug,evidence_root):
    e=Path(evidence_root);out=Path(destination)
    diagnostic.require(not out.exists() and re.fullmatch('[a-z0-9]+(?:-[a-z0-9]+)*',slug),'Fresh build and valid slug required')
    diagnostic.require(not out.resolve().is_relative_to(ROOT) or out.resolve().is_relative_to(ROOT/'outputs'),'Ignored output destination required')
    package=e/'outputs/kaggle/trackA-coatnet224-fold01-20261009'
    result=e/'outputs/kaggle/trackA-coatnet224-fold01-20261009-result/trackA'
    record=json.loads((result/'run_fold0.json').read_text())
    member={'fold':0,'kernel':own.KERNELS[0],'run_file':'run_fold0.json','run_sha256':diagnostic.file_sha(result/'run_fold0.json'),
        'checkpoint_file':'raptor_ft_fold0_swa.pt','checkpoint_sha256':diagnostic.CHECKPOINT_SHA,'selected_epochs':[8,9,7]}
    own.verify_run(record,member)
    diagnostic.checked(result/'raptor_ft_fold0_swa.pt',diagnostic.CHECKPOINT_SHA)
    diagnostic.checked(result/'oof_fold0.csv',diagnostic.OOF_SHA)
    diagnostic.checked(package/'tracka_entry.py',own.ENTRY_SHA)
    source=diagnostic.checked(package/'train_knee_patched.py',diagnostic.PATCHED_SHA).read_text()
    model=definitions(source,('StudyWindows','build_backbone','RaptorClassifier'))
    utilities=definitions((ROOT/'knee/validation.py').read_text(),('_auc','_report_key','_identity_columns'))
    utilities+=definitions((ROOT/'knee/public_adaptation_fit.py').read_text(),('load_binary_truth','metric_report','paired_bootstrap','report_components'),remove_imports=True)
    compile(model,'model','exec');compile(utilities,'utilities','exec')
    files=('knee/__init__.py','knee/own224_window_diagnostic.py','knee/fresh_reader_setup.py')
    sources={name:(ROOT/name).read_text() for name in files}
    private={name:(package/name).read_text() for name in ('folds_v0.json','excluded_ids.json')}
    for name,key in [('folds_v0.json','folds_sha256'),('excluded_ids.json','excluded_ids_sha256')]:
        diagnostic.checked(package/name,own.POLICY[key])
    audit=e/'outputs/analysis/public-train-recipe-20261009';manifest=json.loads((audit/'evidence-manifest.json').read_text())
    cache_pins={}
    for directory,prefix in [('corpus','all'),('corpus-ext','extra')]:
        listing=json.loads((audit/(directory+'-files.json')).read_text())
        sizes={r['name']:r['size'] for r in listing}
        for suffix in ('ids','masks','vols'):
            name=prefix+'_'+suffix+'.npy';cache_pins[name]={'bytes':sizes[name]}
            if suffix!='vols':
                pin=manifest['files'][directory+'/'+name]['sha256'];diagnostic.checked(audit/directory/name,pin);cache_pins[name]['sha256']=pin
    contract={'schema':1,'checkpoint_source':own.KERNELS[0],'checkpoint_version':3,
        'version_binding':'Cached completed v3 handoff; exact mounted run/checkpoint/OOF hashes required regardless of mutable latest mount',
        'source_patched_sha256':diagnostic.PATCHED_SHA,'exporter_sha256':own.ENTRY_SHA,'model_code_sha256':diagnostic.fingerprint(model),'utility_code_sha256':diagnostic.fingerprint(utilities),
        'code_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'working_source_status':subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True).strip(),
        'sources':{n:diagnostic.fingerprint(s) for n,s in sources.items()},
        'owned_pins':{'run_fold0.json':member['run_sha256'],'raptor_ft_fold0_swa.pt':diagnostic.CHECKPOINT_SHA,'oof_fold0.csv':diagnostic.OOF_SHA},
        'private_pins':{n:diagnostic.file_sha(package/n) for n in private},'cache_pins':cache_pins,
        'corpora':['dreaddevelopment/knee-raptor-corpus','dreaddevelopment/knee-raptor-corpus-ext'],'hpo_source':'nartaa/rsna-knee-hpo-assets',
        'settings':{'studies':870,'control_k':16,'candidate_k':42,'res':224,'norm':'imagenet','precision':'cuda_float16','batch_studies':4,'win_chunk':0,
            'tta':False,'roi':False,'seed':42,'optimizer_updates':0,'bootstrap_draws':2000,'bootstrap_seed':42,'parity_atol':diagnostic.PARITY_ATOL},
        'utility_sources':{n:diagnostic.file_sha(ROOT/n) for n in ('knee/validation.py','knee/public_adaptation_fit.py')},
        'limits':['K42 is fixed legacy selection including duplicate centers; masked and boundary windows are reported, not repaired',
            'Initial volume hash observed, not independently pinned; K16 historical parity gates interpretation',
            'Same fold selected epochs/SWA; repeatedly exposed development, not independent generalization','One model used sequentially; no retraining, fitting, source96, ROI or TTA']}
    entry=("import json,sys,hashlib\nfrom pathlib import Path\n"
        f'SOURCES={sources!r}\nPRIVATE={private!r}\nCONTRACT={contract!r}\nMODEL={model!r}\nUTILITIES={utilities!r}\n'
        "base=Path('/kaggle/working/own224_code')\nbase.mkdir(exist_ok=False)\n"
        "def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()\n"
        "for name,source in SOURCES.items():\n    if digest(source)!=CONTRACT['sources'][name]:raise ValueError('Packaged source differs')\n    p=base/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(source)\n"
        "for name,source in PRIVATE.items():\n    p=base/name;p.write_text(source)\n    if hashlib.sha256(p.read_bytes()).hexdigest()!=CONTRACT['private_pins'][name]:raise ValueError('Private membership bytes differ')\n"
        "sys.path.insert(0,str(base))\nfrom knee.own224_window_diagnostic import run\n"
        "run('/kaggle/input','/kaggle/working/own224_window_diagnostic',CONTRACT,MODEL,UTILITIES)\n")
    compile(entry,'diagnostic-notebook','exec')
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python3','language':'python'}},'cells':[
        {'cell_type':'markdown','metadata':{},'source':['# Fixed own224 fold0 evaluation coverage\n','K16 versus fixed legacy K42; same checkpoint, same870, no optimizer updates. Saved K16 parity required before interpretation.\n']},
        {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':entry.splitlines(True)}]}
    blob=(json.dumps(notebook,indent=2)+'\n').encode();diagnostic.require(len(blob)<900000,'Notebook source limit')
    metadata={'id':'ozgung/'+slug,'title':slug.replace('-',' ').title(),'code_file':'notebook.ipynb','language':'python','kernel_type':'notebook','is_private':True,
        'enable_gpu':True,'enable_tpu':False,'enable_internet':False,'competition_sources':['rsna-knee-abnormality-detection'],
        'dataset_sources':[*contract['corpora'],contract['hpo_source']],'kernel_sources':[contract['checkpoint_source']],
        'model_sources':[],'docker_image':DOCKER_IMAGE,'machine_shape':'NvidiaTeslaT4'}
    out.mkdir(parents=True);(out/'notebook.ipynb').write_bytes(blob)
    diagnostic.save(out/'kernel-metadata.json',metadata);diagnostic.save(out/'build-contract-private.json',contract)
    receipt={'build_only':True,'launched':False,'notebook_sha256':diagnostic.file_sha(out/'notebook.ipynb'),'notebook_bytes':len(blob),
        'metadata_sha256':diagnostic.file_sha(out/'kernel-metadata.json'),'contract_sha256':diagnostic.fingerprint(contract)}
    diagnostic.save(out/'build-receipt.json',receipt);return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('destination',type=Path);p.add_argument('slug');p.add_argument('--evidence-root',type=Path,required=True)
    print(json.dumps(build(**vars(p.parse_args())),indent=2))
