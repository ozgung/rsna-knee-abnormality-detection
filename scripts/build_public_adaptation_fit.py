"""Package a private immutable fit only after exact actual two-stage qualification."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from knee.reference import audit_source, canonical_bytes, sha256, WEIGHT_DATASET
from knee.reference_blend import CONVNEXT_DATASET, DOCKER_IMAGE
from knee.coverage_inference import CODECS
from knee.mirror_input import policy_contract
from knee.public_adaptation_pilot import cache_contract, stream_contract, file_sha
from knee.public_adaptation_fit import (qualification_gate, fit_config, fingerprint,
    SCIENTIFIC_REVISION, SCIENTIFIC_BUNDLE, FOLD_SHA, EXCLUDED_SHA)

LABEL_DATASET='nartaa/rsna-knee-hpo-assets'
SCIENTIFIC_SOURCES=('knee/__init__.py','knee/reference.py','knee/reference_blend.py','knee/mirror_input.py',
    'knee/raptor_input.py','knee/public_reader_contract.py','knee/fresh_reader_setup.py',
    'knee/runtime.py','knee/public_adaptation_pilot.py','knee/public_study_stream.py')
SOURCES=(*SCIENTIFIC_SOURCES,'knee/validation.py','knee/public_adaptation_fit.py')


def reviewed_json(path, expected):
    if not re.fullmatch('[0-9a-f]{64}',expected or '') or file_sha(path)!=expected:
        raise ValueError('Exact parent-reviewed receipt bytes required')
    return json.loads(Path(path).read_text())


def build_fit(source_path,cache_receipt,cpu_stream_receipt,stage1_receipt,stage2_receipt,
              recovery_manifest,preflight,folds_path,excluded_path,train_csv,destination,slug, *,
              expected_stage1_sha256,expected_stage2_sha256,expected_manifest_sha256,
              expected_preflight_sha256,resume_source=None,resume_receipt=None,expected_resume_sha256=None,
              resume_terminal_receipt=None,expected_resume_terminal_sha256=None):
    """Preparation is local; this function never pushes, downloads or starts jobs."""
    dest=Path(destination); root=Path(__file__).resolve().parents[1]
    if dest.resolve().is_relative_to(root) and not dest.resolve().is_relative_to(root/'outputs'):
        raise ValueError('Private notebook builds must remain in ignored outputs')
    if dest.exists(): raise ValueError('Fresh immutable build required')
    if not re.fullmatch('[a-z0-9]+(?:-[a-z0-9]+)*',slug): raise ValueError('Invalid slug')
    if (resume_source is None)!=(resume_receipt is None): raise ValueError('Resume source and reviewed receipt required together')
    if resume_source and not re.fullmatch('[a-z0-9_-]+/[a-z0-9]+(?:-[a-z0-9]+)*',resume_source): raise ValueError('Invalid resume source')
    s1=reviewed_json(stage1_receipt,expected_stage1_sha256)
    s2=reviewed_json(stage2_receipt,expected_stage2_sha256)
    recovery=reviewed_json(recovery_manifest,expected_manifest_sha256)
    pre=reviewed_json(preflight,expected_preflight_sha256)
    contract=cache_contract(json.loads(Path(cache_receipt).read_text()))
    stream=stream_contract(json.loads(Path(cpu_stream_receipt).read_text()),contract)
    qualified=qualification_gate(s1,s2,recovery,contract,stream)
    if file_sha(folds_path)!=FOLD_SHA or file_sha(excluded_path)!=EXCLUDED_SHA: raise ValueError('Pinned private membership bytes differ')
    folds=json.loads(Path(folds_path).read_text())['folds']
    excluded=json.loads(Path(excluded_path).read_text())['quarantined_study_ids']
    if fingerprint(folds)!=pre['fingerprints']['fold_mapping']: raise ValueError('Preflight fold mapping differs')
    public,decoder,manifest=audit_source(Path(source_path).read_text(),profile='mirror')
    if manifest['source_sha256']!=contract['source_sha256'] or fingerprint(manifest)!=contract['manifest_sha256']:
        raise ValueError('Audited public source binding differs')
    sources={name:(root/name).read_text() for name in SOURCES}
    scientific={name:sources[name] for name in SCIENTIFIC_SOURCES}
    if fingerprint(scientific)!=SCIENTIFIC_BUNDLE: raise ValueError('Qualified scientific source bundle differs')
    for name,value in scientific.items():
        committed=subprocess.check_output(['git','show',SCIENTIFIC_REVISION+':'+name],cwd=root)
        if committed!=value.encode(): raise ValueError('Qualified scientific source bytes differ')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    digest=fingerprint(sources)
    binding={'config':fit_config(),'qualification':qualified,'preflight':fingerprint(pre),
             'source_bundle_sha256':digest,'public_manifest':fingerprint(manifest),'input_policy':policy_contract(),
             'fold_file_sha256':FOLD_SHA,'excluded_file_sha256':EXCLUDED_SHA,
             'excluded_content_sha256':fingerprint(excluded),'official_train_sha256':file_sha(train_csv),
             'source_revision':revision,'scientific_revision':SCIENTIFIC_REVISION}
    prior=reviewed_json(resume_receipt,expected_resume_sha256) if resume_source else None
    terminal=reviewed_json(resume_terminal_receipt,expected_resume_terminal_sha256) if resume_source else None
    if terminal and (terminal.get('kernel_source')!=resume_source or
                     terminal.get('status') not in ('ERROR','COMPLETE','CANCELLED','CANCELED') or
                     type(terminal.get('version')) is not int or terminal['version']<=0 or
                     terminal.get('checkpoint_sha256')!=prior.get('checkpoint_sha256')):
        raise ValueError('Exact terminal-kernel/version/checkpoint evidence required')
    if prior:
        original_binding=prior.get('binding',{})
        if {k:v for k,v in original_binding.items() if k!='source_revision'}!={k:v for k,v in binding.items() if k!='source_revision'}:
            raise ValueError('Resume scientific/input/source binding differs')
        binding=original_binding  # Packaging-only commits do not change runtime identity.
    if prior and (prior.get('status') not in ('INTERRUPTED','FAIL','RUNNING') or prior.get('binding')!=binding or
                  not re.fullmatch('[0-9a-f]{64}',prior.get('checkpoint_sha256','')) or
                  not isinstance(prior.get('checkpoint_bytes'),int) or prior['checkpoint_bytes']<=0):
        raise ValueError('Exact interrupted boundary receipt required')
    gate={'stage1':s1,'stage2':s2,'manifest':recovery,'contract':contract,'stream':stream}
    entry=(f'PUBLIC_SOURCE={public!r}\nDECODER_SOURCE={decoder!r}\nMANIFEST={manifest!r}\n'
           f'FOLDS={folds!r}\nEXCLUDED={excluded!r}\nPREFLIGHT={pre!r}\nGATE={gate!r}\nBINDING={binding!r}\n'
           f'EXPECTED_CODECS={CODECS!r}\nRESUME_SOURCE={resume_source!r}\nPRIOR={prior!r}\n'+'''
import importlib.metadata, shutil, subprocess
from knee.fresh_reader_setup import dataset_root,notebook_root
from knee.reference_blend import CONVNEXT_DATASET,select_codec_wheels
from knee.reference import WEIGHT_DATASET
from knee.runtime import competition_root
from knee.public_adaptation_pilot import file_sha
from knee.public_adaptation_fit import run
out=Path('/kaggle/working/public-adaptation-fit')
try:
    if sys.version_info[:2]!=(3,12) or importlib.metadata.version('torch')!='2.10.0+cu128' or importlib.metadata.version('timm')!='1.0.26':
        raise ValueError('Pinned CUDA environment differs')
    codec_root=dataset_root('/kaggle/input',CONVNEXT_DATASET,['pylibjpeg-2.1.0-py3-none-any.whl'])
    wheels=select_codec_wheels(codec_root)
    if {p.name:file_sha(p) for p in wheels}!=EXPECTED_CODECS: raise ValueError('Codec bytes differ')
    subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps',*map(str,wheels)],check=True)
    weights=dataset_root('/kaggle/input',WEIGHT_DATASET,[MANIFEST['checkpoint_file']])
    labels=dataset_root('/kaggle/input','nartaa/rsna-knee-hpo-assets',['labels_v1_blend.parquet','train_labels_v1.parquet'])
    resume=None
    if RESUME_SOURCE is not None:
        previous=notebook_root('/kaggle/input',RESUME_SOURCE,'public-adaptation-fit',['receipt.json','resume.pt'])
        if json.loads((previous/'receipt.json').read_text())!=PRIOR or file_sha(previous/'resume.pt')!=PRIOR['checkpoint_sha256'] or (previous/'resume.pt').stat().st_size!=PRIOR['checkpoint_bytes']:
            raise ValueError('Resume artifact bytes differ')
        out.mkdir(parents=True,exist_ok=False)
        shutil.copyfile(previous/'resume.pt',out/'resume.pt'); resume=out/'resume.pt'
    report=run(PUBLIC_SOURCE,DECODER_SOURCE,MANIFEST,weights/MANIFEST['checkpoint_file'],
               labels/'labels_v1_blend.parquet',labels/'train_labels_v1.parquet',competition_root(),out,
               FOLDS,EXCLUDED,PREFLIGHT,GATE,BINDING,resume=resume)
    print(json.dumps({'event':'fixed_fit_done','status':report['status'],'updates':report['updates'],'elapsed_seconds':report['elapsed_seconds']}),flush=True)
except BaseException as error:
    if not (out/'receipt.json').exists():
        out.mkdir(parents=True,exist_ok=True)
        (out/'receipt.json').write_text(json.dumps({'status':'FAIL','failure_stage':'offline_setup','failure_type':type(error).__name__,'binding':BINDING},indent=2)+'\\n')
    raise RuntimeError('Fixed fit failed; inspect sanitized aggregate receipt') from None
''')
    code=("import os,sys,json\nfrom pathlib import Path\n"
          "os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')\n"
          "project=Path('/kaggle/working/project')\n"+f'sources={sources!r}\n'+
          "for name,content in sources.items():\n    p=project/name\n    p.parent.mkdir(parents=True,exist_ok=True)\n    p.write_text(content)\n"
          "sys.path.insert(0,str(project))\n"+f"os.environ['KNEE_CODE_REVISION']={revision!r}\nos.environ['KNEE_SOURCE_SHA256']={digest!r}\n"+entry)
    compile(code,'<fixed-public-adaptation>','exec')
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python3','language':'python','name':'python3'}},'cells':[
        {'cell_type':'markdown','id':'protocol','metadata':{},'source':['# Fixed public-state adaptation\n',
            f"{manifest['attribution']}. [Public source]({manifest['source_url']}).\n",
            'Two epochs/fold0,3455contributors,216updates per epoch,432total,15tail. No intermediate outcomes or checkpoint selection. Exact original binary report truth and K94 control/endpoint evaluation; paired component bootstrap. Private population mappings remain in this ignored private notebook.\n']},
        {'cell_type':'code','id':'run','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(True)}]}
    meta={'id':f'ozgung/{slug}','title':slug.replace('-',' ').title(),'code_file':'notebook.ipynb',
        'language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_tpu':False,
        'enable_internet':False,'competition_sources':['rsna-knee-abnormality-detection'],
        'dataset_sources':[WEIGHT_DATASET,LABEL_DATASET,CONVNEXT_DATASET],
        'kernel_sources':[resume_source] if resume_source else [],'model_sources':[],'docker_image':DOCKER_IMAGE}
    encoded=json.dumps(nb,indent=2)+'\n'
    if len(encoded.encode())>900000: raise ValueError('Notebook exceeds source limit')
    # Runtime images come directly from competition data; no large cache attachment.
    dest.mkdir(parents=True)
    (dest/'notebook.ipynb').write_text(encoded)
    (dest/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    result={'binding':binding,'manifest':manifest,'code_revision':revision,'bundle_sha256':digest,
            'sources':{name:sha256(value.encode()) for name,value in sources.items()},
            'notebook_sha256':sha256(encoded.encode()),'resume_source':resume_source,'resume_terminal':terminal,
            'qualification_status':'ACTUAL_TWO_STAGE_PASS','launched':False}
    (dest/'build-receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source_path','cache_receipt','cpu_stream_receipt','stage1_receipt','stage2_receipt',
                 'recovery_manifest','preflight','folds_path','excluded_path','train_csv','destination'):
        parser.add_argument(name,type=Path)
    parser.add_argument('slug')
    for name in ('stage1','stage2','manifest','preflight'):
        parser.add_argument('--expected-'+name+'-sha256',required=True)
    parser.add_argument('--resume-source'); parser.add_argument('--resume-receipt',type=Path)
    parser.add_argument('--expected-resume-sha256')
    parser.add_argument('--resume-terminal-receipt',type=Path);parser.add_argument('--expected-resume-terminal-sha256')
    args=vars(parser.parse_args()); print(json.dumps(build_fit(**args),indent=2))
