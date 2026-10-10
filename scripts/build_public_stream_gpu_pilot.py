"""Immutable two-stage streamed CUDA and private artifact recovery qualification."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from knee.reference import audit_source,canonical_bytes,sha256,WEIGHT_DATASET
from knee.reference_blend import CONVNEXT_DATASET,DOCKER_IMAGE
from knee.coverage_inference import CODECS
from knee.public_adaptation_pilot import CACHE_SOURCE,cache_contract,stream_contract,scientific_config,validate_recovery_manifest

LABEL_DATASET='nartaa/rsna-knee-hpo-assets'
SOURCES=('knee/__init__.py','knee/reference.py','knee/reference_blend.py','knee/mirror_input.py',
         'knee/raptor_input.py','knee/public_reader_contract.py','knee/fresh_reader_setup.py',
         'knee/runtime.py','knee/public_adaptation_pilot.py','knee/public_study_stream.py')


def build_pilot(source_path,cache_receipt,cpu_stream_receipt,destination,slug, *,
                recovery_source=None,recovery_manifest=None,expected_manifest_sha256=None):
    dest=Path(destination)
    if dest.exists(): raise ValueError('Fresh immutable build required')
    if not re.fullmatch('[a-z0-9]+(?:-[a-z0-9]+)*',slug): raise ValueError('Invalid slug')
    receipt=json.loads(Path(cache_receipt).read_text()); contract=cache_contract(receipt)
    qualified_stream=stream_contract(json.loads(Path(cpu_stream_receipt).read_text()),contract)
    if (recovery_source is None)!=(recovery_manifest is None): raise ValueError('Both recovery source and reviewed manifest required')
    if recovery_source is not None and not re.fullmatch(r'[a-z0-9_-]+/[a-z0-9]+(?:-[a-z0-9]+)*',recovery_source):
        raise ValueError('Invalid completed private recovery source')
    expected=json.loads(Path(recovery_manifest).read_text()) if recovery_manifest is not None else None
    if expected is not None and sha256(canonical_bytes(expected))!=expected_manifest_sha256:
        raise ValueError('Parent-reviewed manifest hash required')
    public,decoder,manifest=audit_source(Path(source_path).read_text(),profile='mirror')
    if manifest['source_sha256']!=contract['source_sha256'] or sha256(canonical_bytes(manifest))!=contract['manifest_sha256']:
        raise ValueError('Audited public/cache source differs')
    entry=(f'PUBLIC_SOURCE={public!r}\nDECODER_SOURCE={decoder!r}\nMANIFEST={manifest!r}\n'
           f'CONTRACT={contract!r}\nQUALIFIED_STREAM={qualified_stream!r}\nEXPECTED_MANIFEST={expected!r}\nRECOVERY_SOURCE={recovery_source!r}\nEXPECTED_CODECS={CODECS!r}\n' + '''
import importlib.metadata, subprocess
from knee.fresh_reader_setup import dataset_root,notebook_root
from knee.reference_blend import CONVNEXT_DATASET,select_codec_wheels
from knee.reference import WEIGHT_DATASET
from knee.runtime import competition_root
from knee.public_adaptation_pilot import run,CACHE_SOURCE,file_sha
if sys.version_info[:2]!=(3,12) or importlib.metadata.version('torch')!='2.10.0+cu128' or importlib.metadata.version('timm')!='1.0.26':
    raise ValueError('Pinned CUDA environment differs')
codec_root=dataset_root('/kaggle/input',CONVNEXT_DATASET,['pylibjpeg-2.1.0-py3-none-any.whl'])
wheels=select_codec_wheels(codec_root)
if {p.name:file_sha(p) for p in wheels}!=EXPECTED_CODECS: raise ValueError('Codec bytes differ')
subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps',*map(str,wheels)],check=True)
cache=notebook_root('/kaggle/input',CACHE_SOURCE,'public-rich-cache-pilot',['receipt.json','private-index.json'])
weights=dataset_root('/kaggle/input',WEIGHT_DATASET,[MANIFEST['checkpoint_file']])
labels=dataset_root('/kaggle/input','nartaa/rsna-knee-hpo-assets',['labels_v1_blend.parquet','train_labels_v1.parquet'])
recovery=notebook_root('/kaggle/input',RECOVERY_SOURCE,'public-stream-gpu-pilot',['recovery-manifest.json','recovery.pt']) if RECOVERY_SOURCE is not None else None
report=run(PUBLIC_SOURCE,DECODER_SOURCE,MANIFEST,weights/MANIFEST['checkpoint_file'],cache,
           labels/'labels_v1_blend.parquet',labels/'train_labels_v1.parquet',competition_root(),
           '/kaggle/working/public-stream-gpu-pilot',CONTRACT,streamed=True,qualified_stream=QUALIFIED_STREAM,
           recovery_root=recovery,expected_manifest=EXPECTED_MANIFEST,export_recovery=RECOVERY_SOURCE is None)
print(json.dumps({'event':'stream_gpu_pilot_done','status':report['status'],'elapsed_seconds':report['elapsed_seconds']}),flush=True)
''')
    root=Path(__file__).resolve().parents[1]
    sources={name:(root/name).read_text() for name in SOURCES}
    digest=sha256(canonical_bytes(sources))
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    scientific_revision=revision if expected is None else expected.get('binding',{}).get('source_revision')
    binding={'cache':contract,'stream':qualified_stream,'scientific_config':scientific_config(),'source_revision':scientific_revision,'bundle_sha256':digest}
    if expected is not None:
        validate_recovery_manifest(expected,binding)
        if not isinstance(scientific_revision,str) or not re.fullmatch('[0-9a-f]{40}',scientific_revision):
            raise ValueError('Valid pinned scientific commit required')
        subprocess.run(['git','cat-file','-e',scientific_revision+'^{commit}'],cwd=root,check=True,capture_output=True)
        for name,content in sources.items():
            committed=subprocess.check_output(['git','show',scientific_revision+':'+name],cwd=root)
            if committed!=content.encode():raise ValueError('Scientific sources differ from pinned committed bytes')
    code=("import os,sys,json\nfrom pathlib import Path\n"
          "os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')\n"
          "project=Path('/kaggle/working/project')\n"+f'sources={sources!r}\n'+
          "for name,content in sources.items():\n    p=project/name\n    p.parent.mkdir(parents=True,exist_ok=True)\n    p.write_text(content)\n"
          "sys.path.insert(0,str(project))\n"+f"os.environ['KNEE_CODE_REVISION']={revision!r}\nos.environ['KNEE_SCIENTIFIC_REVISION']={scientific_revision!r}\nos.environ['KNEE_SOURCE_SHA256']={digest!r}\n"+entry)
    compile(code,'<adaptation-pilot>','exec')
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python3','language':'python','name':'python3'}},'cells':[
        {'cell_type':'markdown','id':'protocol','metadata':{},'source':[
            '# Streamed CUDA and external recovery qualification\n',
            f"{manifest['attribution']}. [Public source]({manifest['source_url']}).\n",
            'Pinned32 inputs/30 contributors; actual16+14 production study-mean updates,3-tail streamed baseline versus independent exact cached arithmetic. Reordered oracle has fixed global/encoder/head relative-L2 and scale budget1e-5. Stage1 retains one private disposable pre-tail checkpoint and aggregate expected-state manifest; stage2 independently mounts and verifies those bytes then requires exact streamed gradient/state/RNG/sampler/prediction continuation. No fit, outcomes or retained candidate. No full fit before external stage2 PASS.\n']},
        {'cell_type':'code','id':'run','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(True)}]}
    meta={'id':f'ozgung/{slug}','title':slug.replace('-',' ').title(),'code_file':'notebook.ipynb',
          'language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_tpu':False,
          'enable_internet':False,'competition_sources':['rsna-knee-abnormality-detection'],
          'dataset_sources':[WEIGHT_DATASET,LABEL_DATASET,CONVNEXT_DATASET],'kernel_sources':[CACHE_SOURCE]+([recovery_source] if recovery_source is not None else []),
          'model_sources':[],'docker_image':DOCKER_IMAGE}
    dest.mkdir(parents=True)
    (dest/'notebook.ipynb').write_text(json.dumps(nb,indent=2)+'\n')
    if (dest/'notebook.ipynb').stat().st_size>900000: raise ValueError('Notebook source exceeds limit')
    (dest/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    result={'manifest':manifest,'cache_contract':contract,'cache_source':CACHE_SOURCE,
            'qualified_cpu_stream':qualified_stream,'recovery_binding':binding,'stage':2 if expected is not None else 1,
            'recovery_source':recovery_source,'expected_manifest_sha256':expected_manifest_sha256,
            'code_revision':scientific_revision,'packaging_revision':revision,'bundle_sha256':digest,'sources':{p:sha256(v.encode()) for p,v in sources.items()},
            'notebook_sha256':sha256((dest/'notebook.ipynb').read_bytes())}
    (dest/'build-receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_path',type=Path); parser.add_argument('cache_receipt',type=Path)
    parser.add_argument('cpu_stream_receipt',type=Path); parser.add_argument('destination',type=Path); parser.add_argument('slug')
    parser.add_argument('--recovery-source'); parser.add_argument('--recovery-manifest',type=Path); parser.add_argument('--expected-manifest-sha256')
    a=parser.parse_args(); print(json.dumps(build_pilot(a.source_path,a.cache_receipt,a.cpu_stream_receipt,a.destination,a.slug,
        recovery_source=a.recovery_source,recovery_manifest=a.recovery_manifest,expected_manifest_sha256=a.expected_manifest_sha256),indent=2))
