"""Immutable private CPU streaming qualification; exact cache, no models or launch."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from knee.reference import audit_source,canonical_bytes,sha256
from knee.reference_blend import CONVNEXT_DATASET,DOCKER_IMAGE
from knee.coverage_inference import CODECS
from knee.public_adaptation_pilot import CACHE_SOURCE,cache_contract

SOURCES=('knee/__init__.py','knee/reference.py','knee/reference_blend.py','knee/mirror_input.py',
         'knee/raptor_input.py','knee/public_reader_contract.py','knee/fresh_reader_setup.py',
         'knee/runtime.py','knee/public_adaptation_pilot.py','knee/public_rich_cache.py',
         'knee/public_study_stream.py','knee/public_stream_pilot.py')


def build_pilot(source_path,cache_receipt,destination,slug):
    dest=Path(destination)
    if dest.exists(): raise ValueError('Fresh immutable build required')
    if not re.fullmatch('[a-z0-9]+(?:-[a-z0-9]+)*',slug): raise ValueError('Invalid slug')
    receipt=json.loads(Path(cache_receipt).read_text()); contract=cache_contract(receipt)
    public,decoder,manifest=audit_source(Path(source_path).read_text(),profile='mirror')
    if manifest['source_sha256']!=contract['source_sha256'] or sha256(canonical_bytes(manifest))!=contract['manifest_sha256']:
        raise ValueError('Audited public/cache source differs')
    entry=(f'PUBLIC_SOURCE={public!r}\nDECODER_SOURCE={decoder!r}\nMANIFEST={manifest!r}\n'
           f'CONTRACT={contract!r}\nEXPECTED_CODECS={CODECS!r}\n' + '''
import importlib.metadata, subprocess
from knee.fresh_reader_setup import dataset_root,notebook_root
from knee.reference_blend import CONVNEXT_DATASET,select_codec_wheels
from knee.runtime import competition_root
from knee.public_adaptation_pilot import CACHE_SOURCE,file_sha
from knee.public_stream_pilot import run
if sys.version_info[:2]!=(3,12): raise ValueError('Pinned Python environment differs')
codec_root=dataset_root('/kaggle/input',CONVNEXT_DATASET,['pylibjpeg-2.1.0-py3-none-any.whl'])
wheels=select_codec_wheels(codec_root)
if {p.name:file_sha(p) for p in wheels}!=EXPECTED_CODECS: raise ValueError('Codec bytes differ')
subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps',*map(str,wheels)],check=True)
cache=notebook_root('/kaggle/input',CACHE_SOURCE,'public-rich-cache-pilot',['receipt.json','private-index.json'])
report=run(PUBLIC_SOURCE,DECODER_SOURCE,MANIFEST,cache,competition_root(),
           '/kaggle/working/public-stream-pilot',CONTRACT)
print(json.dumps({'event':'stream_pilot_done','status':report['status'],'elapsed_seconds':report['elapsed_seconds']}),flush=True)
''')
    root=Path(__file__).resolve().parents[1]
    sources={name:(root/name).read_text() for name in SOURCES}
    digest=sha256(canonical_bytes(sources))
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    code=("import os,sys,json\nfrom pathlib import Path\n"
          "os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')\n"
          "project=Path('/kaggle/working/project')\n"+f'sources={sources!r}\n'+
          "for name,content in sources.items():\n    p=project/name\n    p.parent.mkdir(parents=True,exist_ok=True)\n    p.write_text(content)\n"
          "sys.path.insert(0,str(project))\n"+f"os.environ['KNEE_CODE_REVISION']={revision!r}\nos.environ['KNEE_SOURCE_SHA256']={digest!r}\n"+entry)
    compile(code,'<stream-pilot>','exec')
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python3','language':'python','name':'python3'}},'cells':[
        {'cell_type':'markdown','id':'protocol','metadata':{},'source':[
            '# Exact public CPU streaming qualification\n',
            f"{manifest['attribution']}. [Public source]({manifest['source_url']}).\n",
            'Exact pinned32-study bytes/masks and sampled12 windows; crop after quantization. Ordered bounded prefetch, acknowledged-cursor replay, forced cache hit/miss/eviction and producer failure. CPU input mechanics only; no model, labels, GPU, training or outcomes.\n']},
        {'cell_type':'code','id':'run','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(True)}]}
    meta={'id':f'ozgung/{slug}','title':slug.replace('-',' ').title(),'code_file':'notebook.ipynb',
          'language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':False,'enable_tpu':False,
          'enable_internet':False,'competition_sources':['rsna-knee-abnormality-detection'],
          'dataset_sources':[CONVNEXT_DATASET],'kernel_sources':[CACHE_SOURCE],
          'model_sources':[],'docker_image':DOCKER_IMAGE}
    dest.mkdir(parents=True)
    (dest/'notebook.ipynb').write_text(json.dumps(nb,indent=2)+'\n')
    if (dest/'notebook.ipynb').stat().st_size>900000: raise ValueError('Notebook source exceeds limit')
    (dest/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    result={'manifest':manifest,'cache_contract':contract,'cache_source':CACHE_SOURCE,
            'code_revision':revision,'bundle_sha256':digest,'sources':{p:sha256(v.encode()) for p,v in sources.items()},
            'notebook_sha256':sha256((dest/'notebook.ipynb').read_bytes())}
    (dest/'build-receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_path',type=Path); parser.add_argument('cache_receipt',type=Path)
    parser.add_argument('destination',type=Path); parser.add_argument('slug')
    a=parser.parse_args(); print(json.dumps(build_pilot(a.source_path,a.cache_receipt,a.destination,a.slug),indent=2))
