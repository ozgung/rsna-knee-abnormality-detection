"""Build only: private CPU three-arm32 span/depth pilot, explicit external evidence root."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from knee import source_span_pilot as pilot, public_training_smoke as smoke
from knee.coverage_inference import CODECS
from knee.reference_blend import DOCKER_IMAGE, CONVNEXT_DATASET

FILES = ('knee/__init__.py', 'knee/source_span_pilot.py', 'knee/corpus44_parity.py',
         'knee/public_training_smoke.py', 'knee/public_rich_cache.py', 'knee/reference.py',
         'knee/fresh_reader_setup.py', 'knee/raptor_input.py')


def build(destination, slug, *, evidence_root, public_source, training_source, cache_contract):
    out, evidence = Path(destination), Path(evidence_root)
    if out.exists(): raise ValueError('Fresh immutable build directory required')
    if not re.fullmatch('[a-z0-9]+(?:-[a-z0-9]+)*', slug): raise ValueError('Invalid notebook slug')
    def supplied(path):
        path = Path(path)
        return path if path.is_absolute() else evidence/path
    codes = pilot.recipes(supplied(public_source).read_text())
    sampler = pilot.sampler_source(supplied(training_source).read_text())
    identity = evidence/'outputs/kaggle/sealed-identity-full-result/sealed_identity/sealed-manifest-private.json'
    smoke.require_file(identity, smoke.MANIFEST_SHA)
    items = json.loads(identity.read_bytes())['records']
    exclusions = {'quarantined_study_ids':[r['uid'] for r in items if r['future_role']=='quarantined']}
    raw = evidence/'data/raw'
    ids, _, membership = pilot.select_inputs(raw, smoke.TARGETS, exclusions)
    cache_pins = json.loads(supplied(cache_contract).read_bytes())['cache_pins']
    expected = set(smoke.PINS) | {'all_vols.npy', 'extra_vols.npy'}
    if set(cache_pins) != expected: raise ValueError('Corpus pin inventory changed')
    for name, pin in cache_pins.items():
        if not isinstance(pin, dict) or set(pin) != {'sha256','bytes'} or not re.fullmatch('[a-f0-9]{64}',pin['sha256']) or type(pin['bytes']) is not int or pin['bytes']<=0:
            raise ValueError('Invalid corpus pin')
        if name in smoke.PINS and pin['sha256'] != smoke.PINS[name]: raise ValueError('Corpus metadata pin differs')
    sources={name:(ROOT/name).read_text() for name in FILES}
    for name,source in sources.items(): compile(source,name,'exec')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    contract={'version':pilot.VERSION,'config':pilot.CONFIG,'readers':{k:smoke.digest(v.encode()) for k,v in codes.items()},
              'reader_original_sha256':pilot.parity.SOURCE_SHA,'training_original_sha256':smoke.SOURCE_SHA,
              'sampler_sha256':smoke.digest(sampler.encode()),'competition_pins':{n:smoke.file_sha(raw/n) for n in ('train.csv','train_series.csv')},
              'identity_sha256':smoke.MANIFEST_SHA,'membership':membership,'selected_sha256':smoke.digest(smoke.canonical(ids)),
              'cache_pins':cache_pins,'codecs':CODECS,'sources':{k:smoke.digest(v.encode()) for k,v in sources.items()},
              'code_revision':revision,'working_source_status':subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True).strip(),
              'acceptance':'all32 narrow control uint8 slot hashes and masks exactly equal; all3 arms finite K8/K16/reload/repeat',
              'limits':['input mechanics only; no model/metric/training','legacy cross-slot/masked sampler retained', '96wide is a named combined depth/span representation, not an isolated source-span intervention',
                        'selected-slice intensity quantile algorithm fixed; statistics differ with span',
                        'ordinal source coverage only; no physical coordinates exported','actual CPU platform12h limit only']}
    code=("import time,sys,json,hashlib,subprocess\nfrom pathlib import Path\nSTARTED=time.monotonic()\n"
          f'SOURCES={sources!r}\nCONTRACT={contract!r}\nREADERS={codes!r}\nSAMPLER={sampler!r}\n'
          "base=Path('/kaggle/working/span_code')\n"
          "for name,source in SOURCES.items():\n"
          "    if hashlib.sha256(source.encode()).hexdigest()!=CONTRACT['sources'][name]:raise ValueError('Packaged source differs')\n"
          "    p=base/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(source)\n"
          "sys.path.insert(0,str(base))\n"
          "from knee.fresh_reader_setup import dataset_root\nfrom knee.public_training_smoke import require_file\n"
          f"codecs=dataset_root('/kaggle/input',{CONVNEXT_DATASET!r},list(CONTRACT['codecs']))\n"
          "wheels=[]\nfor name,sha in CONTRACT['codecs'].items():\n"
          "    require_file(codecs/name,sha);wheels.append(str(codecs/name))\n"
          "if sys.version_info[:2]!=(3,12):raise ValueError('Pinned Python environment changed')\n"
          "subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps',*wheels],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n"
          "from knee.source_span_pilot import run\n"
          "r=run('/kaggle/input','/kaggle/working/source_span_pilot',CONTRACT,READERS,SAMPLER,started=STARTED)\n"
          "print(json.dumps({'event':'span_pilot_done','status':r['status'],'elapsed_seconds':r['elapsed_seconds']}))\n")
    compile(code,'source-span-notebook','exec')
    # Check every official identity, including excluded members, never just pilot IDs.
    if any(r['uid'] in code for r in items): raise ValueError('Private study identity embedded in notebook')
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python3','language':'python'}},'cells':[
        {'cell_type':'markdown','id':'protocol','metadata':{},'source':['# Three-arm source span/depth CPU pilot\n',
            '32 deterministic nongold/nonquarantine studies; no outcomes or models. Fixed336/140mm, slots12/10/8/6/8;15–85% versus2–98%.\n',
            'Third96wide arm changes only slot allocation26/22/18/12/18 at2–98%. Control must exactly match mounted44 pixels/masks. Offline pinned codecs; aggregate receipt only; MRI temporary files removed.\n']},
        {'cell_type':'code','id':'run','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(True)}]}
    blob=(json.dumps(nb,indent=2)+'\n').encode()
    if len(blob)>900000:raise ValueError('Notebook source limit')
    metadata={'id':'ozgung/'+slug,'title':slug.replace('-',' ').title(),'code_file':'notebook.ipynb','language':'python','kernel_type':'notebook',
              'is_private':True,'enable_gpu':False,'enable_tpu':False,'enable_internet':False,
              'competition_sources':['rsna-knee-abnormality-detection'],'dataset_sources':[*smoke.CORPORA,CONVNEXT_DATASET],
              'kernel_sources':[smoke.IDENTITY_KERNEL],'model_sources':[],'docker_image':DOCKER_IMAGE}
    receipt={'notebook_sha256':smoke.digest(blob),'notebook_bytes':len(blob),'contract_sha256':smoke.digest(smoke.canonical(contract)),
             'bundle_sha256':smoke.digest(smoke.canonical(sources)),'code_revision':revision,'build_only':True}
    out.mkdir(parents=True)
    (out/'notebook.ipynb').write_bytes(blob)
    for name,value in [('kernel-metadata.json',metadata),('build-contract.json',contract),('build-receipt.json',receipt)]:
        (out/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('destination',type=Path);p.add_argument('slug')
    for name in ('evidence-root','public-source','training-source','cache-contract'):p.add_argument('--'+name,type=Path,required=True)
    print(json.dumps(build(**vars(p.parse_args())),indent=2))
