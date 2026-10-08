"""Export selected TRAIN04 EMA tensors and build private offline inference; no launch."""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from knee.coverage_inference import (CODECS, MODEL_RECIPE, PLATFORM_SECONDS, POLICY, TRAIN_COUNT,
                                    require, require_rehearsal, validate_manifest)
from knee.coverage_cache import VERSION
from knee.fresh_reader import (SOURCE_HASHES, check_sources, fingerprint, load_source, validate_plan,
                               weak_validation_evidence)
from knee.fresh_reader_setup import reference_parts
from knee.reference_blend import CONVNEXT_DATASET, DOCKER_IMAGE
from knee.validation import file_sha256

BUNDLE = ('knee/__init__.py', 'knee/validation.py', 'knee/fresh_reader.py', 'knee/fresh_reader_setup.py',
          'knee/coverage_cache.py', 'knee/coverage_inference.py', 'knee/reference_blend.py',
          'knee/reference.py', 'knee/raptor_input.py', 'knee/public_reader_contract.py')


# The original TRAIN04 objective only. Future objective versions need review.
TRAINING_RUNNER_SHA256 = '81594a1d9b4030cdd4dda1e6d18e36c3642d7b482eaf669f5d15d3b5a9bb4a72'
INITIAL_ENCODER_SHA256 = '355cb8e236d6793f7ec440278ad14a59a5484663a99f9ed54d8b6ea105b8bf66'
TRAINING_BUNDLE = ('knee/__init__.py', 'knee/validation.py', 'knee/features.py', 'knee/experiment.py',
                  'knee/fresh_reader.py', 'knee/fresh_reader_setup.py', 'knee/coverage_cache.py',
                  'knee/coverage_training.py')
TRAINING_RUNTIME = ('knee/coverage_training.py', 'knee/coverage_cache.py', 'knee/fresh_reader.py',
                    'knee/fresh_reader_setup.py', 'knee/validation.py')


def assignment(tree, name):
    matches = [n for n in tree.body if isinstance(n, ast.Assign) and len(n.targets) == 1 and
               isinstance(n.targets[0], ast.Name) and n.targets[0].id == name]
    require(len(matches) == 1, 'Archived payload assignment missing or duplicated')
    return matches[0].value


def config_literal(node):
    """Only literals, dict keyword syntax and the original numeric products/powers."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'dict':
        require(not node.args and all(k.arg is not None for k in node.keywords) and
                len({k.arg for k in node.keywords}) == len(node.keywords), 'Unsafe archived config')
        return {k.arg: config_literal(k.value) for k in node.keywords}
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Pow)):
        left, right = config_literal(node.left), config_literal(node.right)
        require(type(left) is int and type(right) is int and 0 <= left <= 2**30 and
                0 <= right <= (3 if isinstance(node.op, ast.Pow) else 2**30), 'Unsafe archived arithmetic')
        value = left ** right if isinstance(node.op, ast.Pow) else left * right
        require(value <= 2**30, 'Archived config numeric bound exceeded')
        return value
    return ast.literal_eval(node)


def read_training_build(path):
    """Inspect saved notebook literals only; never import or execute archived code."""
    path = Path(path)
    build = json.loads((path / 'build-contract.json').read_text())
    notebook = json.loads((path / 'notebook.ipynb').read_text())
    entry = ast.parse('\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code'))
    code, external, embedded = [ast.literal_eval(assignment(entry, n)) for n in ('CODE', 'EXTERNAL', 'CONTRACT')]
    require(embedded == build, 'Archived entry/build contract differs')
    require(set(code) == set(TRAINING_BUNDLE) and
            {n: hashlib.sha256(v.encode()).hexdigest() for n, v in code.items()} == build['sources'],
            'Archived training source payload changed')
    require(set(external) == set(SOURCE_HASHES) and build['external'] == SOURCE_HASHES and
            {n: hashlib.sha256(v.encode()).hexdigest() for n, v in external.items()} == SOURCE_HASHES,
            'Archived public source payload changed')
    require(build['sources']['knee/coverage_training.py'] == TRAINING_RUNNER_SHA256,
            'Unreviewed training objective; original TRAIN04 runner required')
    runner = ast.parse(code['knee/coverage_training.py'])
    require(config_literal(assignment(runner, 'CONFIG')) == build['config'] and
            ast.literal_eval(assignment(runner, 'INITIAL_ENCODER_SHA256')) == INITIAL_ENCODER_SHA256,
            'Archived training config/initialization differs')
    require(build['sources']['knee/coverage_cache.py'] == file_sha256(ROOT / 'knee/coverage_cache.py'),
            'Inference cache preprocessing changed; review required')
    source_sha = hashlib.sha256(json.dumps(build, sort_keys=True).encode()).hexdigest()
    # The original builder writes these literal environment assignments into entry.
    environment = {}
    for node in entry.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Attribute) and
                isinstance(target.value.value, ast.Name) and target.value.value.id == 'os' and
                target.value.attr == 'environ' and isinstance(target.slice, ast.Constant) and
                target.slice.value in ('KNEE_CODE_REVISION', 'KNEE_SOURCE_SHA256')):
            require(target.slice.value not in environment, 'Duplicate archived environment assignment')
            environment[target.slice.value] = ast.literal_eval(node.value)
    require(environment == {'KNEE_CODE_REVISION': build['revision'], 'KNEE_SOURCE_SHA256': source_sha},
            'Archived entry source identity differs')
    return build, {'training_build_sha256': file_sha256(path / 'build-contract.json'),
                   'training_notebook_sha256': file_sha256(path / 'notebook.ipynb'),
                   'training_source_sha256': source_sha, 'training_revision': build['revision']}


def export_weights(sources, training_result, cache_receipts, destination, *, training_build):
    """Private evidence verifies selection locally; only model tensors are exported."""
    import torch
    from knee.paired_validation import metrics
    check_sources(sources)
    result, cache, out = map(Path, (training_result, cache_receipts, destination))
    require(not out.exists(), 'Weights destination must be new')
    archived, archive_evidence = read_training_build(training_build)
    report = json.loads((result / 'training-report.json').read_text())
    run = json.loads((result / 'run-contract.json').read_text())
    private = json.loads((result / 'best-weak-validation-private.json').read_text())
    plan = json.loads((cache / 'plan-private.json').read_text()); validate_plan(plan)
    require(Counter(r['split'] for r in plan['records']) == {'train': 2000, 'weak_validation': 200} and
            len({r['uid'] for r in plan['records']}) == 2200, 'Exact TRAIN04 cohort required')
    contract = run['contract']; validate_plan(contract)
    require(report['status'] == 'PASS' and report['completed_epochs'] == 12 and
            [row['epoch'] for row in report['history']] == list(range(1,13)) and
            report['contract'] == private['contract'] == contract, 'Completed matching TRAIN04 run required')
    require(contract['version'] == 'train04-v1' and contract['variant'] in ('v1','consensus') and
            contract['config'] == archived['config'] and contract['source_hashes'] == SOURCE_HASHES and
            contract['runtime_sources'] == {n: archived['sources'][n] for n in TRAINING_RUNTIME} and
            contract['runner_sha256'] == TRAINING_RUNNER_SHA256, 'Training recipe/source changed')
    require(run.get('environment', {}).get('source_sha256') == archive_evidence['training_source_sha256'] and
            run['environment'].get('code_revision') == archived['revision'], 'Run/archive source identity differs')
    require(archived['cache_pins'] == {n: file_sha256(cache / n) for n in
            ('plan-private.json', 'cache-report.json', 'index-private.json')}, 'Archived cache attachment differs')
    require(plan['version'] == VERSION and contract['plan_fingerprint'] == plan['fingerprint'] and
            contract['index_sha256'] == file_sha256(cache / 'index-private.json') and
            contract['complete_sha256'] == file_sha256(cache / 'cache-report.json'), 'Selected model/cache provenance differs')
    complete = json.loads((cache / 'cache-report.json').read_text())
    require(complete['status'] == 'PASS' and complete['stage'] == 'complete' and complete['cached_studies'] == 2200 and
            complete['plan_fingerprint'] == plan['fingerprint'], 'Completed2200 cache receipt required')
    checkpoint_sha = file_sha256(result / 'best.pt')
    require(private['best_checkpoint_sha256'] == checkpoint_sha, 'Private selected checkpoint hash differs')
    checkpoint = torch.load(result / 'best.pt', map_location='cpu', weights_only=True)
    selected = min(report['history'], key=lambda row: row['weak_bce'])
    require(checkpoint['contract'] == contract and checkpoint['epoch'] == selected['epoch'] and
            checkpoint['weak_bce'] == selected['weak_bce'] == report['selected_bce'] and
            checkpoint['initial_encoder_sha256'] == run['initial_encoder_sha256'] == report['initial_encoder_sha256'] == INITIAL_ENCODER_SHA256,
            'Selected EMA epoch/BCE/initialization differs')
    indices = [i for i,r in enumerate(plan['records']) if r['split'] == 'weak_validation']
    evidence = private['evidence']
    expected = weak_validation_evidence(plan, indices, evidence['probabilities'], logits=evidence['logits'],
                                        epoch=selected['epoch'], weak_bce=selected['weak_bce'])
    require(len(indices) == 200 and checkpoint['targets'] == plan['targets'] and
            checkpoint['weak_validation_evidence'] == evidence == expected, 'Selected private evidence does not match exact plan')
    score = metrics([plan['records'][i]['labels'] for i in indices], evidence['probabilities'],
                    plan['targets'], logits=evidence['logits'])
    require(abs(score['macro_confidence_weighted_bce'] - selected['weak_bce']) <= 2e-6,
            'Selected logits do not reproduce recorded BCE')
    code = load_source(sources, 'knee')
    require(code.LABELS == plan['targets'], 'Public model/official target order differs')
    state = checkpoint['model']
    require(state and all(torch.is_tensor(v) and torch.isfinite(v).all() for v in state.values()), 'Invalid selected EMA tensors')
    # Strict shape/key validation does not download or initialize pretrained weights.
    model = code.KneeNet(MODEL_RECIPE['backbone'], pretrained=False, img_size=256)
    model.load_state_dict(state, strict=True)
    out.mkdir(parents=True, exist_ok=False)
    torch.save({'model': {k:v.detach().cpu().contiguous() for k,v in state.items()}}, out / 'weights.pt')
    manifest = {'version': 1, 'recipe': MODEL_RECIPE, 'cache_version': VERSION, 'external': SOURCE_HASHES,
        'targets': plan['targets'], 'variant': contract['variant'], 'selected_epoch': selected['epoch'],
        'weights_sha256': file_sha256(out / 'weights.pt'), 'training_checkpoint_sha256': checkpoint_sha,
        'training_contract_fingerprint': contract['fingerprint'], 'plan_fingerprint': plan['fingerprint'],
        'index_sha256': contract['index_sha256'], 'run_sha256': file_sha256(result / 'run-contract.json'),
        'report_sha256': file_sha256(result / 'training-report.json'),
        'initial_encoder_sha256': INITIAL_ENCODER_SHA256, 'training_environment': contract['runtime_environment'],
        **archive_evidence}
    manifest['fingerprint'] = fingerprint(manifest); validate_manifest(manifest)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def build(sources, assets, destination, slug, *, weights_dataset, cache_kernel=None, rehearsal_report=None):
    check_sources(sources); reference_parts(weights_dataset, owner='ozgung'); reference_parts('ozgung/' + slug)
    assets, out = Path(assets), Path(destination)
    require(set(p.name for p in assets.iterdir()) == {'weights.pt', 'manifest.json'}, 'Only weights and manifest may be packaged')
    manifest = json.loads((assets / 'manifest.json').read_text()); validate_manifest(manifest)
    require(file_sha256(assets / 'weights.pt') == manifest['weights_sha256'], 'Own exported weights changed')
    rehearsal = rehearsal_report is None
    require(bool(cache_kernel) == rehearsal, 'Only rehearsal attaches the exact cache')
    if cache_kernel: reference_parts(cache_kernel, owner='ozgung')
    own = {n: (ROOT / n).read_text() for n in BUNDLE}
    external = {n: (Path(sources) / n).read_text() for n in SOURCE_HASHES}
    own_hashes = {n: hashlib.sha256(v.encode()).hexdigest() for n,v in own.items()}
    qualifier = {'manifest_fingerprint': manifest['fingerprint'], 'own_sources': own_hashes,
        'builder_sha256': file_sha256(__file__), 'external': SOURCE_HASHES, 'policy': POLICY,
        'codecs': CODECS, 'python': '3.12', 'torch': '2.10.0+cu128', 'timm': '1.0.26'}
    qualification = fingerprint(qualifier)
    report_sha = None
    if not rehearsal:
        report = json.loads(Path(rehearsal_report).read_text()); require_rehearsal(report, qualification)
        report_sha = file_sha256(rehearsal_report)
    revision = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    contract = {'revision': revision, 'manifest': manifest, 'qualification': qualifier,
        'qualification_fingerprint': qualification, 'rehearsal': rehearsal, 'weights_dataset': weights_dataset,
        'cache_kernel': cache_kernel, 'rehearsal_report_sha256': report_sha, 'platform_seconds': PLATFORM_SECONDS}
    license_text = (ROOT / 'docs/licenses/fresh_reader_Apache-2.0.txt').read_text()
    entry = f'''import os,sys,time,json,subprocess,shutil
from pathlib import Path
STARTED=time.monotonic()
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
CODE={own!r}
EXTERNAL={external!r}
CONTRACT={contract!r}
base=Path('/kaggle/working/coverage_inference_code')
for name,text in CODE.items():
    p=base/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
sources=base/'external';sources.mkdir()
for name,text in EXTERNAL.items():(sources/name).write_text(text)
(sources/'LICENSE').write_text({license_text!r})
sys.path.insert(0,str(base))
from knee.coverage_inference import run,CODECS
from knee.fresh_reader import fingerprint,write_json
from knee.fresh_reader_setup import exact_root,dataset_root,notebook_root,time_limit
from knee.reference_blend import select_codec_wheels
from knee.validation import file_sha256
import importlib.metadata
try:
    with time_limit(CONTRACT['platform_seconds']-(time.monotonic()-STARTED),'Competition9h runtime'):
        if fingerprint(CONTRACT['qualification'])!=CONTRACT['qualification_fingerprint']:raise ValueError('Qualification contract changed')
        if sys.version_info[:2]!=(3,12):raise RuntimeError('Pinned Python3.12 environment required')
        for package in ('torch','timm'):
            if importlib.metadata.version(package)!=CONTRACT['qualification'][package]:raise RuntimeError('Model environment changed; rehearse again')
        inputs=Path('/kaggle/input')
        root=exact_root([inputs/'competitions/rsna-knee-abnormality-detection',inputs/'rsna-knee-abnormality-detection'],['test.csv','test_series.csv','sample_submission.csv'])
        weights=dataset_root(inputs,CONTRACT['weights_dataset'],['weights.pt','manifest.json'])
        codecs=dataset_root(inputs,{CONVNEXT_DATASET!r},[next(iter(CODECS))])
        wheels=select_codec_wheels(codecs)
        if {{p.name:file_sha256(p) for p in wheels}}!=CODECS:raise ValueError('Offline codec hashes differ')
        subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps',*map(str,wheels)],check=True)
        cache=notebook_root(inputs,CONTRACT['cache_kernel'],'coverage_cache',['index-private.json']) if CONTRACT['rehearsal'] else None
        out=Path('/kaggle/working/coverage_inference')
        report=run(root,weights,sources,CONTRACT,out,cache=cache)
        report['inference_seconds']=report['elapsed_seconds'];report['elapsed_seconds']=time.monotonic()-STARTED
        write_json(out/'rehearsal.json',report)
        if not CONTRACT['rehearsal']:
            target=Path('/kaggle/working/submission.csv')
            if target.exists():raise FileExistsError('Submission already exists')
            shutil.copyfile(out/'submission.csv',target)
        print(json.dumps({{'status':report['status'],'test_studies':report['test_studies'],'elapsed_seconds':report['elapsed_seconds']}}),flush=True)
except BaseException as exc:
    write_json('/kaggle/working/coverage-inference-failure.json',{{'failure_type':type(exc).__name__,'elapsed_seconds':time.monotonic()-STARTED}})
    raise RuntimeError('Image-only TRAIN04 inference failed; see aggregate receipts') from None
'''
    for name,value in {**own,**external,'entry':entry}.items(): compile(value,name,'exec')
    nb = {'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}},'cells':[
        {'cell_type':'markdown','id':'attribution','metadata':{},'source':['# Own TRAIN04 image-only EMA reader\nPublic model/preprocessing code by goodpjw2008, Apache-2.0: https://www.kaggle.com/code/goodpjw2008/rsna-knee-stack-2-5d-convnext-mil-lb-0-944\nOwn selected EMA weights; no public knee checkpoint, report input, fitting, or gold evaluation. Healthy cache math is unchanged. Corrupt-input masking is separately qualified; whole-empty/resource failures remain fatal.\n']},
        {'cell_type':'code','id':'run','metadata':{},'execution_count':None,'outputs':[],'source':entry.splitlines(True)}]}
    metadata = {'id':'ozgung/'+slug,'title':slug.replace('-',' ').title(),'code_file':'notebook.ipynb',
        'language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_tpu':False,
        'enable_internet':False,'dataset_sources':[weights_dataset,CONVNEXT_DATASET],
        'competition_sources':['rsna-knee-abnormality-detection'],'kernel_sources':[cache_kernel] if rehearsal else [],
        'model_sources':[],'docker_image':DOCKER_IMAGE,'machine_shape':'NvidiaTeslaT4'}
    text=json.dumps(nb,indent=2)+'\n'
    require(len(text.encode())<900000 and len(entry.encode())<900000, 'Notebook exceeds reviewed source-size allowance')
    out.mkdir(parents=True,exist_ok=False)
    for name,value in [('notebook.ipynb',nb),('kernel-metadata.json',metadata),('build-contract.json',contract)]:
        (out/name).write_text(json.dumps(value,indent=2)+'\n')
    return contract


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__); stages=parser.add_subparsers(dest='stage',required=True)
    export=stages.add_parser('export')
    for name in ('sources','training_result','cache_receipts','destination'):export.add_argument(name)
    export.add_argument('--training-build', required=True, help='Original saved training notebook/build contract directory')
    package=stages.add_parser('build')
    for name in ('sources','assets','destination','slug'):package.add_argument(name)
    package.add_argument('--weights-dataset',required=True);package.add_argument('--cache-kernel')
    package.add_argument('--rehearsal-report')
    args=vars(parser.parse_args()); stage=args.pop('stage')
    (export_weights if stage=='export' else build)(**args)
