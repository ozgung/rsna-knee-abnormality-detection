"""Build private Kaggle CPU-cache or fresh-ImageNet training pilots; never launch."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import textwrap
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from knee.fresh_reader import (CONFIG, SOURCE_HASHES, DURATION_PROFILE, TRAINING_SOURCE_FILES,
                               check_sources, cache_time_budget, duration_experiment, duration_resume_report)
from knee.fresh_reader_setup import reference_parts

DOCKER = 'gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461'


def build(source_dir, destination, slug, stage, *, cache_kernel=None, plan_fingerprint=None, resume_kernel=None,
          cache_seconds=None, training_profile=None, resume_contract=None):
    source_dir, destination = Path(source_dir), Path(destination)
    check_sources(source_dir)
    if stage not in ('cache-pilot', 'cache', 'train', 'probe'):
        raise ValueError('Unknown stage')
    effective_cache_seconds = cache_time_budget(stage, cache_seconds)
    stage_seconds = (CONFIG['pilot_seconds'] if stage == 'cache-pilot' else
                     effective_cache_seconds if stage == 'cache' else 600 if stage == 'probe' else CONFIG['training_seconds'])
    if stage != 'cache-pilot' and (not cache_kernel or not plan_fingerprint):
        raise ValueError('Expansion/training requires a cache kernel and its reviewed plan fingerprint')
    if resume_kernel and stage != 'train':
        raise ValueError('Resume applies to training only')
    if training_profile not in (None, DURATION_PROFILE) or (training_profile is not None and stage != 'train'):
        raise ValueError('Named duration profile applies only to training')
    experiment = duration_experiment(plan_fingerprint, CONFIG) if training_profile else None
    resume_contract_sha256 = None
    resume_report_sha256 = None
    prior_elapsed_seconds = 0.
    if resume_contract and not (experiment and resume_kernel):
        raise ValueError('Named resume contract requires its own duration resume attachment')
    if experiment and resume_kernel:
        if not resume_contract:
            raise ValueError('Duration resume requires a downloaded matching run-contract; no old checkpoint attachment')
        prior = json.loads(Path(resume_contract).read_text())
        if (prior.get('experiment') != experiment or prior.get('experiment_fingerprint') != experiment['fingerprint'] or
                prior.get('runner_sha256') != experiment['runner_sha256'] or
                prior.get('plan_fingerprint') != plan_fingerprint or prior.get('config') != experiment['config']):
            raise ValueError('Duration resume attachment contract differs')
        resume_contract_sha256 = hashlib.sha256(Path(resume_contract).read_bytes()).hexdigest()
        prior_elapsed_seconds = duration_resume_report(resume_contract, experiment, time.monotonic())
        resume_report_sha256 = hashlib.sha256((Path(resume_contract).parent/'training-report.json').read_bytes()).hexdigest()
        stage_seconds -= prior_elapsed_seconds
    for reference in (cache_kernel, resume_kernel):
        if reference:
            reference_parts(reference, owner='ozgung')
    own = {name: (ROOT / name).read_text() for name in TRAINING_SOURCE_FILES}
    external = {name: (source_dir/name).read_text() for name in SOURCE_HASHES}
    license_text = (ROOT/'docs/licenses/fresh_reader_Apache-2.0.txt').read_text()
    for name, content in {**own, **external}.items():
        compile(content, name, 'exec')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    contract = {'stage': stage, 'revision': revision, 'sources': SOURCE_HASHES,
                'own_sources': {name: hashlib.sha256(content.encode()).hexdigest() for name, content in own.items()},
                'license_sha256': hashlib.sha256(license_text.encode()).hexdigest(),
                'config': CONFIG, 'cache_kernel': cache_kernel, 'plan_fingerprint': plan_fingerprint,
                'experiment': experiment, 'experiment_fingerprint': experiment['fingerprint'] if experiment else None,
                'resume_kernel': resume_kernel, 'setup_seconds': 300,
                'resume_contract_sha256': resume_contract_sha256,
                'resume_report_sha256': resume_report_sha256, 'prior_elapsed_seconds': prior_elapsed_seconds,
                'cache_seconds_override': cache_seconds, 'effective_stage_seconds': stage_seconds}
    source_hash = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    setup = f'''from pathlib import Path
import hashlib, json, os, shutil, subprocess, sys
STAGE = {stage!r}
EXPECTED_PLAN = {plan_fingerprint!r}
RESUME_KERNEL = {resume_kernel!r}
CACHE_KERNEL = {cache_kernel!r}
STAGE_BUDGET = {stage_seconds}
CACHE_SECONDS_OVERRIDE = {cache_seconds!r}
EXPERIMENT = {experiment!r}
RESUME_CONTRACT_SHA256 = {resume_contract_sha256!r}
RESUME_REPORT_SHA256 = {resume_report_sha256!r}
CODE = {own!r}
EXTERNAL = {external!r}
LICENSE_TEXT = {license_text!r}
base = Path('/kaggle/working/fresh_reader_code')
for name, content in CODE.items():
    target = base/name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
sources = base/'external'
sources.mkdir(exist_ok=True)
(sources/'LICENSE').write_text(LICENSE_TEXT)
(sources/'NOTICE').write_text('Architecture/preprocessing: goodpjw2008, Apache-2.0. https://www.kaggle.com/datasets/goodpjw2008/rsna-knee-2-5d-convnext-reader. Public source files are unmodified; fresh_reader is a separate wrapper.')
for name, content in EXTERNAL.items():
    (sources/name).write_text(content)
sys.path.insert(0, str(base))
os.environ['KNEE_CODE_REVISION'] = {revision!r}
os.environ['KNEE_SOURCE_SHA256'] = {source_hash!r}
from knee.fresh_reader import prepare_cache, train, duration_resume_report, remaining_training_budget
from knee.fresh_reader_probe import probe
from knee.fresh_reader_setup import dataset_root, exact_root, cache_root, resume_checkpoint, progress, time_limit
with time_limit(STAGE_BUDGET - (time.monotonic() - _stage_started), 'whole ' + STAGE + ' stage'):
    with time_limit(300, 'attachment and decoder setup'):
        progress('setup_begin', stage=STAGE)
        inputs = Path('/kaggle/input')
        if STAGE not in ('train', 'probe'):
            data = exact_root([inputs/'competitions/rsna-knee-abnormality-detection',
                               inputs/'rsna-knee-abnormality-detection'], ['train.csv', 'train_series.csv'])
            if sys.version_info[:2] not in ((3,11), (3,12)):
                raise RuntimeError('Pinned decoder bundle requires Python 3.11/3.12')
            assets = dataset_root(inputs, 'goodpjw2008/rsna-knee-2-5d-convnext-reader',
                                  ['pylibjpeg-2.1.0-py3-none-any.whl'])
            weak = dataset_root(inputs, 'stevenleehans/rsna-knee-llm-report-labels',
                                ['llm_labels_full.csv']) / 'llm_labels_full.csv'
            progress('attachments_resolved', stage=STAGE)
            progress('decoder_install_begin')
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '--find-links', str(assets),
                            'pylibjpeg', 'pylibjpeg-libjpeg', 'pylibjpeg-openjpeg'], check=True, timeout=120)
            progress('decoder_install_complete')
        cache = Path('/kaggle/working/fresh_reader_cache')
        if STAGE != 'cache-pilot':
            previous = cache_root(inputs, CACHE_KERNEL, EXPECTED_PLAN, complete=STAGE == 'train')
            progress('cache_attachment_verified', stage=STAGE)
            if STAGE == 'cache':
                progress('pilot_cache_copy_begin')
                cache.mkdir(exist_ok=True)
                for name in ('plan-private.json','cache-index-private.json','cache-pilot.json'):
                    shutil.copy2(previous/name, cache/name)
                shutil.copytree(previous/'arrays', cache/'arrays', dirs_exist_ok=True)
                progress('pilot_cache_copy_complete')
            else:
                cache = previous
        resume = resume_checkpoint(inputs, RESUME_KERNEL, EXPECTED_PLAN) if RESUME_KERNEL else None
        if resume and EXPERIMENT:
            if hashlib.sha256((resume.parent/'run-contract.json').read_bytes()).hexdigest() != RESUME_CONTRACT_SHA256:
                raise ValueError('Attached duration resume contract differs from reviewed local contract')
            if hashlib.sha256((resume.parent/'training-report.json').read_bytes()).hexdigest() != RESUME_REPORT_SHA256:
                raise ValueError('Attached duration resume report differs from reviewed local report')
        progress('setup_complete', elapsed_seconds=time.monotonic() - _stage_started)
    progress('stage_execution_begin', stage=STAGE)
    if STAGE == 'probe':
        result = probe(cache, sources, '/kaggle/working/fresh_reader_probe', EXPECTED_PLAN)
    elif STAGE == 'train':
        if EXPERIMENT:
            prior_elapsed = duration_resume_report(resume, EXPERIMENT, _stage_started) if resume else 0.
            with time_limit(remaining_training_budget(prior_elapsed, _stage_started, reserve=0), 'cumulative duration training'):
                result = train(cache, sources, '/kaggle/working/fresh_reader_training', resume=resume,
                               experiment=EXPERIMENT, started_at=_stage_started)
        else:
            result = train(cache, sources, '/kaggle/working/fresh_reader_training', resume=resume)
    else:
        result = prepare_cache(data, weak, sources, cache, pilot=STAGE == 'cache-pilot', cache_seconds=CACHE_SECONDS_OVERRIDE)
    print(json.dumps(result, indent=2))
'''
    setup = ('import time, json\nfrom pathlib import Path\n_stage_started = time.monotonic()\ntry:\n'
             + textwrap.indent(setup, '    ')
             + '\nexcept Exception as exc:\n'
             + '    Path(\'/kaggle/working/fresh-reader-failure.json\').write_text(json.dumps({\n'
             + f'        \'stage\': {stage!r}, \'code_revision\': {revision!r},\n'
             + '        \'error\': f\'{type(exc).__name__}: {exc}\', \'elapsed_seconds\': time.monotonic() - _stage_started}, indent=2))\n'
             + '    raise\n')
    compile(setup, 'fresh-reader-notebook', 'exec')
    notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, 'cells': [
        {'cell_type': 'markdown', 'id': 'contract', 'metadata': {}, 'source': [
            '# Fresh ImageNet ConvNeXt 2.5D pilot\n',
            'Public architecture/preprocessing by goodpjw2008 (Apache-2.0): https://www.kaggle.com/datasets/goodpjw2008/rsna-knee-2-5d-convnext-reader\n',
            'Own split, compact cache, confidence-masked supervision, resource gates and resumable training. No public knee checkpoint loaded. No gold evaluation.\n',
            ('Diagnostic: cached training32 only, ≤64 updates/600 seconds, no saved weights or model selection.\n' if stage == 'probe' else
             'Named same512-duration12-v1: fresh ImageNet/optimizer/EMA/cosine, exact512/200, all12epochs must fit5400s; no4epoch warm start.\n' if experiment else
             'Predeclared: 224px, six slots × four fixed windows, ≤512 training/200 weak validation, ≤4 epochs/90 GPU minutes.\n')]},
        {'cell_type': 'code', 'id': 'pilot', 'metadata': {}, 'execution_count': None, 'outputs': [],
         'source': setup.splitlines(True)}]}
    metadata = {'id': 'ozgung/' + slug, 'title': slug.replace('-', ' ').title(), 'code_file': 'notebook.ipynb',
                'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
                'enable_gpu': stage in ('train', 'probe'), 'enable_tpu': False, 'enable_internet': stage in ('train', 'probe'),
                'dataset_sources': [] if stage in ('train', 'probe') else
                    ['stevenleehans/rsna-knee-llm-report-labels', 'goodpjw2008/rsna-knee-2-5d-convnext-reader'],
                'competition_sources': [] if stage in ('train', 'probe') else ['rsna-knee-abnormality-detection'],
                'kernel_sources': [x for x in (cache_kernel, resume_kernel) if x], 'model_sources': [],
                'docker_image': DOCKER}
    if stage in ('train', 'probe'):
        metadata['machine_shape'] = 'NvidiaTeslaT4'
    destination.mkdir(parents=True, exist_ok=True)
    (destination/'notebook.ipynb').write_text(json.dumps(notebook, indent=2))
    (destination/'kernel-metadata.json').write_text(json.dumps(metadata, indent=2))
    (destination/'build-contract.json').write_text(json.dumps(contract | {'source_sha256': source_hash}, indent=2))
    return contract


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source_dir'); ap.add_argument('destination'); ap.add_argument('slug')
    ap.add_argument('--stage', required=True, choices=('cache-pilot', 'cache', 'train', 'probe'))
    ap.add_argument('--cache-seconds', type=int)
    ap.add_argument('--cache-kernel'); ap.add_argument('--plan-fingerprint'); ap.add_argument('--resume-kernel')
    ap.add_argument('--training-profile', choices=(DURATION_PROFILE,)); ap.add_argument('--resume-contract')
    args = ap.parse_args()
    build(args.source_dir, args.destination, args.slug, args.stage, cache_kernel=args.cache_kernel,
          plan_fingerprint=args.plan_fingerprint, resume_kernel=args.resume_kernel, cache_seconds=args.cache_seconds,
          training_profile=args.training_profile, resume_contract=args.resume_contract)
