"""Build a private CUDA full-model smoke using the pinned public image contract."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from knee.reference import audit_source, canonical_bytes, sha256, WEIGHT_DATASET
from knee.reference_blend import CONVNEXT_DATASET, DOCKER_IMAGE
from knee.coverage_inference import CODECS

SOURCES = ('knee/__init__.py', 'knee/reference.py', 'knee/reference_blend.py',
           'knee/mirror_input.py', 'knee/raptor_input.py', 'knee/mirror_submission.py',
           'knee/public_reader_contract.py', 'knee/fresh_reader_setup.py',
           'knee/runtime.py', 'knee/public_warmstart.py')


def build_smoke(source_path, destination, slug):
    destination = Path(destination)
    if destination.exists():
        raise ValueError('Use a fresh immutable build directory')
    if not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', slug):
        raise ValueError('Invalid private notebook slug')
    public, decoder, manifest = audit_source(Path(source_path).read_text(), profile='mirror')
    entry = (f'PUBLIC_SOURCE={public!r}\nDECODER_SOURCE={decoder!r}\nMANIFEST={manifest!r}\n'
             f'EXPECTED_CODECS={CODECS!r}\n' + '''
import hashlib, importlib.metadata, subprocess, sys
from knee.fresh_reader_setup import dataset_root
from knee.reference_blend import CONVNEXT_DATASET, select_codec_wheels
from knee.runtime import competition_root
from knee.public_warmstart import run
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
if sys.version_info[:2] != (3, 12) or importlib.metadata.version('torch') != '2.10.0+cu128' or importlib.metadata.version('timm') != '1.0.26':
    raise ValueError('Pinned warm-start CUDA environment changed')
codecs = dataset_root('/kaggle/input', CONVNEXT_DATASET, ['pylibjpeg-2.1.0-py3-none-any.whl'])
wheels = select_codec_wheels(codecs)
if {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in wheels} != EXPECTED_CODECS:
    raise ValueError('Pinned codec bytes changed')
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', *map(str, wheels)], check=True)
weights = dataset_root('/kaggle/input', MANIFEST['weight_dataset'], [MANIFEST['checkpoint_file']])
report = run(PUBLIC_SOURCE, DECODER_SOURCE, MANIFEST, weights/MANIFEST['checkpoint_file'],
             competition_root(), '/kaggle/working/public-warmstart-smoke')
print(json.dumps({'event':'warmstart_smoke_done', 'status':report['status'], 'elapsed_seconds':report['elapsed_seconds']}), flush=True)
''')
    compile(entry, '<warmstart-entry>', 'exec')
    root = Path(__file__).resolve().parents[1]
    sources = {name: (root / name).read_text() for name in SOURCES}
    digest = sha256(canonical_bytes(sources))
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    code = ("import os, sys, json\nfrom pathlib import Path\nproject=Path('/kaggle/working/project')\n"
            f'sources={sources!r}\n'
            "for name, content in sources.items():\n    path=project/name\n    path.parent.mkdir(parents=True,exist_ok=True)\n    path.write_text(content)\n"
            "sys.path.insert(0,str(project))\n"
            f"os.environ['KNEE_CODE_REVISION']={revision!r}\n"
            f"os.environ['KNEE_SOURCE_SHA256']={digest!r}\n" + entry)
    compile(code, '<warmstart-notebook>', 'exec')
    description = ['# Full public knee-model warm-start qualification\n',
        f"{manifest['attribution']}. [Public source]({manifest['source_url']}).\n",
        'Three visible image studies, strict full-state loading, native320/K94 frozen parity, and one synthetic-target update on two windows. No validation metric or competition submission; smoke weights are deleted.\n',
        'Public weights include competition-training exposure. This check cannot establish generalization.\n']
    nb = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.12'}}, 'cells': [
        {'cell_type': 'markdown', 'id': 'method', 'metadata': {}, 'source': description},
        {'cell_type': 'code', 'id': 'run', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': code.splitlines(True)}]}
    meta = {'id': f'ozgung/{slug}', 'title': slug.replace('-', ' ').title(),
            'code_file': 'notebook.ipynb', 'language': 'python', 'kernel_type': 'notebook',
            'is_private': True, 'enable_gpu': True, 'enable_tpu': False, 'enable_internet': False,
            'competition_sources': ['rsna-knee-abnormality-detection'],
            'dataset_sources': [WEIGHT_DATASET, CONVNEXT_DATASET], 'model_sources': [],
            'kernel_sources': [], 'docker_image': DOCKER_IMAGE}
    destination.mkdir(parents=True)
    (destination / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    nb_path = destination / 'notebook.ipynb'
    nb_path.write_text(json.dumps(nb, indent=2) + '\n')
    if nb_path.stat().st_size > 900000:
        raise ValueError('Notebook exceeds conservative source limit')
    receipt = {'bundle_sha256': digest, 'manifest': manifest, 'code_revision': revision,
               'sources': {name: sha256(value.encode()) for name, value in sources.items()},
               'runtime_sha256': sha256((Path(__file__).resolve().parents[1] / 'knee/public_warmstart.py').read_bytes()),
               'notebook_sha256': sha256(nb_path.read_bytes()), 'contract_sha256': sha256(canonical_bytes(manifest))}
    (destination / 'build-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_path', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('slug')
    args = parser.parse_args()
    print(json.dumps(build_smoke(args.source_path, args.destination, args.slug), indent=2))
