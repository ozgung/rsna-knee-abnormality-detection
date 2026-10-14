"""Build-only Colab bundle: four arm packages, VM tools and the accepted cache notebooks.

Output DIR/raptor64-colab-bundle.tar.gz extracts under /content to raptor64-ARM-pkg (x4),
raptor64-tools/ (knee + scripts used on the VM) and raptor64-accepted/shardN.ipynb.
The private stage config, Kaggle credential and pinned persister are uploaded separately.
"""
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tarfile
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from knee import raptor64_training as guard

TOOLS = ('knee/__init__.py', 'knee/raptor64_cache.py', 'knee/raptor64_loader.py', 'knee/raptor64_training.py', 'knee/source96_cache.py',
         'knee/source96_loader.py', 'knee/source96_training.py', 'knee/source_span_pilot.py', 'knee/corpus44_parity.py',
         'knee/public_training_smoke.py', 'knee/public_rich_cache.py', 'knee/reference.py', 'knee/fresh_reader_setup.py', 'knee/raptor_input.py',
         'scripts/prepare_raptor64_initializer.py', 'scripts/prepare_source96_initializer.py', 'scripts/colab_raptor64_prepare.py',
         'scripts/colab_source96_prepare.py', 'scripts/colab_source96_inputs.py', 'scripts/colab_raptor64_run.py', 'scripts/colab_raptor64_vm.py',
         'scripts/colab_raptor64_export_terminal.py', 'scripts/vm_raptor64_supervise.py', 'scripts/vm_raptor64_autopilot.py',
         'scripts/vm_raptor64_infer.py')
PERSISTER = 'outputs/analysis/colab-fold3-recovery-20261011/fold4-uploaded-persist.py'
PERSISTER_SHA = '8923bd872500b87f85be78aefd9fc799a6f88942e94503045b5e01ea66c118df'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(destination, *, evidence_root, full_swa_epochs=None, iteration2_teacher_sha256=None, iteration3_teacher_sha256=None):
    out = Path(destination); evidence = Path(evidence_root)
    if out.exists():
        raise ValueError('Fresh bundle directory required')
    spec = importlib.util.spec_from_file_location('build_raptor64_training', ROOT / 'scripts/build_raptor64_training.py')
    builder = importlib.util.module_from_spec(spec); spec.loader.exec_module(builder)
    staging = out / 'content'; staging.mkdir(parents=True)
    packages = {}
    for arm in guard.ARM_ORDER:
        receipt = builder.build(staging / f'raptor64-{arm}-pkg', evidence_root=evidence, arm=arm)
        packages[arm] = {'manifest_sha256': receipt['manifest_sha256'], 'scientific_ast_equal': receipt['scientific_ast_equal']}
    if iteration2_teacher_sha256 is not None:
        for arm, kw in (('in-mirror', {}), ('in-distil', {'teacher_sha256': iteration2_teacher_sha256})):
            receipt = builder.build(staging / f'raptor64-{arm}-pkg', evidence_root=evidence, arm=arm, **kw)
            packages[arm] = {'manifest_sha256': receipt['manifest_sha256'], 'scientific_ast_equal': receipt['scientific_ast_equal'], **kw}
    if iteration3_teacher_sha256 is not None:
        for arm in ('in-distil-w1', 'in-distil-flip'):
            receipt = builder.build(staging / f'raptor64-{arm}-pkg', evidence_root=evidence, arm=arm, teacher_sha256=iteration3_teacher_sha256)
            packages[arm] = {'manifest_sha256': receipt['manifest_sha256'], 'scientific_ast_equal': receipt['scientific_ast_equal'], 'teacher_sha256': iteration3_teacher_sha256}
    if full_swa_epochs is not None:
        receipt = builder.build(staging / 'raptor64-dr-full-pkg', evidence_root=evidence, arm='dr-full', swa_epochs=list(full_swa_epochs))
        packages['dr-full'] = {'manifest_sha256': receipt['manifest_sha256'], 'scientific_ast_equal': receipt['scientific_ast_equal'],
                               'swa_epochs': list(full_swa_epochs)}
    for name in TOOLS:
        target = staging / 'raptor64-tools' / name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT / name, target)
    notebooks = {}
    for i in range(4):
        src = evidence / f'outputs/kaggle/knee-raptor64-cache-shard{i}-20261013/notebook.ipynb'
        if not src.is_file():
            src = evidence / f'outputs/kaggle/raptor64-cache-shard{i}-20261013/notebook.ipynb'
        target = staging / 'raptor64-accepted' / f'shard{i}.ipynb'; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target); notebooks[i] = sha(target)
    persister = evidence / PERSISTER
    if sha(persister) != PERSISTER_SHA:
        raise ValueError('Pinned persister differs')
    shutil.copyfile(persister, out / 'colab_tracka_persist.py')
    tar = out / 'raptor64-colab-bundle.tar.gz'
    with tarfile.open(tar, 'w:gz') as t:
        for path in sorted(staging.rglob('*')):
            if path.is_file():
                t.add(path, arcname=str(path.relative_to(staging)))
    receipt = {'bundle_sha256': sha(tar), 'bundle_bytes': tar.stat().st_size, 'packages': packages,
               'tools_sha256': {name: sha(ROOT / name) for name in TOOLS}, 'accepted_notebooks_sha256': notebooks,
               'persister_sha256': PERSISTER_SHA, 'build_only': True}
    (out / 'bundle-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    shutil.rmtree(staging)
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('destination', type=Path); p.add_argument('--evidence-root', type=Path, required=True)
    p.add_argument('--iteration3-teacher-sha256', help='also build in-distil-w1 and in-distil-flip (teacher pinned)')
    p.add_argument('--iteration2-teacher-sha256', help='also build in-mirror and in-distil (teacher pinned)')
    p.add_argument('--full-swa-epochs', type=int, nargs=3, help='also build the dr-full package with these pre-registered SWA epochs')
    print(json.dumps(build(**vars(p.parse_args())), indent=2))
