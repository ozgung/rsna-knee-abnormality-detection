"""Build an export-only Track A Kaggle notebook package: score a checkpoint trained elsewhere (e.g. Colab).

The notebook copies one fold checkpoint from an attached checkpoint dataset into WORKDIR, writes a
train_meta_fold{K}.json with cmd ["colab-trained", <checkpoint sha256>], then runs
`python tracka_entry.py export K` (held-out fold + gold-58 predictions, oof_fold{K}.csv, run_fold{K}.json).
Model geometry (arch/res) comes from the checkpoint dict; --arch/--res are recorded for the run record only.
Builds only; never pushes. Usage: python scripts/build_tracka_export_kaggle.py --help
"""
import argparse, json, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from knee.selected_export import validate_expectation, sha256_file as sha
ASSET_FILES = ['train_knee_patched.py', 'train_knee.patch', 'excluded_ids.json']
ORIGINAL_SHA256 = 'd39966791ef1c48d9317656e8346c60cae1c064192c87b3f350e298ec4c567e8'
DOCKER_IMAGE = None  # resolved from the reviewed package metadata at build time

SRC_DIR = ROOT / "outputs/kaggle/trackA-convnext-20261009"
DEFAULT_OUT = ROOT / "outputs/kaggle/trackA-export-coat384-fold0-20261010"
PKG_FILES = ["tracka_entry.py", "train_knee_original.py"] + ASSET_FILES + ["folds_v0.json"]
DATASETS = ["dreaddevelopment/knee-raptor-corpus", "dreaddevelopment/knee-raptor-corpus-ext", "nartaa/rsna-knee-hpo-assets",
            "ozgung/knee-bootstrap-labels-v1-20261009"]

RUNNER = r'''import os, sys, json, shutil, hashlib, runpy, subprocess, pathlib
pkg = pathlib.Path('/kaggle/working/trackA_pkg'); pkg.mkdir(parents=True, exist_ok=True)
for name, text in ASSETS.items(): (pkg / name).write_text(text)
(pkg / 'tracka_entry.py').write_text(ENTRY)
(pkg / 'selected_export_audit.py').write_text(EXPORT_AUDIT_SOURCE)
sys.path.insert(0, str(pkg))
try:
    import timm
except ImportError:
    raise SystemExit('timm is missing and internet is off; use the Track A docker image')
work = pathlib.Path('/kaggle/working/trackA'); work.mkdir(parents=True, exist_ok=True)

def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''): h.update(c)
    return h.hexdigest()

def locate(name):
    # preferred: /kaggle/input/<dataset slug>/<name>; fall back to a walk because Kaggle mount layouts vary
    direct = pathlib.Path('/kaggle/input') / CKPT_DATASET.split('/')[-1] / name
    if direct.exists(): return direct
    for d, _, fs in os.walk('/kaggle/input'):
        if name in fs and CKPT_DATASET.split('/')[-1] in d: return pathlib.Path(d) / name
    raise SystemExit(f'{name} not found under /kaggle/input; attach dataset {CKPT_DATASET}')

ck = work / f'raptor_resume_fold{FOLD}.pt'
src = locate(ck.name)
if sha256(src) != EXPECTED['resume_sha256']:
    raise SystemExit('mounted resume differs from the expected version artifact')
shutil.copy(src, ck)
# A reused working directory must not select a stale sibling checkpoint.
for name in (f'raptor_ft_fold{FOLD}_swa.pt', f'raptor_ft_fold{FOLD}.pt'):
    (work / name).unlink(missing_ok=True)
seconds = 0
try:   # sibling json carries hist / n_train / partial for the run record
    js = locate(f'raptor_ft_fold{FOLD}.json'); shutil.copy(js, work / js.name)
    meta = json.load(open(js)); seconds = meta.get('train_wall_seconds') or meta.get('seconds') or 0
except SystemExit as e:
    print('[export] no sibling json:', e)
ck_sha = sha256(ck)
print('[export] checkpoint', ck, ck_sha, flush=True)
json.dump(dict(cmd=['colab-trained', ck_sha], rc=0, seconds=seconds, t0=0, t1=seconds), open(work / f'train_meta_fold{FOLD}.json', 'w'), indent=1)
os.environ.update(RAPTOR_INPUT='/kaggle/input', WORKDIR=str(work), ASSET_DIR=str(pkg), K_EVAL=str(K_EVAL),
                  LABELS_FILE=str(LABELS_FILE), WORKERS='2', SMOKE='0', ARCH=str(ARCH), RES=str(RES), K=str(K_TRAIN), EPOCHS=str(EPOCHS_TRAINED),
                  PYTHONUNBUFFERED='1')
# prepare() fingerprints the original script, folds and labels exactly as the training notebooks do
import importlib.util
spec = importlib.util.spec_from_file_location('tracka_entry', pkg / 'tracka_entry.py'); te = importlib.util.module_from_spec(spec); spec.loader.exec_module(te)
te.prepare(te.cfg())
import torch
import train_knee_patched as tk
import selected_export_audit as audit
current = {'script_sha256': sha256(pkg / 'train_knee_patched.py'),
           'labels_sha256': sha256(tk._find(LABELS_FILE)),
           'oof_mask_sha256': sha256(tk._find('train_labels_v1.parquet')),
           'fold_file_sha256': sha256(work / 'folds_v0.json'),
           'excluded_ids_sha256': sha256(work / 'excluded_ids.json'),
           'corpus_ids_sha256': {n: sha256(tk._find(n)) for n in ('all_ids.npy', 'extra_ids.npy')}}
R = torch.load(ck, map_location='cpu', weights_only=False)
audit.validate_resume(R, EXPECTED, current, tk.LAB)
del R
# The original selector rebuilds its original FP64 sum; the independent helper
# reloads that saved artifact and checks a stacked FP64 oracle before inference.
st, kind, epochs, selected_path, hist, partial = te.select_checkpoint(work, FOLD, torch)
del st
provenance = audit.check_export(ck, selected_path, EXPECTED, current, tk.LAB)
provenance['audit_source_sha256'] = sha256(pkg / 'selected_export_audit.py')
provenance['selector_source_sha256'] = sha256(pkg / 'tracka_entry.py')
json.dump(provenance, open(work / f'export_provenance_fold{FOLD}.json', 'w'), indent=1)
sys.stdout.flush()
subprocess.run([sys.executable, '-u', str(pkg / 'tracka_entry.py'), 'export', str(FOLD)], check=True)
# The subprocess selects from the same immutable resume; bind its actual output
# hash and epochs to the pre-inference oracle, instead of trusting a placeholder.
rp = work / f'run_fold{FOLD}.json'
record = json.load(open(rp))
if record['checkpoint_sha256'] != provenance['selected_file_sha256'] or record['selected_epochs'] != provenance['selected_epochs']:
    raise SystemExit('inference selected a different checkpoint than the qualified export')
record['export_params'] = record.pop('params')
record['training_config'] = provenance['training_config']
record['training_identity'] = provenance['training_identity']
record['stopped_by_time_budget'] = False
record['train_wall_seconds'] = provenance['training_wall_seconds']
record['approx_train_images_per_second_incl_eval'] = None
record['time_budget_h'] = None
record['time_budget_note'] = 'Completed source training; export has no experiment time-budget gate.'
record['export_provenance'] = provenance
record['caveat'] = 'Full-fold epoch selection and prior development exposure; gold58 is small and not an untouched independent evaluation.'
json.dump(record, open(rp, 'w'), indent=1)
print(open(work / f'run_fold{FOLD}.json').read()[:600])
'''


def build(out, ckpt_dataset, fold, k_eval, labels_file, kernel_id, title, arch, res, k_train=8, epochs=10,
          *, expected_version, expected_resume_sha256, expected_config):
    expected = {'dataset_version': expected_version, 'resume_sha256': expected_resume_sha256,
                'fold': fold, 'config': expected_config}
    validate_expectation(expected)
    for key, value in {'arch': arch, 'res': res, 'k': k_train, 'k_eval': k_eval, 'epochs': epochs}.items():
        if expected_config[key] != value:
            raise ValueError(f'export arguments differ from expected training config: {key}')
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    for n in PKG_FILES:
        shutil.copy(SRC_DIR / n, out / n)
    if sha(out / "train_knee_original.py") != ORIGINAL_SHA256:
        raise SystemExit("train_knee_original.py is not the reviewed original")
    if not json.loads((out / "excluded_ids.json").read_text()).get("quarantined_study_ids"):
        raise SystemExit("excluded_ids.json empty")
    ref = json.loads((SRC_DIR / "manifest.json").read_text())
    manifest = {"original_sha256": ORIGINAL_SHA256, "patched_sha256": sha(out / "train_knee_patched.py"),
                "patch_sha256": sha(out / "train_knee.patch"), "folds_sha256": sha(out / "folds_v0.json"),
                "excluded_ids_sha256": sha(out / "excluded_ids.json")}
    if manifest != ref:
        raise SystemExit("package files differ from the reviewed Track A manifest")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    assets = {n: (out / n).read_text() for n in ASSET_FILES + ["manifest.json"]}
    entry = (out / "tracka_entry.py").read_text()
    compile(entry, "tracka_entry.py", "exec")
    params = ("# Export-only: scores the Colab-trained checkpoint, no training. ARCH/RES are recorded only; the model comes from the checkpoint.\n"
              f"CKPT_DATASET = {ckpt_dataset!r}\nFOLD = {fold}\nK_EVAL = {k_eval}\nLABELS_FILE = {labels_file!r}\n"
              f"ARCH = {arch!r}\nRES = {res}\nK_TRAIN = {k_train}   # training windows/study, record only\nEPOCHS_TRAINED = {epochs}   # record only\n")
    audit_source = (ROOT / 'knee/selected_export.py').read_text()
    (out / 'selected_export_audit.py').write_text(audit_source)
    (out / 'export-expectation.json').write_text(json.dumps(expected, indent=2) + '\n')
    runner = f"ASSETS = {assets!r}\nENTRY = {entry!r}\nEXPORT_AUDIT_SOURCE = {audit_source!r}\nEXPECTED = {expected!r}\n" + RUNNER
    compile(runner, "runner", "exec")
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": "3.12"}},
          "cells": [
              {"cell_type": "markdown", "id": "intro", "metadata": {}, "source": [
                  f"# Track A export: fold {fold} checkpoint trained on Colab\n",
                  f"Checks resume bytes against expected dataset version {expected_version}, training identity and selected tensor oracle, then runs `tracka_entry.py export {fold}` "
                  f"(held-out fold + gold-58, K_EVAL={k_eval}). Internet off; the backbone is built with pretrained=False. "
                  f"Outputs: oof_fold{fold}.csv, run_fold{fold}.json under /kaggle/working/trackA."]},
              {"cell_type": "code", "id": "params", "metadata": {}, "execution_count": None, "outputs": [], "source": params.splitlines(True)},
              {"cell_type": "code", "id": "run", "metadata": {}, "execution_count": None, "outputs": [], "source": runner.splitlines(True)}]}
    (out / "notebook.ipynb").write_text(json.dumps(nb, indent=1) + "\n")
    meta = {"id": kernel_id, "title": title, "code_file": "notebook.ipynb", "language": "python", "kernel_type": "notebook",
            "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
            "competition_sources": ["rsna-knee-abnormality-detection"],
            "dataset_sources": DATASETS + [ckpt_dataset], "kernel_sources": [], "model_sources": [],
            "docker_image": DOCKER_IMAGE or json.loads((SRC_DIR / 'kernel-metadata.json').read_text())['docker_image']}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(DEFAULT_OUT))
    ap.add_argument("--k-train", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--ckpt-dataset", default="ozgung/knee-coat384-fold0-ckpt-20261010")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--k-eval", type=int, default=24)
    ap.add_argument("--labels-file", default="labels_bootstrap_v1.parquet")
    ap.add_argument("--kernel-id", default="ozgung/knee-tracka-export-coat384-fold0-20261010")
    ap.add_argument("--title", default="Knee TrackA export coat384 fold0 20261010")
    ap.add_argument("--arch", default="coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k")
    ap.add_argument("--res", type=int, default=384)
    ap.add_argument('--expected-version', type=int, required=True)
    ap.add_argument('--expected-resume-sha256', required=True)
    ap.add_argument('--expected-config-json', type=Path, required=True,
                    help='Prospective training config JSON, using the trainer identity field names.')
    a = ap.parse_args()
    print(json.dumps(build(a.dir, a.ckpt_dataset, a.fold, a.k_eval, a.labels_file, a.kernel_id, a.title, a.arch, a.res, a.k_train, a.epochs,
                          expected_version=a.expected_version, expected_resume_sha256=a.expected_resume_sha256,
                          expected_config=json.loads(a.expected_config_json.read_text())), indent=1))
