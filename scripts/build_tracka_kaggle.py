"""Build the Track A Kaggle package: manifest.json, notebook.ipynb, kernel-metadata.json.

Inputs already in outputs/kaggle/trackA-convnext-20261009/: tracka_entry.py, train_knee_original.py,
train_knee_patched.py, train_knee.patch, folds_v0.json, excluded_ids.json. Builds only; never pushes.
Usage: python scripts/build_tracka_kaggle.py [--dir DIR]
"""
import argparse, hashlib, json, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_SHA256 = "d39966791ef1c48d9317656e8346c60cae1c064192c87b3f350e298ec4c567e8"
KERNEL_ID = "ozgung/knee-tracka-convnext-fold01-20261009"
TITLE = "Knee TrackA ConvNeXt fold01 20261009"
DOCKER_IMAGE = json.load(open(ROOT / "outputs/kaggle/mavit-train2000-diagnostic-20261009/kernel-metadata.json"))["docker_image"]
ASSET_FILES = ["train_knee_patched.py", "train_knee.patch", "excluded_ids.json"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build(d, fold_a=0, fold_b=1, epochs=10, smoke=True, kernel_id=KERNEL_ID, title=TITLE, arch="convnext_tiny.fb_in22k_ft_in1k", res=256, labels_file="labels_v1_blend.parquet", extra_datasets=(),
          k=8, k_eval=16, bs=8, grad_ckpt=False, win_chunk=0, accum=1, bb_lr="3e-5", head_lr="1e-3", wd="0.02", seed=42):
    d = Path(d)
    if sha(d / "train_knee_original.py") != ORIGINAL_SHA256:
        raise SystemExit("train_knee_original.py is not the reviewed original")
    # the stored patch must reproduce the patched script exactly from the original bytes
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t) / "train_knee.py"
        tmp.write_bytes((d / "train_knee_original.py").read_bytes())
        subprocess.run(["patch", "-s", str(tmp), str(d / "train_knee.patch")], check=True)
        if sha(tmp) != sha(d / "train_knee_patched.py"):
            raise SystemExit("train_knee.patch does not reproduce train_knee_patched.py")
    if not json.load(open(d / "excluded_ids.json")).get("quarantined_study_ids"):
        raise SystemExit("excluded_ids.json empty")
    manifest = {"original_sha256": ORIGINAL_SHA256, "patched_sha256": sha(d / "train_knee_patched.py"),
                "patch_sha256": sha(d / "train_knee.patch"), "folds_sha256": sha(d / "folds_v0.json"),
                "excluded_ids_sha256": sha(d / "excluded_ids.json")}
    (d / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    assets = {n: (d / n).read_text() for n in ASSET_FILES}
    assets["manifest.json"] = (d / "manifest.json").read_text()
    entry = (d / "tracka_entry.py").read_text()
    compile(entry, "tracka_entry.py", "exec")

    params = ('# Parameters (edit and re-run). SMOKE=True: 1 epoch, 40 training studies, k_eval 4, whole notebook ~10 min.\n'
              f'FOLD_A = {fold_a}\nFOLD_B = {fold_b!r}        # set to "" to train a single fold\nEPOCHS = {epochs}\nTIME_BUDGET_H = 10.0  # 12 h session minus checkpoint/export margin\n'
              f'SMOKE = {smoke}\nRES = {res}\nARCH = {arch!r}\nLABELS_FILE = {labels_file!r}\n'
              f'K = {k}\nK_EVAL = {k_eval}\nBS = {bs}\nGRAD_CKPT = {bool(grad_ckpt)}\nWIN_CHUNK = {win_chunk}\nACCUM = {accum}\nSEED = {seed}\n'
              f'BB_LR = {str(bb_lr)!r}\nHEAD_LR = {str(head_lr)!r}\nWD = {str(wd)!r}\n')
    runner = (f"import os, sys, runpy, subprocess, pathlib\n"
              f"ASSETS = {assets!r}\nENTRY = {entry!r}\n"
              "pkg = pathlib.Path('/kaggle/working/trackA_pkg'); pkg.mkdir(parents=True, exist_ok=True)\n"
              "for name, text in ASSETS.items(): (pkg / name).write_text(text)\n"
              "(pkg / 'tracka_entry.py').write_text(ENTRY)\n"
              "try:\n    import timm\nexcept ImportError:\n    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'timm'])\n"
              "os.environ.update(FOLD_A=str(FOLD_A), FOLD_B=str(FOLD_B), EPOCHS=str(EPOCHS), TIME_BUDGET_H=str(TIME_BUDGET_H),\n"
              "                  SMOKE='1' if SMOKE else '0', RES=str(RES), ARCH=str(ARCH), LABELS_FILE=str(LABELS_FILE), ASSET_DIR=str(pkg),\n"
              "                  K=str(K), K_EVAL=str(K_EVAL), BS=str(BS), GRAD_CKPT='1' if GRAD_CKPT else '0', WIN_CHUNK=str(WIN_CHUNK),\n"
              "                  ACCUM=str(ACCUM), BB_LR=str(BB_LR), HEAD_LR=str(HEAD_LR), WD=str(WD), SEED=str(SEED))\n"
              "runpy.run_path(str(pkg / 'tracka_entry.py'), run_name='__main__')\n")
    compile(runner, "runner", "exec")
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": "3.12"}},
          "cells": [
              {"cell_type": "markdown", "id": "intro", "metadata": {}, "source": [
                  "# Track A: ConvNeXt-T on the CC0 44-slice corpus, two folds on two T4s\n",
                  "Recipe: nartaa `train_knee.py` (sha256 verified, then one recorded patch `train_knee.patch`: checkpoint selection by full held-out-fold masked AUC vs report labels, gold-58 never used for selection). ",
                  "Corpus: dreaddevelopment/knee-raptor-corpus(+ext). Labels: nartaa labels_v1_blend. Outputs: oof_fold{K}.csv, run_fold{K}.json, checkpoints, logs under /kaggle/working/trackA."]},
              {"cell_type": "code", "id": "params", "metadata": {}, "execution_count": None, "outputs": [], "source": params.splitlines(True)},
              {"cell_type": "code", "id": "run", "metadata": {}, "execution_count": None, "outputs": [], "source": runner.splitlines(True)}]}
    (d / "notebook.ipynb").write_text(json.dumps(nb, indent=1) + "\n")
    meta = {"id": kernel_id, "title": title, "code_file": "notebook.ipynb", "language": "python", "kernel_type": "notebook",
            "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
            "competition_sources": ["rsna-knee-abnormality-detection"],
            "dataset_sources": ["dreaddevelopment/knee-raptor-corpus", "dreaddevelopment/knee-raptor-corpus-ext",
                                "nartaa/rsna-knee-hpo-assets"] + list(extra_datasets),
            "kernel_sources": [], "model_sources": [], "docker_image": DOCKER_IMAGE}
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "outputs/kaggle/trackA-convnext-20261009"))
    ap.add_argument("--fold-a", type=int, default=0)
    ap.add_argument("--fold-b", default="1", help='second fold index, or "" for a single fold')
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--no-smoke", action="store_true", help="build the full run (SMOKE = False)")
    ap.add_argument("--kernel-id", default=KERNEL_ID)
    ap.add_argument("--title", default=TITLE)
    ap.add_argument("--arch", default="convnext_tiny.fb_in22k_ft_in1k")
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--labels-file", default="labels_v1_blend.parquet")
    ap.add_argument("--extra-dataset", action="append", default=[])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=8, help="windows per study (training)")
    ap.add_argument("--k-eval", type=int, default=16, help="windows per study (evaluation)")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad-ckpt", action="store_true", help="gradient checkpointing in the backbone")
    ap.add_argument("--win-chunk", type=int, default=0, help="checkpointed backbone chunks of N windows; 0 = off")
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--bb-lr", default="3e-5")
    ap.add_argument("--head-lr", default="1e-3")
    ap.add_argument("--wd", default="0.02")
    a = ap.parse_args()
    fb = int(a.fold_b) if str(a.fold_b).strip() else ""
    print(json.dumps(build(a.dir, a.fold_a, fb, a.epochs, not a.no_smoke, a.kernel_id, a.title, a.arch, a.res, a.labels_file, a.extra_dataset,
                           a.k, a.k_eval, a.bs, a.grad_ckpt, a.win_chunk, a.accum, a.bb_lr, a.head_lr, a.wd, a.seed), indent=1))
