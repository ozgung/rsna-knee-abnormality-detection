#!/usr/bin/env python3
"""Mac-side payload-provenance check for the Colab resume loop (called by scripts/colab_tracka_folds.sh).

Downloads payload_manifest.json and the resume file of a Kaggle resume dataset (kaggle CLI), then checks:
  sha256(resume file) == manifest.resume_sha256 (and manifest.fold == FOLD);
  torch.load(map_location=cpu, weights_only=False) has epoch, hist, topk and an identity block whose fold == FOLD and whose
  labels / folds / excluded-ids / script sha256 equal the CURRENT package's (never the dataset's own claims).
Prints one JSON verdict line and exits 0 (verdict TRAINED or RESUME) or 1 (verdict STOP: anything unverifiable or mismatching).
  Also required (else STOP): all trainer resume keys, history exactly ep 0..epoch, well-formed non-empty top-k, identity n_folds/tag/
  oof-mask/corpus ids and every --expect KEY=VALUE trainer config entry (arch, res, k, ...).
  TRAINED: epoch == EPOCHS-1.      RESUME: epoch < EPOCHS-1: safe to relaunch with --resume-from-dataset --strict-identity.
Time budgets play no part: only the verified payload decides.
"""
import argparse, hashlib, json, os, subprocess, sys, tempfile, zipfile
from pathlib import Path


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def current_hashes(pkg, labels_file=None, labels_sha=None):
    pkg = Path(pkg)
    out = {"script_sha256": sha256_file(pkg / "train_knee_patched.py"), "fold_file_sha256": sha256_file(pkg / "folds_v0.json"),
           "excluded_ids_sha256": sha256_file(pkg / "excluded_ids.json"),
           "labels_sha256": labels_sha or (sha256_file(labels_file) if labels_file else None)}
    if out["labels_sha256"] is None:
        raise ValueError("no labels file / sha256 given: cannot verify labels identity")
    return out


RESUME_KEYS = ("model", "opt", "sched", "best", "best_state", "best_P", "hist", "topk", "epoch")


def structure_problems(R, epoch):
    """Completeness of a resume dict: every key the trainer needs, history exactly ep 0..epoch, non-empty well-formed top-k."""
    bad = [f"resume state missing keys {[k for k in RESUME_KEYS if k not in R]}"] if any(k not in R for k in RESUME_KEYS) else []
    for k in ("model", "opt", "sched", "best_state"):
        if k in R and not R[k]:
            bad.append(f"resume state {k} is empty")
    hist, topk = R.get("hist"), R.get("topk")
    if not isinstance(hist, list) or [h.get("ep") if isinstance(h, dict) else None for h in hist] != list(range(epoch + 1)):
        bad.append(f"history is not exactly epochs 0..{epoch} (a resume cannot repair earlier history)")
    if not isinstance(topk, list) or not topk or not all(isinstance(t, dict) and t.get("model") and "sel" in t and "epoch" in t for t in topk):
        bad.append("top-k states missing, empty or malformed")
    elif len({int(t["epoch"]) for t in topk}) != len(topk) or any(not 0 <= int(t["epoch"]) <= epoch for t in topk):
        bad.append("top-k epochs duplicated or outside 0..epoch")
    return bad


def config_problems(ident, fold, expect, n_folds):
    """Identity fields beyond the file hashes: n_folds, tag, oof mask / corpus ids present, and the expected trainer config."""
    bad = []
    if n_folds is not None and str(ident.get("n_folds")) != str(n_folds):
        bad.append(f"identity.n_folds {ident.get('n_folds')!r} != {n_folds}")
    if ident.get("tag") != f"fold{fold}":
        bad.append(f"identity.tag {ident.get('tag')!r} != 'fold{fold}'")
    if not ident.get("oof_mask_sha256"):
        bad.append("identity.oof_mask_sha256 missing")
    if not ident.get("corpus_ids_sha256") or not all((ident.get("corpus_ids_sha256") or {}).values()):
        bad.append("identity.corpus_ids_sha256 missing")
    cfg = ident.get("config")
    if not isinstance(cfg, dict):
        return bad + ["identity.config missing"]
    for k, v in (expect or {}).items():
        if str(cfg.get(k)) != str(v):
            bad.append(f"identity.config.{k} {cfg.get(k)!r} != expected {v!r}")
    return bad


def check_payload(manifest, resume_path, fold, epochs, current, torch_mod=None, expect=None, n_folds=None):
    """Pure check of a downloaded payload. Returns (verdict, epoch, reasons). verdict in TRAINED / RESUME / STOP."""
    bad = []
    if str(manifest.get("fold")) != str(fold):
        bad.append(f"manifest.fold {manifest.get('fold')!r} != {fold}")
    want = manifest.get("resume_sha256")
    got = sha256_file(resume_path)
    if not want or got != want:
        bad.append(f"resume sha256 {got[:12]} != manifest {str(want)[:12]}")
    if bad:
        return "STOP", None, bad
    torch = torch_mod or __import__("torch")
    try:
        R = torch.load(resume_path, map_location="cpu", weights_only=False)
        epoch, ident = int(R["epoch"]), R.get("identity")
        bad += structure_problems(R, epoch)
        del R
    except Exception as e:
        return "STOP", None, [f"resume file unreadable: {type(e).__name__}: {e}"[:200]]
    if not isinstance(ident, dict):
        return "STOP", epoch, ["no identity block in the resume file"]
    if str(ident.get("fold")) != str(fold):
        bad.append(f"identity.fold {ident.get('fold')!r} != {fold}")
    for k, v in current.items():
        if ident.get(k) != v:
            bad.append(f"identity.{k} differs from the current package")
    if manifest.get("epoch") is not None and int(manifest["epoch"]) != epoch:
        bad.append(f"manifest.epoch {manifest['epoch']} != resume epoch {epoch}")
    bad += config_problems(ident, fold, expect, n_folds)
    if epoch > epochs - 1:
        bad.append(f"epoch {epoch} > EPOCHS-1 ({epochs - 1})")
    if bad:
        return "STOP", epoch, bad
    if epoch == epochs - 1:
        return "TRAINED", epoch, []
    return "RESUME", epoch, [f"epoch {epoch} < {epochs - 1}"]


def dataset_version(kaggle_bin, ds):
    """(status, positive current_version_number or None) via `datasets status --format json`."""
    r = subprocess.run([kaggle_bin, "datasets", "status", ds, "--format", "json(status,current_version_number)"],
                       capture_output=True, text=True)
    for line in reversed(((r.stdout or "") + (r.stderr or "")).strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                j = json.loads(line)
            except ValueError:
                break
            v = j.get("current_version_number")
            return str(j.get("status") or "").lower(), (v if isinstance(v, int) and v > 0 else None)
    return "unknown", None


def fetch(kaggle_bin, ds, name, dest):
    subprocess.run([kaggle_bin, "datasets", "download", ds, "-f", name, "-p", str(dest), "--unzip", "-o", "-q"],
                   capture_output=True, text=True)
    for z in Path(dest).glob("*.zip"):
        zipfile.ZipFile(z).extractall(dest)
        z.unlink()
    hits = [p for p in Path(dest).rglob(name) if p.is_file()]
    return hits[0] if hits else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--fold", required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--pkg-dir", required=True)
    ap.add_argument("--labels-file")
    ap.add_argument("--labels-sha256")
    ap.add_argument("--kaggle", default="kaggle")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--expect", action="append", default=[], metavar="KEY=VALUE",
                    help="trainer config entry the identity block must carry (arch, res, k, k_eval, bs, grad_ckpt, topk, ...)")
    a = ap.parse_args(argv)
    verdict = {"dataset": a.dataset, "fold": a.fold, "epochs": a.epochs, "verdict": "STOP", "epoch": None, "version": None, "reasons": []}
    try:
        expect = dict(x.split("=", 1) for x in a.expect)
        cur = current_hashes(a.pkg_dir, a.labels_file, a.labels_sha256)
        st0, v0 = dataset_version(a.kaggle, a.dataset)
        if v0 is None or not st0.startswith("ready"):
            verdict["reasons"] = [f"dataset status {st0!r} version {v0!r}: a READY dataset with a positive exact version is required"]
            print(json.dumps(verdict)); return 1
        verdict["version"] = v0
        pinned = f"{a.dataset}/{v0}"   # CLI >= 2.2 accepts OWNER/SLUG/N: the download is pinned to this exact version
        with tempfile.TemporaryDirectory() as d:
            mp = fetch(a.kaggle, pinned, "payload_manifest.json", d)
            if mp is None:
                verdict["reasons"] = ["payload_manifest.json could not be downloaded"]
            else:
                manifest = json.load(open(mp))
                rp = fetch(a.kaggle, pinned, manifest.get("resume_file") or f"raptor_resume_fold{a.fold}.pt", d)
                if rp is None:
                    verdict["reasons"] = ["resume file could not be downloaded"]
                else:
                    verdict["verdict"], verdict["epoch"], verdict["reasons"] = check_payload(manifest, rp, a.fold, a.epochs, cur, expect=expect, n_folds=a.n_folds)
                    st1, v1 = dataset_version(a.kaggle, a.dataset)
                    if v1 != v0:
                        verdict["verdict"] = "STOP"
                        verdict["reasons"] = list(verdict["reasons"]) + [f"dataset version changed during verification ({v0} -> {v1}); the pinned version {v0} is no longer the newest"]
    except Exception as e:
        verdict["reasons"] = [f"verifier error: {type(e).__name__}: {e}"[:200]]
    print(json.dumps(verdict))
    return 0 if verdict["verdict"] in ("TRAINED", "RESUME") else 1


if __name__ == "__main__":
    sys.exit(main())
