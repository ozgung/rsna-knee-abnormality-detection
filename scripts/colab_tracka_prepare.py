"""VM-side preparation for the Track A fast path (run via `colab exec -f`, driven by scripts/colab_tracka.sh).

Downloads the Kaggle datasets into /content/input with the Kaggle CLI, verifies the reviewed hashes, and checks
the Python environment. Expects /content/kaggle.json (uploaded by the driver) and /content/trackA_pkg/.
The Kaggle token is only copied to /content/.kaggle/kaggle.json (mode 600) and is never printed; CLI output is
passed through a redactor as a second guard. Prints TRACKA_PREPARE_OK on success; raises otherwise.
Note: `colab exec` runs this in a Jupyter kernel, so there is no __file__/__main__ and sys.exit is avoided.
"""
import hashlib, importlib, json, os, re, shutil, subprocess, sys, threading, time, zipfile

INPUT = "/content/input"
PKG = "/content/trackA_pkg"
KCFG = "/content/.kaggle"
TRAIN_KNEE_SHA256 = "d39966791ef1c48d9317656e8346c60cae1c064192c87b3f350e298ec4c567e8"
COMP = "rsna-knee-abnormality-detection"
# (slug, subdir, files that must exist afterwards, expected exact sizes for the big arrays)
DATASETS = [
    ("dreaddevelopment/knee-raptor-corpus", "knee-raptor-corpus",
     ["all_vols.npy", "all_masks.npy", "all_ids.npy", "window_index.npy"], {"all_vols.npy": 15895756928}),
    ("dreaddevelopment/knee-raptor-corpus-ext", "knee-raptor-corpus-ext",
     ["extra_vols.npy", "extra_masks.npy", "extra_ids.npy", "extra_meta.json"], {"extra_vols.npy": 5995680896}),
    ("nartaa/rsna-knee-hpo-assets", "rsna-knee-hpo-assets",
     ["train_knee.py", "labels_v1_blend.parquet", "train_labels_v1.parquet"], {}),
    # private bootstrap labels; the entry finds labels by basename (LABELS_FILE) anywhere under /content/input
    ("ozgung/knee-bootstrap-labels-v1-20261009", "bootstrap-labels",
     ["labels_bootstrap_v1.parquet", "labels_bootstrap_v1_synovitisprior.parquet"], {}),
]
NEEDED = ["all_vols.npy", "all_masks.npy", "all_ids.npy", "window_index.npy", "extra_vols.npy", "extra_masks.npy",
          "extra_ids.npy", "extra_meta.json", "train_knee.py", "folds_v0.json", "labels_v1_blend.parquet",
          "train_labels_v1.parquet", "train.csv",
          "labels_bootstrap_v1.parquet", "labels_bootstrap_v1_synovitisprior.parquet"]

_SECRETS = []


def redact(s):
    for x in _SECRETS:
        if x:
            s = s.replace(x, "***")
    return s


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def free_gb(path="/content"):
    return shutil.disk_usage(path).free / 1e9


def tree_bytes(path):
    tot = 0
    for d, _, fs in os.walk(path):
        for f in fs:
            try:
                tot += os.path.getsize(os.path.join(d, f))
            except OSError:
                pass
    return tot


def find(name, root=INPUT):
    for d, _, fs in os.walk(root):
        if name in fs:
            return os.path.join(d, name)
    return None


def setup_kaggle_config():
    if not os.path.exists("/content/kaggle.json"):
        raise RuntimeError("/content/kaggle.json missing (driver uploads it before this step)")
    os.makedirs(KCFG, mode=0o700, exist_ok=True)
    shutil.copy("/content/kaggle.json", f"{KCFG}/kaggle.json")
    os.chmod(f"{KCFG}/kaggle.json", 0o600)
    os.chmod("/content/kaggle.json", 0o600)
    try:  # register secrets only for output redaction
        j = json.load(open(f"{KCFG}/kaggle.json"))
        _SECRETS.extend([j.get("key"), j.get("token")])
    except Exception:
        pass
    os.environ["KAGGLE_CONFIG_DIR"] = KCFG


def kaggle(args, dest, label, monitor=True):
    """Run the kaggle CLI (quiet), print a size heartbeat from `dest` every 60 s, return (rc, seconds, bytes_added)."""
    t0, b0 = time.time(), tree_bytes(dest)
    stop = threading.Event()

    def beat():
        while not stop.wait(60):
            print(f"  [{label}] {(tree_bytes(dest) - b0) / 1e9:.1f} GB so far, {time.time() - t0:.0f}s, "
                  f"free {free_gb():.0f} GB", flush=True)
    th = threading.Thread(target=beat, daemon=True)
    if monitor:
        th.start()
    p = subprocess.run(["kaggle"] + args, capture_output=True, text=True, env=dict(os.environ))
    stop.set()
    out = redact((p.stdout or "") + (p.stderr or "")).strip()
    if out:
        print("  " + "\n  ".join(out.splitlines()[-8:]), flush=True)
    return p.returncode, time.time() - t0, tree_bytes(dest) - b0


def have(dest, files, sizes):
    for f in files:
        p = find(f, dest)
        if not p or os.path.getsize(p) == 0:
            return False
        if f in sizes and os.path.getsize(p) != sizes[f]:
            return False
    return True


def unzip_leftovers(dest):
    for d, _, fs in os.walk(dest):
        for f in fs:
            if f.endswith(".zip"):
                zp = os.path.join(d, f)
                with zipfile.ZipFile(zp) as z:
                    z.extractall(d)
                os.remove(zp)
                print(f"  extracted and removed {f}", flush=True)


def download_dataset(slug, sub, files, sizes):
    dest = os.path.join(INPUT, sub)
    os.makedirs(dest, exist_ok=True)
    if have(dest, files, sizes):
        print(f"[skip] {slug}: required files present with expected size", flush=True)
        return
    need = 40 if sizes else 5
    if free_gb() < need:
        raise RuntimeError(f"only {free_gb():.0f} GB free; need about {need} GB for {slug}")
    print(f"[download] {slug} -> {dest}  (free {free_gb():.0f} GB)", flush=True)
    # --unzip also deletes the zip. Positional dataset slug works across kaggle CLI 1.x/2.x; fall back to -d.
    for form in ([slug], ["-d", slug]):
        rc, sec, added = kaggle(["datasets", "download", *form, "-p", dest, "--unzip", "-o", "-q"], dest, slug)
        if rc == 0:
            break
        print(f"  kaggle exited {rc} for form {form}; trying the next form if any", flush=True)
    else:
        raise RuntimeError(f"kaggle download of {slug} failed (see output above; check token and dataset access)")
    unzip_leftovers(dest)
    print(f"  done: {added / 1e9:.2f} GB in {sec:.0f}s = {added / 1e6 / max(sec, 1e-6):.0f} MB/s (extracted bytes / wall)", flush=True)
    if not have(dest, files, sizes):
        missing = [f for f in files if not find(f, dest)]
        raise RuntimeError(f"{slug}: after download missing {missing} or size mismatch (expected {sizes})")


def download_train_csv():
    dest = os.path.join(INPUT, COMP)
    os.makedirs(dest, exist_ok=True)
    if find("train.csv", dest) and os.path.getsize(find("train.csv", dest)) > 0:
        print("[skip] train.csv present", flush=True)
        return
    print(f"[download] {COMP}/train.csv -> {dest}", flush=True)
    for form in ([COMP], ["-c", COMP]):
        rc, sec, added = kaggle(["competitions", "download", *form, "-f", "train.csv", "-p", dest, "-q"], dest, "train.csv", monitor=False)
        if rc == 0:
            break
    else:
        raise RuntimeError("train.csv download failed; the account must have accepted the competition rules on kaggle.com "
                           "(a 403 means not accepted), which this script does not do")
    unzip_leftovers(dest)
    if not find("train.csv", dest):
        raise RuntimeError("train.csv not found after competition download")


PARAMS = "/content/tracka_params.json"
WORKDIR = "/content/trackA"
RESUME_KEYS = ("model", "opt", "sched", "best", "best_state", "best_P", "hist", "topk", "epoch")


def check_payload(manifest, rp, fold, ident, cur):
    """Provenance of a downloaded resume file (mirrors scripts/colab_tracka_verify_payload.py). Returns a list of problems."""
    bad = []
    if manifest is None:
        return ["payload_manifest.json missing from the dataset download"]
    if str(manifest.get("fold")) != str(fold):
        bad.append(f"manifest.fold {manifest.get('fold')!r} != {fold}")
    got = sha256(rp)
    if not manifest.get("resume_sha256") or got != manifest["resume_sha256"]:
        bad.append(f"resume sha256 {got[:12]} != manifest {str(manifest.get('resume_sha256'))[:12]}")
    if not isinstance(ident, dict):
        bad.append("no identity block")
        return bad
    if str(ident.get("fold")) != str(fold):
        bad.append(f"identity.fold {ident.get('fold')!r} != {fold}")
    for key, want in cur.items():
        if ident.get(key) != want:
            bad.append(f"identity.{key} differs from the current package")
    return bad


def current_identity_hashes(labels_name):
    lp = find(labels_name or "labels_v1_blend.parquet")
    if not lp:
        raise RuntimeError(f"RESUME RESTORE FAILED: current labels file {labels_name!r} not found, cannot verify identity")
    return {"script_sha256": sha256(f"{PKG}/train_knee_patched.py"), "labels_sha256": sha256(lp),
            "fold_file_sha256": sha256(find("folds_v0.json")), "excluded_ids_sha256": sha256(f"{PKG}/excluded_ids.json")}


def ensure_kaggle_cli(minimum=(2, 2, 4)):
    """The persister/restore need `datasets status --format json(status,current_version_number)` (CLI >= 2.2): upgrade the
    VM's CLI when older. Raises if the upgraded CLI still cannot report a version."""
    def ver():
        r = subprocess.run(["kaggle", "--version"], capture_output=True, text=True, env=dict(os.environ))
        m = re.search(r"(\d+)\.(\d+)\.(\d+)", (r.stdout or "") + (r.stderr or ""))
        return tuple(int(x) for x in m.groups()) if m else None
    v = ver()
    print(f"kaggle CLI: {'.'.join(map(str, v)) if v else 'unknown'}", flush=True)
    if v is None or v < minimum:
        spec = "kaggle>=" + ".".join(map(str, minimum))
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", spec], capture_output=True, text=True)
        v = ver()
        print(f"kaggle CLI upgraded to {'.'.join(map(str, v)) if v else 'unknown'} (pip rc {r.returncode})", flush=True)
        if v is None or v < minimum:
            raise RuntimeError(f"kaggle CLI {spec} could not be installed: {redact((r.stderr or '')[-300:])}")
    return v


def dataset_version(slug):
    """(status, current_version_number) from `datasets status --format json`; version None when not retrievable."""
    r = subprocess.run(["kaggle", "datasets", "status", slug, "--format", "json(status,current_version_number)"],
                       capture_output=True, text=True, env=dict(os.environ))
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


def restore_resume(slug_tmpl):
    """Download the latest version of the resume dataset into WORKDIR, verify each resume state loads, set RESUME=1."""
    import torch
    params = json.load(open(PARAMS))
    strict = str(params.get("STRICT_IDENTITY", "0")).strip().lower() in ("1", "true", "yes")
    folds = [str(params["FOLD_A"])]
    if "{K}" in slug_tmpl and str(params.get("FOLD_B", "")).strip() not in ("", "none", "None", "-1"):
        folds.append(str(params["FOLD_B"]))
    os.makedirs(WORKDIR, exist_ok=True)
    for k in folds:
        slug = slug_tmpl.replace("{K}", k)
        dest = f"{os.path.dirname(WORKDIR)}/resume_dl_fold{k}"
        shutil.rmtree(dest, ignore_errors=True)
        os.makedirs(dest)
        st0, v0 = dataset_version(slug)
        if v0 is None or not st0.startswith("ready"):
            raise RuntimeError(f"RESUME RESTORE FAILED: {slug} status {st0!r} version {v0!r}: a READY dataset with a positive "
                               f"exact version is required before a restore (no unversioned restore)")
        print(f"[resume] downloading {slug} version {v0} (status {st0}; pinned as {slug}/{v0})", flush=True)
        for form in ([f"{slug}/{v0}"], ["-d", f"{slug}/{v0}"]):   # CLI >= 2.2: OWNER/SLUG/N pins the version; never restore the unpinned slug
            rc, sec, added = kaggle(["datasets", "download", *form, "-p", dest, "--unzip", "-o", "-q"], dest, "resume", monitor=False)
            if rc == 0:
                break
        else:
            raise RuntimeError(f"RESUME RESTORE FAILED: could not download resume dataset {slug} (missing, not ready, or not owned by this "
                               f"account); refusing to fall back to training from scratch")
        st1, v1 = dataset_version(slug)
        if v1 != v0:
            raise RuntimeError(f"RESUME RESTORE FAILED: {slug} version changed during the download ({v0} -> {v1}); refusing to "
                               f"continue from a version that is no longer the newest")
        unzip_leftovers(dest)
        rp = find(f"raptor_resume_fold{k}.pt", dest)
        if not rp or os.path.getsize(rp) == 0:
            raise RuntimeError(f"RESUME RESTORE FAILED: {slug} has no (or an empty) raptor_resume_fold{k}.pt; "
                               f"refusing to fall back to training from scratch")
        try:
            R = torch.load(rp, map_location="cpu", weights_only=False)
            miss = [x for x in RESUME_KEYS if x not in R]
            if miss:
                raise KeyError(f"missing keys {miss}")
            epoch = int(R["epoch"])
            n_topk = len(R["topk"])
            hist = R["hist"]
            ident = R.get("identity")
            tk = R["topk"]
            struct = [f"{x} empty" for x in ("model", "opt", "sched", "best_state") if not R[x]]
            if [h.get("ep") if isinstance(h, dict) else None for h in hist] != list(range(epoch + 1)):
                struct.append(f"history is not exactly epochs 0..{epoch}")
            if not tk or not all(isinstance(t, dict) and t.get("model") and "sel" in t and "epoch" in t for t in tk):
                struct.append("top-k states missing, empty or malformed")
            del R
        except Exception as e:
            raise RuntimeError(f"resume state from {slug} is unreadable: {type(e).__name__}: {e}")
        print(f"[resume] fold {k}: {slug} resume file loads; epoch {epoch}; sha256 {sha256(rp)}", flush=True)
        mpath = find("payload_manifest.json", dest)
        manifest = json.load(open(mpath)) if mpath else None
        problems = check_payload(manifest, rp, k, ident, current_identity_hashes(params.get("LABELS_FILE")))
        if ident is None and manifest is not None and not strict:
            problems = [x for x in problems if x != "no identity block"]   # legacy block-less state: warned below, still sha-checked
        problems = struct + problems
        if manifest is not None and manifest.get("epoch") is not None and int(manifest["epoch"]) != epoch:
            problems.append(f"manifest.epoch {manifest['epoch']} != resume epoch {epoch}")
        if problems:
            raise RuntimeError(f"RESUME RESTORE FAILED: {slug} payload provenance not verified: {'; '.join(problems)}")
        if ident is None:
            msg = (f"fold {k}: saved state has NO identity block (written by the old persister/trainer): fold, script, labels, folds, "
                   f"exclusions and config of the state cannot be verified")
            if strict:
                raise RuntimeError(f"RESUME RESTORE FAILED: {msg}; aborting because STRICT_IDENTITY is set")
            print("!" * 100 + f"\n[resume] IDENTITY WARNING: {msg}; continuing\n" + "!" * 100, flush=True)
        else:
            brief = {x: ident.get(x) for x in ("fold", "n_folds", "tag", "script_sha256", "labels_sha256", "fold_file_sha256",
                                               "excluded_ids_sha256", "oof_mask_sha256", "epochs_requested")}
            print(f"[resume] fold {k}: identity block {json.dumps(brief)}", flush=True)
            if ident.get("fold") is not None and str(ident["fold"]) != k:
                raise RuntimeError(f"RESUME RESTORE FAILED: {slug} holds fold {ident['fold']}, expected fold {k}")
        # keep the previous session's log next to the new one; the trainer rewrites log_fold{k}.txt
        for name in (f"raptor_resume_fold{k}.pt", f"raptor_ft_fold{k}.json"):
            src = find(name, dest)
            if src:
                shutil.move(src, os.path.join(WORKDIR, name))
        json.dump({"sha256": sha256(os.path.join(WORKDIR, f"raptor_resume_fold{k}.pt")), "epoch": epoch, "dataset": slug,
                   "version": v0, "manifest_sha256": manifest["resume_sha256"] if manifest else None},
                  open(os.path.join(WORKDIR, f"restored_verified_fold{k}.json"), "w"))   # persister trusts only a matching marker
        prev = find(f"log_fold{k}.txt", dest)
        if prev:
            shutil.move(prev, os.path.join(WORKDIR, f"log_fold{k}.prev.txt"))
        shutil.rmtree(dest, ignore_errors=True)
        print(f"[resume] fold {k}: restored and verified {slug} version {v0}: last completed epoch {epoch} "
              f"(training continues at epoch {epoch + 1}), top-k kept {n_topk}, "
              f"{os.path.getsize(os.path.join(WORKDIR, f'raptor_resume_fold{k}.pt')) / 1e6:.0f} MB", flush=True)
    params["RESUME"] = "1"
    json.dump(params, open(PARAMS, "w"))
    print("[resume] params updated: RESUME=1", flush=True)


def check_env():
    import torch
    print(f"python {sys.version.split()[0]} | torch {torch.__version__} | cuda {torch.cuda.is_available()} "
          f"| gpus {torch.cuda.device_count()}", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device visible; session is not a GPU runtime")
    print(f"gpu {torch.cuda.get_device_name(0)}", flush=True)
    try:
        importlib.import_module("timm")
    except ImportError:
        print("timm missing -> pip install timm", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm"], check=True)
    importlib.invalidate_caches()
    mods = {}
    for m in ("timm", "numpy", "pandas", "sklearn", "pyarrow"):
        try:
            mods[m] = importlib.import_module(m).__version__
        except ImportError:
            raise RuntimeError(f"required module {m} not importable on this VM")
    print("modules:", mods, flush=True)
    mem = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo") if l.split()[1].isdigit()}
    print(f"RAM available {mem.get('MemAvailable', 0) / 1e6:.0f} GB | CPUs {os.cpu_count()}", flush=True)


def main():
    t_all = time.time()
    print(f"free disk before: {free_gb():.0f} GB", flush=True)
    setup_kaggle_config()
    ensure_kaggle_cli()
    manifest = json.load(open(f"{PKG}/manifest.json"))
    os.makedirs(INPUT, exist_ok=True)
    for ds in DATASETS:
        download_dataset(*ds)
    download_train_csv()
    # leave no archives behind
    for d, _, fs in os.walk(INPUT):
        for f in fs:
            if f.endswith(".zip"):
                os.remove(os.path.join(d, f))
    # folds_v0.json normally ships in the assets dataset; otherwise place the reviewed package copy.
    if not find("folds_v0.json"):
        shutil.copy(f"{PKG}/folds_v0.json", f"{INPUT}/folds_v0.json")
        print("folds_v0.json not in datasets; placed the package copy", flush=True)
    # hash checks
    tk = find("train_knee.py")
    if sha256(tk) != TRAIN_KNEE_SHA256:
        raise RuntimeError(f"train_knee.py sha256 mismatch: {sha256(tk)}")
    if sha256(find("folds_v0.json")) != manifest["folds_sha256"]:
        raise RuntimeError("folds_v0.json sha256 differs from manifest.json")
    for name, key in (("train_knee_patched.py", "patched_sha256"), ("train_knee.patch", "patch_sha256"),
                      ("excluded_ids.json", "excluded_ids_sha256")):
        if sha256(f"{PKG}/{name}") != manifest[key]:
            raise RuntimeError(f"{name} sha256 differs from manifest.json")
    print("hash checks ok (train_knee.py, folds_v0.json, patched script, patch, excluded_ids)", flush=True)
    missing = [n for n in NEEDED if not find(n)]
    if missing:
        raise RuntimeError(f"missing under {INPUT}: {missing}")
    print("all required inputs found:", {n: find(n) for n in NEEDED}, flush=True)
    check_env()
    rslug = str(json.load(open(PARAMS)).get("RESUME_FROM_DATASET", "")).strip()
    if rslug:
        restore_resume(rslug)
    print(f"free disk after: {free_gb():.0f} GB | prepare took {time.time() - t_all:.0f}s", flush=True)
    print("TRACKA_PREPARE_OK", flush=True)


main()
