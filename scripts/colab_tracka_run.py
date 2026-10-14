"""VM-side runner for the Track A fast path (run via `colab exec -f`, driven by scripts/colab_tracka.sh).

`colab exec` cannot pass environment variables, so the driver uploads /content/tracka_params.json and this script
turns it into os.environ for tracka_entry.py, then runs the entry as a subprocess and streams its output.
Prints TRACKA_RUN_OK only if the entry exited 0 AND (when persistence is requested) the final resume state was
verified READY in the dataset: after the entry exits every persister is told to publish + verify the newest state
(SIGUSR1) and this script waits up to FINAL_BACKUP_WAIT_S (15 min) for it, printing `FINAL BACKUP VERIFIED epoch N
version V` or `FINAL BACKUP UNVERIFIED (reason)` per fold. The finished resume file is deleted only AFTER that line.
Runs in a Jupyter kernel: no sys.exit, raise instead. TRACKA_CONTENT (default /content) relocates all paths for tests.
"""
import glob, json, os, signal, subprocess, sys, threading, time

CONTENT = os.environ.get("TRACKA_CONTENT", "/content")
PARAMS = f"{CONTENT}/tracka_params.json"
PKG = f"{CONTENT}/trackA_pkg"
WORKDIR = f"{CONTENT}/trackA"
PERSIST_PY = f"{CONTENT}/colab_tracka_persist.py"   # uploaded by the driver
KCFG = f"{CONTENT}/.kaggle"
FINAL_BACKUP_WAIT_S = float(os.environ.get("FINAL_BACKUP_WAIT_S", 900))


def mem_available_gb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return 0.0


def truthy(v):
    return str(v).strip().lower() in ("1", "true", "yes")


def relay(proc, tag, logfile):
    """Echo a background process's output into this cell's output (and a file in WORKDIR)."""
    def pump():
        with open(logfile, "a") as lf:
            for line in proc.stdout:
                print(f"{tag} {line.rstrip()}", flush=True)
                lf.write(line); lf.flush()
    threading.Thread(target=pump, daemon=True).start()


def start_persisters(params, env):
    """One background persister per fold: publishes WORKDIR/raptor_resume_fold{K}.pt as a private Kaggle dataset version."""
    slug = str(params.get("RESUME_DATASET", "")).strip()
    if not slug:
        print("[run] RESUME_DATASET not set: per-epoch resume state stays on this VM only", flush=True)
        return []
    folds = [str(params["FOLD_A"])] + ([str(params["FOLD_B"])] if str(params.get("FOLD_B", "")).strip() not in ("", "none", "None", "-1") else [])
    out = []
    for k in folds:
        if "{K}" not in slug and k != folds[0]:
            print(f"[run] RESUME_DATASET has no {{K}} placeholder: fold {k} is NOT persisted (only fold {folds[0]})", flush=True)
            continue
        ds = slug.replace("{K}", k)
        penv = dict(env, KAGGLE_CONFIG_DIR=KCFG, PYTHONUNBUFFERED="1")
        cmd = [sys.executable, "-u", PERSIST_PY, "--workdir", WORKDIR, "--fold", k, "--dataset", ds,
               "--staging", f"{CONTENT}/resume_staging_fold{k}", "--status-file", f"{CONTENT}/persist_status_fold{k}.json"]
        if str(params.get("RESUME_FROM_DATASET", "")).strip():
            cmd.append("--restored-from-dataset")   # the restored resume file is already in the dataset
        p = subprocess.Popen(cmd, env=penv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        relay(p, f"[persist fold{k}]", f"{WORKDIR}/persist_fold{k}.log")
        out.append((k, p))
        print(f"[run] persister started for fold {k} -> {ds} (pid {p.pid})", flush=True)
    return out


def read_status(k):
    with open(f"{CONTENT}/persist_status_fold{k}.json") as f:
        return json.load(f)


def show_persist_status(persisters, label="persist status"):
    """Print each fold's persist_status_fold{K}.json (accepted vs verified state)."""
    for k, _ in persisters:
        try:
            st = read_status(k)
            print(f"[run] {label} fold {k}: VERIFIED epoch {st.get('verified_epoch')} (version {st.get('verified_version')}), "
                  f"accepted epoch {st.get('accepted_epoch')} (version {st.get('accepted_version')}), "
                  f"verification_attempts {st.get('verification_attempts')}, last_error {st.get('last_error')!r}, at {st.get('timestamp')}", flush=True)
        except Exception as e:
            print(f"[run] {label} fold {k}: unavailable ({type(e).__name__}: {e})", flush=True)


def expected_final_epoch(params, rc, k):
    """Epoch the final backup must hold: EPOCHS-1 for a clean, complete (not time-budget-stopped, non-smoke) run, else None."""
    if rc != 0 or truthy(params.get("SMOKE", "0")) or not str(params.get("EPOCHS", "")).strip():
        return None
    try:
        if json.load(open(f"{WORKDIR}/raptor_ft_fold{k}.json")).get("partial"):
            return None
    except Exception:
        pass
    return int(params["EPOCHS"]) - 1


def final_backup(persisters, params, rc, wait_s=None):
    """Signal every persister (SIGUSR1 = publish newest state + verify READY), wait up to wait_s (15 min) in total, print
    `FINAL BACKUP VERIFIED epoch N version V` / `FINAL BACKUP UNVERIFIED (reason)` per fold. Returns {fold: bool}."""
    wait_s = FINAL_BACKUP_WAIT_S if wait_s is None else wait_s
    for k, p in persisters:
        if p.poll() is None:
            p.send_signal(signal.SIGUSR1)
    deadline = time.time() + wait_s
    res = {}
    for k, p in persisters:
        timed_out = False
        try:
            p.wait(timeout=max(0.0, deadline - time.time()))
        except subprocess.TimeoutExpired:
            p.kill(); p.wait(); timed_out = True
        reason, st = None, {}
        try:
            st = read_status(k)
        except Exception as e:
            reason = f"persist status unreadable ({type(e).__name__})"
        fin = st.get("final") or {}
        if timed_out:
            reason = f"persister still running after {wait_s:.0f}s and was killed; last_error {st.get('last_error')!r}"
        elif reason is None and (p.returncode != 0 or fin.get("state") != "verified"):
            reason = f"persister rc={p.returncode}, final state {fin.get('state')!r}: {fin.get('reason') or st.get('last_error')}"
        exp = expected_final_epoch(params, rc, k)
        if reason is None and exp is not None and st.get("verified_epoch") != exp:
            reason = f"verified epoch {st.get('verified_epoch')} != expected final epoch {exp}"
        if reason is None and st.get("verified_epoch") is None:
            reason = "verified epoch unknown"
        if reason is None:
            ver = st.get("verified_version")
            print(f"FINAL BACKUP VERIFIED epoch {st['verified_epoch']} version {ver if ver is not None else 'unknown'} (fold {k})", flush=True)
        else:
            print(f"FINAL BACKUP UNVERIFIED ({reason}) (fold {k})", flush=True)
        res[k] = reason is None
    show_persist_status(persisters, "final persist status")
    return res


def drop_finished_resume(k):
    """Free disk: delete a finished fold's resume state. Only called AFTER `FINAL BACKUP VERIFIED`."""
    rp = f"{WORKDIR}/raptor_resume_fold{k}.pt"
    if os.path.exists(rp) and os.path.getsize(rp) > 300e6 and os.path.exists(f"{WORKDIR}/raptor_ft_fold{k}_swa.pt"):
        os.unlink(rp)
        print(f"[run] removed finished resume state {os.path.basename(rp)} after the verified final backup", flush=True)


def heartbeat(t0, stop, persisters):
    while not stop.wait(300):
        print(f"[run] heartbeat {(time.time() - t0) / 60:.0f} min", flush=True)
        show_persist_status(persisters)


def main():
    params = json.load(open(PARAMS))
    env = dict(os.environ)
    for k, v in params.items():  # every JSON key becomes an env var for tracka_entry (PARAM_DEFAULTS names, uppercase)
        env[str(k).upper()] = str(v)
    env.update(RAPTOR_INPUT=f"{CONTENT}/input", WORKDIR=WORKDIR, ASSET_DIR=PKG, WORKERS="10", PYTHONUNBUFFERED="1")
    # The corpus is 22 GB of npy (all_vols 15.9 GB + extra_vols 6.0 GB). Random slice reads through np.load(mmap_mode="r")
    # are slow on the VM disk, so preload it into RAM once in the parent (forked loader workers share it) when it
    # comfortably fits: this VM has ~83 GB RAM. Otherwise keep the memmap.
    avail = mem_available_gb()
    if avail > 40:
        env["RAPTOR_PRELOAD"] = "1"
    else:
        env.pop("RAPTOR_PRELOAD", None)
    # Single GPU note: tracka_entry runs the two folds concurrently only when torch.cuda.device_count() >= len(folds)
    # (one process per GPU). With one A100 and FOLD_B set, it runs fold A then fold B SEQUENTIALLY on GPU 0, and
    # TIME_BUDGET_H applies to each fold's training separately. For a single fold, FOLD_B is "" and nothing changes.
    shown = {k: env[k] for k in sorted(set(k.upper() for k in params) | {"WORKERS", "RAPTOR_PRELOAD"}) if k in env}
    print(f"[run] params {shown} | MemAvailable {avail:.0f} GB | preload {'on' if avail > 40 else 'off'}", flush=True)
    os.makedirs(WORKDIR, exist_ok=True)
    t0 = time.time()
    hb_stop = threading.Event()
    persisters = start_persisters(params, env)
    threading.Thread(target=heartbeat, args=(t0, hb_stop, persisters), daemon=True).start()
    p = subprocess.Popen([sys.executable, "-u", f"{PKG}/tracka_entry.py"], env=env, cwd=CONTENT,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line.rstrip("\n"), flush=True)
    rc = p.wait()
    hb_stop.set()
    folds_run = [str(params["FOLD_A"])] + ([str(params["FOLD_B"])] if str(params.get("FOLD_B", "")).strip() not in ("", "none", "None", "-1") else [])
    print(f"[run] tracka_entry exited rc={rc} after {(time.time() - t0) / 3600:.2f} h", flush=True)
    backup = final_backup(persisters, params, rc) if persisters else {}
    if not persisters:
        print("[run] persistence not requested: no final backup to verify", flush=True)
    for k, ok in backup.items():
        if ok and rc == 0:
            drop_finished_resume(k)
    for f in sorted(glob.glob(f"{WORKDIR}/run_fold*.json")):
        r = json.load(open(f))
        print(f"[summary] fold {r['fold']}: epochs {r.get('epochs_completed')}/{r.get('epochs_requested')} "
              f"| checkpoint {r.get('checkpoint_kind')} | selected epochs {r.get('selected_epochs')} "
              f"| val masked AUC {r.get('val_masked_macro_auc')} (n={r['export']['n_val']}) "
              f"| gold AUC {r.get('gold_macro_auc')} (n={r['export']['n_gold']}) "
              f"| train wall {r.get('train_wall_seconds')} s | time-budget stop {r.get('stopped_by_time_budget')}", flush=True)
    if rc != 0:
        raise RuntimeError(f"tracka_entry failed with rc={rc}; partial outputs (if any) are in {WORKDIR}")
    if not all(backup.values()):
        raise RuntimeError("FINAL BACKUP UNVERIFIED for fold(s) " + ",".join(k for k, ok in backup.items() if not ok)
                           + ": training finished but the latest resume state is not confirmed in the dataset")
    print("TRACKA_RUN_OK", flush=True)


if os.environ.get("TRACKA_RUN_NOMAIN") != "1":
    main()
