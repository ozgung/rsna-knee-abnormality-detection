#!/usr/bin/env python3
"""VM-side persister for the Track A per-epoch resume state (launched by colab_tracka_run.py, runs on the Colab VM).

Every --interval seconds (default 60) it checks WORKDIR/raptor_resume_fold{K}.pt (size + mtime). When the file
changed and at least --min-gap seconds (default 480) passed since the last accepted publish, it stages the file
(hard link, so the trainer's atomic os.replace can never alter the staged bytes; copy as a fallback), together with
raptor_ft_fold{K}.json and log_fold{K}.txt, writes an immutable payload_manifest.json (file names, sizes, sha256 of the
resume file, the epoch read from the STAGED resume dict, publish_started_at) plus a copy of the status file, and
publishes the staging dir as a new version of a private Kaggle dataset (`kaggle datasets create -p DIR` on first use,
`kaggle datasets version -p DIR -m "epoch N"` afterwards). Old dataset versions are KEPT unless --delete-old-versions.
Staged copies are deleted after every attempt.

Two distinct states are tracked (and written to persist_status_fold{K}.json next to the staging dir):
  published/accepted : the CLI accepted an upload (accepted_epoch, accepted_version = number parsed from the CLI output,
                       else from `kaggle datasets versions`, else None; local_publish_count is only a local counter).
  verified           : `kaggle datasets status` says ready AND the `datasets files --csv` listing shows BOTH the resume
                       file and payload_manifest.json with exactly the manifest's size and a creation time >= publish
                       start (minus --clock-skew). An unparseable/failed listing is NEVER ready (verified_epoch,
                       verified_version).
An accepted-but-unverified payload is re-VERIFIED on later cycles (no re-upload); it is re-uploaded only when the server
still does not show it after --reupload-after seconds, or shows a mismatching file, or reports an error status.
The status JSON has: accepted_epoch, accepted_version, verified_epoch, verified_version, last_error,
verification_attempts, published_signature, verified_signature, final (state of the last final request).

Signals: SIGTERM/SIGINT = stop now; SIGUSR1 = final request: keep publishing/verifying (ignoring throttles) until the
newest resume file is verified READY or --final-timeout (default 840 s) passes, log `FINAL PERSIST VERIFIED epoch N
version V` or `FINAL PERSIST UNVERIFIED (reason)`, and exit 0 (verified / nothing pending) or 3 (unverified).
The status copy staged INSIDE the dataset carries payload_epoch (epoch of that very version's resume file), so a
recovery tool can learn the latest stored epoch with `kaggle datasets download -f persist_status_fold{K}.json`.
Needs the Kaggle token in KAGGLE_CONFIG_DIR (the prepare step puts it in /content/.kaggle). License CC0.
"""
import argparse, calendar, csv, hashlib, io, json, os, re, shutil, signal, subprocess, sys, time
from pathlib import Path

STOP = {"now": False, "final": False}
SOFT_CAP_BYTES = 2_000_000_000


def log(msg):
    print(f"[persist {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sig_of(p):
    try:
        st = os.stat(p)
        return (st.st_size, st.st_mtime_ns)
    except OSError:
        return None


def redact(s):
    try:
        j = json.load(open(os.path.join(os.environ.get("KAGGLE_CONFIG_DIR", os.path.expanduser("~/.kaggle")), "kaggle.json")))
        for v in (j.get("key"), j.get("token")):
            if v:
                s = s.replace(v, "***")
    except Exception:
        pass
    return s


def kaggle(args, timeout=3600):
    try:
        p = subprocess.run(["kaggle"] + args, capture_output=True, text=True, timeout=timeout, env=dict(os.environ))
        return p.returncode, redact((p.stdout or "") + (p.stderr or "")).strip()
    except subprocess.TimeoutExpired:
        return 124, f"kaggle {args[:2]} timed out after {timeout}s"
    except FileNotFoundError:
        return 127, "kaggle CLI not found on PATH"


def atomic_write_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    os.replace(tmp, path)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def payload_epoch(resume_path, ft_json_path=None):
    """(epoch, source) of a STAGED payload: the resume dict's `epoch` (torch.load) only. Never taken from the training log
    or the raptor_ft json (ft_json_path is accepted for compatibility and ignored). (None, 'unknown') when unreadable."""
    try:
        import torch
        try:
            R = torch.load(resume_path, map_location="cpu", weights_only=False, mmap=True)
        except Exception:
            R = torch.load(resume_path, map_location="cpu", weights_only=False)
        ep = int(R["epoch"])
        del R
        return ep, "resume_dict"
    except Exception as e:
        err = f"{type(e).__name__}"
    return None, f"unknown ({err})"


def parse_created(text):
    """UTC epoch seconds from a listing timestamp ('2026-10-10 12:34:56[.123]'), or None."""
    m = re.search(r"(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})", text or "")
    if not m:
        return None
    try:
        return calendar.timegm(time.strptime(f"{m.group(1)} {m.group(2)}", "%Y-%m-%d %H:%M:%S"))
    except ValueError:
        return None


def parse_version(out):
    """Accepted version number printed by the CLI ('... version 7 ...'), or None."""
    m = re.search(r"\bversion\s*(?:number\s*)?[#:]?\s*(\d+)\b", out or "", re.I)
    return int(m.group(1)) if m else None


def server_version(slug, started_at=None):
    """Newest server version number, or None when it cannot be retrieved.
    First `datasets status --format json(status,current_version_number)` (CLI >= 2.2), then the versions-style CSV listing
    (older CLIs; columns containing 'version'). `started_at` filters CSV versions created before this publish."""
    rc, out = kaggle(["datasets", "status", slug, "--format", "json(status,current_version_number)"], timeout=120)
    if rc == 0:
        for line in reversed((out or "").strip().splitlines()):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                v = json.loads(line).get("current_version_number")
            except ValueError:
                break
            if isinstance(v, int) and v > 0:
                return v
            if isinstance(v, str) and v.strip().isdigit():
                return int(v)
            break
    rc, out = kaggle(["datasets", "versions", slug, "--csv"], timeout=120)
    if rc != 0:
        return None
    lines = out.splitlines()
    idx = next((i for i, l in enumerate(lines) if "version" in l.lower() and "," in l), None)
    if idx is None:
        return None
    best = None
    for r in csv.DictReader(io.StringIO("\n".join(lines[idx:]))):
        keys = {(k or "").strip().lower(): v for k, v in r.items()}
        vn = next((v for k, v in keys.items() if "version" in k and (v or "").strip().isdigit()), None)
        if vn is None:
            continue
        when = next((parse_created(v) for k, v in keys.items() if "creat" in k or "date" in k), None)
        if started_at is not None and when is not None and when < int(started_at):
            continue
        best = max(best or 0, int(vn))
    return best


def check_files(slug, expect, started_at, skew):
    """Compare the server listing (`datasets files --csv`) with the manifest. expect: {name: size}.
    Returns (state, detail); state is ok / absent / stale / bad / unknown. Only 'ok' means verified."""
    rc, out = kaggle(["datasets", "files", slug, "--csv"], timeout=120)
    if rc != 0:
        return "unknown", f"files listing failed rc={rc}"
    lines = out.splitlines()
    idx = next((i for i, l in enumerate(lines) if l.strip().lower().startswith("name,")), None)
    if idx is None:
        return "unknown", "files listing unparseable (no csv header)"
    rows = {}
    for r in csv.DictReader(io.StringIO("\n".join(lines[idx:]))):
        k = {(a or "").strip().lower(): (b or "").strip() for a, b in r.items()}
        if k.get("name"):
            rows[k["name"]] = k
    floor = int(started_at) - int(skew)
    states, details = [], []
    for name, size in expect.items():
        r = rows.get(name)
        if r is None:
            states.append("absent"); details.append(f"{name} not listed")
            continue
        if not re.fullmatch(r"\d+", r.get("size", "")):
            states.append("unknown"); details.append(f"{name} size {r.get('size')!r} not exact bytes")
            continue
        created = parse_created(r.get("creationdate") or r.get("creation_date") or r.get("created") or "")
        if int(r["size"]) != int(size):
            # A listing that still shows a file created BEFORE this publish started is the previous version (Kaggle publishes
            # the new version with a lag); only a file created by this publish with the wrong size is a bad upload.
            if created is not None and created < floor:
                states.append("stale"); details.append(f"{name} size {r['size']} != manifest {size} but created {created} predates publish start {floor} (previous version still listed)")
            elif created is None:
                states.append("unknown"); details.append(f"{name} size {r['size']} != manifest {size} and no parseable creation time")
            else:
                states.append("bad"); details.append(f"{name} size {r['size']} != manifest {size} (created {created} >= publish start {floor})")
            continue
        if created is None:
            states.append("unknown"); details.append(f"{name} has no parseable creation time")
        elif created < floor:
            states.append("stale"); details.append(f"{name} created {created} predates publish start {floor}")
        else:
            states.append("ok")
    for s in ("unknown", "bad", "absent", "stale"):
        if s in states:
            return s, "; ".join(d for d in details)
    return "ok", "listing matches manifest (" + ", ".join(f"{n} {s}" for n, s in expect.items()) + ")"


def bind_check(slug, acc):
    """Bind the listing to THIS accepted payload: download the dataset's payload_manifest.json and compare its resume sha256,
    epoch and publish start with what was staged, then require a positive exact server version equal to the accepted one when
    that is known (an unrelated/newer/older version never confirms). Returns (ok, detail): True bound, False a wrong/foreign
    payload, None unqualified (server version not retrievable; the payload is not wrong, re-uploading would not help)."""
    import tempfile, zipfile
    # positive exact version required (fail-closed): a listing without a retrievable server version never qualifies
    sv = server_version(slug)
    if not isinstance(sv, int) or sv <= 0:
        return None, "server version unavailable (positive exact version required; CLI must support `datasets status --format json`)"
    if sv < int(acc.get("min_version") or 1):
        # the version counter also publishes with a lag: right after acceptance the server can still report the previous
        # version; that is not a wrong payload, just not visible yet
        return None, f"server version {sv} predates this publish (expected >= {acc.get('min_version')}); not visible yet"
    if acc.get("version") is not None and sv != acc["version"]:
        return False, f"server newest version {sv} != accepted version {acc['version']}"
    with tempfile.TemporaryDirectory() as d:
        # CLI >= 2.2 accepts OWNER/SLUG/N: the manifest is fetched from exactly that version
        rc, out = kaggle(["datasets", "download", f"{slug}/{sv}", "-f", "payload_manifest.json", "-p", d, "--unzip", "-o", "-q"], timeout=300)
        for z in Path(d).glob("*.zip"):
            zipfile.ZipFile(z).extractall(d)
        hits = list(Path(d).rglob("payload_manifest.json"))
        if rc != 0 or not hits:
            return False, f"payload_manifest.json could not be downloaded for binding (rc={rc})"
        try:
            m = json.load(open(hits[0]))
        except ValueError:
            return False, "downloaded payload_manifest.json is not valid JSON"
    if m.get("resume_sha256") != acc["sha256"]:
        return False, f"server manifest sha256 {str(m.get('resume_sha256'))[:12]} != staged {acc['sha256'][:12]}"
    if m.get("epoch") != acc["epoch"]:
        return False, f"server manifest epoch {m.get('epoch')} != staged {acc['epoch']}"
    if m.get("publish_started_at_epoch") != acc["started_at"]:
        return False, "server manifest publish_started_at differs from this publish"
    acc["version"] = sv
    return True, f"manifest binds sha256 {acc['sha256'][:12]} epoch {acc['epoch']} version {sv}"


def verify(slug, acc, timeout, poll, interrupted, skew):
    """Poll until the accepted payload is verified. Returns (kind, detail): ready / pending / absent / unknown / unbound / bad / error."""
    deadline = time.time() + timeout
    expect = {acc["name"]: acc["size"], "payload_manifest.json": acc["manifest_size"]}
    last = ("pending", "no status obtained")
    while True:
        rc, out = kaggle(["datasets", "status", slug], timeout=120)
        line = (out.strip().splitlines() or [""])[-1].strip()
        if rc != 0 or not line:
            last = ("unknown", f"status rc={rc}: {line[:120] or 'empty'}")
        elif re.search(r"^\W*(status\W+)?ready\b", line, re.I):
            state, detail = check_files(slug, expect, acc["started_at"], skew)
            if state == "ok":
                bound, bdetail = bind_check(slug, acc)
                if bound:
                    return "ready", detail + "; " + bdetail
                if bound is None:
                    last = ("unbound", f"status ready but not qualified yet: {bdetail}")   # keep polling within the timeout
                else:
                    last = ("bad", f"status ready but not bound to the accepted payload: {bdetail}")
                    return last
            else:
                last = ("absent" if state == "stale" else state, f"status ready but {detail}")
            # 'bad' (a file created by this publish with the wrong size) is re-checked within the timeout as well: the listing
            # can briefly show a partially published version; the caller re-uploads only after repeated bad verifications.
        elif re.search(r"error|fail|deleted", line, re.I):
            return "error", f"dataset status reports: {line[:120]}"
        else:
            last = ("pending", f"status '{line[:60]}'")
        if time.time() + poll > deadline:
            return last[0], f"not verified after {timeout:.0f}s ({last[1]})"
        if interrupted():
            return last[0], f"stopped while waiting ({last[1]})"
        time.sleep(poll)


def stage(work, fold, staging, slug, status_snapshot, n_pub):
    """Refresh the staging dir with the latest resume file (+ ft json + log) and write payload_manifest.json,
    the in-dataset status copy and dataset-metadata.json. Returns an `acc` dict (what is about to be sent) or None."""
    src = work / f"raptor_resume_fold{fold}.pt"
    s0 = sig_of(src)
    if s0 is None:
        return None
    staging.mkdir(parents=True, exist_ok=True)
    for old in staging.glob("raptor_resume_fold*.pt"):   # keep only the latest resume file
        old.unlink()
    dst = staging / src.name
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    if sig_of(src) != s0 or dst.stat().st_size != s0[0]:   # changed while staging: try again next cycle
        log(f"{src.name} changed while staging; will retry")
        return None
    for extra in (f"raptor_ft_fold{fold}.json", f"log_fold{fold}.txt"):
        if (work / extra).exists():
            shutil.copy2(work / extra, staging / extra)
    ep, ep_src = payload_epoch(dst, staging / f"raptor_ft_fold{fold}.json")   # from the STAGED payload, not the log
    digest = sha256_file(dst)
    started = time.time()
    files = {p.name: {"size": p.stat().st_size} for p in sorted(staging.iterdir()) if p.is_file()}
    files[dst.name]["sha256"] = digest
    manifest = {"fold": fold, "dataset": slug, "epoch": ep, "epoch_source": ep_src, "resume_file": dst.name,
                "resume_sha256": digest, "files": files, "local_publish_count": n_pub,
                "publish_started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
                "publish_started_at_epoch": started}
    atomic_write_json(staging / "payload_manifest.json", manifest)
    snap = dict(status_snapshot, payload_epoch=ep, payload_sha256=digest, staged_at=manifest["publish_started_at"])
    atomic_write_json(staging / f"persist_status_fold{fold}.json", snap)
    name = slug.split("/", 1)[1]
    title = name.replace("-", " ").replace("_", " ")
    title = (title if len(title) >= 6 else title + " resume")[:50]
    (staging / "dataset-metadata.json").write_text(json.dumps(
        {"title": title, "id": slug, "licenses": [{"name": "CC0-1.0"}]}, indent=1) + "\n")
    return {"sig": s0, "epoch": ep, "epoch_source": ep_src, "name": dst.name, "size": s0[0], "sha256": digest,
            "manifest_size": (staging / "payload_manifest.json").stat().st_size, "started_at": started,
            "version": None, "attempts": 0, "accepted_at": None}


def dataset_exists(slug):
    rc, out = kaggle(["datasets", "status", slug], timeout=120)
    if rc == 0 and not re.search(r"404|not found|forbidden|403|error", out, re.I):
        return True
    log(f"dataset {slug} not found yet (status rc={rc}); will create it")
    return False


def publish(slug, staging, note, state, delete_old):
    """Returns (accepted, output, created)."""
    if state["exists"] is None:
        state["exists"] = dataset_exists(slug)
    if not state["exists"]:
        rc, out = kaggle(["datasets", "create", "-p", str(staging), "-q"])
        if rc == 0 and not re.search(r"error|invalid|forbidden|failed", out, re.I):
            state["exists"] = True
            return True, out, True
        if re.search(r"already|exists|409|taken", out, re.I):
            state["exists"] = True      # created earlier; fall through to a new version
        else:
            return False, out, False
    args = ["datasets", "version", "-p", str(staging), "-m", note, "-q"] + (["-d"] if delete_old else [])
    rc, out = kaggle(args)
    if rc == 0 and not re.search(r"error|invalid|forbidden|failed|not found", out, re.I):
        return True, out, False
    if re.search(r"404|not found", out, re.I):
        state["exists"] = False         # create it on the next attempt
    return False, out, False


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="/content/trackA")
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--dataset", required=True, help="owner/slug of the private Kaggle dataset")
    ap.add_argument("--staging", default=None)
    ap.add_argument("--interval", type=float, default=float(os.environ.get("PERSIST_INTERVAL_S", 60)))
    ap.add_argument("--min-gap", type=float, default=float(os.environ.get("PERSIST_MIN_GAP_S", 480)), help="minimum seconds between accepted versions")
    ap.add_argument("--retry-gap", type=float, default=float(os.environ.get("PERSIST_RETRY_GAP_S", 120)), help="minimum seconds after a failed publish / verification")
    ap.add_argument("--keep-old-versions", action="store_true", help="(default behaviour; accepted for compatibility)")
    ap.add_argument("--delete-old-versions", action="store_true", help="pass -d to `kaggle datasets version` (DELETES older versions)")
    ap.add_argument("--ready-timeout", type=float, default=float(os.environ.get("PERSIST_READY_TIMEOUT_S", 600)))
    ap.add_argument("--ready-poll", type=float, default=float(os.environ.get("PERSIST_READY_POLL_S", 15)))
    ap.add_argument("--reupload-after", type=float, default=float(os.environ.get("PERSIST_REUPLOAD_AFTER_S", 1800)),
                    help="re-upload an accepted payload only if the server still does not list it this long after acceptance")
    ap.add_argument("--final-timeout", type=float, default=float(os.environ.get("PERSIST_FINAL_TIMEOUT_S", 840)))
    ap.add_argument("--clock-skew", type=float, default=float(os.environ.get("PERSIST_CLOCK_SKEW_S", 2)),
                    help="seconds of VM-vs-Kaggle clock skew tolerated when comparing creation time with publish start")
    ap.add_argument("--restored-from-dataset", action="store_true",
                    help="a resume file was restored from this dataset: treat it as verified only with a matching restored_verified marker")
    ap.add_argument("--status-file", default=None, help="default: persist_status_fold{K}.json next to the staging dir")
    a = ap.parse_args(argv)
    if not re.fullmatch(r"[a-z0-9_-]+/[a-z0-9-]{3,50}", a.dataset):
        raise SystemExit(f"--dataset must look like owner/lowercase-slug, got {a.dataset!r}")
    work = Path(a.workdir)
    staging = Path(a.staging or f"/content/resume_staging_fold{a.fold}")
    signal.signal(signal.SIGTERM, lambda *_: STOP.update(now=True))
    signal.signal(signal.SIGINT, lambda *_: STOP.update(now=True))
    signal.signal(signal.SIGUSR1, lambda *_: STOP.update(final=True))
    src = work / f"raptor_resume_fold{a.fold}.pt"
    status_path = Path(a.status_file) if a.status_file else staging.parent / f"persist_status_fold{a.fold}.json"
    ft_json = work / f"raptor_ft_fold{a.fold}.json"
    S = {"exists": None, "published_sig": None, "verified_sig": None, "acc": None, "n_pub": 0, "last_ok": 0.0,
         "last_try": 0.0, "last_verify": 0.0}
    status = {"fold": a.fold, "dataset": a.dataset, "accepted_epoch": None, "accepted_version": None,
              "verified_epoch": None, "verified_version": None, "last_error": None, "verification_attempts": 0,
              "local_publish_count": 0, "published_signature": None, "verified_signature": None, "final": None}

    def save():
        status["published_signature"] = list(S["published_sig"]) if S["published_sig"] else None
        status["verified_signature"] = list(S["verified_sig"]) if S["verified_sig"] else None
        status["local_publish_count"] = S["n_pub"]
        atomic_write_json(status_path, dict(status, timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))

    restored = sig_of(src)
    # A restored state counts as verified only if prepare left a marker whose sha256 equals the file now on disk
    # (prepare writes it after the sha256-vs-manifest and identity checks passed); otherwise it is published like any new state.
    marker = work / f"restored_verified_fold{a.fold}.json"
    trusted = False
    if restored and a.restored_from_dataset:
        mk = {}
        try:
            mk = json.load(open(marker))
            trusted = mk.get("sha256") == sha256_file(src) and isinstance(mk.get("version"), int) and mk["version"] > 0
        except (OSError, ValueError):
            trusted = False
        if not trusted:
            log(f"restored {src.name} has no matching {marker.name} (sha256 + positive dataset version required); NOT treated as verified")
    if restored and trusted:
        S["published_sig"] = S["verified_sig"] = restored
        ep, _ = payload_epoch(src, ft_json)
        status.update(accepted_epoch=ep, verified_epoch=ep, accepted_version=mk["version"], verified_version=mk["version"])
    save()
    log(f"watching {src} every {a.interval:.0f}s -> {a.dataset} (min gap {a.min_gap:.0f}s, ready timeout {a.ready_timeout:.0f}s, "
        f"old versions {'DELETED' if a.delete_old_versions else 'kept'}, status {status_path})"
        + (f"; restored state {restored[0] / 1e6:.0f} MB treated as verified (epoch {status['verified_epoch']})"
           if restored and trusted else (f"; pre-existing state {restored[0] / 1e6:.0f} MB will be published" if restored else "")))

    base_ready_timeout = a.ready_timeout

    def do_verify(final):
        acc = S["acc"]
        acc["attempts"] += 1
        status["verification_attempts"] += 1
        S["last_verify"] = time.time()
        tmo = a.ready_timeout
        kind, why = verify(a.dataset, acc, tmo, a.ready_poll, lambda: STOP["now"], a.clock_skew)
        epl = f"epoch {acc['epoch']}"
        if kind == "ready":
            ver = acc["version"] or server_version(a.dataset, acc["started_at"])
            S["verified_sig"] = acc["sig"]
            status.update(verified_epoch=acc["epoch"], verified_version=ver, last_error=None)
            if acc["version"] is None:
                status["accepted_version"] = ver
            log(f"PUBLISHED {a.dataset} READY (local #{S['n_pub']}, server version {ver}, {epl}, {acc['size'] / 1e6:.0f} MB resume state): {why}")
        else:
            status["last_error"] = f"PUBLISH-UNVERIFIED ({epl}, attempt {acc['attempts']}): {kind}: {why}"
            log(f"PUBLISH-UNVERIFIED {a.dataset} ({epl}): {kind}: {why}")
            overdue = time.time() - acc["accepted_at"] > (min(a.reupload_after, base_ready_timeout) if final else a.reupload_after)
            acc["bad_strikes"] = acc.get("bad_strikes", 0) + (1 if kind == "bad" else 0)
            # re-upload only after a confirmed failure: an error status, a bad listing on two separate verifications,
            # or a payload the server still has not listed long after acceptance (never on the first bad/stale probe)
            if kind == "error" or (kind == "bad" and acc["bad_strikes"] >= 2) or (kind in ("absent", "bad") and overdue):
                log(f"server never showed the accepted payload ({kind}, {acc['bad_strikes']} bad verification(s)); it will be re-uploaded")
                S["published_sig"] = None     # re-upload next
                S["acc"] = None
        save()

    def attempt(final=False):
        cur = sig_of(src)
        now = time.time()
        acc = S["acc"]
        # 1) an accepted payload that is not verified yet: re-VERIFY it (no re-upload)
        if acc and S["verified_sig"] != acc["sig"] and (final or now - S["last_verify"] >= a.retry_gap):
            do_verify(final)
        # 2) publish a changed state
        cur = sig_of(src)
        if cur is None or cur == S["published_sig"]:
            return
        gap = a.min_gap if S["last_ok"] else 0.0
        if not final and (time.time() - S["last_ok"] < gap or time.time() - S["last_try"] < a.retry_gap):
            return
        S["last_try"] = time.time()
        snap = dict(status)
        acc_new = stage(work, a.fold, staging, a.dataset, snap, S["n_pub"] + 1)
        if acc_new is None:
            shutil.rmtree(staging, ignore_errors=True)
            return
        note = f"epoch {acc_new['epoch']}" if acc_new["epoch"] is not None else "epoch ?"
        if acc_new["size"] > SOFT_CAP_BYTES:
            log(f"WARNING: resume state is {acc_new['size'] / 1e9:.2f} GB, over the {SOFT_CAP_BYTES / 1e9:.0f} GB per-version soft cap; uploading anyway")
        t0 = time.time()
        sv_before = server_version(a.dataset) if S["n_pub"] or dataset_exists(a.dataset) else None   # newest version BEFORE this publish
        try:
            ok, out, created = publish(a.dataset, staging, note, S, a.delete_old_versions)
        finally:
            shutil.rmtree(staging, ignore_errors=True)   # staged hard links/copies are never needed after the attempt
        tail = " | ".join(out.splitlines()[-3:])[:400]
        if not ok:
            status["last_error"] = f"publish failed ({note}): {tail[:200]}"
            save()
            log(f"PUBLISH FAILED ({note}) after {time.time() - t0:.0f}s, retrying in >= {a.retry_gap:.0f}s: {tail}")
            return
        S["n_pub"] += 1
        S["published_sig"], S["last_ok"] = acc_new["sig"], time.time()
        acc_new["accepted_at"] = time.time()
        # The CLI rarely prints the new version number and the server reports it with a lag, so it is NOT read here:
        # bind_check assigns it once the server shows a version >= min_version that carries this payload's manifest.
        acc_new["version"] = parse_version(out) or (1 if created else None)
        acc_new["min_version"] = (sv_before or 0) + 1
        S["acc"] = acc_new
        status.update(accepted_epoch=acc_new["epoch"], accepted_version=acc_new["version"])
        save()
        log(f"accepted {a.dataset} local #{S['n_pub']} (server version {acc_new['version']}, {note}, {acc_new['size'] / 1e6:.0f} MB) "
            f"in {time.time() - t0:.0f}s: {tail}; verifying")
        do_verify(final)

    def safe_attempt(final=False):
        try:
            attempt(final)
        except Exception as e:   # never let a persistence problem kill the loop or the training run
            status["last_error"] = f"persist cycle error: {type(e).__name__}: {e}"[:300]
            try:
                save()
            except Exception:
                pass
            log(f"ERROR in persist cycle: {type(e).__name__}: {e}")

    def converged():
        cur = sig_of(src)
        if cur is None:   # file gone (e.g. removed after a verified backup): nothing newer to publish
            return S["acc"] is None or S["verified_sig"] == S["acc"]["sig"]
        return cur == S["verified_sig"]

    while not STOP["now"] and not STOP["final"]:
        safe_attempt(False)
        for _ in range(int(max(a.interval, 1))):
            if STOP["now"] or STOP["final"]:
                break
            time.sleep(1)
    rc = 0
    if STOP["final"] and not STOP["now"]:
        log("final publish + verification requested")
        deadline = time.time() + a.final_timeout
        a_ready_timeout = base_ready_timeout
        while True:
            a.ready_timeout = max(5.0, min(a_ready_timeout, deadline - time.time()))
            safe_attempt(True)
            if converged() or STOP["now"] or time.time() >= deadline:
                break
            time.sleep(min(a.ready_poll, max(0.0, deadline - time.time())))
        ok = converged()
        if sig_of(src) is None and S["verified_sig"] is None and S["acc"] is None:
            reason = "no resume file present and nothing was ever published"
            state = "unverified"
        elif ok:
            state = "verified"; reason = None
        else:
            state = "unverified"; reason = status.get("last_error") or "latest resume state not verified before the final timeout"
        status["final"] = {"state": state, "reason": reason, "epoch": status["verified_epoch"], "version": status["verified_version"]}
        save()
        if state == "verified":
            log(f"FINAL PERSIST VERIFIED epoch {status['verified_epoch']} version {status['verified_version']}")
        else:
            log(f"FINAL PERSIST UNVERIFIED ({reason})")
            rc = 3
    log(f"stopped; {S['n_pub']} version(s) accepted; verified epoch {status['verified_epoch']}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
