"""Tests for the Colab Track A persistence/retry hardening (scripts/colab_tracka_{persist,run}.py, colab_tracka_folds.sh).

No Colab or Kaggle is contacted: a fake `kaggle` CLI (a Python script on PATH backed by a state dir) stands in for the
Kaggle CLI, and a stub script stands in for the Mac-side driver. Needs torch (resume payloads are torch files); skipped
without it.
"""
import json
import os
import signal
import stat
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

ROOT = Path(__file__).resolve().parents[1]
PERSIST = ROOT / "scripts" / "colab_tracka_persist.py"
RUN = ROOT / "scripts" / "colab_tracka_run.py"
FOLDS = ROOT / "scripts" / "colab_tracka_folds.sh"
DRIVER = ROOT / "scripts" / "colab_tracka.sh"

FAKE_KAGGLE = textwrap.dedent('''\
    #!%(py)s
    import json, os, shutil, sys, time
    S = os.environ["FAKE_STATE"]; MODE = os.environ.get("FAKE_MODE", "ok")
    st_path = os.path.join(S, "state.json")
    st = json.load(open(st_path)) if os.path.exists(st_path) else {"exists": False, "version": 0, "files": {}, "polls": 0, "uploads": 0}
    def save():
        json.dump(st, open(st_path, "w"))
    a = sys.argv[1:]
    if a[:2] in (["datasets", "create"], ["datasets", "version"]):
        d = a[a.index("-p") + 1]
        st["exists"] = True; st["version"] += 1; st["uploads"] += 1; st["polls"] = 0
        if MODE != "never":
            now = time.time()
            if MODE == "lag":   # Kaggle lists the previous version for a while after the new one is accepted
                st["prev_files"] = dict(st["files"]); st["lag_left"] = int(os.environ.get("FAKE_LAG_POLLS", "2"))
            for n in os.listdir(d):
                shutil.copy(os.path.join(d, n), os.path.join(S, "store_" + n))
                st["files"][n] = {"size": os.path.getsize(os.path.join(d, n)), "t": now}
        save()
        print("Dataset version is being created. Please check progress" + (" (version %%d)" %% st["version"] if MODE != "noversion" else ""))
    elif a[:2] == ["datasets", "status"]:
        if not st["exists"]:
            print("404 Not found"); sys.exit(1)
        st["polls"] += 1; save()
        ready = st["polls"] >= int(os.environ.get("FAKE_READY_AFTER", "1"))
        if "--format" in a:
            vn = None if MODE == "noversion" else st["version"]
            if MODE == "lag" and st.get("lag_left", 0) > 0 and st["version"] > 1:
                vn = st["version"] - 1   # the version counter lags too (fold3 2026-10-10 23:46 UTC: newest 2 reported as 1)
            print(json.dumps({"status": "ready" if ready else "pending", "current_version_number": vn})); sys.exit(0)
        print("ready" if ready else "pending")
    elif a[:2] == ["datasets", "files"]:
        if MODE == "garbage":
            print("this is not a listing"); sys.exit(0)
        print("name,size,creationDate")
        files = st["files"]
        if MODE == "lag" and st.get("lag_left", 0) > 0 and st.get("prev_files"):
            st["lag_left"] -= 1; save(); files = st["prev_files"]
        for n, f in sorted(files.items()):
            t = f["t"] - (3600 if MODE == "stale" else 0)
            size = f["size"] + (1 if MODE == "badsize" and n.endswith(".pt") else 0)   # a freshly created file with the wrong size
            print("%%s,%%d,%%s" %% (n, size, time.strftime("%%Y-%%m-%%d %%H:%%M:%%S", time.gmtime(t))))
    elif a[:2] == ["datasets", "download"]:
        name = a[a.index("-f") + 1]; d = a[a.index("-p") + 1]
        st["last_download_ref"] = a[2]; save()   # OWNER/SLUG or OWNER/SLUG/N (version-pinned)
        src = os.path.join(S, "store_" + name)
        if not os.path.exists(src):
            sys.exit(1)
        if MODE == "foreign" and name == "payload_manifest.json":
            m = json.load(open(src)); m["resume_sha256"] = "0" * 64
            json.dump(m, open(os.path.join(d, name), "w"))
        else:
            shutil.copy(src, os.path.join(d, name))
    elif a[:2] == ["datasets", "versions"]:
        if MODE in ("noversion", "lag"):   # 'lag' models a CLI without `datasets versions` (status --format only)
            sys.exit(1)
        print("versionNumber,creationDate"); print("%%d,%%s" %% (st["version"], time.strftime("%%Y-%%m-%%d %%H:%%M:%%S", time.gmtime())))
    else:
        print("unsupported", a); sys.exit(2)
''')


def make_fake_kaggle(bindir, state):
    bindir.mkdir(parents=True, exist_ok=True)
    exe = bindir / "kaggle"
    exe.write_text(FAKE_KAGGLE % {"py": sys.executable})
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    state.mkdir(parents=True, exist_ok=True)


def fake_state(state):
    p = state / "state.json"
    return json.loads(p.read_text()) if p.exists() else {}


def wait_for(pred, timeout=40.0, step=0.2):
    end = time.time() + timeout
    while time.time() < end:
        v = pred()
        if v:
            return v
        time.sleep(step)
    return None


def write_resume(work, fold, epoch, pad=2048):
    work.mkdir(parents=True, exist_ok=True)
    tmp = work / "tmp_resume.pt"
    torch.save({"epoch": epoch, "model": {"w": torch.zeros(pad)}}, tmp)
    os.replace(tmp, work / f"raptor_resume_fold{fold}.pt")
    (work / f"raptor_ft_fold{fold}.json").write_text(json.dumps({"hist": [{"ep": epoch}]}))
    # a log that DISAGREES with the payload: the reported epoch must come from the staged payload
    (work / f"log_fold{fold}.txt").write_text(f"ep{epoch + 5} loss 0.1\n  [resume] state saved after ep{epoch + 5}\n")


@unittest.skipIf(torch is None, "torch not installed")
class PersisterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        self.work, self.state, self.bin = t / "work", t / "state", t / "bin"
        make_fake_kaggle(self.bin, self.state)
        self.status = t / "persist_status_fold0.json"
        self.procs = []

    def tearDown(self):
        for p in self.procs:
            if p.poll() is None:
                p.kill()
        self.tmp.cleanup()

    def start(self, mode="ok", ready_after="1", extra=()):
        env = dict(os.environ, PATH=f"{self.bin}:{os.environ['PATH']}", FAKE_STATE=str(self.state), FAKE_MODE=mode,
                   FAKE_READY_AFTER=ready_after, KAGGLE_CONFIG_DIR=str(self.tmp.name))
        p = subprocess.Popen([sys.executable, str(PERSIST), "--workdir", str(self.work), "--fold", "0", "--dataset", "own/fake-resume-ds",
                              "--staging", str(Path(self.tmp.name) / "stg"), "--status-file", str(self.status), "--interval", "1",
                              "--min-gap", "0", "--retry-gap", "0", "--ready-poll", "0.2", *extra],
                             env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.procs.append(p)
        return p

    def st(self):
        try:
            return json.loads(self.status.read_text())
        except (OSError, ValueError):
            return {}

    def test_exact_version_verified_with_manifest_and_payload_epoch(self):
        write_resume(self.work, 0, 4)
        p = self.start(ready_after="3", extra=["--ready-timeout", "20"])
        st = wait_for(lambda: self.st().get("verified_epoch") == 4 and self.st())
        self.assertTrue(st, p.poll())
        self.assertEqual((st["accepted_epoch"], st["accepted_version"], st["verified_version"]), (4, 1, 1))   # item 1/2
        self.assertEqual(st["published_signature"], st["verified_signature"])
        self.assertIsNone(st["last_error"])
        manifest = json.loads((self.state / "store_payload_manifest.json").read_text())
        self.assertEqual(manifest["epoch"], 4)                     # item 5: payload epoch, NOT the log's ep9
        self.assertEqual(manifest["epoch_source"], "resume_dict")
        self.assertEqual(len(manifest["resume_sha256"]), 64)
        self.assertIn("publish_started_at", manifest)
        self.assertIn("raptor_resume_fold0.pt", manifest["files"])
        staged_status = json.loads((self.state / "store_persist_status_fold0.json").read_text())
        self.assertEqual(staged_status["payload_epoch"], 4)         # what the loop downloads from the dataset
        write_resume(self.work, 0, 5)                               # newer epoch -> second version, version number parsed
        st = wait_for(lambda: self.st().get("verified_epoch") == 5 and self.st())
        self.assertTrue(st)
        self.assertEqual(st["verified_version"], 2)
        p.send_signal(signal.SIGUSR1)
        out, _ = p.communicate(timeout=60)
        self.assertEqual(p.returncode, 0, out)
        self.assertIn("FINAL PERSIST VERIFIED epoch 5 version 2", out)
        self.assertEqual(self.st()["final"]["state"], "verified")

    def test_unretrievable_server_version_never_verifies(self):
        """Sol 2026-10-10 22:38 UTC: binding requires a positive exact server version; a CLI that reports none (no version in
        the create output, status JSON without a number, no `datasets versions`) must not qualify the payload."""
        write_resume(self.work, 0, 2)
        p = self.start(mode="noversion", extra=["--ready-timeout", "1.5", "--reupload-after", "1000", "--final-timeout", "3"])
        st = wait_for(lambda: self.st().get("verification_attempts", 0) >= 1 and self.st(), timeout=30)
        self.assertTrue(st, p.poll())
        self.assertIsNone(st["verified_epoch"])
        self.assertIn("positive exact version required", st["last_error"] or "")
        p.send_signal(signal.SIGUSR1)
        out, _ = p.communicate(timeout=60)
        self.assertEqual(p.returncode, 3, out)
        self.assertEqual(fake_state(self.state)["uploads"], 1)   # not re-uploaded: the payload is not bad, just unqualified

    def test_never_listed_stays_unverified_and_is_reverified_not_reuploaded(self):
        write_resume(self.work, 0, 2)
        p = self.start(mode="never", extra=["--ready-timeout", "1.5", "--reupload-after", "1000", "--final-timeout", "4"])
        st = wait_for(lambda: self.st().get("verification_attempts", 0) >= 2 and self.st(), timeout=40)
        self.assertTrue(st)
        self.assertEqual(st["accepted_epoch"], 2)                  # item 2: accepted ...
        self.assertIsNone(st["verified_epoch"])                    # ... but never verified
        self.assertIn("PUBLISH-UNVERIFIED", st["last_error"])
        self.assertEqual(fake_state(self.state)["uploads"], 1)     # re-verified, NOT re-uploaded
        p.send_signal(signal.SIGUSR1)
        out, _ = p.communicate(timeout=60)
        self.assertEqual(p.returncode, 3, out)
        self.assertIn("FINAL PERSIST UNVERIFIED", out)
        self.assertEqual(self.st()["final"]["state"], "unverified")

    def test_listing_not_bound_to_payload_manifest_is_not_ready(self):
        write_resume(self.work, 0, 3)
        p = self.start(mode="foreign", extra=["--ready-timeout", "1.5", "--reupload-after", "1000", "--final-timeout", "3"])
        st = wait_for(lambda: self.st().get("verification_attempts", 0) >= 1 and self.st(), timeout=30)
        self.assertTrue(st)
        self.assertIsNone(st["verified_epoch"])
        self.assertIn("not bound", st["last_error"])
        p.send_signal(signal.SIGUSR1)
        out, _ = p.communicate(timeout=60)
        self.assertEqual(p.returncode, 3, out)

    def test_restored_state_needs_matching_marker_to_count_as_verified(self):
        write_resume(self.work, 0, 6)
        p = self.start(mode="never", extra=["--restored-from-dataset", "--ready-timeout", "1.5", "--reupload-after", "1000"])
        time.sleep(3)
        p.terminate(); out, _ = p.communicate(timeout=30)
        self.assertIsNone(self.st().get("verified_epoch"), out)       # no marker: not verified (it gets republished instead)
        self.assertIn("NOT treated as verified", out)
        for f in (self.status, self.state / "state.json"):
            f.exists() and f.unlink()
        import hashlib
        digest = hashlib.sha256((self.work / "raptor_resume_fold0.pt").read_bytes()).hexdigest()
        (self.work / "restored_verified_fold0.json").write_text(json.dumps({"sha256": digest}))   # no version: not trusted
        p = self.start(mode="never", extra=["--restored-from-dataset", "--ready-timeout", "1.5", "--reupload-after", "1000"])
        time.sleep(3)
        p.terminate(); out, _ = p.communicate(timeout=30)
        self.assertIsNone(self.st().get("verified_epoch"), out)
        self.assertIn("positive dataset version required", out)
        for f in (self.status, self.state / "state.json"):
            f.exists() and f.unlink()
        (self.work / "restored_verified_fold0.json").write_text(json.dumps({"sha256": digest, "version": 9}))
        p = self.start(mode="never", extra=["--restored-from-dataset"])
        st = wait_for(lambda: self.st().get("verified_epoch") == 6 and self.st(), timeout=20)
        p.terminate(); p.communicate(timeout=30)
        self.assertTrue(st)
        self.assertEqual((st["accepted_version"], st["verified_version"]), (9, 9))

    def test_lagged_listing_of_previous_version_is_not_bad_and_is_not_reuploaded(self):
        """Sol 2026-10-10 21:43 UTC live fold2 report: after a new version is accepted the files listing still shows the
        PREVIOUS version's file (different size, older creation time). That is publication lag, not a bad upload: the
        verifier must keep polling, verify the accepted payload once the listing catches up, and never re-upload it."""
        write_resume(self.work, 0, 1)
        os.environ["FAKE_LAG_POLLS"] = "3"
        try:
            p = self.start(mode="lag", extra=["--ready-timeout", "20", "--reupload-after", "1000"])
            st = wait_for(lambda: self.st().get("verified_epoch") == 1 and self.st(), timeout=30)
            self.assertTrue(st, p.poll())
            self.assertEqual(fake_state(self.state)["uploads"], 1)
            self.assertEqual((st["accepted_version"], st["verified_version"]), (1, 1))   # via status --format json
            write_resume(self.work, 0, 2)   # second version: listing lags again, showing epoch 1's (smaller/older) file
            st = wait_for(lambda: self.st().get("verified_epoch") == 2 and self.st(), timeout=30)
            self.assertTrue(st, p.poll())
            self.assertEqual(fake_state(self.state)["uploads"], 2)       # no re-upload of an accepted payload
            self.assertEqual(st["verified_version"], 2)
            self.assertEqual(fake_state(self.state)["last_download_ref"], "own/fake-resume-ds/2")   # manifest fetched from the exact version
            self.assertIsNone(st["last_error"])
            p.terminate(); out, _ = p.communicate(timeout=30)
            self.assertNotIn("will be re-uploaded", out)
        finally:
            os.environ.pop("FAKE_LAG_POLLS", None)

    def test_check_files_classifies_size_mismatch_by_creation_time(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("persist_mod", PERSIST)
        P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
        start = 1_800_000_000
        old_t = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(start - 600))
        new_t = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(start + 60))
        def fake(listing):
            P.kaggle = lambda args, timeout=0: (0, listing)
        fake(f"name,size,creationDate\nraptor_resume_fold0.pt,100,{old_t}\npayload_manifest.json,10,{old_t}\n")
        self.assertEqual(P.check_files("o/s", {"raptor_resume_fold0.pt": 120, "payload_manifest.json": 11}, start, 2)[0], "stale")
        fake(f"name,size,creationDate\nraptor_resume_fold0.pt,100,{new_t}\npayload_manifest.json,11,{new_t}\n")
        self.assertEqual(P.check_files("o/s", {"raptor_resume_fold0.pt": 120, "payload_manifest.json": 11}, start, 2)[0], "bad")
        fake(f"name,size,creationDate\nraptor_resume_fold0.pt,120,{new_t}\npayload_manifest.json,11,{new_t}\n")
        self.assertEqual(P.check_files("o/s", {"raptor_resume_fold0.pt": 120, "payload_manifest.json": 11}, start, 2)[0], "ok")

    def test_server_version_prefers_status_json_then_versions_csv(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("persist_mod2", PERSIST)
        P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
        calls = []
        def k(args, timeout=0):
            calls.append(args[:3])
            if args[1] == "status" and "--format" in args:
                return 0, '{"status": "ready", "current_version_number": 7}'
            return 1, ""
        P.kaggle = k
        self.assertEqual(P.server_version("o/s"), 7)
        def k2(args, timeout=0):
            if args[1] == "status":
                return 0, "ready"            # old CLI: ignores --format, prints plain status
            if args[1] == "versions":
                return 0, "versionNumber,creationDate\n3,2026-10-10 12:00:00\n"
            return 1, ""
        P.kaggle = k2
        self.assertEqual(P.server_version("o/s"), 3)
        P.kaggle = lambda args, timeout=0: (1, "")
        self.assertIsNone(P.server_version("o/s"))

    def test_bad_listing_needs_two_verifications_before_reupload(self):
        """A file created by this publish with the wrong size is 'bad', but one bad verification does not discard the
        accepted payload; the second one triggers the re-upload."""
        write_resume(self.work, 0, 1)
        p = self.start(mode="badsize", extra=["--ready-timeout", "2", "--retry-gap", "4", "--reupload-after", "1000"])
        st = wait_for(lambda: self.st().get("verification_attempts", 0) >= 1 and "bad" in (self.st().get("last_error") or "") and self.st(), timeout=30)
        self.assertTrue(st, p.poll())
        self.assertEqual(fake_state(self.state)["uploads"], 1)       # not re-uploaded after the first bad verification
        self.assertIsNone(st["verified_epoch"])
        wait_for(lambda: fake_state(self.state).get("uploads", 0) >= 2, timeout=40)
        p.terminate(); out, _ = p.communicate(timeout=30)
        self.assertIn("2 bad verification(s)", out)
        self.assertGreaterEqual(fake_state(self.state)["uploads"], 2)   # re-uploaded after the second

    def test_stale_or_unparseable_listing_is_not_ready(self):
        for mode in ("stale", "garbage"):
            with self.subTest(mode=mode):
                for stale in (self.state / "state.json", self.status):
                    if stale.exists():
                        stale.unlink()
                write_resume(self.work, 0, 1)
                p = self.start(mode=mode, extra=["--ready-timeout", "1.5", "--reupload-after", "1000", "--final-timeout", "3"])
                st = wait_for(lambda: self.st().get("verification_attempts", 0) >= 1 and self.st(), timeout=30)
                self.assertTrue(st)
                self.assertIsNone(st["verified_epoch"])
                p.send_signal(signal.SIGUSR1)
                out, _ = p.communicate(timeout=60)
                self.assertEqual(p.returncode, 3, out)


@unittest.skipIf(torch is None, "torch not installed")
class RunFinalBackupTests(unittest.TestCase):
    ENTRY = textwrap.dedent('''\
        import json, os, sys, time, torch
        w = os.environ["WORKDIR"]
        def save(ep):
            torch.save({"epoch": ep, "model": {"w": torch.zeros(1024)}}, w + "/t.pt"); os.replace(w + "/t.pt", w + "/raptor_resume_fold0.pt")
            open(w + "/raptor_ft_fold0.json", "w").write(json.dumps({"hist": [{"ep": ep}]}))
        save(10); time.sleep(3); save(11)
        print("entry done", flush=True)
        ''')

    def run_case(self, mode):
        with tempfile.TemporaryDirectory() as td:
            t = Path(td)
            make_fake_kaggle(t / "bin", t / "state")
            content = t / "content"
            (content / "trackA_pkg").mkdir(parents=True)
            (content / "trackA_pkg" / "tracka_entry.py").write_text(self.ENTRY)
            (content / "colab_tracka_persist.py").write_text(PERSIST.read_text())
            (content / "tracka_params.json").write_text(json.dumps(
                {"FOLD_A": "0", "FOLD_B": "", "SMOKE": "0", "EPOCHS": "12", "RESUME_DATASET": "own/fake-resume-ds"}))
            env = dict(os.environ, PATH=f"{t / 'bin'}:{os.environ['PATH']}", FAKE_STATE=str(t / "state"), FAKE_MODE=mode,
                       TRACKA_CONTENT=str(content), PERSIST_INTERVAL_S="1", PERSIST_MIN_GAP_S="0", PERSIST_RETRY_GAP_S="0",
                       PERSIST_READY_POLL_S="0.2", PERSIST_READY_TIMEOUT_S="3", PERSIST_FINAL_TIMEOUT_S="6", FINAL_BACKUP_WAIT_S="30",
                       KAGGLE_CONFIG_DIR=str(t))
            p = subprocess.run([sys.executable, str(RUN)], env=env, capture_output=True, text=True, timeout=180)
            try:
                status = json.loads((content / "persist_status_fold0.json").read_text())
            except (OSError, ValueError):
                status = {}
            return p, status, (content / "trackA" / "raptor_resume_fold0.pt").exists()

    def test_final_backup_verified_before_ok(self):
        p, status, _ = self.run_case("ok")
        out = p.stdout + p.stderr
        self.assertEqual(p.returncode, 0, out)
        self.assertRegex(out, r"FINAL BACKUP VERIFIED epoch 11 version \d+")
        self.assertIn("TRACKA_RUN_OK", out)
        self.assertLess(out.index("FINAL BACKUP VERIFIED"), out.index("TRACKA_RUN_OK"))
        self.assertEqual(status["verified_epoch"], 11)

    def test_unverified_final_backup_fails_run_and_keeps_resume(self):
        p, status, resume_exists = self.run_case("never")
        out = p.stdout + p.stderr
        self.assertNotEqual(p.returncode, 0, out)
        self.assertIn("FINAL BACKUP UNVERIFIED", out)
        self.assertNotIn("TRACKA_RUN_OK", out)
        self.assertTrue(resume_exists)
        self.assertIsNone(status["verified_epoch"])


class PrepareProvenanceTests(unittest.TestCase):
    def ns(self):
        src = (ROOT / "scripts" / "colab_tracka_prepare.py").read_text().rstrip()
        assert src.endswith("\nmain()")
        ns = {}
        exec(compile(src[:-len("main()")], "prepare", "exec"), ns)
        return ns

    def test_check_payload_gates_restore(self):
        ns = self.ns()
        with tempfile.TemporaryDirectory() as td:
            rp = Path(td) / "r.pt"; rp.write_bytes(b"abc")
            good = ns["sha256"](rp)
            cur = {"script_sha256": "s", "labels_sha256": "l"}
            ident = {"fold": 2, "script_sha256": "s", "labels_sha256": "l"}
            self.assertEqual(ns["check_payload"]({"fold": 2, "resume_sha256": good}, rp, "2", ident, cur), [])
            self.assertTrue(ns["check_payload"]({"fold": 2, "resume_sha256": "0" * 64}, rp, "2", ident, cur))      # hash
            self.assertTrue(ns["check_payload"]({"fold": 2, "resume_sha256": good}, rp, "1", dict(ident, fold=1), cur))   # manifest fold
            self.assertTrue(ns["check_payload"]({"fold": 2, "resume_sha256": good}, rp, "2", dict(ident, labels_sha256="x"), cur))
            self.assertTrue(ns["check_payload"](None, rp, "2", ident, cur))                                      # no manifest

    @unittest.skipIf(torch is None, "torch unavailable")
    def test_failed_pinned_restore_never_downloads_unpinned_dataset(self):
        ns = self.ns()
        with tempfile.TemporaryDirectory() as td:
            params = Path(td) / "params.json"
            params.write_text(json.dumps({"FOLD_A": "3", "FOLD_B": ""}))
            ns["PARAMS"] = str(params)
            ns["WORKDIR"] = str(Path(td) / "trackA")
            ns["dataset_version"] = lambda slug: ("ready", 13)
            calls = []
            def fail_download(args, *extra, **kwargs):
                calls.append(args)
                return 1, 0, 0
            ns["kaggle"] = fail_download
            with self.assertRaisesRegex(RuntimeError, "could not download resume dataset"):
                ns["restore_resume"]("own/resume")
            self.assertEqual(len(calls), 2)
            self.assertTrue(all("own/resume/13" in args for args in calls))
            self.assertFalse(any("own/resume" in args for args in calls))


class FoldLoopTests(unittest.TestCase):
    STUB_DRIVER = textwrap.dedent('''\
        #!/usr/bin/env bash
        echo "$@" >> "$STUB_DIR/calls.txt"
        n=$(wc -l < "$STUB_DIR/calls.txt" | tr -d ' ')
        rc=$(echo "$STUB_RCS" | tr ',' '\\n' | sed -n "${n}p")
        exit "${rc:-0}"
        ''')
    # serves files from $STUB_DIR/ds (the "dataset"); exit 1 when the file is absent
    STUB_KAGGLE = textwrap.dedent('''\
        #!/usr/bin/env bash
        if [ "$1" = datasets ] && [ "$2" = status ]; then   # version-bound verification: READY + positive exact version
          echo '{"status": "ready", "current_version_number": '"${STUB_VERSION:-4}"'}'; exit 0
        fi
        while [ $# -gt 0 ]; do case "$1" in -f) f="$2"; shift 2 ;; -p) d="$2"; shift 2 ;; *) shift ;; esac; done
        [ -f "$STUB_DIR/ds/$f" ] || exit 1
        cp "$STUB_DIR/ds/$f" "$d/$f"
        ''')

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        for name, body in (("driver.sh", self.STUB_DRIVER), ("kaggle", self.STUB_KAGGLE)):
            (self.t / name).write_text(body)
            (self.t / name).chmod(0o755)
        (self.t / "calls.txt").write_text("")
        self.lock = self.t / "x.lock"
        (self.t / "ds").mkdir()
        self.pkg = self.t / "pkg"
        self.pkg.mkdir()
        for n, c in (("train_knee_patched.py", "# script"), ("folds_v0.json", "{}"), ("excluded_ids.json", "[]")):
            (self.pkg / n).write_text(c)
        (self.t / "labels.parquet").write_bytes(b"labels")
        import hashlib
        h = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
        self.ident = {"fold": 0, "script_sha256": h(self.pkg / "train_knee_patched.py"), "labels_sha256": h(self.t / "labels.parquet"),
                      "fold_file_sha256": h(self.pkg / "folds_v0.json"), "excluded_ids_sha256": h(self.pkg / "excluded_ids.json"),
                      "n_folds": 5, "tag": "fold0", "oof_mask_sha256": "a" * 64, "corpus_ids_sha256": {"all_ids.npy": "b" * 64},
                      "config": {"arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "k": 12, "k_eval": 24, "bs": 8, "grad_ckpt": True}}

    def tearDown(self):
        self.tmp.cleanup()

    def payload(self, epoch, hist_len=None, ident=None, topk=1, tamper=False, manifest_fold=0, drop=(), hist=None, topk_list=None, cfg=None):
        """Write a synthetic dataset payload (resume file + payload_manifest.json) into the stub dataset."""
        if torch is None:
            self.skipTest("torch not installed")
        import hashlib
        rp = self.t / "ds" / "raptor_resume_fold0.pt"
        identity = dict(self.ident, **(ident or {}))
        if cfg:
            identity["config"] = dict(identity["config"], **cfg)
        state = {"epoch": epoch, "model": {"w": 1}, "opt": {"s": 1}, "sched": {"last_epoch": epoch}, "best": 0.5, "best_state": {"epoch": 0},
                 "best_P": [0.5], "hist": hist if hist is not None else [{"ep": i} for i in range(epoch + 1 if hist_len is None else hist_len)],
                 "topk": topk_list if topk_list is not None else [{"model": {"w": 1}, "sel": 0.5, "epoch": i} for i in range(topk)],
                 "identity": identity}
        for k in drop:
            state.pop(k)
        torch.save(state, rp)
        digest = hashlib.sha256(rp.read_bytes()).hexdigest()
        if tamper:
            digest = "0" * 64
        (self.t / "ds" / "payload_manifest.json").write_text(json.dumps(
            {"fold": manifest_fold, "epoch": epoch, "resume_file": rp.name, "resume_sha256": digest}))

    def loop(self, rcs, epoch, folds=("0",), extra_env=None, args=()):
        """epoch: int -> a correct payload at that epoch (12-epoch run); None -> empty dataset; "keep" -> payload already written."""
        if isinstance(epoch, int):
            self.payload(epoch)
        env = dict(os.environ, STUB_DIR=str(self.t), STUB_RCS=rcs, FOLDS_DRIVER=str(self.t / "driver.sh"),
                   FOLDS_KAGGLE=str(self.t / "kaggle"), FOLDS_LOCK_FILE=str(self.lock), FOLDS_SLEEP_S="0", FOLDS_EPOCHS="12",
                   FOLDS_PYTHON=sys.executable, FOLDS_PKG_DIR=str(self.pkg), FOLDS_LABELS_FILE=str(self.t / "labels.parquet"))
        env.update(extra_env or {})
        p = subprocess.run(["bash", str(FOLDS), *args, *folds], env=env, capture_output=True, text=True, timeout=120)
        calls = [l for l in (self.t / "calls.txt").read_text().splitlines() if l]
        return p, calls

    def test_wrong_fold_identity_stops(self):
        self.payload(11, ident={"fold": 3})
        p, calls = self.loop_existing("75,0")
        self.assertEqual(p.returncode, 75, p.stdout)
        self.assertEqual(len(calls), 1)
        self.assertIn("identity.fold", p.stdout)
        self.assertNotIn("TRAINED", p.stdout)

    def test_hash_mismatch_stops(self):
        self.payload(11, tamper=True)
        p, calls = self.loop_existing("75,0")
        self.assertEqual((p.returncode, len(calls)), (75, 1))
        self.assertIn("sha256", p.stdout)

    def test_unversioned_dataset_stops_even_with_a_trained_payload(self):
        """Sol 2026-10-10 22:38 UTC: a restore/verification must be bound to a positive exact dataset version."""
        self.payload(11)
        os.environ["STUB_VERSION"] = "null"
        try:
            p, calls = self.loop_existing("75,0")
        finally:
            os.environ.pop("STUB_VERSION", None)
        self.assertEqual((p.returncode, len(calls)), (75, 1), p.stdout)
        self.assertIn("positive exact version", p.stdout)
        self.assertNotIn("TRAINED (final", p.stdout)

    def test_foreign_labels_identity_stops(self):
        self.payload(11, ident={"labels_sha256": "f" * 64})
        p, calls = self.loop_existing("75,0")
        self.assertEqual((p.returncode, len(calls)), (75, 1))
        self.assertIn("identity.labels_sha256", p.stdout)

    def test_staged_status_file_alone_is_not_trusted(self):
        (self.t / "ds" / "persist_status_fold0.json").write_text('{"payload_epoch": 11, "verified_epoch": 11}')
        p, calls = self.loop("75,0", None)
        self.assertEqual((p.returncode, len(calls)), (75, 1))
        self.assertNotIn("TRAINED", p.stdout)

    def assert_stopped(self, reason):
        p, calls = self.loop_existing("75,0")
        self.assertEqual((p.returncode, len(calls)), (75, 1), p.stdout)
        self.assertIn("STOPPED", p.stdout)
        self.assertNotIn("TRAINED (final", p.stdout)
        self.assertIn(reason, p.stdout)

    # ---- probes from outputs/analysis/colab-payload-rereview-20261010/report.json (each returned TRAINED/RESUME before)
    def test_probe_missing_model_opt_sched_stops(self):
        self.payload(11, drop=("model", "opt", "sched"))
        self.assert_stopped("missing keys")

    def test_probe_foreign_config_stops(self):
        for cfg, key in (({"arch": "convnext_tiny"}, "arch"), ({"res": 256}, "res"), ({"k": 8}, "config.k")):
            with self.subTest(cfg=cfg):
                (self.t / "calls.txt").write_text("")
                self.payload(11, cfg=cfg)
                self.assert_stopped(f"identity.config.{key.split('.')[-1]}")

    def test_probe_foreign_n_folds_or_tag_stops(self):
        self.payload(11, ident={"n_folds": 3})
        self.assert_stopped("identity.n_folds")

    def test_probe_duplicate_history_stops(self):
        self.payload(11, hist=[{"ep": 11}] * 12)
        self.assert_stopped("history is not exactly")

    def test_probe_empty_topk_state_stops(self):
        self.payload(11, topk_list=[{}])
        self.assert_stopped("top-k")

    def test_probe_final_epoch_short_history_stops_not_resume(self):
        self.payload(11, hist_len=7)
        self.assert_stopped("history is not exactly")

    def test_early_epoch_malformed_history_stops(self):
        self.payload(5, hist=[{"ep": 5}] * 6)
        self.assert_stopped("history is not exactly")

    def test_probe_driver_time_budget_forced_off(self):
        p, calls = self.loop("75,0", 11, extra_env={"TIME_BUDGET_H": "0.01", "FOLDS_DRIVER_ARGS": "--time-budget-h 0.01"})
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("TRAINED", p.stdout)
        self.assertEqual(calls[0].count("--time-budget-h"), 1)
        self.assertIn("--time-budget-h 0 ", calls[0] + " ")
        self.assertNotIn("0.01", calls[0])

    def test_probe_persist_epoch_never_from_ft_json(self):
        src = PERSIST.read_text()
        ns = {}
        exec(compile(src[:src.index("\ndef parse_created")], "persist", "exec"), ns)
        with tempfile.TemporaryDirectory() as td:
            bad, ft = Path(td) / "r.pt", Path(td) / "ft.json"
            bad.write_bytes(b"not a torch file"); ft.write_text(json.dumps({"hist": [{"ep": 9}]}))
            ep, _ = ns["payload_epoch"](bad, ft)
            self.assertIsNone(ep)

    def test_probe_prepare_rejects_malformed_state(self):
        src = (ROOT / "scripts" / "colab_tracka_prepare.py").read_text()
        self.assertIn("history is not exactly epochs", src)
        self.assertIn("manifest.epoch", src)
        self.assertNotIn("WARNING history length", src)

    def loop_existing(self, rcs):
        return self.loop(rcs, "keep")

    def test_rc75_with_final_epoch_verified_is_trained_and_moves_on(self):
        p, calls = self.loop("75,0", 11, folds=("0", "1"))
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("FOLD 0 TRAINED (final state verified in dataset); export pending", p.stdout)
        self.assertEqual(len(calls), 2)                      # fold 0 once (no new session), fold 1 once
        self.assertIn("--fold-a 0", calls[0]); self.assertIn("--fold-a 1", calls[1])
        self.assertFalse(self.lock.exists())

    def test_rc75_with_early_epoch_relaunches_with_resume_and_strict_identity(self):
        p, calls = self.loop("75,0", 5)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertEqual(len(calls), 2)
        self.assertNotIn("--resume-from-dataset", calls[0])
        self.assertNotIn("--strict-identity", calls[0])
        self.assertIn("--resume-from-dataset ozgung/knee-colab-resume-c384r-fold0", calls[1])
        self.assertIn("--strict-identity", calls[1])
        self.assertRegex(p.stdout, r"\[\d{4}-\d\d-\d\dT[\d:]+Z\] FOLD 0 DONE")

    def test_other_rc_stops_loop_without_retry(self):
        p, calls = self.loop("3,0", 5, folds=("0", "1"))
        self.assertEqual(p.returncode, 3)
        self.assertEqual(len(calls), 1)
        self.assertIn("FOLD 0 FAILED rc=3", p.stdout)
        self.assertNotIn("ALL FOLDS LOOP FINISHED", p.stdout)

    def test_exhausted_attempts_fail_overall(self):
        p, calls = self.loop("75,75,75,75", 5)
        self.assertEqual(p.returncode, 1)
        self.assertEqual(len(calls), 3)
        self.assertIn("EXHAUSTED", p.stdout)
        self.assertNotIn("ALL FOLDS LOOP FINISHED", p.stdout)

    def test_unknown_dataset_epoch_stops_instead_of_blind_retry(self):
        p, calls = self.loop("75,0", None)
        self.assertEqual(p.returncode, 75)
        self.assertEqual(len(calls), 1)

    def test_continue_flag_resumes_first_attempt(self):
        p, calls = self.loop("0", 5, args=("--continue",))
        self.assertEqual(p.returncode, 0)
        self.assertIn("--resume-from-dataset", calls[0]); self.assertIn("--strict-identity", calls[0])

    def test_live_lock_refuses_and_stale_lock_is_replaced(self):
        sleeper = subprocess.Popen(["sleep", "30"])
        try:
            self.lock.write_text(f"pid={sleeper.pid}\nstarted=x\nsessions=a\n")
            p, calls = self.loop("0", 5)
            self.assertEqual(p.returncode, 3)
            self.assertEqual(calls, [])
            self.assertIn("REFUSING TO START", p.stdout)
            self.assertTrue(self.lock.exists())              # not removed by the refused instance
        finally:
            sleeper.kill(); sleeper.wait()
        self.lock.write_text("pid=999999\nstarted=x\nsessions=a\n")   # dead pid
        p, calls = self.loop("0", 5)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("removing stale lock", p.stdout)
        self.assertFalse(self.lock.exists())

    def test_driver_header_prints_lock_owner(self):
        self.lock.write_text(f"pid={os.getpid()}\nstarted=T\nsessions=s1,s2\n")
        env = dict(os.environ, TRACKA_LOCK_FILE=str(self.lock))
        p = subprocess.run(["bash", str(DRIVER), "--dry-run", "--session", "x1", "--fold-a", "0"], env=env, capture_output=True,
                           text=True, timeout=60)
        self.assertIn("lock owner (live)", p.stdout, p.stdout + p.stderr)
        self.assertIn("sessions=s1,s2", p.stdout)


if __name__ == "__main__":
    unittest.main()
