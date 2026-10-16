"""Restore the s6v2 pixel cache on a fresh VM from the private Kaggle backups (index dataset + 17 shard datasets) into
/content/px, verify every shard and index file against the sha256s bound in the original preflight receipt
(/content/preflight_ref.json), then write FINALIZED.  Fails closed.  System kaggle CLI; prints counts only."""
import hashlib, json, os, subprocess, sys, time, zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PX = Path("/content/px"); BASE = "ozgung/knee-s6-px-v4-320-20261015"
ENV = dict(os.environ, KAGGLE_CONFIG_DIR="/content")
REF = json.load(open("/content/preflight_ref.json"))


def log(*a):
    print(time.strftime("%H:%M:%SZ", time.gmtime()), *a, flush=True)


def fsha(p):
    m = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            m.update(b)
    return m.hexdigest()


def fetch(ref, name):
    dst = PX / name
    for i in range(5):
        if dst.exists():
            return
        r = subprocess.run(["/usr/local/bin/kaggle", "datasets", "download", ref, "-f", name, "-p", str(PX), "-q"], env=ENV,
                           capture_output=True, text=True)
        z = PX / (name + ".zip")
        if z.exists():
            with zipfile.ZipFile(z) as zf:
                zf.extract(name, PX)
            z.unlink()
        if dst.exists():
            return
        log("retry", name, r.returncode, (r.stdout + r.stderr)[-200:].replace("\n", " "))
        time.sleep(30 * (i + 1))
    raise RuntimeError("download failed: " + name)


def main():
    PX.mkdir(exist_ok=True)
    for f in ("index.json", "px.png.off.npy", "px.png.shards.json", "meta.f32.npy"):
        fetch(BASE, f)
    bad = [f for f in ("index.json", "px.png.off.npy", "px.png.shards.json", "meta.f32.npy") if fsha(PX / f) != REF["bound_sha256"][f]]
    if bad:
        log("RESTORE_FAIL index files differ", bad); sys.exit(1)
    shards = json.load(open(PX / "px.png.shards.json"))
    jobs = [("%s-s%s" % (BASE, x["file"].rsplit(".", 1)[-1]), x["file"]) for x in shards]
    t0 = time.time()
    with ThreadPoolExecutor(4) as ex:
        for f in [ex.submit(fetch, r, n) for r, n in jobs]:
            f.result()
    log("shards downloaded", len(jobs), "%.0f s" % (time.time() - t0))
    with ThreadPoolExecutor(6) as ex:
        hs = dict(zip([n for _, n in jobs], ex.map(lambda n: fsha(PX / n), [n for _, n in jobs])))
    bad = [n for n, h in hs.items() if h != REF["shard_sha256"][n]]
    if bad:
        log("RESTORE_FAIL shard sha256 differs", len(bad)); sys.exit(1)
    (PX / "FINALIZED").write_text("restored from private Kaggle backup; all shard sha256 match preflight_ref\n")
    log("RESTORE_OK shards", len(hs), "all sha256 match the original preflight")


if __name__ == "__main__":
    main()
