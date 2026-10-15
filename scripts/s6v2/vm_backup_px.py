"""Incremental private-Kaggle backup of the s6v2 pixel cache: every completed PNG shard becomes its own private dataset
ozgung/knee-s6-px-v4-320-20261015-sNNN (hard link, no copy); when the build is final, the small index files go to
ozgung/knee-s6-px-v4-320-20261015.  Uses the VM's system kaggle CLI (private by default).  Log: /content/backup_px.log.
Restore: download every -sNNN dataset and the index dataset into one directory."""
import json, os, subprocess, time
from pathlib import Path

PX = Path("/content/px"); BK = Path("/content/bk_px"); BASE = "ozgung/knee-s6-px-v4-320-20261015"
KAGGLE = ["nice", "-n", "15", "/usr/local/bin/kaggle"]
ENV = dict(os.environ, KAGGLE_CONFIG_DIR="/content")
done_path = BK / "uploaded.json"


def log(*a):
    print(time.strftime("%H:%M:%SZ", time.gmtime()), *a, flush=True)


def create(folder, ref, title):
    (folder / "dataset-metadata.json").write_text(json.dumps({"title": title, "id": ref, "licenses": [{"name": "other"}]}))
    for i in range(4):
        r = subprocess.run(KAGGLE + ["datasets", "create", "-p", str(folder), "--dir-mode", "skip", "-q"], env=ENV,
                           capture_output=True, text=True)
        if r.returncode == 0:
            return True
        log("create failed", ref.split("/")[-1], r.returncode, (r.stdout + r.stderr)[-300:].replace("\n", " "))
        time.sleep(60 * (i + 1))
    return False


def main():
    BK.mkdir(exist_ok=True)
    done = set(json.load(open(done_path))) if done_path.exists() else set()
    while True:
        try:
            shards = json.load(open(PX / "px.png.shards.json"))
            prog = json.load(open(PX / "progress.json"))
        except Exception:
            time.sleep(60); continue
        for sh in shards:
            if not sh["complete"] or sh["file"] in done:
                continue
            if not (PX / sh["file"]).exists() or os.path.getsize(PX / sh["file"]) == 0:   # empty tail shard: nothing to back up
                done.add(sh["file"]); continue
            n = sh["file"].rsplit(".", 1)[-1]
            d = BK / ("s" + n); d.mkdir(exist_ok=True)
            if not (d / sh["file"]).exists():
                os.link(PX / sh["file"], d / sh["file"])
            t0 = time.time()
            if create(d, "%s-s%s" % (BASE, n), "knee-s6-px-v4-320-20261015-s%s" % n):
                done.add(sh["file"]); json.dump(sorted(done), open(done_path, "w"))
                log("uploaded shard", n, "%.1f GB in %.0f s" % (os.path.getsize(d / sh["file"]) / 2 ** 30, time.time() - t0))
        if prog.get("final") and all(s["file"] in done for s in shards if (PX / s["file"]).exists()):
            if (PX / "FINALIZED").exists() and "index" not in done:
                d = BK / "index"; d.mkdir(exist_ok=True)
                for f in ("index.json", "px.png.off.npy", "px.png.shards.json", "meta.f32.npy", "progress.json"):
                    if not (d / f).exists():
                        os.link(PX / f, d / f)
                if create(d, BASE, "knee-s6-px-v4-320-20261015"):
                    done.add("index"); json.dump(sorted(done), open(done_path, "w")); log("uploaded index; BACKUP_COMPLETE")
                    return
        time.sleep(120)


if __name__ == "__main__":
    main()
