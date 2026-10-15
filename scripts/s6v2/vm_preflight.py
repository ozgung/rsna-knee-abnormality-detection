"""Complete-cache preflight before the J0/V2 pair launches; fails closed (exit 1) and writes /content/px/preflight.json.

Order / identity: order.json has no duplicates, equals index.json's order, and its set equals the UID universe of every targets
CSV (identical across arms); gold = the CSV is_gold rows = order[:58] (58 studies).
Disposition: every ordered study is done (rows > 0) or errored, never both; every error is on the reviewed allowlist
(/content/px/error_allowlist.json, {"uids": [...], "review": "..."}; its sha256 is bound); all gold studies are done.
Rows: done ranges [start, start + n) are disjoint, inside [0, cursor) and cover it exactly; every (series, view) block lies inside
its study's range; meta.f32.npy is float32 [cursor, 12]; offsets start at 0, strictly increase, have cursor + 1 entries, and the
shards are complete, contiguous, start on row boundaries and sum to the final offset.  Optional on the VM:
S6V2_PREFLIGHT_DECODE=1 decodes sampled rows (uint8 320x320); S6V2_PREFLIGHT_HASH=1 hashes every shard.
Arms: pool (non-gold, review_holdout 0, targets present, in the cache), fold-0 train / holdout and gold counts + ID sha256 are
identical across arms, non-empty, and train never contains gold.  The receipt binds content hashes of the cache index files and
of the trainer, launcher, targets, init and anatomy files when present.  Prints counts only.
Usage: python -I vm_preflight.py <targets csv> [...]"""
import csv, hashlib, json, os, sys
from pathlib import Path

import numpy as np

PX = Path("/content/px")
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
          "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
BOUND = ["/content/code/compact_train_s6v2.py", "/content/vm_train.sh", "/content/vm_launch_pair.sh", "/content/model.safetensors",
         "/content/helios/anatomy.pt", "/content/order.json", "/content/vm_build_cache.py", "/content/vm_build_cache_v2.py"]


def h(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def fsha(p, big=False):
    m = hashlib.sha256()
    with open(p, "rb") as fh:
        while True:
            b = fh.read(1 << 24)
            if not b:
                break
            m.update(b)
    return m.hexdigest()


def main():
    bad = []
    order = json.load(open("/content/order.json"))
    idx = json.load(open(PX / "index.json"))
    if len(order) != len(set(order)):
        bad.append("duplicate entries in order.json")
    if idx.get("order") != order:
        bad.append("index.json order differs from order.json")
    oset = set(order)
    done_all = idx["done"]
    done = {u for u, v in done_all.items() if v[1] > 0}
    errors = set(idx["errors"])
    if done & errors:
        bad.append("studies both done and errored")
    if set(done_all) - oset or errors - oset:
        bad.append("cache holds studies outside the order")
    missing = [u for u in order if u not in done and u not in errors]
    if missing:
        bad.append("%d ordered studies neither done nor errored" % len(missing))
    allow_path = PX / "error_allowlist.json"
    allow = json.load(open(allow_path)) if allow_path.exists() else {"uids": [], "review": None}
    unreviewed = errors - set(allow["uids"])
    if unreviewed:
        bad.append("%d errored studies are not on the reviewed allowlist" % len(unreviewed))
    if not (PX / "FINALIZED").exists():
        bad.append("FINALIZED missing")
    # ---- row ownership and block ranges
    cursor = int(idx["cursor"])
    spans = sorted((int(v[0]), int(v[1]), u) for u, v in done_all.items() if u in done)
    pos = 0
    for st, n, u in spans:
        if st != pos or n <= 0:
            bad.append("study row ranges are not disjoint and contiguous"); break
        pos = st + n
    if pos != cursor:
        bad.append("study row ranges do not cover [0, cursor)")
    for st, n, u in spans:
        ok = True
        for b in idx["blocks"].get(u, []):
            ns = int(b["n_slices"])
            for v in b["views"]:
                r0 = int(v[1])
                if ns <= 0 or r0 < st or r0 + ns > st + n:
                    ok = False
        if not idx["blocks"].get(u) or not ok:
            bad.append("a study's view rows fall outside its row range"); break
    # ---- offsets, shards, meta
    off = np.load(PX / "px.png.off.npy")
    sh = json.load(open(PX / "px.png.shards.json"))
    if off.dtype != np.int64 or len(off) != cursor + 1 or int(off[0]) != 0 or not bool((np.diff(off) > 0).all()):
        bad.append("offsets are not int64, 0-based, strictly increasing with cursor + 1 entries")
    if not all(x.get("complete") for x in sh):
        bad.append("incomplete shard")
    b0 = 0
    offset_set = set(off.tolist()) if len(off) < 5_000_000 else None
    for x in sh:
        size = os.path.getsize(PX / x["file"]) if (PX / x["file"]).exists() else -1
        if int(x["byte0"]) != b0 or size < 0 or (offset_set is not None and b0 not in offset_set):
            bad.append("shard boundaries are not contiguous row boundaries"); break
        b0 += size
    if b0 != int(off[-1]):
        bad.append("shard bytes do not sum to the final offset")
    meta = np.load(PX / "meta.f32.npy", mmap_mode="r")
    if meta.dtype != np.float32 or meta.shape != (cursor, 12):
        bad.append("meta.f32.npy is not float32 [cursor, 12]")
    if os.environ.get("S6V2_PREFLIGHT_DECODE") == "1":
        import cv2
        byte0 = np.array([x["byte0"] for x in sh if os.path.getsize(PX / x["file"]) > 0], np.int64)
        files = [x["file"] for x in sh if os.path.getsize(PX / x["file"]) > 0]
        for i in np.linspace(0, cursor - 1, 2000).astype(int):
            lo, hi = int(off[i]), int(off[i + 1]); s = int(np.searchsorted(byte0, lo, side="right")) - 1
            with open(PX / files[s], "rb") as fh:
                fh.seek(lo - int(byte0[s])); buf = fh.read(hi - lo)
            x = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_UNCHANGED)
            if x is None or x.dtype != np.uint8 or x.shape != (320, 320):
                bad.append("sampled PNG row does not decode to uint8 320x320"); break
    rec = {"studies_ordered": len(order), "done": len(done), "errors": len(errors), "allowlisted_errors": len(errors & set(allow["uids"])),
           "error_allowlist_sha256": fsha(allow_path) if allow_path.exists() else None, "rows": cursor, "png_bytes": int(off[-1]),
           "shards": len(sh), "error_kinds": {}, "arms": {}}
    for e in idx["errors"].values():
        k = e.split("(")[0].split(":")[0][:30]
        rec["error_kinds"][k] = rec["error_kinds"].get(k, 0) + 1
    # ---- arms
    gold_order = order[:58]
    universes = []
    for path in sys.argv[1:]:
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        uids = [r["StudyInstanceUID"] for r in rows]
        universes.append(frozenset(uids))
        if len(uids) != len(set(uids)) or set(uids) != oset:
            bad.append("targets UID universe differs from the order (%s)" % Path(path).name)
        gold = {r["StudyInstanceUID"] for r in rows if r["is_gold"] == "1"}
        if gold != set(gold_order) or len(gold) != 58:
            bad.append("targets gold rows differ from order[:58] (%s)" % Path(path).name)
        if gold - done:
            bad.append("gold studies missing from the cache")
        pool = [r["StudyInstanceUID"] for r in rows if r["is_gold"] == "0" and r["review_holdout"] == "0"
                and all(r[L + "|prior"] != "" for L in LABELS) and r["StudyInstanceUID"] in done]
        fold = {r["StudyInstanceUID"]: int(r["fold"]) for r in rows}
        tr = [u for u in pool if fold[u] != 0]; ho = [u for u in pool if fold[u] == 0]
        g = sorted(gold & done)
        if not tr or not ho:
            bad.append("empty fold-0 train or holdout (%s)" % Path(path).name)
        rec["arms"][Path(path).name] = {"pool": len(pool), "pool_sha256": h(pool), "fold0_train": len(tr), "fold0_train_sha256": h(tr),
                                        "fold0_holdout": len(ho), "fold0_holdout_sha256": h(ho), "gold": len(g), "gold_sha256": h(g),
                                        "train_gold_overlap": len(set(tr) & gold), "csv_sha256": fsha(path)}
    arms = list(rec["arms"].values())
    if len(set(universes)) > 1:
        bad.append("arms have different UID universes")
    if len(arms) > 1 and len({(a["pool_sha256"], a["fold0_train_sha256"], a["fold0_holdout_sha256"], a["gold_sha256"]) for a in arms}) != 1:
        bad.append("arms differ in membership")
    if any(a["gold"] != 58 or a["train_gold_overlap"] for a in arms):
        bad.append("gold coverage / overlap")
    # ---- bound identities
    rec["bound_sha256"] = {f: fsha(PX / f) for f in ("index.json", "px.png.off.npy", "px.png.shards.json", "meta.f32.npy")}
    rec["bound_sha256"].update({p: fsha(p) for p in BOUND if os.path.exists(p)})
    if os.environ.get("S6V2_PREFLIGHT_HASH") == "1":
        rec["shard_sha256"] = {x["file"]: fsha(PX / x["file"]) for x in sh}
    rec["bad"] = bad
    rec["status"] = "PASS" if not bad else "FAIL"
    json.dump(rec, open(PX / "preflight.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in rec.items() if k not in ("bound_sha256", "shard_sha256")}, indent=1))
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
