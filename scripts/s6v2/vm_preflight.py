"""Complete-cache preflight before the J0/V2 pair launches (fails closed, exit 1).  Checks: FINALIZED present; every study of
order.json disposed (done with rows or error); all 58 gold studies done; shards complete with byte total = offsets; meta rows
= cursor; for each targets CSV the trainer's pool (non-gold, review_holdout 0, targets present, in the cache) and fold-0
train / holdout counts + sha256 of the sorted IDs, identical between arms.  Writes /content/px/preflight.json; prints counts only.
Usage: python -I vm_preflight.py <targets csv> [...]"""
import csv, hashlib, json, os, sys
from pathlib import Path

import numpy as np

PX = Path("/content/px")
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
          "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def h(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def main():
    bad = []
    order = json.load(open("/content/order.json"))
    idx = json.load(open(PX / "index.json"))
    done = {u for u, v in idx["done"].items() if v[1] > 0}
    errors = set(idx["errors"])
    if not (PX / "FINALIZED").exists():
        bad.append("FINALIZED missing")
    missing = [u for u in order if u not in done and u not in errors]
    if missing:
        bad.append("%d studies neither done nor errored" % len(missing))
    gold = set(order[:58])
    if gold - done:
        bad.append("%d gold studies not in the cache" % len(gold - done))
    sh = json.load(open(PX / "px.png.shards.json"))
    off = np.load(PX / "px.png.off.npy")
    if not all(x["complete"] for x in sh):
        bad.append("incomplete shard")
    if sum(os.path.getsize(PX / x["file"]) for x in sh) != int(off[-1]) or len(off) != idx["cursor"] + 1:
        bad.append("byte / row totals differ")
    meta = np.load(PX / "meta.f32.npy", mmap_mode="r")
    if meta.shape[0] != idx["cursor"]:
        bad.append("meta rows differ from cursor")
    rec = {"studies_ordered": len(order), "done": len(done), "errors": len(errors), "gold_done": len(gold & done),
           "rows": idx["cursor"], "png_bytes": int(off[-1]), "shards": len(sh), "error_kinds": {}, "arms": {}}
    for e in idx["errors"].values():
        k = e.split("(")[0].split(":")[0][:30]
        rec["error_kinds"][k] = rec["error_kinds"].get(k, 0) + 1
    for path in sys.argv[1:]:
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        pool = [r["StudyInstanceUID"] for r in rows if r["is_gold"] == "0" and r["review_holdout"] == "0"
                and all(r[L + "|prior"] != "" for L in LABELS) and r["StudyInstanceUID"] in done]
        fold = {r["StudyInstanceUID"]: int(r["fold"]) for r in rows}
        tr = [u for u in pool if fold[u] != 0]; ho = [u for u in pool if fold[u] == 0]
        g = [r["StudyInstanceUID"] for r in rows if r["is_gold"] == "1" and r["StudyInstanceUID"] in done]
        rec["arms"][Path(path).name] = {"pool": len(pool), "pool_sha256": h(pool), "fold0_train": len(tr), "fold0_train_sha256": h(tr),
                                        "fold0_holdout": len(ho), "fold0_holdout_sha256": h(ho), "gold": len(g), "gold_sha256": h(g),
                                        "train_gold_overlap": len(set(tr) & gold), "csv_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
    arms = list(rec["arms"].values())
    if len(arms) > 1 and len({(a["pool_sha256"], a["fold0_train_sha256"], a["fold0_holdout_sha256"], a["gold_sha256"]) for a in arms}) != 1:
        bad.append("arms differ in membership")
    if any(a["gold"] != 58 or a["train_gold_overlap"] for a in arms):
        bad.append("gold coverage / overlap")
    rec["bad"] = bad
    rec["status"] = "PASS" if not bad else "FAIL"
    json.dump(rec, open(PX / "preflight.json", "w"), indent=1)
    print(json.dumps(rec, indent=1))
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
