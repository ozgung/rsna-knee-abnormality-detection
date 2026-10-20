"""Lane F: rewrite the fold column of an s6v2 targets table to the R6/R7 hash rule, sha1(uid)[:8] % 4 (compact_train.py:2015,
confirmed on all 4,349 Lane B out-of-fold rows), so that compact_train_s6v2.py --folds 4 --only-folds k trains on three hash folds
and predicts the held-out hash fold k.  Gold rows keep fold -1; every other column is copied byte for byte.
usage: python3.12 -I make_hash4_targets.py <in targets csv> <out targets csv>   (prints counts and sha256 only)"""
import csv, hashlib, sys
from collections import Counter


def hash_fold(uid, k=4):
    return int(hashlib.sha1(uid.encode()).hexdigest()[:8], 16) % k


def main(src, dst):
    with open(src, newline="") as f:
        rows = list(csv.DictReader(f))
        fields = f.seek(0) or next(csv.reader(f))
    assert fields[:4] == ["StudyInstanceUID", "is_gold", "review_holdout", "fold"], fields[:4]
    changed = Counter()
    for r in rows:
        if r["is_gold"] == "1":
            assert r["fold"] == "-1"
            continue
        r["fold"] = str(hash_fold(r["StudyInstanceUID"]))
        changed[r["fold"]] += 1
    with open(dst, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\r\n")
        w.writeheader()
        w.writerows(rows)
    print({"rows": len(rows), "gold": sum(r["is_gold"] == "1" for r in rows), "hash4_nongold": dict(sorted(changed.items())),
           "in_sha256": hashlib.sha256(open(src, "rb").read()).hexdigest(),
           "out_sha256": hashlib.sha256(open(dst, "rb").read()).hexdigest()})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
