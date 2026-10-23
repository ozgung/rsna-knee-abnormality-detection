"""Lane D (VM): main pixel store for the joint fit = the restored competition store (/content/px, unchanged files, symlinked)
with the fastMRI studies added to index.json 'done'/'blocks', so that compact_train's --px-override /content/pxfm reads every
fastMRI study from its own store (row space, meta) and the competition rows stay bit-identical.  Writes /content/pxm.
Usage: python -I vm_fm_merge.py"""
import json, os
from pathlib import Path

PX, FM, OUT = Path("/content/px"), Path("/content/pxfm"), Path("/content/pxm")
assert (PX / "FINALIZED").exists() and (FM / "FINALIZED").exists()
OUT.mkdir(exist_ok=False)
for f in os.listdir(PX):
    if f != "index.json":
        os.symlink(PX / f, OUT / f)
a, b = json.load(open(PX / "index.json")), json.load(open(FM / "index.json"))
dup = set(a["done"]) & set(b["done"])
assert not dup, "fastMRI and competition study IDs collide"
a["done"].update(b["done"]); a["blocks"].update(b["blocks"])
a["lane_d_merge"] = {"competition_done": len(a["done"]) - len(b["done"]), "fastmri_done": len(b["done"]),
                     "note": "fastMRI blocks are placeholders; --px-override /content/pxfm supplies their rows"}
json.dump(a, open(OUT / "index.json", "w"))
print(json.dumps(a["lane_d_merge"]))
