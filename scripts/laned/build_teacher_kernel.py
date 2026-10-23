"""Lane D (local): derive a private teacher-inference kernel from P2's notebook (57340072, R6F-pure v2) with the frozen payload
byte-identical; only the bootstrap's test-root list is swapped to a private fastMRI dataset in the competition layout.
Usage: python build_teacher_kernel.py <P2 kernel dir> <out dir> <dataset slug> <kernel slug>"""
import hashlib, json, sys
from pathlib import Path

OLD = "roots=[Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),Path('/kaggle/input/rsna-knee-abnormality-detection')]"
P2_NOTEBOOK_SHA = "3af761f636535777c8d5d1462679085bb3222e230929588416af44fe3808d9ca"


def main(src, out, ds, slug):
    src, out = Path(src), Path(out)
    raw = (src / "notebook.ipynb").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == P2_NOTEBOOK_SHA, "not P2's notebook"
    nb = json.loads(raw)
    cell = "".join(nb["cells"][0]["source"])
    assert cell.count(OLD) == 1
    new = "roots=[Path('/kaggle/input/datasets/ozgung/%s'),Path('/kaggle/input/%s')]" % (ds, ds)
    payload_before = cell.split("encoded='")[1].split("'")[0]
    cell = cell.replace(OLD, new)
    assert cell.split("encoded='")[1].split("'")[0] == payload_before
    nb["cells"][0]["source"] = cell
    meta = json.loads((src / "kernel-metadata.json").read_text())
    meta.update(id="ozgung/" + slug, title=slug, competition_sources=[], is_private=True,
                dataset_sources=meta["dataset_sources"] + ["ozgung/" + ds])
    out.mkdir(parents=True, exist_ok=True)
    (out / "notebook.ipynb").write_text(json.dumps(nb, indent=1))
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    rec = {"p2_notebook_sha256": P2_NOTEBOOK_SHA, "payload_unchanged": True, "root_swap": [OLD, new],
           "notebook_sha256": hashlib.sha256((out / "notebook.ipynb").read_bytes()).hexdigest(), "kernel": meta["id"],
           "datasets": meta["dataset_sources"]}
    (out / "build-receipt.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:5])
