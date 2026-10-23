"""Lane D (VM): hard-link the selected fastMRI series into the competition layouts (no pixel change, no copies).
  /content/fmdata/{train.csv,train_series.csv,train_series/<study>/<series>/NNNN.dcm}  -> cache builder (CStudies 'train')
  /content/fmteach/<part>/{test.csv,test_series.csv,sample_submission.csv,test_series/...,dataset-metadata.json}
                                                                                   -> private Kaggle datasets for the teacher kernels
Usage: python -I vm_fm_layout.py <fm_train_series.csv> <n parts> <dataset slug prefix>"""
import csv, json, os, sys
from pathlib import Path

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
          "Baker's", "Contusion", "Fracture"]
COLS = ["StudyInstanceUID", "SeriesInstanceUID", "Fluid_Sensitive", "Fat_Suppression", "Anatomical_Plane"]


def link_series(files, dst):
    dst.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(files):
        os.link(f, dst / ("%04d.dcm" % i))


def main(table, parts, prefix):
    rows = list(csv.DictReader(open(table)))
    assert rows and list(rows[0]) == COLS
    files = json.load(open("/content/fm_meta/fm_files.json"))
    studies = list(dict.fromkeys(r["StudyInstanceUID"] for r in rows))
    by = {}
    for r in rows:
        by.setdefault(r["StudyInstanceUID"], []).append(r)
    D = Path("/content/fmdata"); (D / "train_series").mkdir(parents=True, exist_ok=False)
    with open(D / "train.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["StudyInstanceUID"]); w.writerows([[s] for s in studies])
    with open(D / "train_series.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS); w.writeheader(); w.writerows(rows)
    nfiles = 0
    for s in studies:
        for r in by[s]:
            fl = files["%s/%s" % (s, r["SeriesInstanceUID"])]
            link_series(fl, D / "train_series" / s / r["SeriesInstanceUID"]); nfiles += len(fl)
    T = Path("/content/fmteach")
    n = int(parts)
    for k in range(n):
        mine = studies[k::n]
        P = T / ("part%d" % k); (P / "test_series").mkdir(parents=True, exist_ok=False)
        with open(P / "test.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["StudyInstanceUID"]); w.writerows([[s] for s in mine])
        with open(P / "sample_submission.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["StudyInstanceUID"] + LABELS); w.writerows([[s] + ["0.5"] * 12 for s in mine])
        with open(P / "test_series.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLS); w.writeheader()
            for s in mine:
                w.writerows(by[s])
        for s in mine:
            for r in by[s]:
                src = D / "train_series" / s / r["SeriesInstanceUID"]
                dst = P / "test_series" / s / r["SeriesInstanceUID"]; dst.mkdir(parents=True)
                for f in sorted(src.iterdir()):
                    os.link(f, dst / f.name)
        slug = "%s-p%d" % (prefix, k)
        json.dump({"title": slug, "id": "ozgung/" + slug, "licenses": [{"name": "other"}]}, open(P / "dataset-metadata.json", "w"))
        print("part", k, "studies", len(mine), flush=True)
    print("LAYOUT_DONE studies", len(studies), "series", len(rows), "files", nfiles, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
