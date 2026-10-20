"""Lane C read-outs on the A100 through the pinned DICOM -> compact_infer path (vm_gold_infer.py's guards and arguments).
  stage <ids json>                               range-read those train studies into $S6V2_DATA/train_series (gold-58 + probes)
  score <models dir> <out dir> <ids json> <amp>  compact_infer.main() over exactly those studies (eval stride 1, v4, 320/288)
Prints no study IDs."""
import io, json, os, sys, zipfile
from pathlib import Path

os.environ.setdefault("S6V2_DATA", "/content/data_gold")
sys.path.insert(0, "/content")
import vm_gold_infer as G  # noqa: E402  (installs the K1 loader guards on compact_infer.CStudies)

B, CI = G.B, G.CI


def stage(ids_path):
    want = set(json.load(open(ids_path)))
    B.DATA.mkdir(parents=True, exist_ok=True)
    for name in ("train.csv", "train_series.csv"):
        dst = B.DATA / name
        if not dst.exists():
            dst.write_bytes((Path("/content/data") / name).read_bytes())
    have = {p.name for p in (B.DATA / "train_series").iterdir()} if (B.DATA / "train_series").exists() else set()
    want -= have
    if not want:
        B.log("all requested studies already staged"); return
    ar = B.Archive()
    zf = zipfile.ZipFile(io.BufferedReader(B.Remote(ar), 4 << 20))
    by = {}
    for z in zf.infolist():
        p = z.filename.split("/")
        if len(p) == 4 and p[0] == "train_series" and p[1] in want and not z.is_dir():
            by.setdefault(p[1], []).append(z)
    assert set(by) == want, "requested study missing from the archive"
    n = sum(B.fetch_study(ar, by[u])[0] for u in sorted(by))
    B.log("staged studies", len(by), "files", n)


def score(models, out, ids_path, amp):
    G._memoise()
    sys.argv = ["compact_infer.py", "--data", str(B.DATA), "--split", "train", "--code", B.CODE, "--models", models,
                "--arm", "E_prior", "--backbone", "convnext_small", "--size", "320", "--crop", "288", "--views", "v4",
                "--slices", "3", "--phys-mm", "3.5", "--eval-stride", "1", "--anchor-weights", B.ANATOMY, "--amp", amp,
                "--chunk", "192", "--workers", "3", "--item-timeout", "900", "--assignment", ids_path,
                "--output", out, "--dump-logits", os.path.join(out, "logits.json")]
    CI.main()


if __name__ == "__main__":
    if sys.argv[1] == "stage":
        stage(sys.argv[2])
    else:
        score(*sys.argv[2:6])
