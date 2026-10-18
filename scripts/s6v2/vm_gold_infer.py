"""Gold-58 scoring from raw DICOM with heliosli's pinned compact_infer.py (nartaa's S6F0 arguments: v4, 320/288, 3 slices,
3.5 mm, eval stride 1, SKM-TEA anchors, no side model).  Stage:  python -I vm_gold_infer.py stage   (range-reads the gold
studies into $S6V2_DATA/train_series).  Score:  python -I vm_gold_infer.py score <models dir> <out dir> [amp]
(runs compact_infer.main() in this process through nartaa's launcher guards).  Prints no study IDs."""
import json, os, sys
from pathlib import Path

os.environ.setdefault("S6V2_DATA", "/content/data_gold")
sys.path.insert(0, "/content")
import vm_build_cache as B  # noqa: E402  (DATA, install_paths, Archive, Remote, fetch_study)


def gold_ids():
    order = json.load(open("/content/order.json"))
    return order[:58]                       # order.json = sorted gold-58 first, then the pool


def stage():
    import io, zipfile
    B.DATA.mkdir(parents=True, exist_ok=True)
    for name in ("train.csv", "train_series.csv"):
        dst = B.DATA / name
        if not dst.exists():
            dst.write_bytes((Path("/content/data") / name).read_bytes())
    ar = B.Archive()
    zf = zipfile.ZipFile(io.BufferedReader(B.Remote(ar), 4 << 20))
    want = set(gold_ids()); by = {}
    for z in zf.infolist():
        p = z.filename.split("/")
        if len(p) == 4 and p[0] == "train_series" and p[1] in want and not z.is_dir():
            by.setdefault(p[1], []).append(z)
    assert set(by) == want, "gold study missing from the archive"
    n = sum(B.fetch_study(ar, by[u])[0] for u in sorted(by))
    json.dump(sorted(want), open(B.DATA / "gold_assignment.json", "w"))
    B.log("gold staged studies", len(by), "files", n)


def _memoise():
    import probe_infer as PI
    if getattr(PI, "_s6v2_memo", False):
        return
    orig, cache = PI.install_native, {}

    def memo(lean=True):
        if lean not in cache:
            cache[lean] = orig(lean)
        return cache[lean]
    PI.install_native = memo
    PI._s6v2_memo = True


B.install_paths()
import compact_infer as CI  # noqa: E402


class LocalCStudies(CI.CStudies):
    """nartaa's K1 guards: paths/stubs and memoised install_native in every (spawned) worker; a failed setup resets."""
    def _setup(self):
        B.install_paths(); _memoise()
        try:
            super()._setup()
            if self.bank is None:
                raise RuntimeError("SKM-TEA anchor bank did not load")
        except BaseException:
            self.native = None
            raise

    def __getitem__(self, i):
        out = super().__getitem__(i)
        if out.get("error") is None and out.get("anchor") is None:
            out = dict(out, error="item built without an SKM-TEA anchor", images=out["images"][:0], meta=out["meta"][:0])
        return out


CI.CStudies = LocalCStudies


def score(models, out, amp="bf16"):
    _memoise()
    sys.argv = ["compact_infer.py", "--data", str(B.DATA), "--split", "train", "--code", B.CODE, "--models", models,
                "--arm", "E_prior", "--backbone", "convnext_small", "--size", "320", "--crop", "288", "--views", "v4",
                "--slices", "3", "--phys-mm", "3.5", "--eval-stride", "1", "--anchor-weights", B.ANATOMY, "--amp", amp,
                "--chunk", "192", "--workers", "3", "--item-timeout", "900", "--assignment", str(B.DATA / "gold_assignment.json"),
                "--output", out, "--dump-logits", os.path.join(out, "logits.json")]
    CI.main()


if __name__ == "__main__":
    if sys.argv[1] == "stage":
        stage()
    else:
        score(*sys.argv[2:])
