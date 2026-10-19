"""Lane B parity reference: heliosli's pinned compact_infer v4c main(), source unchanged, on gold-58 with all four fold models.
  stage:            python -I vm_gold_parity.py stage            (range-reads the gold studies into LB_GOLD_DATA/train_series)
  score r6|r7 OUT:  python -I vm_gold_parity.py score r6 OUT     (R6: notebook WORKER_ARGS incl. --flip-tta; R7: no flip)
Both eager (no --compile, as the OOF driver), --amp auto, --dump-logits/--dump-state; the driver's gold records must reproduce
predictions.csv.  Prints no study IDs."""
import json, os, sys
from pathlib import Path

GOLD = Path(os.environ.get("LB_GOLD_DATA", "/content/data_gold"))
os.environ["S6V2_DATA"] = str(GOLD)
os.environ.setdefault("S6V2_SRC", "/content/r6src")
os.environ.setdefault("S6V2_CODE", "/content/code/cloud_code")
sys.path.insert(0, "/content")
import vm_build_cache as B  # noqa: E402

SRC, CODE = os.environ["S6V2_SRC"], os.environ["S6V2_CODE"]
BUNDLE = {"r6": Path(os.environ.get("LB_R6", "/content/r6")), "r7": Path(os.environ.get("LB_R7", "/content/r7"))}


def gold_ids():
    return [u for u, md in json.load(open("/content/worklist.json")) if md == "all"][:58]


def stage():
    import io, zipfile
    B.DATA.mkdir(parents=True, exist_ok=True)
    for name in ("train.csv", "train_series.csv"):
        (B.DATA / name).write_bytes((Path("/content/data") / name).read_bytes())
    ar = B.Archive()
    zf = zipfile.ZipFile(io.BufferedReader(B.Remote(ar), 4 << 20))
    want = set(gold_ids()); by = {}
    for z in zf.infolist():
        p = z.filename.split("/")
        if len(p) == 4 and p[0] == "train_series" and p[1] in want and not z.is_dir():
            by.setdefault(p[1], []).append(z)
    assert set(by) == want and len(want) == 58, "gold study missing from the archive"
    n = sum(B.fetch_study(ar, by[u])[0] for u in sorted(by))
    json.dump(sorted(want), open(B.DATA / "gold_assignment.json", "w"))
    B.log("gold staged studies", len(by), "files", n)


def install_paths():
    B.install_paths()
    for p in (CODE, SRC):
        while p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)


def _memoise():
    import probe_infer as PI
    if getattr(PI, "_lb_memo", False):
        return
    orig, cache = PI.install_native, {}

    def memo(lean=True):
        if lean not in cache:
            cache[lean] = orig(lean)
        return cache[lean]
    PI.install_native = memo
    PI._lb_memo = True


install_paths()
import compact_infer as CI  # noqa: E402


class LocalCStudies(CI.CStudies):
    """launcher guards in every spawned worker: paths, memoised install_native; an item without an SKM-TEA anchor is an error."""
    def _setup(self):
        install_paths(); _memoise()
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


def score(which, out):
    _memoise()
    d = BUNDLE[which]
    m = json.load(open(d / "manifest.json"))
    args = ["--data", str(GOLD), "--split", "train", "--code", CODE, "--arm", m["arm"], "--assignment", str(GOLD / "gold_assignment.json"),
            "--output", out, "--amp", "auto", "--workers", "3", "--models", str(d), "--backbone", m["backbone"], "--size", str(m["size"]),
            "--crop", str(m["crop"]), "--eval-stride", str(m["eval_stride"]), "--views", m["views"], "--state-spec", str(d / m["state_spec"]),
            "--side-model", str(d / m["side_model"]), "--side-threshold", str(m["side_threshold"]), "--side-position-mm",
            str(m["side_position_mm"]), "--side-max-x", str(m["side_max_x"]), "--slices", str(m["slices"]), "--phys-mm", str(m["phys_mm"]),
            "--trainer-md5", m["trainer_md5"], "--fluid-stacker", str(d / m["fluid_stacker"])]
    if which == "r6":
        args += ["--flip-tta"]
    args += ["--state-stacker", str(d / m["state_stacker"]), "--anchor-weights", "/content/helios/anatomy.pt",
             "--dump-logits", os.path.join(out, "logits.json"), "--dump-state"]
    sys.argv = ["compact_infer.py"] + args
    CI.main()


if __name__ == "__main__":
    if sys.argv[1] == "stage":
        stage()
    else:
        score(sys.argv[2], sys.argv[3])
