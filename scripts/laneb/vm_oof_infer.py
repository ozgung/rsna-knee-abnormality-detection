"""Lane B (Oct 19): per-study R6 / R7 fold-model columns on the training pool and gold-58, on a Colab A100.

One render per study through heliosli's pinned compact_infer.CStudies (native -> geometry -> image side check -> SKM-TEA anchors
-> v4 384 px rows -> D2 fluid volumes; the R6 and R7 bundles share the recipe and side_cls.pt), then on the GPU, per selected
fold model, the eager forward compact_infer v4c runs (bf16 autocast, chunk 192, fp32 rescue of a non-finite output, centre crop
352, return_aux state read-out):
  R6: plain pass + mirrored pass (compact_infer.mirror_windows; combined as score_v4c: 0.5 (z + P z_m), MCL from the plain pass;
      the state read-out combined the same way);  R7: plain pass (+ mirrored pass with LB_R7_FLIP=1).
mode 'oof' = only the model of the study's hash fold (sha1(uid)[:8] % 4, compact_train.py:2015); mode 'all' = the four fold
models (gold-58 and the mapping-check sample).  No torch.compile (offline analysis; compiled vs eager is a <= 1e-3 matter).
Stackers are applied offline from the recorded logits, state read-outs and fluid volumes.
Staging: HTTP range reads of the official archive per study (vm_build_cache.Archive / fetch_study), DICOMs deleted after the
render.  Output: LB_OUT/records.jsonl (one JSON per study; resumable), progress.json, receipt.json.  Prints no study IDs."""
import hashlib, io, json, multiprocessing as mp, os, queue, shutil, sys, threading, time, zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

E = os.environ.get
DATA = Path(E("LB_DATA", "/content/data"))
OUT = Path(E("LB_OUT", "/content/lb_out"))
SRC = E("LB_SRC", "/content/r6src")
CODE = E("LB_CODE", "/content/code/cloud_code")
ANATOMY = E("LB_ANATOMY", "/content/helios/anatomy.pt")
R6 = Path(E("LB_R6", "/content/r6"))
R7 = Path(E("LB_R7", "/content/r7"))
WORKLIST = Path(E("LB_WORKLIST", "/content/worklist.json"))
TMP = Path(E("LB_TMP", "/content/tmp_lb"))
WORKERS = int(E("LB_WORKERS", "10"))
STAGERS = int(E("LB_STAGERS", "8"))
AHEAD = int(E("LB_AHEAD", "36"))
LIMIT = int(E("LB_LIMIT", "0"))
R7_FLIP = E("LB_R7_FLIP", "0") == "1"      # R7 flip dropped after the first 49 gold studies (GPU-bound; plan: if time allows)
SIZE, CROP, VIEWS, SLICES, PHYS, STRIDE, CHUNK, K = 384, 352, "v4", 3, 3.5, 1, 192, 4
PINS = {"compact_infer.py": "8401812cd63737ec99839d317c4037c4efb445d1baec892fa7481bb30399cc2d",
        "compact_train.py": "04df7e2e154b2c2521696908f272940c63ce17e9b6337b85847fee64504eeaba",
        "px_cache.py": "b13e38a7ad3bd496f8027e02209653e224b0926db4f0853eeb0d076a2d6e2632",
        "probe_infer.py": "a6cc31448019c84be5868d0a2ff1d99719b23b0f86110efc03efff3ce6bd6980",
        "knee_anchor_bank_v2.py": "e564807a129dfcd3c984348917436ee6083066a4d685340587ebd1d3ea14f309",
        "side_common.py": "bbcc5d027834e6a6fd48c4a8cf228e25661b7522c792587c840fe4d7cd5c0566"}
BUNDLES = {"r6": {"dir": R6, "manifest": "6508ce6921eda986c885b2226fa55dd4d2aa8e5fc570b4c007c9a60c05bfc648"},
           "r7": {"dir": R7, "manifest": "c4bb0f7399ada9a020e0c15cae6252033cf0c4335e999961793a176063074ec6"}}
SIDE_SHA = "ab623e2b619db4917639220fed15202e8c8aedd07c8e298a66a2afc0f153650a"
ANATOMY_SHA = "5904dd16eb9a4fd7a1caf90ac977f8fe2ef364908598c23c209db3fe7fd1563e"

os.environ.setdefault("S6V2_DATA", str(DATA))
os.environ.setdefault("S6V2_SRC", SRC)
os.environ.setdefault("S6V2_CODE", CODE)
sys.path.insert(0, "/content")
import vm_build_cache as B  # noqa: E402  (Archive, Remote, fetch_study, install_paths, log)

log = B.log


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def install_paths():
    B.install_paths()
    for p in (CODE, SRC):                     # the pinned R6 sources shadow anything of the same name in the code dataset
        while p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)


def check_modules():
    import compact_infer, compact_train, px_cache, probe_infer, knee_anchor_bank_v2, side_common  # noqa: F401
    got = {}
    for name in PINS:
        mod = sys.modules[name[:-3]]
        f = Path(mod.__file__).resolve()
        assert f.parent == Path(SRC).resolve(), "module %s imported from %s" % (name, f.parent)
        got[name] = sha(f)
        assert got[name] == PINS[name], "source pin %s" % name
    return got


def fold_of(uid):
    return int(hashlib.sha1(uid.encode()).hexdigest()[:8], 16) % K


# ---------------------------------------------------------------- loader workers (spawned): one CStudies each
_CS = None


def worker_init():
    global _CS
    install_paths()
    import torch
    torch.set_num_threads(1)
    import probe_infer as PI
    orig, cache = PI.install_native, {}

    def memo(lean=True):
        if lean not in cache:
            cache[lean] = orig(lean)
        return cache[lean]
    PI.install_native = memo
    import compact_infer as CI
    check_modules()
    wtmp = TMP / ("w%d" % os.getpid())
    wtmp.mkdir(parents=True, exist_ok=True)
    _CS = CI.CStudies(str(DATA), "train", [], str(wtmp), CODE, SIZE, eval_stride=STRIDE, views=VIEWS, slices=SLICES,
                      anchor_weights=ANATOMY, phys_mm=PHYS, side_model=str(R6 / "side_cls.pt"), side_threshold=2.0, side_max_x=0.0,
                      side_position_mm=20.0, extent_meta=True, fluid=True)
    _CS._setup()
    if _CS.bank is None or _CS.side_net is None:
        raise RuntimeError("anchor bank / side net did not load")


def build_one(uid):
    t0 = time.time()
    try:
        _CS.uids = [uid]
        it = _CS[0]
        if it.get("error") is None and it.get("anchor") is None:
            it = dict(it, error="item built without an SKM-TEA anchor")
        return {"uid": uid, "error": it.get("error"), "images": it["images"].numpy() if it.get("error") is None else None,
                "meta": it["meta"].numpy() if it.get("error") is None else None, "anchor": it.get("anchor"), "side": it.get("side"),
                "seconds": it.get("seconds"), "fluid": it.get("fluid"), "fluid_s": it.get("fluid_s"), "wall": time.time() - t0}
    except Exception as exc:
        return {"uid": uid, "error": "worker: " + repr(exc)[:300], "wall": time.time() - t0}


# ---------------------------------------------------------------- GPU side (main process)
class Scorer:
    def __init__(self):
        import torch
        import compact_infer as CI
        import compact_train as CT
        self.torch, self.CI, self.CT = torch, CI, CT
        self.dev = torch.device("cuda")
        self.LABELS = CT.LABELS
        self.mcl = self.LABELS.index("MCL")
        self.b = {}
        for name, spec in BUNDLES.items():
            d = spec["dir"]
            assert sha(d / "manifest.json") == spec["manifest"], name + " manifest pin"
            man = json.load(open(d / "manifest.json"))
            for f, h in man["sha256"].items():
                assert sha(d / f) == h, "%s asset %s" % (name, f)
            assert man["sha256"]["side_cls.pt"] == SIDE_SHA and man["anchor_weights_sha256"] == ANATOMY_SHA
            for key, want in (("size", SIZE), ("crop", CROP), ("views", VIEWS), ("slices", SLICES), ("phys_mm", PHYS), ("eval_stride", STRIDE),
                              ("backbone", "convnext_small"), ("arm", "E_lrprior"), ("freeze_blocks", 16)):
                assert man[key] == want, "%s manifest %s" % (name, key)
            assert man.get("models2") is None and man["gold_in_training"] == 0
            spec_json = json.load(open(d / man["state_spec"]))
            models = []
            for k in range(K):
                p = d / ("E_lrprior_s0_f%d.pt" % k)
                assert man["models"][k] == p.name
                sd = torch.load(p, map_location="cpu", weights_only=True)
                kw = {}
                if any(x.split(".")[-1] in ("ext_u", "ext_view", "ext_gamma") for x in sd):
                    kw["extent"] = True
                if any(x == "sp_q" or x.endswith(".sp_q") for x in sd):
                    kw["spatial"] = True
                if any(x == "syn_beta" or x.endswith(".syn_beta") for x in sd):
                    kw["syn_decomp"] = True
                assert kw.get("extent"), "%s f%d has no extent head (flip TTA needs the plane column)" % (name, k)
                m = CT.Net("convnext_small", None, freeze_blocks=16, crop=CROP, state_spec=spec_json, in_chans=SLICES, **kw)
                m.load_state_dict(sd, strict=True)
                models.append(m.eval().requires_grad_(False).to(self.dev).to(memory_format=torch.channels_last))
                kw_rec = sorted(kw)
            perm, bin_perm = CI.mirror_perm(self.LABELS, spec_json, list(models[0].bin_names))
            layout = CI.state_layout(spec_json, self.LABELS)
            mcl_cols = [i for c in layout["columns"] if c["class"] == "MCL"
                        for i in (range(c["start"], c["stop"]) if "categories" in c else [c["column"]])]
            self.b[name] = {"models": models, "perm": perm, "bin_perm": bin_perm, "mcl_cols": mcl_cols, "kw": kw_rec,
                            "width": layout["width"], "flip": True if name == "r6" else R7_FLIP}
        self.amp = "bf16" if torch.cuda.get_device_capability()[0] >= 8 else "fp16"

    def fwd(self, m, img, meta, mask):
        torch = self.torch
        with torch.autocast("cuda", dtype=torch.bfloat16 if self.amp == "bf16" else torch.float16):
            z, sl, bl = m(img, meta, mask, chunk=CHUNK, return_aux=True)
        z, sl, bl = z.float(), sl.float(), bl.float()
        rescue = False
        if not torch.isfinite(z).all():
            z, sl, bl = m(img, meta, mask, chunk=CHUNK, return_aux=True)
            z, sl, bl = z.float(), sl.float(), bl.float()
            rescue = True
        return z, sl, bl, rescue

    def score(self, images, meta_np, folds, flip_folds):
        torch, CI = self.torch, self.CI
        with torch.inference_mode():
            full = torch.from_numpy(images).to(self.dev, non_blocking=True)
            off = (SIZE - CROP) // 2
            img = full[None, :, :, off:off + CROP, off:off + CROP]
            meta = torch.from_numpy(meta_np).to(self.dev)[None]
            mask = torch.ones((1, img.shape[1]), dtype=torch.bool, device=self.dev)
            img_m, meta_m = CI.mirror_windows(img, meta)
            rec, rescues = {}, 0
            for name, g in self.b.items():
                per = {}
                for k in folds:
                    m = g["models"][k]
                    z, sl, bl, r1 = self.fwd(m, img, meta, mask)
                    st = CI.state_readout(m, sl, bl)
                    rescues += int(r1)
                    r = {"z": z[0].tolist(), "st": st[0].tolist()}
                    if g["flip"] and k in flip_folds:
                        zm, slm, blm, r2 = self.fwd(m, img_m, meta_m, mask)
                        rescues += int(r2)
                        P = g["perm"]
                        zt = 0.5 * (z[0] + zm[0][P])
                        zt[self.mcl] = z[0][self.mcl]
                        stm = CI.state_readout(m, slm[:, P], blm[:, g["bin_perm"]])
                        stt = 0.5 * (st[0] + stm[0])
                        stt[g["mcl_cols"]] = st[0][g["mcl_cols"]]
                        r.update(zm_raw=zm[0].tolist(), zt=zt.tolist(), stt=stt.tolist())
                    per[str(k)] = r
                rec[name] = per
            torch.cuda.synchronize()
        return rec, rescues


def watchdog():
    def avail_gb():
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2 ** 20
        return 1e9
    n = 0
    while True:
        m = avail_gb(); d = shutil.disk_usage("/content").free / 2 ** 30
        if n % 10 == 0:
            log("WATCH mem_avail_gb %.1f disk_free_gb %.1f" % (m, d))
        if m < 6 or d < 6:
            log("WATCHDOG ABORT mem_avail_gb %.1f disk_free_gb %.1f" % (m, d)); os._exit(9)
        n += 1; time.sleep(30)


def main():
    import torch
    threading.Thread(target=watchdog, daemon=True).start()
    install_paths()
    OUT.mkdir(parents=True, exist_ok=True); TMP.mkdir(parents=True, exist_ok=True)
    pins = check_modules()
    assert sha(ANATOMY) == ANATOMY_SHA
    work = json.load(open(WORKLIST))                     # [[uid, mode], ...] in processing order
    assert len({u for u, _ in work}) == len(work) and all(md in ("oof", "all", "map") for _, md in work)
    if LIMIT:
        work = work[:LIMIT]
    rec_path = OUT / "records.jsonl"
    done = set()
    if rec_path.exists():
        for line in open(rec_path):
            try:
                done.add(json.loads(line)["uid"])
            except Exception:
                pass
    todo = [(u, md) for u, md in work if u not in done]
    mode_of = dict(todo)
    log("worklist", len(work), "done", len(done), "todo", len(todo))
    sc = Scorer()
    receipt = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "gpu": torch.cuda.get_device_name(), "amp": sc.amp,
               "torch": torch.__version__, "python": sys.version.split()[0], "source_sha256": pins, "worklist_sha256": sha(WORKLIST),
               "driver_sha256": sha(__file__), "bundles": {k: {"manifest_sha256": v["manifest"], "net_kwargs": sc.b[k]["kw"],
                                                               "flip": sc.b[k]["flip"], "state_width": sc.b[k]["width"]} for k, v in BUNDLES.items()},
               "recipe": {"size": SIZE, "crop": CROP, "views": VIEWS, "slices": SLICES, "phys_mm": PHYS, "eval_stride": STRIDE, "chunk": CHUNK,
                          "compile": False, "side_model": SIDE_SHA, "fluid": True, "fold_rule": "sha1(uid)[:8] % 4"},
               "workers": WORKERS, "stagers": STAGERS, "ahead": AHEAD}
    json.dump(receipt, open(OUT / "receipt.json", "w"), indent=1)
    log("RECEIPT", json.dumps({k: receipt[k] for k in ("gpu", "amp", "torch", "bundles")}))
    if not todo:
        log("DONE nothing to do"); return
    ar = B.Archive()
    zf = zipfile.ZipFile(io.BufferedReader(B.Remote(ar), 4 << 20))
    by_study = {}
    for z in zf.infolist():
        parts = z.filename.split("/")
        if len(parts) == 4 and parts[0] == "train_series" and not z.is_dir():
            by_study.setdefault(parts[1], []).append(z)
    missing = [u for u, _ in todo if u not in by_study]
    log("archive studies", len(by_study), "todo missing from archive", len(missing))
    fh = open(rec_path, "a", encoding="utf-8")

    def write(rec):
        fh.write(json.dumps(rec, separators=(",", ":")) + "\n"); fh.flush()

    for u in missing:
        write({"uid": u, "mode": mode_of[u], "fold": fold_of(u), "error": "absent from archive"})
    todo = [(u, md) for u, md in todo if u in by_study]
    sem = threading.Semaphore(AHEAD)
    staged_q = queue.Queue()
    stats = {"staged_bytes": 0, "scored": 0, "errors": 0, "gpu_s": 0.0, "rescues": 0, "t0": time.time(), "by_mode": {"oof": 0, "all": 0, "map": 0}}

    def stage(u):
        sem.acquire()
        try:
            n, b = B.fetch_study(ar, by_study[u])
            stats["staged_bytes"] += b
            staged_q.put((u, None))
        except Exception as exc:
            staged_q.put((u, "stage: " + repr(exc)[:200]))

    tpool = ThreadPoolExecutor(STAGERS)
    for u, _ in todo:
        tpool.submit(stage, u)
    pool = mp.get_context("spawn").Pool(WORKERS, initializer=worker_init, maxtasksperchild=200)
    results, remaining = [], len(todo)
    last = [0.0]

    def progress():
        last[0] = time.time()
        el = time.time() - stats["t0"]
        n = stats["scored"] + stats["errors"]
        prog = {"done_this_run": n, "todo_this_run": len(todo) + len(missing), "remaining": remaining, "scored": stats["scored"],
                "errors": stats["errors"], "by_mode": stats["by_mode"], "rescues": stats["rescues"], "elapsed_s": round(el),
                "s_per_study": round(el / max(1, n), 2), "gpu_s_per_scored": round(stats["gpu_s"] / max(1, stats["scored"]), 2),
                "eta_min": round(remaining * el / max(1, n) / 60, 1), "staged_gb": round(stats["staged_bytes"] / 2 ** 30, 1),
                "disk_free_gb": round(shutil.disk_usage("/content").free / 2 ** 30, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        json.dump(prog, open(OUT / "progress.json", "w"))
        log("PROGRESS", json.dumps(prog))

    while remaining > 0:
        while True:
            try:
                u, err = staged_q.get_nowait()
            except queue.Empty:
                break
            if err:
                shutil.rmtree(DATA / "train_series" / u, ignore_errors=True); sem.release()
                write({"uid": u, "mode": mode_of[u], "fold": fold_of(u), "error": err}); remaining -= 1; stats["errors"] += 1
                continue
            results.append(pool.apply_async(build_one, (u,)))
        still = []
        for r in results:
            if not r.ready():
                still.append(r); continue
            it = r.get()
            u = it["uid"]
            shutil.rmtree(DATA / "train_series" / u, ignore_errors=True)
            md, fo = mode_of[u], fold_of(u)
            rec = {"uid": u, "mode": md, "fold": fo, "error": it.get("error"), "render_wall": round(it.get("wall", 0), 2),
                   "seconds": it.get("seconds"), "anchor": (it.get("anchor") or {}).get("method"), "side": it.get("side")}
            fl = it.get("fluid")
            rec["fluid"] = None if fl is None else {k: fl.get(k) for k in ("status", "eff_sag_p99_open", "bak_sag_p99_open", "n_sag_fluid_series")}
            if not rec["error"]:
                if len(it["images"]) == 0:
                    rec["error"] = "no usable window"
                else:
                    tg = time.time()
                    try:
                        out, resc = sc.score(it["images"], it["meta"], [fo] if md == "oof" else list(range(K)),
                                             list(range(K)) if md == "all" else [fo])
                        rec.update(out); rec["windows"] = int(len(it["images"])); rec["gpu_s"] = round(time.time() - tg, 3)
                        stats["gpu_s"] += time.time() - tg; stats["rescues"] += resc
                    except Exception as exc:
                        rec["error"] = "gpu: " + repr(exc)[:300]
            del it
            write(rec); sem.release(); remaining -= 1
            if rec["error"]:
                stats["errors"] += 1
            else:
                stats["scored"] += 1; stats["by_mode"][md] += 1
            if time.time() - last[0] > 60:
                progress()
        results = still
        if time.time() - last[0] > 60:
            progress()
        time.sleep(0.05)
    pool.close(); pool.join(); tpool.shutdown()
    fh.close()
    receipt["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    receipt["stats"] = {k: v for k, v in stats.items() if k != "t0"}
    json.dump(receipt, open(OUT / "receipt.json", "w"), indent=1)
    log("DONE", json.dumps(receipt["stats"]))


if __name__ == "__main__":
    main()
