"""VM-side pixel-cache builder for the s6v2 lane: competition archive (HTTP range reads) -> per-study DICOM staging ->
heliosli's compact_infer.CStudies.rows (native_cache.prepare -> geometry_bank -> SKM-TEA anchors -> px_cache.StudySlices.build,
the same path compact_infer scores test studies with) -> px.u8.npy / meta.f32.npy / index.json in px_cache's format, read by
compact_train.  DICOMs are deleted as soon as their study is rendered, so disk stays bounded.  Resumable (index.json).
Pixel rows are stored losslessly as one PNG per row (px.png.bin.NNN shards + px.png.off.npy + px.png.shards.json, ~1.9x smaller than px.u8.npy: the raw
v4/320 cache of 4,407 studies (~310 GB) does not fit the 183 GB VM disk); compact_train_s6v2.py decodes them bit-exactly.
Never prints study IDs or credentials.  Config by environment (S6V2_*), defaults for /content."""
import hashlib, io, json, multiprocessing as mp, os, queue, shutil, struct, sys, threading, time, zipfile, zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

E = os.environ.get
DATA = Path(E("S6V2_DATA", "/content/data"))          # train_series.csv, train.csv (UID column only), train_series/
OUT = Path(E("S6V2_OUT", "/content/px"))
SRC = E("S6V2_SRC", "/content/code/heliosli")
CODE = E("S6V2_CODE", "/content/code/cloud_code")
ANATOMY = E("S6V2_ANATOMY", "/content/helios/anatomy.pt")
ORDER = Path(E("S6V2_ORDER", "/content/order.json"))  # gold first, then the pool
TMP = Path(E("S6V2_TMP", "/content/tmp_px"))
WORKERS = int(E("S6V2_WORKERS", "10"))
STAGERS = int(E("S6V2_STAGERS", "8"))
AHEAD = int(E("S6V2_AHEAD", "40"))                     # studies staged on disk at most
LIMIT = int(E("S6V2_LIMIT", "0"))
N_UPPER = int(E("S6V2_N_UPPER", "3200000"))
SHARD_BYTES = int(float(E("S6V2_SHARD_GB", "8")) * 2 ** 30)   # PNG bytes are split into shard files (rolled at study boundaries)            # sparse row allocation; finalize() truncates to the real count
SIZE, VIEWS, SLICES, PHYS = 320, "v4", 3, 3.5
COMP = "rsna-knee-abnormality-detection"
META_COLS = ["series_idx", "slot", "plane", "view_id", "rel_pos", "log_spacing", "extent_130", "fluid",
             "fatsat", "slice_order", "n_slices", "real_series_idx"]


def log(*a):
    print(time.strftime("%H:%M:%SZ", time.gmtime()), *a, flush=True)


# ---------------------------------------------------------------- worker side (nartaa's K1 launcher guards, same code path)
def _strtobool(val):
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    if val in ("n", "no", "f", "false", "off", "0"):
        return False
    raise ValueError("invalid truth value %r" % (val,))


def install_paths():
    import types
    sys.dont_write_bytecode = True
    for p in (SRC, CODE):
        if p not in sys.path:
            sys.path.insert(0, p)
    if "training.download_data" not in sys.modules:
        m = types.ModuleType("training.download_data"); m.ChunkedArchive = None
        sys.modules["training.download_data"] = m
    if "distutils.util" not in sys.modules:
        try:
            import distutils.util  # noqa: F401
        except Exception:
            d, u = types.ModuleType("distutils"), types.ModuleType("distutils.util")
            d.__path__ = []; u.strtobool = _strtobool; d.util = u
            sys.modules["distutils"], sys.modules["distutils.util"] = d, u


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
    wtmp = TMP / ("w%d" % os.getpid()); wtmp.mkdir(parents=True, exist_ok=True)
    _CS = CI.CStudies(str(DATA), "train", [], str(wtmp), CODE, SIZE, eval_stride=1, views=VIEWS, slices=SLICES,
                      anchor_weights=ANATOMY, phys_mm=PHYS)
    _CS._setup()
    if _CS.bank is None:
        raise RuntimeError("SKM-TEA anchor bank did not load")


def build_one(uid):
    t0 = time.time()
    try:
        item, sides = _CS.rows(uid)
        if item.get("error"):
            return uid, None, None, None, item["error"][:300], time.time() - t0
        import cv2, numpy as np
        px = item["px"].numpy()
        enc = []
        for k in range(len(px)):
            ok, b = cv2.imencode(".png", np.ascontiguousarray(px[k]), [cv2.IMWRITE_PNG_COMPRESSION, 3])
            assert ok
            enc.append(b.tobytes())
        for k in range(0, len(px), 97):                       # lossless spot check
            assert np.array_equal(cv2.imdecode(np.frombuffer(enc[k], np.uint8), cv2.IMREAD_UNCHANGED), px[k]), "PNG not lossless"
        return uid, enc, item["meta"].numpy(), {"blocks": item["blocks"], "removed": int(item.get("removed", 0)),
                                                                "skipped": len(item.get("skipped_series", []))}, None, time.time() - t0
    except Exception as exc:
        return uid, None, None, None, repr(exc)[:300], time.time() - t0


# ---------------------------------------------------------------- archive staging (range reads of the official archive)
class Archive:
    def __init__(self):
        self.lock = threading.Lock()
        self.url = self.signed_url()
        import requests
        self.requests = requests
        h = requests.get(self.url, headers={"Range": "bytes=0-0"}, timeout=60)
        assert h.status_code == 206, "NO_RANGE_SUPPORT"
        self.total = int(h.headers["Content-Range"].split("/")[1])
        log("archive bytes", self.total)

    def signed_url(self):
        import subprocess
        r = subprocess.run(["/usr/local/bin/python", "-I", "/content/vm_signed_url.py"], capture_output=True, text=True, timeout=900)
        if r.returncode != 0:
            raise RuntimeError("SIGNED_URL_UNAVAILABLE " + r.stdout[-200:])
        return open("/content/.signed_url").read().strip()

    def refresh(self, old):
        with self.lock:
            if self.url == old:
                self.url = self.signed_url(); log("signed url refreshed")

    def rng(self, lo, hi):
        for i in range(8):
            url = self.url
            try:
                r = self.requests.get(url, headers={"Range": "bytes=%d-%d" % (lo, hi)}, timeout=300)
                if r.status_code == 206 and len(r.content) == hi - lo + 1:
                    return r.content
                if r.status_code in (400, 401, 403):
                    self.refresh(url)
            except Exception:
                pass
            time.sleep(2 * (i + 1))
        raise IOError("RANGE_FAIL")


class Remote(io.RawIOBase):
    def __init__(s, ar):
        s.ar, s.pos, s.cache, s.B = ar, 0, {}, 4 << 20

    def seekable(s): return True
    def readable(s): return True
    def tell(s): return s.pos

    def seek(s, o, w=0):
        s.pos = o if w == 0 else s.pos + o if w == 1 else s.ar.total + o
        return s.pos

    def block(s, b):
        if b not in s.cache:
            lo = b * s.B; hi = min(s.ar.total, lo + s.B) - 1
            s.cache[b] = s.ar.rng(lo, hi)
        return s.cache[b]

    def read(s, n=-1):
        if n is None or n < 0:
            n = s.ar.total - s.pos
        n = max(0, min(n, s.ar.total - s.pos)); out = bytearray()
        while len(out) < n:
            b, off = divmod(s.pos, s.B); chunk = s.block(b)[off:off + n - len(out)]
            if not chunk:
                break
            out += chunk; s.pos += len(chunk)
        return bytes(out)

    def readinto(s, buf):
        d = s.read(len(buf)); buf[:len(d)] = d; return len(d)


def fetch_study(ar, members):
    zs = sorted(members, key=lambda z: z.header_offset)
    groups, cur = [], [zs[0]]
    for z in zs[1:]:
        last = cur[-1]
        if z.header_offset - (last.header_offset + last.compress_size + 30 + len(last.filename) + 1024) < (8 << 20):
            cur.append(z)
        else:
            groups.append(cur); cur = [z]
    groups.append(cur); n = nbytes = 0
    for g in groups:
        lo = g[0].header_offset
        hi = min(ar.total - 1, g[-1].header_offset + g[-1].compress_size + 30 + len(g[-1].filename) + 65600)
        buf = ar.rng(lo, hi)
        for z in g:
            o = z.header_offset - lo
            assert buf[o:o + 4] == b"PK\x03\x04", "LOCAL_HEADER"
            fn_len, ex_len = struct.unpack("<HH", buf[o + 26:o + 30])
            data = buf[o + 30 + fn_len + ex_len:o + 30 + fn_len + ex_len + z.compress_size]
            assert len(data) == z.compress_size, "SHORT_MEMBER"
            raw = data if z.compress_type == 0 else zlib.decompress(data, -15) if z.compress_type == 8 else None
            if raw is None:
                raise ValueError("COMPRESSION_%d" % z.compress_type)
            assert len(raw) == z.file_size and (zlib.crc32(raw) & 0xffffffff) == z.CRC, "CRC_MISMATCH"
            p = DATA / z.filename; p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(p.suffix + ".part"); tmp.write_bytes(raw); os.replace(tmp, p)
            n += 1; nbytes += len(raw)
    return n, nbytes


# ---------------------------------------------------------------- npy helpers
def open_mm(path, shape, dtype):
    import numpy as np
    if path.exists():
        mm = np.load(path, mmap_mode="r+")
        assert tuple(mm.shape) == tuple(shape), "existing memmap has another shape"
        return mm
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def finalize_npy(path, n_rows):
    """Rewrite the .npy header with the real row count (same header length) and truncate the sparse tail."""
    import numpy as np
    with open(path, "r+b") as fh:
        version = np.lib.format.read_magic(fh)
        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = reader(fh)
        start = fh.tell()
        new = (n_rows,) + tuple(shape[1:])
        hdr = "{'descr': %r, 'fortran_order': False, 'shape': %r, }" % (np.lib.format.dtype_to_descr(dtype), new)
        hlen_field = 2 if version == (1, 0) else 4
        room = start - 6 - 2 - hlen_field
        assert len(hdr) + 1 <= room, "header does not fit"
        hdr = hdr + " " * (room - len(hdr) - 1) + "\n"
        fh.seek(6 + 2 + hlen_field); fh.write(hdr.encode("latin1"))
        row_bytes = int(np.prod(shape[1:])) * dtype.itemsize
        fh.truncate(start + n_rows * row_bytes)
    chk = np.load(path, mmap_mode="r")
    assert chk.shape == new


# ---------------------------------------------------------------- main
def watchdog():
    """Log memory / disk every 30 s; abort the build before the VM runs out of RAM or disk."""
    def avail_gb():
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2 ** 20
        return 1e9
    n = 0
    while True:
        m = avail_gb(); d = shutil.disk_usage("/content").free / 2 ** 30
        if n % 4 == 0:
            log("WATCH mem_avail_gb %.1f disk_free_gb %.1f" % (m, d))
        if m < 8 or d < 6:
            log("WATCHDOG ABORT mem_avail_gb %.1f disk_free_gb %.1f" % (m, d)); os._exit(9)
        n += 1; time.sleep(30)


def binding():
    def h(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    return {"order_sha256": h(ORDER), "anatomy_sha256": h(ANATOMY), "px_cache_sha256": h(Path(SRC) / "px_cache.py"),
            "compact_infer_sha256": h(Path(SRC) / "compact_infer.py"), "train_series_sha256": h(DATA / "train_series.csv"),
            "size": SIZE, "views": VIEWS, "slices": SLICES, "phys_mm": PHYS}


def main():
    import numpy as np
    threading.Thread(target=watchdog, daemon=True).start()
    OUT.mkdir(parents=True, exist_ok=True); TMP.mkdir(parents=True, exist_ok=True)
    order = json.load(open(ORDER))
    if LIMIT:
        order = order[:LIMIT]
    index_path = OUT / "index.json"
    if index_path.exists():
        index = json.load(open(index_path))
        assert index["config"]["size"] == SIZE and index["config"]["views"] == VIEWS
        assert index["order"] == order, "resume: study order differs"
        bind = index["config"].get("binding")
        assert bind is None or bind == binding(), "resume: source / anatomy / config binding differs"
    else:
        index = {"order": order, "n_upper": N_UPPER, "done": {}, "blocks": {}, "cursor": 0, "errors": {}, "removed_series": 0,
                 "meta_cols": META_COLS, "seconds": {},
                 "config": {"size": SIZE, "local_mm": 100.0, "policy": "series_companded", "views": VIEWS, "anchors": "skm-tea (per study, compact_infer.CStudies.rows)",
                            "view_recipe": "v4 via compact_infer.CStudies.rows", "duplicates": "collapse_exact",
                            "rows": "one per (series, view, slice), physical order", "builder": "s6v2 vm_build_cache.py",
                            "binding": binding()}}
    off_path, shards_path = OUT / "px.png.off.npy", OUT / "px.png.shards.json"
    offsets = list(np.load(off_path)[:index["cursor"] + 1].tolist()) if off_path.exists() else [0]
    assert len(offsets) == index["cursor"] + 1, "offsets / cursor mismatch"
    shards = json.load(open(shards_path)) if shards_path.exists() else [{"file": "px.png.bin.000", "byte0": 0, "complete": False}]
    while len(shards) > 1 and shards[-1]["byte0"] > offsets[-1]:
        shards.pop()
    shards[-1]["complete"] = False
    cur_shard = OUT / shards[-1]["file"]
    binf = open(cur_shard, "r+b" if cur_shard.exists() else "w+b")
    binf.truncate(offsets[-1] - shards[-1]["byte0"]); binf.seek(offsets[-1] - shards[-1]["byte0"])
    meta_mm = open_mm(OUT / "meta.f32.npy", (index["n_upper"], len(META_COLS)), np.float32)
    todo = [u for u in order if u not in index["done"] and u not in index["errors"]]
    log("studies", len(order), "todo", len(todo), "done", len(index["done"]), "errors", len(index["errors"]))

    def save(final=False):
        binf.flush(); os.fsync(binf.fileno()); meta_mm.flush()
        np.save(str(off_path) + ".tmp.npy", np.asarray(offsets, np.int64)); os.replace(str(off_path) + ".tmp.npy", off_path)
        if final:
            shards[-1]["complete"] = True
        json.dump(shards, open(str(shards_path) + ".tmp", "w")); os.replace(str(shards_path) + ".tmp", shards_path)
        tmp = str(index_path) + ".tmp"
        json.dump(index, open(tmp, "w")); os.replace(tmp, index_path)
        prog = {"done": len(index["done"]), "errors": len(index["errors"]), "total": len(order), "rows": index["cursor"],
                "gb_raw": round(index["cursor"] * SIZE * SIZE / 2 ** 30, 2), "gb_png": round(offsets[-1] / 2 ** 30, 2),
                "disk_free_gb": round(shutil.disk_usage("/content").free / 2 ** 30, 1), "elapsed_s": round(time.time() - T0, 1),
                "rate_per_min": round(60 * stats["built"] / max(1.0, time.time() - T0), 2),
                "eta_min": round((len(todo) - stats["built"]) / max(1e-6, stats["built"] / max(1.0, time.time() - T0)) / 60, 1),
                "staged_mb": round(stats["staged_bytes"] / 2 ** 20), "final": final, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        json.dump(prog, open(OUT / "progress.json", "w"))
        log("PROGRESS", json.dumps(prog))

    T0 = time.time()
    stats = {"built": 0, "staged_bytes": 0}
    if todo:
        log("authenticating / signed URL")
        ar = Archive()
        zf = zipfile.ZipFile(io.BufferedReader(Remote(ar), 4 << 20))
        infos = zf.infolist(); log("archive members", len(infos))
        by_study = {}
        for z in infos:
            parts = z.filename.split("/")
            if len(parts) == 4 and parts[0] == "train_series" and not z.is_dir():
                by_study.setdefault(parts[1], []).append(z)
        missing = [u for u in todo if u not in by_study]
        log("todo studies missing from the archive", len(missing))
        for u in missing:
            index["errors"][u] = "absent from archive"
        todo = [u for u in todo if u in by_study]
        sem = threading.Semaphore(AHEAD)
        staged_q = queue.Queue()

        def stage(u):
            sem.acquire()
            try:
                n, b = fetch_study(ar, by_study[u])
                stats["staged_bytes"] += b
                staged_q.put((u, None))
            except Exception as exc:
                staged_q.put((u, "stage: " + repr(exc)[:200]))

        tpool = ThreadPoolExecutor(STAGERS)
        for u in todo:
            tpool.submit(stage, u)
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(WORKERS, initializer=worker_init, maxtasksperchild=200)
        pending, last = 0, time.time()
        stage_fail = {}

        def feed():
            nonlocal pending
            while True:
                try:
                    u, err = staged_q.get_nowait()
                except queue.Empty:
                    return
                if err:
                    stage_fail[u] = err; sem.release()
                    shutil.rmtree(DATA / "train_series" / u, ignore_errors=True)
                    continue
                results.append(pool.apply_async(build_one, (u,)))
                pending += 1

        results = []
        remaining = len(todo)
        while remaining > 0:
            feed()
            if stage_fail:
                for u, err in list(stage_fail.items()):
                    index["errors"][u] = err; remaining -= 1; del stage_fail[u]
            still = []
            for r in results:
                if not r.ready():
                    still.append(r); continue
                uid, px, meta, info, err, secs = r.get()
                pending -= 1; remaining -= 1; stats["built"] += 1
                shutil.rmtree(DATA / "train_series" / uid, ignore_errors=True); sem.release()
                index["seconds"][uid] = round(secs, 2)
                if err:
                    index["errors"][uid] = err; continue
                n = len(px); cur = index["cursor"]
                if cur + n > index["n_upper"]:
                    raise RuntimeError("row count exceeds the allocated upper bound")
                if n:
                    for b_ in px:
                        binf.write(b_); offsets.append(offsets[-1] + len(b_))
                    meta_mm[cur:cur + n] = meta
                blocks = info["blocks"]
                for b in blocks:
                    b["views"] = [[v, cur + off] for v, off in b["views"]]
                index["done"][uid] = [cur, n, len(blocks)]
                index["blocks"][uid] = blocks
                index["cursor"] = cur + n
                index["removed_series"] += info["removed"]
                if offsets[-1] - shards[-1]["byte0"] >= SHARD_BYTES:        # roll the shard at a study boundary
                    binf.flush(); os.fsync(binf.fileno()); binf.close()
                    shards[-1]["complete"] = True
                    shards.append({"file": "px.png.bin.%03d" % len(shards), "byte0": offsets[-1], "complete": False})
                    binf = open(OUT / shards[-1]["file"], "w+b")
                    save(); last = time.time()
            results = still
            if time.time() - last > 60:
                save(); last = time.time()
            time.sleep(0.2)
        pool.close(); pool.join(); tpool.shutdown()
    save(final=True)
    log("DONE rows", index["cursor"], "studies", len(index["done"]), "errors", len(index["errors"]))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "finalize":
        idx = json.load(open(OUT / "index.json"))
        n = idx["cursor"]
        finalize_npy(OUT / "meta.f32.npy", n)
        import numpy as np
        sh = json.load(open(OUT / "px.png.shards.json"))
        while len(sh) > 1 and os.path.getsize(OUT / sh[-1]["file"]) == 0:     # empty tail shard rolled after the last study
            os.remove(OUT / sh[-1]["file"]); sh.pop()
        json.dump(sh, open(OUT / "px.png.shards.json", "w"))
        off = np.load(OUT / "px.png.off.npy")
        assert len(off) == n + 1 and all(x["complete"] for x in sh)
        assert sum(os.path.getsize(OUT / x["file"]) for x in sh) == off[-1]
        idx["n_upper"] = n; json.dump(idx, open(OUT / "index.json", "w"))
        (OUT / "FINALIZED").write_text("rows %d\n" % n)
        log("finalized rows", n)
    else:
        main()
