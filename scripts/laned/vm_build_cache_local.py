"""Lane D (VM): run the unchanged s6v2 cache builder (vm_build_cache.py, same CStudies.rows render path, same PNG row store)
on studies that are already on local disk (the hard-linked fastMRI layout), instead of range-reading the competition archive.
Only the archive staging is replaced: the member list comes from the local tree, staging is a no-op and the rendered study's
DICOMs are NOT deleted (they are hard links; the teacher datasets share them).
Env as vm_build_cache.py: S6V2_DATA=/content/fmdata S6V2_OUT=/content/pxfm S6V2_ORDER=/content/fm_order.json ...
Usage: python -I vm_build_cache_local.py   (then: python -I vm_build_cache.py finalize, same env)"""
import os, sys, types
sys.path.insert(0, "/content")
import vm_build_cache as B


class _Ar:
    total = 0
    def __init__(self):
        pass


class _Z:
    def __init__(self, name):
        self.filename = name
    def is_dir(self):
        return False


class _ZipFile:
    def __init__(self, *a, **k):
        pass
    def infolist(self):
        root = B.DATA / "train_series"
        out = []
        for st in sorted(os.listdir(root)):
            for se in sorted(os.listdir(root / st)):
                for f in sorted(os.listdir(root / st / se)):
                    out.append(_Z("train_series/%s/%s/%s" % (st, se, f)))
        return out


class _Shutil:
    def __getattr__(self, k):
        import shutil
        return getattr(shutil, k)
    @staticmethod
    def rmtree(path, ignore_errors=False):
        if str(path).startswith(str(B.DATA)):
            return None                      # keep the hard-linked DICOMs
        import shutil
        return shutil.rmtree(path, ignore_errors=ignore_errors)


B.Archive = _Ar
B.Remote = lambda ar: types.SimpleNamespace()
B.io = types.SimpleNamespace(BufferedReader=lambda raw, n: raw)
B.zipfile = types.SimpleNamespace(ZipFile=_ZipFile)
B.fetch_study = lambda ar, members: (len(members), 0)
B.shutil = _Shutil()

if __name__ == "__main__":
    B.main()        # finalize: run the original vm_build_cache.py finalize with the same env (no archive involved)
