"""352-pixel local64 archive access; kept separate from frozen336 Raptor64 loader."""
from pathlib import Path

from knee import local64_cache as cache


def load_study(path, *, expected_hashes=None):
    """Load and verify one local64 NPZ, optionally binding it to a private index row."""
    import numpy as np
    volume, mask, joint_mask = cache.read_npz(path)
    hashes = cache.array_hashes(volume, mask, joint_mask)
    if expected_hashes is not None and any(hashes.get(k) != expected_hashes.get(k) for k in hashes):
        raise ValueError("local64 index raw-array hash differs")
    if expected_hashes is not None and "slots" in expected_hashes:
        cache.validate_slot_mask(mask, expected_hashes["slots"])
    volume.flags.writeable = False
    mask.flags.writeable = False
    joint_mask.flags.writeable = False
    return volume, mask, joint_mask


def eval_windows(public, volume, mask, *, k=42, resolution=384):
    """Use the exact audited Dread window function to resize 352 storage to model input."""
    import numpy as np
    cache.validate_arrays(volume, mask, np.zeros((cache.STORAGE, cache.STORAGE), dtype="uint8"))
    if type(resolution) is not int or resolution != cache.RENDER_SIZE:
        raise ValueError("local64 model resolution is pinned to 384")
    windows = public.eval_windows(volume, mask, k, resolution)
    if tuple(windows.shape) != (k, 3, resolution, resolution):
        raise ValueError("Pinned Dread window output shape differs")
    import torch
    if not torch.isfinite(windows).all():
        raise ValueError("Pinned Dread window output contains nonfinite pixels")
    return windows


class Local64Proxy:
    """Small lazy per-study proxy over verified local64 private indexes."""
    def __init__(self, root, entries):
        self.root = Path(root)
        self.entries = list(entries)
        if [e.get("ordinal") for e in self.entries] != list(range(len(self.entries))):
            raise ValueError("Proxy entries must be in merged ordinal order")
        self._cached_row = None
        self._cached = None

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, row):
        if type(row) is not int or not 0 <= row < len(self.entries):
            raise IndexError("Integer local64 row required")
        entry = self.entries[row]
        if entry.get("file") != cache.filename(entry["ordinal"]):
            raise ValueError("local64 row filename differs from ordinal")
        path = cache.safe_file(self.root, entry["file"])
        if self._cached_row != row:
            self._cached = load_study(path, expected_hashes=entry)
            self._cached_row = row
        return self._cached
