"""Strict CPU acceptance of the two real checkpoints in the frozen S6 solo pair.

Synthetic inputs only: this does not qualify the DICOM reader, CUDA or runtime.
Uses the captured publisher constructor, with no downloads or initialization file.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from knee.public_adaptation_pilot import state_fingerprint
from knee.s6_solo_pair import CONTROL_SHA256, TARGETS

TRAIN_SHA = "14767c7a8f0daa986477511fe9a83aefe032a208ae978fabf15a5f896992ac80"
INFER_SHA = "34f433fe72b732543813165c3397ec80cef01b0d49b25b7b22668714f0b0ef4a"
V2_SHA = "e32e5976c60a8d2f33954e348f4b310e49e9739daf4b2012335071cbedb8a972"
CONSTRUCTOR = dict(backbone="convnext_small", init=None, hidden=384, drop=0.1,
                   freeze_blocks=16, crop=288, state_spec=None, in_chans=3,
                   spatial=True, extent=False)


def file_sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def require(condition, check):
    if not condition:
        raise ValueError(check)


def qualify(train, infer, control, v2, v2_sha=V2_SHA, v2_bytes=200188027):
    import torch
    import timm

    require(file_sha(train) == TRAIN_SHA, "PINNED_TRAIN_SOURCE")
    require(file_sha(infer) == INFER_SHA, "PINNED_INFER_SOURCE")
    spec = importlib.util.spec_from_file_location("s6solo_pinned_constructor", train)
    ct = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ct)
    require(ct.LABELS == TARGETS, "CANONICAL_TARGETS")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    img = torch.randint(0, 256, (1, 3, 3, 288, 288), dtype=torch.uint8)
    meta = torch.zeros(1, 3, 12)
    meta[0, :, 0] = torch.tensor([0, 9, 0])
    meta[0, :, 1] = torch.tensor([0, 5, 0])
    mask = torch.tensor([[True, False, True]])
    altered_img, altered_meta = img.clone(), meta.clone()
    altered_img[:, 1] = 255 - altered_img[:, 1]
    altered_meta[:, 1] = 4.0
    receipt = dict(status="PASS_SYNTHETIC_CPU_ONLY", python=sys.version.split()[0],
                   torch=torch.__version__, timm=timm.__version__, device="cpu",
                   precision="fp32", constructor=CONSTRUCTOR, targets=TARGETS,
                   constructor_source_sha256=TRAIN_SHA, inference_source_sha256=INFER_SHA,
                   qualifier_sha256=file_sha(__file__), chunk=192,
                   input_shape=list(img.shape), metadata_shape=list(meta.shape),
                   mask=mask.tolist(), synthetic_input_sha256=state_fingerprint([img, meta, mask]),
                   real_image_reader_qualified=False, cuda_qualified=False,
                   runtime_qualified=False, checkpoints={})
    for name, path, expected_hash, expected_size in (
        ("control", control, CONTROL_SHA256, 200185003),
        ("v2", v2, v2_sha, v2_bytes),        # Lane C seeds pass their own export identity
    ):
        path = Path(path)
        require((file_sha(path), path.stat().st_size) == (expected_hash, expected_size), "CHECKPOINT_BYTES_" + name)
        sd = torch.load(path, map_location="cpu", weights_only=True)
        require(isinstance(sd, dict) and sd and all(torch.is_tensor(t) for t in sd.values()), "RAW_STATE_DICT_" + name)
        require(all(bool(torch.isfinite(t).all()) for t in sd.values()), "FINITE_STATE_" + name)
        require("sp_q" in sd and not any(k.startswith(("state_", "bin_", "ext_")) for k in sd), "HEAD_CONTRACT_" + name)
        model = ct.Net(**CONSTRUCTOR).eval()
        model.load_state_dict(sd, strict=True)
        before = state_fingerprint(model.state_dict())
        require(before == state_fingerprint(sd), "STRICT_STATE_IDENTITY_" + name)
        buffers = dict(model.named_buffers())
        started = time.monotonic()
        with torch.inference_mode():
            logits = model(img, meta, mask, chunk=192)
            repeated = model(img, meta, mask, chunk=192)
            masked_changed = model(altered_img, altered_meta, mask, chunk=192)
        require(list(logits.shape) == [1, len(TARGETS)], "OUTPUT_SHAPE_" + name)
        require(all(bool(torch.isfinite(t).all()) for t in (logits, repeated, masked_changed)), "FINITE_OUTPUT_" + name)
        repeat_delta = float((logits - repeated).abs().max())
        mask_delta = float((logits - masked_changed).abs().max())
        require(repeat_delta == 0 and mask_delta == 0, "REPEAT_MASK_INVARIANCE_" + name)
        require(state_fingerprint(model.state_dict()) == before, "UNCHANGED_STATE_" + name)
        probs = torch.sigmoid(logits)
        require(bool(((probs >= 0) & (probs <= 1)).all()), "PROBABILITY_RANGE_" + name)
        receipt["checkpoints"][name] = dict(checkpoint_sha256=expected_hash,
            checkpoint_size=expected_size, state_sha256=before, state_tensors=len(sd),
            constructor_buffers_sha256=state_fingerprint(buffers),
            buffer_inventory={k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in buffers.items()},
            strict_load=True, all_state_finite=True, unchanged_state=True,
            output_shape=list(logits.shape), output_sha256=state_fingerprint(logits),
            repeat_max_abs=repeat_delta, masked_perturbation_max_abs=mask_delta,
            forward_count=3, forward_seconds=time.monotonic() - started)
        print(json.dumps({"checkpoint": name, **{k: receipt["checkpoints"][name][k] for k in
            ("state_sha256", "state_tensors", "repeat_max_abs", "masked_perturbation_max_abs", "forward_seconds")}}), flush=True)
        del model, sd
    return receipt


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for field in ("train", "infer", "control", "v2", "out"):
        p.add_argument("--" + field, type=Path, required=True)
    p.add_argument("--v2-sha", default=V2_SHA, help="expected SHA256 of --v2 (default: the accepted V2-all seed 42 export)")
    p.add_argument("--v2-bytes", type=int, default=200188027)
    a = p.parse_args()
    require(not a.out.exists(), "FRESH_RECEIPT")
    result = qualify(a.train, a.infer, a.control, a.v2, a.v2_sha, a.v2_bytes)
    with a.out.open("x") as f:
        f.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
