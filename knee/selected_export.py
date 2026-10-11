"""Fail-closed provenance and selected-checkpoint checks for remote exports.

Torch is imported only when checking real tensor payloads. No image or label data
is included in the receipt; an expected version records the external provenance,
while the actual mounted bytes must match its independently recorded SHA.
"""
from __future__ import annotations

import hashlib
import json


CONFIG_REQUIRED = frozenset({"arch", "res", "k", "k_eval", "bs", "accum",
                             "epochs", "topk", "seed", "grad_ckpt",
                             "bb_lr", "head_lr", "wd", "ckpt", "win_chunk",
                             "oof_every", "fp16", "xla", "limit", "freeze_blocks",
                             "norm", "smoke", "folds", "fold", "tag", "roi",
                             "roi_mode", "roi_pad", "roi_overlap"})


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_expectation(expected):
    if type(expected.get("dataset_version")) is not int or expected["dataset_version"] <= 0:
        raise ValueError("positive expected dataset version required")
    value = expected.get("resume_sha256", "")
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("expected resume SHA256 required")
    config = expected.get("config", {})
    if not isinstance(config, dict) or not CONFIG_REQUIRED <= config.keys():
        raise ValueError("incomplete expected training config")
    if type(expected.get("fold")) is not int or expected["fold"] not in range(5):
        raise ValueError("expected fold outside 0..4")
    if config["fold"] != expected["fold"] or config["folds"] != 5 or config["tag"] != f"fold{expected['fold']}":
        raise ValueError("expected config fold differs")
    if type(config["epochs"]) is not int or config["epochs"] <= 0 or type(config["topk"]) is not int or config["topk"] < 2:
        raise ValueError("completed multi-epoch SWA expected")
    for key in ('res', 'k', 'k_eval', 'bs', 'accum'):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"invalid expected config: {key}")
    for key in ('grad_ckpt', 'fp16', 'xla', 'smoke', 'roi'):
        if type(config[key]) is not bool:
            raise ValueError(f"invalid expected config: {key}")
    if config['roi'] or config['norm'] != 'imagenet':
        raise ValueError('this exporter supports global windows with imagenet normalization only')


def validate_resume(resume, expected, current, labels):
    validate_expectation(expected)
    ident = resume.get("identity", {})
    for key, value in {"fold": expected["fold"], "n_folds": 5,
                       "tag": f"fold{expected['fold']}", **current}.items():
        if ident.get(key) != value:
            raise ValueError(f"resume identity mismatch: {key}")
    cfg = ident.get("config", {})
    for key, value in expected["config"].items():
        if cfg.get(key) != value:
            raise ValueError(f"resume training config mismatch: {key}")
    epochs = expected["config"]["epochs"]
    if resume.get("epoch") != epochs - 1 or ident.get("epochs_requested") != epochs:
        raise ValueError("resume training is not complete")
    if [h.get("ep") for h in resume.get("hist", [])] != list(range(epochs)):
        raise ValueError("resume history incomplete")
    topk = resume.get("topk", [])
    if len(topk) != min(epochs, expected["config"]["topk"]):
        raise ValueError("selected top-k incomplete")
    selected = [int(t["epoch"]) for t in topk]
    if len(set(selected)) != len(selected) or any(e not in range(epochs) for e in selected):
        raise ValueError("invalid selected epochs")
    for state in topk:
        if (state.get("arch") != cfg["arch"] or state.get("res") != cfg["res"]
                or list(state.get("lab", [])) != list(labels)):
            raise ValueError("selected state geometry/target order mismatch")
    return ident, selected


def tensor_oracle(resume, selected, selected_epochs):
    """Compare reloaded export with an independently stacked FP64 mean.

    Integer/bool buffers follow the source selector's first-state convention.
    Only one key is stacked at a time, bounding additional CPU memory.
    """
    import torch
    topk = resume["topk"]
    states = [t["model"] for t in topk]
    model = selected["model"]
    if selected.get("swa_over") != selected_epochs:
        raise ValueError("export selected epochs differ")
    if set(model) != set(states[0]) or any(set(s) != set(model) for s in states):
        raise ValueError("selected tensor keys differ")
    digest = hashlib.sha256()
    cells = 0
    nonfloating = 0
    for name in sorted(model):
        actual, first = model[name].detach().cpu(), states[0][name].detach().cpu()
        if actual.dtype != first.dtype or actual.shape != first.shape:
            raise ValueError(f"export tensor schema differs: {name}")
        values = [s[name].detach().cpu() for s in states]
        if any(v.dtype != first.dtype or v.shape != first.shape for v in values):
            raise ValueError(f"top-k tensor schema differs: {name}")
        if first.is_floating_point():
            if any(not torch.isfinite(v).all() for v in values):
                raise ValueError(f"nonfinite selected tensor: {name}")
            oracle = torch.stack([v.to(torch.float64) for v in values]).mean(0).to(first.dtype)
        else:
            oracle = first
            nonfloating += 1
        if not torch.equal(actual, oracle):
            raise ValueError(f"selected tensor equality failed: {name}")
        header = json.dumps([name, str(actual.dtype), list(actual.shape)], separators=(",", ":")).encode()
        raw = actual.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(header).to_bytes(8, "big")); digest.update(header)
        digest.update(len(raw).to_bytes(8, "big")); digest.update(raw)
        cells += actual.numel()
    return {"all_tensor_equal": True, "tensor_count": len(model), "tensor_cells": cells,
            "nonfloating_first_state_buffers": nonfloating,
            "selected_tensor_sha256": digest.hexdigest(), "oracle": "stacked_fp64_mean_then_source_dtype"}


def check_export(resume_path, selected_path, expected, current, labels):
    import torch
    validate_expectation(expected)
    actual_sha = sha256_file(resume_path)
    if actual_sha != expected["resume_sha256"]:
        raise ValueError("mounted resume SHA differs from expected version artifact")
    resume = torch.load(resume_path, map_location="cpu", weights_only=False)
    ident, selected_epochs = validate_resume(resume, expected, current, labels)
    selected = torch.load(selected_path, map_location="cpu", weights_only=False)
    for key in ("arch", "res"):
        if selected.get(key) != ident["config"][key]:
            raise ValueError(f"export geometry mismatch: {key}")
    if list(selected.get("lab", [])) != list(labels):
        raise ValueError("export target order mismatch")
    oracle = tensor_oracle(resume, selected, selected_epochs)
    return {"status": "PASS", "expected_dataset_version": expected["dataset_version"],
            "version_binding": "mounted_resume_bytes_match_external_version_sha",
            "resume_sha256": actual_sha, "selected_file_sha256": sha256_file(selected_path),
            "fold": expected["fold"], "training_config": ident["config"],
            "training_identity": {k: v for k, v in ident.items() if k != "config"},
            "epochs_completed": resume["epoch"] + 1, "selected_epochs": selected_epochs,
            "training_wall_seconds": resume.get("elapsed"), **oracle}
