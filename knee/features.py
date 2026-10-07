"""Frozen DINOv2 slice features and explicitly masked prediction heads (ML only)."""
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn


def find_encoder(root):
    root = Path(root)
    direct = [root / "models/metaresearch/dinov2/pytorch/small/1",
              root / "dinov2/pytorch/small/1"]
    matches = []
    candidates = [p / "config.json" for p in direct]
    if (root / "config.json").exists():
        candidates.append(root / "config.json")
    for path in candidates:
        try:
            config = json.loads(path.read_text())
        except (ValueError, OSError):
            continue
        if config.get("model_type") == "dinov2" and config.get("hidden_size") == 384:
            matches.append(path.parent)
    matches = list(dict.fromkeys(matches))
    if len(matches) != 1:
        raise ValueError(f"Expected one DINOv2-Small attachment, found {len(matches)}")
    return matches[0]


def load_encoder(path, device):
    from transformers import AutoModel
    model = AutoModel.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    if model.config.hidden_size != 384 or model.config.model_type != "dinov2":
        raise ValueError("Encoder architecture differs from recorded feature contract")
    return model.to(device).eval().requires_grad_(False)


@torch.inference_mode()
def encode_volume(model, volume, device, batch_size=24):
    """Only real slices enter the encoder; masked padding stays exactly zero."""
    mask = np.asarray(volume.slice_mask, dtype=bool)
    images = np.asarray(volume.images)
    if images.shape[:2] != mask.shape:
        raise ValueError("Image/mask shape mismatch")
    output = np.zeros((*mask.shape, 768), dtype=np.float16)
    flat = images[mask]
    mean = torch.tensor([.485, .456, .406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([.229, .224, .225], device=device).view(1, 3, 1, 1)
    blocks = []
    for offset in range(0, len(flat), batch_size):
        x = torch.from_numpy(flat[offset:offset + batch_size]).to(device).unsqueeze(1).expand(-1, 3, -1, -1)
        tokens = model(pixel_values=(x - mean) / std).last_hidden_state
        feature = torch.cat([tokens[:, 0], tokens[:, 1:].mean(1)], dim=1)
        if not torch.isfinite(feature).all():
            raise ValueError("Nonfinite encoder features")
        blocks.append(feature.cpu().numpy().astype(np.float16))
    if blocks:
        output[mask] = np.concatenate(blocks)
    return output


def fit_feature_stats(features, masks, train_indices):
    """Fit only explicitly selected training slices, never padding/validation."""
    indices = np.asarray(train_indices, dtype=int)
    if not len(indices) or len(set(indices.tolist())) != len(indices):
        raise ValueError("Training indices must be nonempty and unique")
    selected = np.asarray(features[indices], dtype=np.float32)[masks[indices].astype(bool)]
    if not len(selected) or not np.isfinite(selected).all():
        raise ValueError("No finite training features")
    return selected.mean(0), np.maximum(selected.std(0), .05)


class StudyHead(nn.Module):
    """Finding-specific attention or masked mean/max pooling, with missing-study prior."""
    def __init__(self, mean, std, priors, hidden=128, kind="attention", dropout=.2):
        super().__init__()
        if kind not in ("attention", "meanmax"):
            raise ValueError("Unknown head kind")
        self.kind = kind
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        prior = torch.as_tensor(priors, dtype=torch.float32).clamp(.001, .999)
        self.register_buffer("prior_logits", torch.logit(prior))
        width, n_targets = len(mean), len(priors)
        self.proj = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout))
        self.plane = nn.Parameter(torch.zeros(3, hidden))
        self.query = nn.Parameter(torch.randn(n_targets, hidden) * .02)
        self.out = nn.Linear(hidden if kind == "attention" else hidden * 6, n_targets)

    def forward(self, features, masks):
        if features.ndim != 4 or features.shape[1] != 3 or masks.shape != features.shape[:3]:
            raise ValueError("Expected [batch,3,slices,features] and matching mask")
        mask = masks.bool()
        x = (features.float() - self.mean) / self.std
        x = torch.where(mask[..., None], x, torch.zeros_like(x))
        h = self.proj(x) + self.plane[None, :, None]
        if self.kind == "attention":
            h = h.flatten(1, 2)
            weights = torch.einsum("bsh,th->bts", h, self.query) / h.shape[-1] ** .5
            weights = weights.masked_fill(~mask.flatten(1)[:, None], -1e4).softmax(-1)
            pooled = torch.einsum("bts,bsh->bth", weights, h)
            logits = (pooled * self.out.weight[None]).sum(-1) + self.out.bias
        else:
            count = mask.sum(2).clamp_min(1)[..., None]
            avg = (h * mask[..., None]).sum(2) / count
            maximum = h.masked_fill(~mask[..., None], -1e4).amax(2)
            maximum = torch.where(mask.any(2)[..., None], maximum, torch.zeros_like(maximum))
            logits = self.out(torch.cat([avg, maximum], dim=-1).flatten(1))
        return torch.where(mask.flatten(1).any(1)[:, None], logits, self.prior_logits)


def masked_loss(logits, targets, confidence):
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    support = confidence.sum(0)
    valid = support > 0
    if not valid.any():
        raise ValueError("Batch has no supervised target")
    return ((loss * confidence).sum(0)[valid] / support[valid]).mean()
