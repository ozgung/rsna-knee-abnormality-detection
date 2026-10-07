"""Toy multi-plane 2D encoder. No pretrained weights; not a trained baseline."""
import torch
from torch import nn


class StudyEncoder(nn.Module):
    def __init__(self, num_targets=12):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.head = nn.Linear(16, num_targets)

    def forward(self, images):
        # [batch, planes, slices, channels=1, height, width].
        # Fixed complete synthetic bags only; real data needs padding masks.
        if images.ndim != 6 or images.shape[3] != 1 or min(images.shape) < 1:
            raise ValueError("Expected nonempty [B,P,S,1,H,W] input")
        batch, planes, slices, channels, height, width = images.shape
        features = self.encoder(images.reshape(-1, channels, height, width))
        pooled = features.reshape(batch, planes, slices, -1).mean(dim=(1, 2))
        return self.head(pooled)  # logits; apply sigmoid at inference
