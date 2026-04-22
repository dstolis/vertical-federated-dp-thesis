"""Bottom and top model definitions for the VFL split architecture.

The two-party split follows the formalisation in Chapter 3 of the thesis:
each party trains a bottom model f_A / f_B that maps its feature slice to a
d-dimensional embedding; the server fuses the two embeddings through a top
model g that produces class logits.
"""
from __future__ import annotations

import torch
from torch import nn


class BottomNet(nn.Module):
    """Party-side bottom model: flattened MLP producing a fixed-size embedding."""

    def __init__(self, input_width: int, embedding_dim: int = 64, image_height: int = 28):
        super().__init__()
        self.input_width = input_width
        self.image_height = image_height
        self.embedding_dim = embedding_dim
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(image_height * input_width, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
            # LayerNorm stabilises IR scale so that clipping threshold C and
            # injected noise sigma = z*C operate on a predictable signal range
            # (roughly ||e||_2 ~ sqrt(d)). Without this, IR magnitudes drift
            # with training and the adaptive-C loop chases a moving target.
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 1, H, W) -> (B, d)
        return self.net(x)


class TopNet(nn.Module):
    """Server-side fusion model: MLP over concatenated embeddings."""

    def __init__(self, embedding_dim: int = 64, num_parties: int = 2, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim * num_parties, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Update learning rate in-place, preserving Adam's momentum / variance state."""
    for pg in optimizer.param_groups:
        pg["lr"] = lr
