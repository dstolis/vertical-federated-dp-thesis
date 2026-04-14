"""Vertical feature partitioning for image-classification benchmarks.

Each 28x28 image is split vertically into two halves of size 28x14; the left
half is assigned to party A, the right half to party B. This simulates
complementary feature ownership while preserving the standard benchmark
evaluation protocol.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


class VerticalSplitDataset(Dataset):
    """Wraps a torchvision image dataset into (x_left, x_right, y) tuples."""

    def __init__(self, base_dataset: Dataset, num_splits: int = 2):
        self.base = base_dataset
        self.num_splits = num_splits

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        x, y = self.base[idx]
        _, _, width = x.shape
        split_w = width // self.num_splits
        return x[:, :, :split_w], x[:, :, split_w:], y


def _get_torchvision_cls(name: str):
    name = name.lower()
    if name == "mnist":
        return datasets.MNIST
    if name in ("fashion_mnist", "fmnist", "fashion-mnist"):
        return datasets.FashionMNIST
    raise ValueError(f"Unsupported dataset: {name}")


def get_dataloaders(
    dataset: str = "mnist",
    batch_size: int = 64,
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, int]:
    """Build train/test DataLoaders for the vertical split and return input width."""
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    cls = _get_torchvision_cls(dataset)
    train_base = cls(str(DATA_ROOT), train=True, download=True, transform=tfm)
    test_base = cls(str(DATA_ROOT), train=False, download=True, transform=tfm)

    train_ds = VerticalSplitDataset(train_base)
    test_ds = VerticalSplitDataset(test_base)

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=g
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    # Half-width after split (28x28 -> each party sees 28x14).
    sample_left, _, _ = train_ds[0]
    split_width = sample_left.shape[-1]
    return train_loader, test_loader, split_width
