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
from torch.utils.data import DataLoader, Dataset, Subset
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


# Fixed seed for the train/validation split, independent of the run seed, so
# every configuration and every seed is selected on the same validation set.
VAL_SPLIT_SEED = 2026


def _normalised_tensor(base) -> Tuple[torch.Tensor, torch.Tensor]:
    """The whole dataset as one tensor, with exactly the arithmetic of
    ToTensor() followed by Normalize((0.1307,), (0.3081,)) (same float32 ops,
    so the values are bit-identical to the per-sample transform)."""
    x = base.data.unsqueeze(1).to(dtype=torch.float32).div(255)
    mean = torch.as_tensor((0.1307,), dtype=x.dtype).view(-1, 1, 1)
    std = torch.as_tensor((0.3081,), dtype=x.dtype).view(-1, 1, 1)
    x = x.sub_(mean).div_(std)
    return x, torch.as_tensor(base.targets, dtype=torch.int64)


class PrecomputedSplitDataset(Dataset):
    """Vertically split samples held in memory. Indexing and batch fetching
    (``__getitems__``) return the same (x_left, x_right, y) values as
    ``VerticalSplitDataset``, without re-running the image transform on every
    access; the DataLoader's shuffling is unchanged because it depends only
    on the dataset length and the generator."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor, num_splits: int = 2):
        w = x.shape[-1] // num_splits
        self.left, self.right, self.y = x[..., :w].contiguous(), x[..., w:].contiguous(), y

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.left[idx], self.right[idx], int(self.y[idx])

    def __getitems__(self, idxs):
        i = torch.as_tensor(idxs, dtype=torch.long)
        return self.left[i], self.right[i], self.y[i]


def _batch_identity(batch):
    return batch


def get_dataloaders_with_val(
    dataset: str = "mnist",
    batch_size: int = 64,
    seed: int = 42,
    val_size: int = 10_000,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """Train/validation/test loaders for the vertical split.

    ``val_size`` samples are held out of the official training split for model
    selection (best-checkpoint restore) and the noise-lock; the official test
    split is only used to report the selected model. Returns
    ``(train_loader, val_loader, test_loader, split_width)``.
    """
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    cls = _get_torchvision_cls(dataset)
    train_base = cls(str(DATA_ROOT), train=True, download=True, transform=tfm)
    test_base = cls(str(DATA_ROOT), train=False, download=True, transform=tfm)

    x_tr, y_tr = _normalised_tensor(train_base)
    x_te, y_te = _normalised_tensor(test_base)
    perm = torch.randperm(len(y_tr), generator=torch.Generator().manual_seed(VAL_SPLIT_SEED))
    val_idx, train_idx = perm[:val_size], perm[val_size:]
    train_ds = PrecomputedSplitDataset(x_tr[train_idx], y_tr[train_idx])
    val_ds = PrecomputedSplitDataset(x_tr[val_idx], y_tr[val_idx])
    test_ds = PrecomputedSplitDataset(x_te, y_te)

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=g,
        collate_fn=_batch_identity,
    )
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=num_workers,
                            collate_fn=_batch_identity)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=num_workers,
                             collate_fn=_batch_identity)
    return train_loader, val_loader, test_loader, train_ds.left.shape[-1]
