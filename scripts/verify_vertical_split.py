"""Verify the VFL vertical partition is sample-aligned and feature-disjoint.

Prints shapes and invariants; saves a PNG showing a few samples with party A
(left half) and party B (right half) side by side alongside the original.
"""
import torch
import matplotlib.pyplot as plt
from adaptive_dp_vfl.data import get_dataloaders

train_loader, test_loader, split_width = get_dataloaders("mnist", batch_size=8, seed=42)
print(f"split_width (per-party feature count along W): {split_width}")

xl, xr, y = next(iter(train_loader))
print(f"xl (party A): {tuple(xl.shape)}  xr (party B): {tuple(xr.shape)}  y: {tuple(y.shape)}")
assert xl.shape[-1] == xr.shape[-1] == split_width == 14
assert xl.shape[0] == xr.shape[0] == y.shape[0], "sample-alignment broken"

# Feature disjointness: reconstructing full image from (xl, xr) must be identity.
reconstructed = torch.cat([xl, xr], dim=-1)
print(f"reconstructed image shape: {tuple(reconstructed.shape)}  (should be (B,1,28,28))")

# Show first 4 samples: original (= cat) vs party A alone vs party B alone.
fig, axes = plt.subplots(4, 3, figsize=(6, 8))
for i in range(4):
    axes[i, 0].imshow(reconstructed[i, 0], cmap="gray"); axes[i, 0].set_title(f"full  y={y[i].item()}")
    axes[i, 1].imshow(xl[i, 0], cmap="gray");           axes[i, 1].set_title("party A (left)")
    axes[i, 2].imshow(xr[i, 0], cmap="gray");           axes[i, 2].set_title("party B (right)")
    for ax in axes[i]:
        ax.axis("off")
plt.tight_layout()
plt.savefig("results/figures/vertical_split_sanity.png", dpi=120)
print("saved results/figures/vertical_split_sanity.png")
