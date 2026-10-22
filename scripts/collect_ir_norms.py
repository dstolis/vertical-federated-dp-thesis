"""Measure the per-sample IR L2-norm distribution the clipping mechanism sees.

The quantile-matching update of Andrew et al. adapts C from the fraction of
per-sample norms above the current threshold. That signal only exists if the
norm distribution has spread. This script measures the distribution directly,
with and without the terminal LayerNorm, both at warm-up (when C_0 is seeded)
and after a short period of training.

Writes norms_{layernorm,nolayernorm}.npy into results_ablation_noln/ and
prints the summary statistics quoted in the paper.

Usage:
    python scripts/collect_ir_norms.py            # both variants
    python scripts/collect_ir_norms.py --rounds 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adaptive_dp_vfl.task import RunConfig, build_session_with_val

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results_camera_ready" / "ir_norms"


def party_a_norms(parties, loader, device, limit_batches: int | None = None):
    """Raw (pre-clip, pre-noise) IR norms from party A."""
    norms = []
    parties[0].bottom.eval()
    with torch.no_grad():
        for i, (xl, _xr, _y) in enumerate(loader):
            norms.append(torch.norm(parties[0].bottom(xl.to(device)), p=2, dim=1).cpu().numpy())
            if limit_batches is not None and i + 1 >= limit_batches:
                break
    return np.concatenate(norms)


def measure(layer_norm: bool, rounds: int, device: torch.device):
    cfg = RunConfig(dataset="mnist", mode="dp", noise_multiplier=2.0, seed=42,
                    layer_norm=layer_norm)
    train_loader, val_loader, _test_loader, parties, server = build_session_with_val(cfg, device)

    # Warm-up distribution: all 10,000 validation samples through the
    # freshly initialised bottom model (the state in which C_0 is seeded).
    warm = party_a_norms(parties, val_loader, device)

    for _ in range(rounds):
        for xl, xr, y in train_loader:
            raws, san = [], []
            for p, x in zip(parties, [xl, xr]):
                raw, s = p.forward_sanitised(x)
                raws.append(raw)
                san.append(s.clone().detach())
            _loss, grads = server.forward_backward(san, y)
            for p, raw, g in zip(parties, raws, grads):
                p.backward_from_embedding_grad(raw, g)

    trained = party_a_norms(parties, val_loader, device)
    return warm, trained


def describe(name: str, v: np.ndarray) -> None:
    cv = 100.0 * v.std() / max(v.mean(), 1e-12)
    print(f"{name:<26} n={len(v):>6}  mean={v.mean():.3f}  sd={v.std():.3f}  "
          f"CV={cv:.1f}%  min={v.min():.3f}  max={v.max():.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5,
                    help="training rounds before measuring the trained distribution")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)

    for layer_norm, tag in ((True, "layernorm"), (False, "nolayernorm")):
        warm, trained = measure(layer_norm, args.rounds, device)
        np.save(OUT / f"norms_{tag}.npy", trained)
        np.save(OUT / f"norms_{tag}_warmup.npy", warm)
        describe(f"{tag} (warm-up)", warm)
        describe(f"{tag} (trained)", trained)


if __name__ == "__main__":
    main()
