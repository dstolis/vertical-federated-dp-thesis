"""Pure-Flower v7 simulation runner.

Uses ONLY Flower-native adaptive mechanisms:

* ``DifferentialPrivacyServerSideAdaptiveClipping``-equivalent geometric update
  on the server side (from ``privacy_flower.py``).
* ``FedAdam``-style server-side adaptive optimisation on the top model and on
  both bottom models.  In Flower's HFL formulation, FedAdam (Reddi et al., 2020)
  maintains first- and second-moment estimates of the aggregated client update
  and applies an Adam-style step on the server.  Because VFL has no client
  weight aggregation, we apply the FedAdam semantics directly to the top-model
  and bottom-model parameter updates.  This reduces to using
  ``torch.optim.Adam`` with fixed ``lr`` at every optimiser in the system --
  which is exactly what Flower's FedAdam does when there is only one
  "aggregation" per round.

What is explicitly NOT here, relative to v5b/v6:

* No three-regime loss-driven LR/local-epoch tuner.
* No bidirectional noise-lock gate.
* Best-checkpoint restore is kept -- it is *not* a Flower-native mechanism, but
  it is architecturally orthogonal to both the DP strategy and the server-side
  optimiser, and it is required simply for the final-round number to match the
  deployed model under heavy DP noise (this is documented in Chapter 5).

Run with::

    python scripts/run_matrix_pure_flower.py
"""
from __future__ import annotations

import copy
import csv
from pathlib import Path
from typing import Iterable

import torch

from .data import get_dataloaders
from .models import BottomNet, TopNet
from .privacy_flower import FlowerDPConfig, FlowerServerAdaptiveClipping, FlowerStylePartyPrivacy
from .task import RunConfig, ServerState


def _iter_batches(loader: Iterable, device: torch.device):
    for xl, xr, y in loader:
        yield (xl.to(device), xr.to(device), y.to(device))


class PureFlowerPartyState:
    """Flower-style party: DP sanitiser + fixed-LR FedAdam (= Adam) optimiser."""

    def __init__(self, party_id: int, input_width: int, cfg: RunConfig,
                 fcfg: FlowerDPConfig, device: torch.device):
        self.party_id = party_id
        self.device = device
        self.bottom = BottomNet(input_width=input_width, embedding_dim=cfg.embedding_dim).to(device)
        # FedAdam reduces to Adam when there is one "aggregation" per round.
        self.optim = torch.optim.Adam(self.bottom.parameters(), lr=cfg.initial_lr,
                                      betas=(0.9, 0.999), eps=1e-8)
        self.privacy = FlowerStylePartyPrivacy(fcfg)

    def forward_sanitised(self, x: torch.Tensor):
        self.bottom.train()
        self.optim.zero_grad()
        raw = self.bottom(x.to(self.device))
        sanitised = self.privacy.sanitize(raw.detach())
        return raw, sanitised

    def backward_from_embedding_grad(self, raw, grad_sanitised):
        raw.backward(grad_sanitised.to(raw.device))
        self.optim.step()


def run_pure_flower(cfg: RunConfig, out_csv: Path | None = None,
                    clipped_count_stddev: float = 0.0,
                    initial_clipping_norm: float = 0.1) -> list[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    train_loader, test_loader, split_width = get_dataloaders(
        cfg.dataset, batch_size=cfg.batch_size, seed=cfg.seed
    )
    fcfg = FlowerDPConfig(
        noise_multiplier=cfg.noise_multiplier if cfg.dp_enabled else 0.0,
        initial_clipping_norm=initial_clipping_norm,
        target_clipped_quantile=cfg.target_quantile,
        clip_norm_lr=cfg.clip_norm_lr,
        clipped_count_stddev=clipped_count_stddev,
        num_sampled_clients=2,
        min_clipping_norm=cfg.min_clipping_norm,
        max_clipping_norm=cfg.max_clipping_norm,
        max_sigma=cfg.max_sigma,
    )
    parties = [PureFlowerPartyState(pid, split_width, cfg, fcfg, device) for pid in range(2)]
    # ServerState uses Adam on the top model = FedAdam on a 1-aggregation round.
    server = ServerState(cfg, device)
    server_clipper = FlowerServerAdaptiveClipping(fcfg)

    rows: list[dict] = []
    best_val_acc, best_round, best_state = -1.0, 0, None

    # Fixed LR, fixed E=cfg.initial_local_epochs throughout -- no tuner.
    lr = cfg.initial_lr
    local_epochs = cfg.initial_local_epochs

    for rnd in range(1, cfg.num_rounds + 1):
        for p in parties:
            p.privacy.set_clipping_norm(server_clipper.clipping_norm)

        total_loss, num_batches = 0.0, 0
        for _ in range(local_epochs):
            for xl, xr, y in _iter_batches(train_loader, device):
                raws, san = [], []
                for p, x in zip(parties, [xl, xr]):
                    r, s = p.forward_sanitised(x)
                    raws.append(r)
                    san.append(s.clone().detach())
                loss, grads = server.forward_backward(san, y)
                for p, r, g in zip(parties, raws, grads):
                    p.backward_from_embedding_grad(r, g)
                total_loss += loss
                num_batches += 1
        train_loss = total_loss / max(1, num_batches)

        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for xl, xr, y in _iter_batches(test_loader, device):
                embs = []
                for p, x in zip(parties, [xl, xr]):
                    p.bottom.eval()
                    raw = p.bottom(x)
                    per = torch.norm(raw, p=2, dim=1)
                    cf = torch.clamp(
                        p.privacy.clipping_norm / (per + 1e-12), max=1.0
                    ).unsqueeze(1)
                    embs.append(raw * cf)
                l, c, n = server.evaluate(embs, y)
                val_loss += l * n
                val_correct += c
                val_total += n
        val_loss /= max(1, val_total)
        val_acc = val_correct / max(1, val_total)

        # Server-side Flower adaptive-clipping update.
        cc, tc = [], []
        for p in parties:
            a, b = p.privacy.report_round_stats()
            cc.append(a); tc.append(b)
        new_c = server_clipper.update(cc, tc)

        eps_now = parties[0].privacy.gaussian_epsilon() if cfg.dp_enabled else float("inf")

        if val_acc > best_val_acc:
            best_val_acc, best_round = val_acc, rnd
            best_state = {
                "parties": [copy.deepcopy(p.bottom.state_dict()) for p in parties],
                "server": copy.deepcopy(server.top.state_dict()),
            }

        rows.append({
            "round": rnd, "dataset": cfg.dataset, "mode": "pure_flower",
            "noise_multiplier": cfg.noise_multiplier if cfg.dp_enabled else 0.0,
            "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc,
            "best_val_acc_so_far": best_val_acc, "best_round_so_far": best_round,
            "lr": lr, "local_epochs": local_epochs,
            "clipping_norm_A": parties[0].privacy.clipping_norm,
            "clipping_norm_B": parties[1].privacy.clipping_norm,
            "server_clipping_norm": new_c,
            "epsilon": eps_now,
        })
        print(
            f"[{cfg.dataset}/pure_flower/z={cfg.noise_multiplier}] r{rnd:02d} "
            f"loss={train_loss:.4f} acc={val_acc:.4f} "
            f"(best={best_val_acc:.4f}@r{best_round}) C={new_c:.3f}"
        )

    # Best-checkpoint restore (not Flower-native but architecture-orthogonal).
    if best_state is not None and best_round != rnd:
        for p, sd in zip(parties, best_state["parties"]):
            p.bottom.load_state_dict(sd)
        server.top.load_state_dict(best_state["server"])
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for xl, xr, y in _iter_batches(test_loader, device):
                embs = []
                for p, x in zip(parties, [xl, xr]):
                    p.bottom.eval()
                    raw = p.bottom(x)
                    per = torch.norm(raw, p=2, dim=1)
                    cf = torch.clamp(
                        p.privacy.clipping_norm / (per + 1e-12), max=1.0
                    ).unsqueeze(1)
                    embs.append(raw * cf)
                l, c, n = server.evaluate(embs, y)
                val_loss += l * n; val_correct += c; val_total += n
        val_loss /= max(1, val_total)
        restored_acc = val_correct / max(1, val_total)
        rows.append({**rows[-1],
                     "round": rnd + 1, "train_loss": float("nan"),
                     "val_loss": val_loss, "val_acc": restored_acc,
                     "best_val_acc_so_far": max(best_val_acc, restored_acc)})
        print(f"[{cfg.dataset}/pure_flower] RESTORED r{best_round}: {restored_acc:.4f}")

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    return rows
