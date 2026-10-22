"""Simulation runner that uses the Flower-style server-side adaptive clipping
strategy together with the existing adaptive LR/epoch tuner and best-checkpoint
restore.

This is v6: same pipeline as ``simulate.run`` but with
``FlowerStylePartyPrivacy`` + ``FlowerServerAdaptiveClipping`` replacing the
per-party ``AdaptivePrivacyManager`` used by v5b.

Run with::

    from adaptive_dp_vfl.task import RunConfig
    from adaptive_dp_vfl.simulate_flower_strategy import run_flower_strategy
    rows = run_flower_strategy(cfg, out_csv=Path("results_v6/mnist_flower_z2.0.csv"))
"""
from __future__ import annotations

import copy
import csv
from pathlib import Path
from typing import Iterable

import torch

from .data import get_dataloaders
from .models import BottomNet, TopNet, set_lr
from .privacy_flower import FlowerDPConfig, FlowerServerAdaptiveClipping, FlowerStylePartyPrivacy
from .task import RunConfig, ServerState, evaluate_clipped, load_data
from .tuner import AdaptiveTuner, TunerConfig


def _iter_batches(loader: Iterable, device: torch.device):
    for xl, xr, y in loader:
        yield (xl.to(device), xr.to(device), y.to(device))


class FlowerPartyState:
    def __init__(self, party_id: int, input_width: int, cfg: RunConfig,
                 fcfg: FlowerDPConfig, device: torch.device):
        self.party_id = party_id
        self.cfg = cfg
        self.device = device
        self.bottom = BottomNet(input_width=input_width, embedding_dim=cfg.embedding_dim).to(device)
        self.optim = torch.optim.Adam(self.bottom.parameters(), lr=cfg.initial_lr)
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

    def apply_lr(self, lr):
        set_lr(self.optim, lr)


def run_flower_strategy(cfg: RunConfig, out_csv: Path | None = None,
                        clipped_count_stddev: float = 0.0,
                        initial_clipping_norm: float = 0.1) -> list[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    train_loader, val_loader, test_loader, split_width = load_data(cfg)
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
    parties = [FlowerPartyState(pid, split_width, cfg, fcfg, device) for pid in range(2)]
    server = ServerState(cfg, device)
    server_clipper = FlowerServerAdaptiveClipping(fcfg)
    tuner = AdaptiveTuner(TunerConfig(
        initial_lr=cfg.initial_lr, min_lr=cfg.min_lr, max_lr=cfg.max_lr,
        initial_local_epochs=cfg.initial_local_epochs,
        min_local_epochs=cfg.min_local_epochs,
        max_local_epochs=cfg.max_local_epochs,
        patience=cfg.patience, min_improve=cfg.min_improve,
        enabled=cfg.adaptive_enabled,
    ))

    rows: list[dict] = []
    lr = cfg.initial_lr
    local_epochs = cfg.initial_local_epochs

    best_val_acc = -1.0
    best_round = 0
    best_state = None
    val_acc_window: list[float] = []
    WINDOW = 4

    for rnd in range(1, cfg.num_rounds + 1):
        # Broadcast current C to all parties at the start of the round.
        for p in parties:
            p.privacy.set_clipping_norm(server_clipper.clipping_norm)
        server.apply_lr(lr)
        for p in parties:
            p.apply_lr(lr)

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

        # --- validation ---
        # Selection signal on the validation split; test logged only.
        eval_clip = [p.privacy.clipping_norm for p in parties]
        val_loss, val_acc = evaluate_clipped(parties, server, val_loader, device)
        test_loss, test_acc = evaluate_clipped(parties, server, test_loader, device)

        # --- server-side adaptive clipping update ---
        clipped_counts, total_counts = [], []
        for p in parties:
            cc, tc = p.privacy.report_round_stats()
            clipped_counts.append(cc)
            total_counts.append(tc)
        new_c = server_clipper.update(clipped_counts, total_counts)

        # --- tuner with bidirectional noise-lock ---
        eps_now = parties[0].privacy.gaussian_epsilon() if cfg.dp_enabled else float("inf")
        val_acc_window.append(val_acc)
        if len(val_acc_window) > WINDOW:
            val_acc_window.pop(0)
        noise_heavy = eps_now < 5.0
        improving = (len(val_acc_window) >= 2
                     and val_acc_window[-1] > max(val_acc_window[:-1]))
        noise_lock = noise_heavy and not improving
        if rnd > cfg.tuner_warmup_rounds:
            lr, local_epochs = tuner.step(train_loss, noise_lock=noise_lock)
        else:
            tuner._prev_loss = train_loss  # type: ignore[attr-defined]

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_round = rnd
            best_state = {
                "parties": [copy.deepcopy(p.bottom.state_dict()) for p in parties],
                "server": copy.deepcopy(server.top.state_dict()),
                "clip": list(eval_clip),
            }

        row = {
            "round": rnd, "dataset": cfg.dataset, "mode": "flower_adaptive",
            "noise_multiplier": cfg.noise_multiplier if cfg.dp_enabled else 0.0,
            "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc,
            "test_loss": test_loss, "test_acc": test_acc,
            "best_val_acc_so_far": best_val_acc, "best_round_so_far": best_round,
            "lr": lr, "local_epochs": local_epochs,
            "clipping_norm_A": parties[0].privacy.clipping_norm,
            "clipping_norm_B": parties[1].privacy.clipping_norm,
            "server_clipping_norm": new_c,
            "epsilon": eps_now,
            "noise_lock": int(noise_lock) if rnd > cfg.tuner_warmup_rounds else 0,
        }
        rows.append(row)
        print(
            f"[{cfg.dataset}/flower/z={cfg.noise_multiplier}] r{rnd:02d} "
            f"loss={train_loss:.4f} acc={val_acc:.4f} "
            f"(best={best_val_acc:.4f}@r{best_round}) C={new_c:.3f} "
            f"lr={lr:.2e} E={local_epochs}"
        )

    # --- best-checkpoint restore ---
    # Restore the best-validation checkpoint WITH the clipping threshold it was
    # evaluated at (the earlier version re-evaluated at the final threshold),
    # then report it once on the test split.
    if best_state is not None:
        for p, sd in zip(parties, best_state["parties"]):
            p.bottom.load_state_dict(sd)
        server.top.load_state_dict(best_state["server"])
        for p, c in zip(parties, best_state["clip"]):
            p.privacy.set_clipping_norm(c)
        r_val_loss, r_val_acc = evaluate_clipped(parties, server, val_loader, device)
        r_test_loss, r_test_acc = evaluate_clipped(parties, server, test_loader, device)
        if abs(r_val_acc - best_val_acc) > 1e-12:
            print(f"WARNING: restored val_acc {r_val_acc:.6f} != selected {best_val_acc:.6f}")
        rows.append({**rows[-1],
                     "round": rnd + 1, "train_loss": float("nan"),
                     "val_loss": r_val_loss, "val_acc": r_val_acc,
                     "test_loss": r_test_loss, "test_acc": r_test_acc,
                     "best_val_acc_so_far": best_val_acc, "best_round_so_far": best_round,
                     "clipping_norm_A": best_state["clip"][0],
                     "clipping_norm_B": best_state["clip"][1],
                     "noise_lock": -1})
        print(f"[{cfg.dataset}/flower/z={cfg.noise_multiplier}] RESTORED r{best_round}: "
              f"val={r_val_acc:.4f} test={r_test_acc:.4f}")

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            w.writeheader()
            w.writerows(rows)
    return rows
