"""Standalone simulation runner.

Executes the full adaptive DP-VFL training loop in a single process, using
exactly the same ``PartyState`` / ``ServerState`` abstractions as the Flower
entrypoints. This is the primary script used to produce the per-round CSVs
consumed by Chapter 5's visualisation pipeline.
"""
from __future__ import annotations

import copy
import csv
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import torch

from .task import RunConfig, build_session
from .tuner import AdaptiveTuner, TunerConfig


def _iter_batches(loader: Iterable, device: torch.device):
    for xl, xr, y in loader:
        yield (xl.to(device), xr.to(device), y.to(device))


def run(cfg: RunConfig, out_csv: Path | None = None) -> list[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, parties, server = build_session(cfg, device)
    tuner = AdaptiveTuner(TunerConfig(
        initial_lr=cfg.initial_lr,
        min_lr=cfg.min_lr,
        max_lr=cfg.max_lr,
        initial_local_epochs=cfg.initial_local_epochs,
        min_local_epochs=cfg.min_local_epochs,
        max_local_epochs=cfg.max_local_epochs,
        patience=cfg.patience,
        min_improve=cfg.min_improve,
        enabled=cfg.adaptive_enabled,
    ))

    rows: list[dict] = []
    lr = cfg.initial_lr
    local_epochs = cfg.initial_local_epochs

    # Best-checkpoint tracking: at heavy DP noise the final-round model can
    # drift below its best intermediate model because train_loss keeps
    # decreasing on injected noise rather than signal. We keep the best
    # val_acc snapshot and restore it at the end.
    best_val_acc = -1.0
    best_round = 0
    best_state = None

    # Rolling window of recent val_acc used to drive the bidirectional
    # noise-lock: under heavy noise (eps < 5) the tuner's regime-3 LR growth
    # is locked UNLESS val_acc has just hit a recent new high, in which case
    # we temporarily unlock because we have utility evidence that the
    # optimiser is finding signal. Replaces the previous static-per-config
    # ``noise_lock = eps < 5`` criterion.
    val_acc_window: list[float] = []
    WINDOW = 4  # rolling window length (current round + 3 previous)

    # Warmup: seed each party's clipping norm C from the empirical target-
    # quantile of raw IR L2-norms on a few initial batches. Prevents the
    # "C=1, fraction_clipped=1, C blows up to max_clipping_norm in round 1"
    # pathology observed in the v1 matrix.
    if cfg.adaptive_enabled:
        seen = 0
        for xl, xr, _ in _iter_batches(train_loader, device):
            with torch.no_grad():
                for p, x in zip(parties, [xl, xr]):
                    p.bottom.eval()
                    raw = p.bottom(x)
                    p.privacy.warmup_initialise(raw)
            seen += 1
            if seen >= 8:
                break

    for rnd in range(1, cfg.num_rounds + 1):
        # --- training -----------------------------------------------------
        # Apply current LR to all optimisers (server top + both parties).
        server.apply_lr(lr)
        for p in parties:
            p.apply_lr(lr)

        total_loss, num_batches = 0.0, 0
        for _ in range(local_epochs):
            for xl, xr, y in _iter_batches(train_loader, device):
                inputs = [xl, xr]
                raws, sanitised = [], []
                for p, x in zip(parties, inputs):
                    raw, san = p.forward_sanitised(x)
                    raws.append(raw)
                    sanitised.append(san.clone().detach())
                loss, grads = server.forward_backward(sanitised, y)
                for p, raw, g in zip(parties, raws, grads):
                    p.backward_from_embedding_grad(raw, g)
                total_loss += loss
                num_batches += 1
        train_loss = total_loss / max(1, num_batches)

        # --- evaluation ---------------------------------------------------
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for xl, xr, y in _iter_batches(test_loader, device):
                embs = []
                for p, x in zip(parties, [xl, xr]):
                    p.bottom.eval()
                    # At evaluation we still apply clipping (bounded sensitivity is
                    # a property of the deployed model) but no fresh noise.
                    raw = p.bottom(x)
                    per_sample_norms = torch.norm(raw, p=2, dim=1)
                    clip_factor = torch.clamp(
                        p.privacy.clipping_norm / (per_sample_norms + 1e-12), max=1.0
                    ).unsqueeze(1)
                    embs.append(raw * clip_factor)
                l, c, n = server.evaluate(embs, y)
                val_loss += l * n
                val_correct += c
                val_total += n
        val_loss /= max(1, val_total)
        val_acc = val_correct / max(1, val_total)

        # --- finalise adaptive clipping (one geometric C update per round) -
        for p in parties:
            p.privacy.finalize_round()

        # --- adapt LR and local epochs, but wait for tuner warmup so the DP
        # loss has time to stabilise before the regime logic kicks in ------
        eps_now = parties[0].privacy.gaussian_epsilon() if cfg.dp_enabled else float("inf")
        # Bidirectional noise-lock: under heavy noise, default to locking the
        # tuner's regime-3 LR growth (loss-up and stagnation still fire), but
        # UNLOCK if val_acc just hit a new recent high -- utility evidence
        # that the optimiser is tracking true signal rather than injected
        # noise. val_acc (not train_loss) because DP noise inflates train_loss
        # improvements independently of utility.
        val_acc_window.append(val_acc)
        if len(val_acc_window) > WINDOW:
            val_acc_window.pop(0)
        noise_heavy = eps_now < 5.0
        improving = (
            len(val_acc_window) >= 2
            and val_acc_window[-1] > max(val_acc_window[:-1])
        )
        noise_lock = noise_heavy and not improving
        if rnd > cfg.tuner_warmup_rounds:
            lr, local_epochs = tuner.step(train_loss, noise_lock=noise_lock)
        else:
            tuner._prev_loss = train_loss  # type: ignore[attr-defined]

        eps = eps_now

        # --- best-checkpoint tracking ------------------------------------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_round = rnd
            best_state = {
                "parties": [copy.deepcopy(p.bottom.state_dict()) for p in parties],
                "server": copy.deepcopy(server.top.state_dict()),
            }

        row = {
            "round": rnd,
            "dataset": cfg.dataset,
            "mode": cfg.mode,
            "noise_multiplier": cfg.noise_multiplier if cfg.dp_enabled else 0.0,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_val_acc_so_far": best_val_acc,
            "best_round_so_far": best_round,
            "lr": lr,
            "local_epochs": local_epochs,
            "clipping_norm_A": parties[0].privacy.clipping_norm,
            "clipping_norm_B": parties[1].privacy.clipping_norm,
            "epsilon": eps,
            "noise_lock": int(noise_lock) if rnd > cfg.tuner_warmup_rounds else 0,
        }
        rows.append(row)
        print(
            f"[{cfg.dataset}/{cfg.mode}/z={cfg.noise_multiplier}] "
            f"round {rnd:02d}  train_loss={train_loss:.4f}  "
            f"val_acc={val_acc:.4f}  (best={best_val_acc:.4f}@r{best_round})  "
            f"lr={lr:.2e}  E={local_epochs}  "
            f"C_A={parties[0].privacy.clipping_norm:.3f}"
            + (
                "  [NL]" if (rnd > cfg.tuner_warmup_rounds and noise_lock)
                else "  [NL?]" if (rnd > cfg.tuner_warmup_rounds and noise_heavy and improving)
                else ""
            )
        )

    # --- restore best checkpoint and re-evaluate so the CSV's final row
    # reflects the model we would actually deploy (not a noise-drifted copy).
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
                    per_sample_norms = torch.norm(raw, p=2, dim=1)
                    clip_factor = torch.clamp(
                        p.privacy.clipping_norm / (per_sample_norms + 1e-12), max=1.0
                    ).unsqueeze(1)
                    embs.append(raw * clip_factor)
                l, c, n = server.evaluate(embs, y)
                val_loss += l * n
                val_correct += c
                val_total += n
        val_loss /= max(1, val_total)
        restored_acc = val_correct / max(1, val_total)
        rows.append({
            "round": rnd + 1,
            "dataset": cfg.dataset,
            "mode": cfg.mode,
            "noise_multiplier": cfg.noise_multiplier if cfg.dp_enabled else 0.0,
            "train_loss": float("nan"),
            "val_loss": val_loss,
            "val_acc": restored_acc,
            "best_val_acc_so_far": max(best_val_acc, restored_acc),
            "best_round_so_far": best_round,
            "lr": lr,
            "local_epochs": local_epochs,
            "clipping_norm_A": parties[0].privacy.clipping_norm,
            "clipping_norm_B": parties[1].privacy.clipping_norm,
            "epsilon": eps,
            "noise_lock": -1,  # sentinel: post-training restored eval
        })
        print(
            f"[{cfg.dataset}/{cfg.mode}/z={cfg.noise_multiplier}] "
            f"RESTORED best r{best_round}: val_acc={restored_acc:.4f}"
        )

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return rows
