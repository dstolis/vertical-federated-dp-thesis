"""Flower ServerApp: drives the VFL protocol from the coordinating server.

The server owns the top model, the adaptive tuner, and the global training
schedule. It iterates over the configured number of rounds; within each round
it sweeps mini-batches, for each batch it issues a FORWARD message to every
party via ``grid.send_and_receive``, assembles the sanitised embeddings, runs
the top model forward and backward to compute dL/de per party, and replies
with BACKWARD messages carrying those gradients. Between rounds the adaptive
tuner updates the learning rate and local-epochs schedule and the new LR is
broadcast to all parties via CONFIGURE.

This file targets the Flower 1.13+ ``ServerApp`` + ``Grid`` API.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List

import numpy as np
import torch
from flwr.common import ArrayRecord, ConfigRecord, Context, Message, MessageType, MetricRecord, RecordDict
from flwr.server import Grid, ServerApp

from .task import RunConfig, ServerState
from .tuner import AdaptiveTuner, TunerConfig

app = ServerApp()


def _build_cfg(context: Context) -> RunConfig:
    get = context.run_config.get
    return RunConfig(
        dataset=get("dataset", "mnist"),
        mode=get("mode", "adaptive"),
        noise_multiplier=float(get("noise-multiplier", 1.0)),
        clipping_norm=float(get("clipping-norm", 1.0)),
        initial_lr=float(get("initial-lr", 1e-3)),
        min_lr=float(get("min-lr", 1e-5)),
        max_lr=float(get("max-lr", 1e-2)),
        initial_local_epochs=int(get("initial-local-epochs", 1)),
        min_local_epochs=int(get("min-local-epochs", 1)),
        max_local_epochs=int(get("max-local-epochs", 5)),
        patience=int(get("patience", 2)),
        min_improve=float(get("min-improve", 0.01)),
        batch_size=int(get("batch-size", 64)),
        embedding_dim=int(get("embedding-dim", 64)),
        num_rounds=int(get("num-server-rounds", 20)),
        seed=int(get("seed", 42)),
    )


def _control(phase: str, **kw) -> RecordDict:
    rec = ConfigRecord({"phase": phase, **{k: _encode(v) for k, v in kw.items()}})
    return RecordDict({"control": rec})


def _encode(v):
    if isinstance(v, (int, float, str, bool)):
        return v
    return str(v)


def _broadcast_configure(grid: Grid, node_ids: List[int], lr: float) -> None:
    msgs = [
        Message(content=_control("configure", lr=lr), dst_node_id=nid, message_type=MessageType.QUERY)
        for nid in node_ids
    ]
    list(grid.send_and_receive(msgs))


def _forward_batch(grid: Grid, node_ids: List[int], batch_id: int):
    msgs = [
        Message(content=_control("forward", batch_id=batch_id), dst_node_id=nid,
                message_type=MessageType.QUERY)
        for nid in node_ids
    ]
    replies = list(grid.send_and_receive(msgs))
    # Preserve party ordering (node_ids order).
    ordered = sorted(replies, key=lambda r: node_ids.index(r.metadata.src_node_id))
    embeddings, labels = [], None
    for r in ordered:
        emb = torch.from_numpy(r.content.array_records["emb"]["e"].numpy())
        embeddings.append(emb)
        if "labels" in r.content.array_records:
            labels = torch.from_numpy(r.content.array_records["labels"]["y"].numpy()).long()
    return embeddings, labels


def _backward_batch(grid: Grid, node_ids: List[int], batch_id: int, grads: List[torch.Tensor]) -> None:
    msgs = []
    for nid, g in zip(node_ids, grads):
        content = _control("backward", batch_id=batch_id)
        content["grad"] = ArrayRecord({"g": g.detach().cpu().numpy().astype(np.float32)})
        msgs.append(Message(content=content, dst_node_id=nid, message_type=MessageType.QUERY))
    list(grid.send_and_receive(msgs))


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = _build_cfg(context)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    server_state = ServerState(cfg, device)
    tuner = AdaptiveTuner(TunerConfig(
        initial_lr=cfg.initial_lr, min_lr=cfg.min_lr, max_lr=cfg.max_lr,
        initial_local_epochs=cfg.initial_local_epochs,
        min_local_epochs=cfg.min_local_epochs,
        max_local_epochs=cfg.max_local_epochs,
        patience=cfg.patience, min_improve=cfg.min_improve,
        enabled=cfg.adaptive_enabled,
    ))

    node_ids = sorted(list(grid.get_node_ids()))
    assert len(node_ids) == 2, f"VFL protocol expects exactly 2 parties; got {len(node_ids)}"

    # Derive batches per epoch from the training-set size; clients iterate their
    # own loader with the same seed so their mini-batches align index-for-index.
    from .data import get_dataloaders
    train_loader, test_loader, _ = get_dataloaders(cfg.dataset, cfg.batch_size, cfg.seed)
    batches_per_epoch = len(train_loader)

    lr, local_epochs = cfg.initial_lr, cfg.initial_local_epochs
    rows: list[dict] = []
    out_csv = Path(__file__).resolve().parent.parent / "results" / f"flower_{cfg.dataset}_{cfg.mode}_z{cfg.noise_multiplier}.csv"

    for rnd in range(1, cfg.num_rounds + 1):
        server_state.apply_lr(lr)
        _broadcast_configure(grid, node_ids, lr)

        total_loss = 0.0
        num_batches = 0
        for _ in range(local_epochs):
            for b_idx in range(batches_per_epoch):
                batch_id = rnd * 10_000 + b_idx
                embeddings, labels = _forward_batch(grid, node_ids, batch_id)
                if labels is None:
                    continue
                loss, grads = server_state.forward_backward(embeddings, labels)
                _backward_batch(grid, node_ids, batch_id, grads)
                total_loss += loss
                num_batches += 1
        train_loss = total_loss / max(1, num_batches)

        lr, local_epochs = tuner.step(train_loss)
        print(f"[flower] round {rnd:02d}  train_loss={train_loss:.4f}  lr={lr:.2e}  E={local_epochs}")
        rows.append({
            "round": rnd, "dataset": cfg.dataset, "mode": cfg.mode,
            "noise_multiplier": cfg.noise_multiplier,
            "train_loss": train_loss, "lr": lr, "local_epochs": local_epochs,
        })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
