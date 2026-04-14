"""Flower ClientApp: exposes a VFL party's bottom model over the Flower API.

Each party runs one ``ClientApp``. The server coordinates training by sending
typed messages: a ``CONFIGURE`` message carries the per-round hyperparameters
(learning rate, clipping norm), a ``FORWARD`` message contains the indices of
the batch the client should process and returns a sanitised embedding, and a
``BACKWARD`` message carries dL/de from the server and triggers an optimiser
step on the bottom model.

This file targets the Flower 1.13+ message API (``ClientApp`` with ``@app.query``
handlers). Under ``flwr run`` it is instantiated once per simulated SuperNode.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from flwr.client import ClientApp
from flwr.common import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict

from .data import get_dataloaders
from .task import PartyState, RunConfig

app = ClientApp()


# --------------------------------------------------------------------------
# Per-process, per-SuperNode state.
# --------------------------------------------------------------------------
_PARTY: PartyState | None = None
_LOADER_ITER = None
_LOADER = None
_SPLIT_WIDTH: int | None = None
_RAW_CACHE: Dict[int, torch.Tensor] = {}          # batch_id -> raw embedding
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _cfg_from_run_config(run_config) -> RunConfig:
    """Map Flower's run-config dict into our typed ``RunConfig``."""
    get = run_config.get
    return RunConfig(
        dataset=get("dataset", "mnist"),
        mode=get("mode", "adaptive"),
        noise_multiplier=float(get("noise-multiplier", 1.0)),
        clipping_norm=float(get("clipping-norm", 1.0)),
        target_quantile=float(get("target-quantile", 0.5)),
        clip_norm_lr=float(get("clip-norm-lr", 0.01)),
        min_clipping_norm=float(get("min-clipping-norm", 0.01)),
        max_clipping_norm=float(get("max-clipping-norm", 100.0)),
        initial_lr=float(get("initial-lr", 1e-3)),
        batch_size=int(get("batch-size", 64)),
        embedding_dim=int(get("embedding-dim", 64)),
        seed=int(get("seed", 42)),
    )


def _ensure_party(party_id: int, context: Context) -> PartyState:
    """Lazy-construct the bottom-model state for this SuperNode."""
    global _PARTY, _LOADER, _LOADER_ITER, _SPLIT_WIDTH
    if _PARTY is not None:
        return _PARTY
    cfg = _cfg_from_run_config(context.run_config)
    train_loader, _, split_width = get_dataloaders(cfg.dataset, cfg.batch_size, cfg.seed)
    _LOADER = train_loader
    _LOADER_ITER = iter(train_loader)
    _SPLIT_WIDTH = split_width
    _PARTY = PartyState(party_id=party_id, input_width=split_width, cfg=cfg, device=_DEVICE)
    return _PARTY


def _next_batch():
    global _LOADER_ITER
    try:
        return next(_LOADER_ITER)
    except StopIteration:
        _LOADER_ITER = iter(_LOADER)
        return next(_LOADER_ITER)


# --------------------------------------------------------------------------
# Message handlers.
# --------------------------------------------------------------------------
@app.query()
def handle(msg: Message, context: Context) -> Message:
    """Dispatch based on the ``phase`` field in the incoming ConfigRecord."""
    party_id = int(context.node_config.get("partition-id", 0))
    phase = str(msg.content.config_records["control"]["phase"])
    party = _ensure_party(party_id, context)

    if phase == "configure":
        lr = float(msg.content.config_records["control"]["lr"])
        party.apply_lr(lr)
        out = RecordDict({"metrics": MetricRecord({"ack": 1.0})})
        return Message(content=out, reply_to=msg)

    if phase == "forward":
        xl, xr, y = _next_batch()
        x = xl if party_id == 0 else xr
        batch_id = int(msg.content.config_records["control"]["batch_id"])
        raw, sanitised = party.forward_sanitised(x)
        _RAW_CACHE[batch_id] = raw
        emb_np = sanitised.detach().cpu().numpy().astype(np.float32)
        # Party 0 additionally returns labels so the server has y for the batch.
        records = {"emb": ArrayRecord({"e": emb_np})}
        if party_id == 0:
            records["labels"] = ArrayRecord({"y": y.detach().cpu().numpy().astype(np.int64)})
        records["metrics"] = MetricRecord({
            "clipping_norm": party.privacy.clipping_norm,
            "noise_multiplier": party.privacy.noise_multiplier,
        })
        return Message(content=RecordDict(records), reply_to=msg)

    if phase == "backward":
        batch_id = int(msg.content.config_records["control"]["batch_id"])
        grad_np = msg.content.array_records["grad"]["g"].numpy()
        grad = torch.from_numpy(grad_np).to(_DEVICE)
        raw = _RAW_CACHE.pop(batch_id)
        party.backward_from_embedding_grad(raw, grad)
        return Message(content=RecordDict({"metrics": MetricRecord({"ack": 1.0})}), reply_to=msg)

    raise ValueError(f"Unknown phase: {phase}")
