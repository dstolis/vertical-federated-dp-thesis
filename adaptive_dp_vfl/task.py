"""Per-party training state shared by the Flower client and the simulation runner.

A ``PartyState`` bundles together the bottom model, its optimiser, a per-party
data iterator, and the embedding-level privacy manager. The Flower client and
the standalone simulation runner both drive the same ``PartyState``, so that
the code paths exercised by the two entrypoints are identical.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Tuple

import torch
from torch import nn

from .data import get_dataloaders, get_dataloaders_with_val
from .models import BottomNet, TopNet, set_lr
from .privacy import AdaptivePrivacyManager, PrivacyConfig


@dataclass
class RunConfig:
    dataset: str = "mnist"
    mode: str = "adaptive"                 # "baseline" | "dp" | "adaptive"
    # Privacy
    noise_multiplier: float = 1.0
    clipping_norm: float = 1.0
    # Under LayerNorm, ||e||_2 ~ sqrt(d) ~ 8 is stable. Track the MEDIAN IR
    # norm rather than the 90th percentile so sigma = z*C stays moderate
    # while C still has room to float around its natural scale. v5a erred by
    # also pinning max_clipping_norm=5, which slammed the warmup-initialised
    # C into the ceiling and killed adaptation; v5b restores max_clipping=20.
    target_quantile: float = 0.5        # track the median, not the tail
    clip_norm_lr: float = 0.2           # per-ROUND update rate
    min_clipping_norm: float = 0.01
    max_clipping_norm: float = 20.0     # headroom above sqrt(d) ~ 8
    max_sigma: float = 10.0             # hard cap on injected noise std
    # Tuner
    initial_lr: float = 1e-3
    min_lr: float = 1e-5
    max_lr: float = 1e-2
    initial_local_epochs: int = 1
    min_local_epochs: int = 1
    max_local_epochs: int = 5
    patience: int = 2
    min_improve: float = 0.01
    tuner_warmup_rounds: int = 3        # let DP loss stabilise before adapting
    # General
    batch_size: int = 64
    embedding_dim: int = 64
    layer_norm: bool = True   # False = no-LayerNorm ablation on the bottom nets
    num_rounds: int = 40                # DP needs longer than non-private run
    seed: int = 42
    # Hold out a validation split from the training data for checkpoint
    # selection and the noise-lock; the test split is then used only to
    # report the selected model. False reproduces the original protocol.
    val_split: bool = True
    val_size: int = 10_000
    # Fixed-threshold control only: draw the same data order as the adaptive
    # configuration, whose warm-up pass consumes one shuffle of the training
    # loader before round 1.
    match_warmup_order: bool = False

    @property
    def dp_enabled(self) -> bool:
        return self.mode in ("dp", "adaptive")

    @property
    def adaptive_enabled(self) -> bool:
        return self.mode == "adaptive"


class PartyState:
    """Bottom-model state for a single VFL party."""

    def __init__(self, party_id: int, input_width: int, cfg: RunConfig, device: torch.device):
        self.party_id = party_id
        self.cfg = cfg
        self.device = device
        self.bottom = BottomNet(input_width=input_width, embedding_dim=cfg.embedding_dim,
                                layer_norm=cfg.layer_norm).to(device)
        self.optim = torch.optim.Adam(self.bottom.parameters(), lr=cfg.initial_lr)
        pcfg = PrivacyConfig(
            noise_multiplier=cfg.noise_multiplier if cfg.dp_enabled else 0.0,
            clipping_norm=cfg.clipping_norm,
            adaptive=cfg.adaptive_enabled,
            target_quantile=cfg.target_quantile,
            clip_norm_lr=cfg.clip_norm_lr if cfg.adaptive_enabled else 0.0,
            min_clipping_norm=cfg.min_clipping_norm,
            max_clipping_norm=cfg.max_clipping_norm,
            max_sigma=cfg.max_sigma,
        )
        self.privacy = AdaptivePrivacyManager(pcfg)

    def forward_sanitised(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (raw_embedding, sanitised_embedding). Raw kept for backward."""
        self.bottom.train()
        self.optim.zero_grad()
        raw = self.bottom(x.to(self.device))
        # Sanitisation is non-differentiable w.r.t. model params (clip+noise are
        # detached noise); for backward we need gradients on ``raw`` and propagate
        # them through the bottom network directly. The server therefore returns
        # dL/d(sanitised), which we treat as dL/d(raw) up to the clip factor.
        sanitised = self.privacy.sanitize(raw.detach())
        return raw, sanitised

    def backward_from_embedding_grad(self, raw: torch.Tensor, grad_sanitised: torch.Tensor) -> None:
        """Backpropagate the server-provided gradient w.r.t. the embedding."""
        raw.backward(grad_sanitised.to(raw.device))
        self.optim.step()

    def apply_lr(self, lr: float) -> None:
        set_lr(self.optim, lr)


class ServerState:
    """Top-model and adaptive-tuner state held by the coordinating server."""

    def __init__(self, cfg: RunConfig, device: torch.device, num_classes: int = 10):
        self.cfg = cfg
        self.device = device
        self.top = TopNet(embedding_dim=cfg.embedding_dim, num_parties=2, num_classes=num_classes).to(device)
        self.optim = torch.optim.Adam(self.top.parameters(), lr=cfg.initial_lr)
        self.criterion = nn.CrossEntropyLoss()

    def forward_backward(
        self, embeddings: list[torch.Tensor], labels: torch.Tensor
    ) -> Tuple[float, list[torch.Tensor]]:
        """Run top model forward + backward, return (loss, grads-per-embedding)."""
        self.top.train()
        self.optim.zero_grad()
        # Detach each embedding then re-attach grad so we can read dL/de per party.
        emb_requires_grad = [e.detach().clone().requires_grad_(True) for e in embeddings]
        concat = torch.cat(emb_requires_grad, dim=1)
        logits = self.top(concat.to(self.device))
        loss = self.criterion(logits, labels.to(self.device))
        loss.backward()
        self.optim.step()
        grads = [e.grad.detach() for e in emb_requires_grad]
        return float(loss.detach().cpu()), grads

    @torch.no_grad()
    def evaluate(self, embeddings: list[torch.Tensor], labels: torch.Tensor) -> Tuple[float, int, int]:
        self.top.eval()
        concat = torch.cat([e.to(self.device) for e in embeddings], dim=1)
        logits = self.top(concat)
        loss = self.criterion(logits, labels.to(self.device))
        pred = logits.argmax(dim=1)
        correct = int((pred == labels.to(self.device)).sum().cpu())
        return float(loss.cpu()), correct, int(labels.size(0))

    def apply_lr(self, lr: float) -> None:
        set_lr(self.optim, lr)


def load_data(cfg: RunConfig):
    """Return (train_loader, val_loader, test_loader, split_width).

    With ``cfg.val_split`` the validation loader is a held-out part of the
    training data; without it, validation falls back to the test split (the
    original protocol)."""
    if cfg.val_split:
        return get_dataloaders_with_val(cfg.dataset, batch_size=cfg.batch_size,
                                        seed=cfg.seed, val_size=cfg.val_size)
    train_loader, test_loader, split_width = get_dataloaders(
        cfg.dataset, batch_size=cfg.batch_size, seed=cfg.seed)
    return train_loader, test_loader, test_loader, split_width


def build_session_with_val(cfg: RunConfig, device: torch.device | None = None):
    """Return (train_loader, val_loader, test_loader, parties, server)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    train_loader, val_loader, test_loader, split_width = load_data(cfg)
    parties = [PartyState(pid, split_width, cfg, device) for pid in range(2)]
    server = ServerState(cfg, device)
    return train_loader, val_loader, test_loader, parties, server


@torch.no_grad()
def evaluate_clipped(parties, server, loader, device, clip_norms=None) -> Tuple[float, float]:
    """Accuracy of the deployed model: each party's embedding is clipped to its
    threshold (bounded sensitivity) but no fresh noise is added.

    ``clip_norms`` overrides the thresholds (used when re-evaluating a restored
    checkpoint at the threshold it was selected with)."""
    total_loss, correct, total = 0.0, 0, 0
    for xl, xr, y in loader:
        embs = []
        for i, (p, x) in enumerate(zip(parties, [xl, xr])):
            p.bottom.eval()
            raw = p.bottom(x.to(device))
            c = clip_norms[i] if clip_norms is not None else p.privacy.clipping_norm
            per = torch.norm(raw, p=2, dim=1)
            embs.append(raw * torch.clamp(c / (per + 1e-12), max=1.0).unsqueeze(1))
        l, k, n = server.evaluate(embs, y.to(device))
        total_loss += l * n
        correct += k
        total += n
    return total_loss / max(1, total), correct / max(1, total)


def build_session(cfg: RunConfig, device: torch.device | None = None):
    """Return (train_loader, test_loader, parties, server) for a complete session."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    train_loader, test_loader, split_width = get_dataloaders(
        cfg.dataset, batch_size=cfg.batch_size, seed=cfg.seed
    )
    parties = [PartyState(pid, split_width, cfg, device) for pid in range(2)]
    server = ServerState(cfg, device)
    return train_loader, test_loader, parties, server
