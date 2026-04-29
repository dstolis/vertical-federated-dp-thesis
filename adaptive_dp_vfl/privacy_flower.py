"""Flower-style server-side adaptive clipping for DP-VFL.

This module implements a VFL analog of Flower's
``DifferentialPrivacyServerSideAdaptiveClipping`` strategy
(https://flower.ai/docs/framework/ref-api/flwr.serverapp.strategy.DifferentialPrivacyServerSideAdaptiveClipping.html).

In Flower's HFL formulation the server receives client weight-update vectors,
clips each to an adaptive threshold, adds Gaussian noise, and updates the
threshold per round via a privatised quantile estimator (Andrew et al., 2022)
with parameters:

* ``noise_multiplier``            -- z
* ``num_sampled_clients``         -- number of clients reporting per round
* ``initial_clipping_norm``       -- C_0 (default 0.1)
* ``target_clipped_quantile``     -- gamma (default 0.5)
* ``clip_norm_lr``                -- eta_C (default 0.2)
* ``clipped_count_stddev``        -- std of the Gaussian noise added to the
                                     privatised fraction-clipped estimate

The VFL analog used in this thesis keeps the per-party architecture of
Chapter 4 (each party clips its own embedding before transmission, so no raw
IR ever leaves the party) but replaces the per-party geometric update with a
single *server-level* update that matches Flower's semantics:

1.  Each party sends the number of samples it saw and the number that were
    clipped at the current C.
2.  The server aggregates these counts, adds ``clipped_count_stddev`` Gaussian
    noise to the fraction-clipped estimate (this is the "server-side"
    privatisation that the Flower API exposes), and runs the geometric update
    once per round.
3.  The new C is broadcast back to every party for the next round.

The sanitisation (clip + noise) itself still happens at each party so that raw
embeddings never leave their owner; this is a stricter privacy boundary than
Flower's pure-HFL default but matches the VFL threat model of Chapter 3.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class FlowerDPConfig:
    noise_multiplier: float = 1.0
    initial_clipping_norm: float = 0.1
    target_clipped_quantile: float = 0.5
    clip_norm_lr: float = 0.2
    clipped_count_stddev: float = 0.0
    num_sampled_clients: int = 2
    min_clipping_norm: float = 0.01
    max_clipping_norm: float = 20.0
    max_sigma: float = 10.0


class FlowerStylePartyPrivacy:
    """Per-party sanitiser that accepts a server-provided C each round.

    Identical clip-then-noise template to ``AdaptivePrivacyManager`` but with
    the adaptive update REMOVED from the party: the server drives C.
    """

    def __init__(self, cfg: FlowerDPConfig):
        self.cfg = cfg
        self.clipping_norm = float(cfg.initial_clipping_norm)
        self.noise_multiplier = float(cfg.noise_multiplier)
        self.max_sigma = float(cfg.max_sigma)
        self._samples_seen = 0
        self._samples_clipped = 0

    def set_clipping_norm(self, c: float) -> None:
        self.clipping_norm = float(c)

    def sanitize(self, embedding: torch.Tensor) -> torch.Tensor:
        per_sample_norms = torch.norm(embedding, p=2, dim=1)
        # Record counts for the server-side quantile estimator.
        b = per_sample_norms.shape[0]
        clipped = int((per_sample_norms > self.clipping_norm).sum().item())
        self._samples_seen += b
        self._samples_clipped += clipped

        clip_factor = torch.clamp(
            self.clipping_norm / (per_sample_norms + 1e-12), max=1.0
        ).unsqueeze(1)
        out = embedding * clip_factor
        if self.noise_multiplier > 0.0:
            sigma = min(self.noise_multiplier * self.clipping_norm, self.max_sigma)
            out = out + torch.randn_like(out) * sigma
        return out

    def warmup_initialise(self, raw_embedding: torch.Tensor) -> None:
        """No-op for Flower-style; C is driven by the server, not the party."""
        return None

    def report_round_stats(self) -> tuple[int, int]:
        seen = self._samples_seen
        clipped = self._samples_clipped
        self._samples_seen = 0
        self._samples_clipped = 0
        return clipped, seen

    def finalize_round(self) -> None:
        """Server-side logic drives C; party-side has nothing to finalise."""
        return None

    def gaussian_epsilon(self, delta: float = 1e-5) -> float:
        if self.noise_multiplier <= 0.0:
            return float("inf")
        return float(math.sqrt(2.0 * math.log(1.25 / delta)) / self.noise_multiplier)


class FlowerServerAdaptiveClipping:
    """Server-side adaptive-clipping controller with privatised counting.

    Implements the Andrew et al. (2022) geometric update

        C <- C * exp(-eta * (q - gamma))

    where q is computed from a Gaussian-noised aggregate of the per-party
    fraction-clipped counts. The noise on the count is the "server-side"
    privatisation exposed by Flower's strategy.
    """

    def __init__(self, cfg: FlowerDPConfig):
        self.cfg = cfg
        self.clipping_norm = float(cfg.initial_clipping_norm)

    def ceiling(self) -> float:
        sigma_cap_c = self.cfg.max_sigma / max(self.cfg.noise_multiplier, 1e-6)
        return float(min(self.cfg.max_clipping_norm, sigma_cap_c))

    def update(self, clipped_counts: list[int], total_counts: list[int]) -> float:
        total = float(sum(total_counts))
        clipped = float(sum(clipped_counts))
        if total <= 0:
            return self.clipping_norm
        # Privatised fraction: add Gaussian noise to the clipped count, then
        # divide by the (public) total count. This is the estimator Flower
        # uses when privatising the quantile counter.
        if self.cfg.clipped_count_stddev > 0.0:
            clipped_priv = clipped + float(
                torch.randn(1).item() * self.cfg.clipped_count_stddev
            )
        else:
            clipped_priv = clipped
        frac = max(0.0, min(1.0, clipped_priv / total))
        gap = frac - self.cfg.target_clipped_quantile
        self.clipping_norm *= math.exp(self.cfg.clip_norm_lr * gap)
        self.clipping_norm = max(
            self.cfg.min_clipping_norm,
            min(self.ceiling(), self.clipping_norm),
        )
        return self.clipping_norm
