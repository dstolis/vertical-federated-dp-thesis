"""Embedding-level differential privacy with optional adaptive clipping.

The mechanism follows the Gaussian DP formulation of Dwork & Roth (2014) and
Abadi et al. (2016), applied to intermediate representations rather than to
per-sample gradients. Each embedding is first clipped so that its L2 norm is
at most C, bounding the sensitivity of the party-to-server message to C; then
isotropic Gaussian noise of scale sigma = z * C is added. When adaptive
clipping is enabled, C is updated per batch via the quantile-matching
geometric rule of Andrew et al. (2022).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class PrivacyConfig:
    noise_multiplier: float = 1.0          # z
    clipping_norm: float = 1.0             # C
    adaptive: bool = False
    target_quantile: float = 0.5           # gamma
    clip_norm_lr: float = 0.1              # eta_C (per ROUND, applied once at end)
    min_clipping_norm: float = 0.01
    max_clipping_norm: float = 20.0        # hard ceiling to cap sigma = z*C
    max_sigma: float = 10.0                # absolute ceiling on noise std


class AdaptivePrivacyManager:
    """Sanitises embeddings with per-sample clipping + Gaussian noise.

    The object is stateful: when ``adaptive`` is ``True`` the per-batch fraction
    of samples whose raw norm exceeds ``C`` is accumulated inside the round,
    and ``C`` is updated ONCE at the end of the round via
    :meth:`finalize_round`. This mirrors Andrew et al. (2022), who apply the
    geometric quantile-matching update at federation-round granularity rather
    than per optimiser step.
    """

    def __init__(self, cfg: PrivacyConfig):
        self.noise_multiplier = float(cfg.noise_multiplier)
        self.clipping_norm = float(cfg.clipping_norm)
        self.adaptive = bool(cfg.adaptive)
        self.target_quantile = float(cfg.target_quantile)
        self.clip_norm_lr = float(cfg.clip_norm_lr)
        self.min_clipping_norm = float(cfg.min_clipping_norm)
        self.max_clipping_norm = float(cfg.max_clipping_norm)
        self.max_sigma = float(getattr(cfg, "max_sigma", 10.0))
        # Running stats for the adaptive update within a round.
        self._frac_above_sum = 0.0
        self._frac_above_count = 0

    # ------------------------------------------------------------------
    # Warmup: pick C_0 from an empirical quantile of raw IR norms.
    # Avoids the "C=1 tiny, fraction_clipped=1, C runs away to the ceiling"
    # failure mode observed in the initial experiments.
    # ------------------------------------------------------------------
    def warmup_initialise(self, raw_embedding: torch.Tensor) -> None:
        if not self.adaptive:
            return
        norms = torch.norm(raw_embedding.detach(), p=2, dim=1)
        q = float(torch.quantile(norms, self.target_quantile).item())
        # Honour the [min, max] clamp, but also back off from the ceiling so
        # sigma = z * C stays well below max_sigma.
        sigma_cap_c = self.max_sigma / max(self.noise_multiplier, 1e-6)
        c0 = max(self.min_clipping_norm, min(self.max_clipping_norm, min(q, sigma_cap_c)))
        self.clipping_norm = c0

    def sanitize(self, embedding: torch.Tensor) -> torch.Tensor:
        """Return a DP-sanitised version of a (B, d) embedding tensor."""
        per_sample_norms = torch.norm(embedding, p=2, dim=1)
        clip_factor = torch.clamp(
            self.clipping_norm / (per_sample_norms + 1e-12), max=1.0
        ).unsqueeze(1)
        clipped = embedding * clip_factor

        if self.noise_multiplier > 0.0:
            sigma = min(self.noise_multiplier * self.clipping_norm, self.max_sigma)
            noise = torch.randn_like(clipped) * sigma
            sanitized = clipped + noise
        else:
            sanitized = clipped

        # Accumulate stats for the end-of-round C update but do NOT mutate C
        # here — per-batch mutation runs away with the batches/round factor.
        if self.adaptive and self.clip_norm_lr > 0.0:
            fraction_clipped = float(
                (per_sample_norms > self.clipping_norm).float().mean().detach().cpu()
            )
            self._frac_above_sum += fraction_clipped
            self._frac_above_count += 1

        return sanitized

    def finalize_round(self) -> float | None:
        """Apply one geometric quantile-matching update to C, return new C.

        Returns ``None`` when adaptive mode is off or no batches were seen.
        """
        if not self.adaptive or self._frac_above_count == 0:
            return None
        mean_frac_above = self._frac_above_sum / self._frac_above_count
        diff = mean_frac_above - self.target_quantile
        self.clipping_norm *= math.exp(self.clip_norm_lr * diff)
        # Honour the hard C ceiling and also the sigma-cap-derived ceiling so
        # that z * C stays within max_sigma.
        sigma_cap_c = self.max_sigma / max(self.noise_multiplier, 1e-6)
        self.clipping_norm = max(
            self.min_clipping_norm,
            min(self.max_clipping_norm, min(self.clipping_norm, sigma_cap_c)),
        )
        # Reset running stats for the next round.
        self._frac_above_sum = 0.0
        self._frac_above_count = 0
        return self.clipping_norm

    # ------------------------------------------------------------------
    # Simple (eps, delta)-DP accounting via the basic Gaussian bound.
    # The thesis uses this only for reporting indicative privacy levels; a
    # tighter bound via Renyi DP accounting is discussed as future work.
    # ------------------------------------------------------------------
    def gaussian_epsilon(self, delta: float = 1e-5) -> float:
        if self.noise_multiplier <= 0.0:
            return float("inf")
        return float(math.sqrt(2.0 * math.log(1.25 / delta)) / self.noise_multiplier)
