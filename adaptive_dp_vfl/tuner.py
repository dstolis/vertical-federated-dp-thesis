"""Round-level adaptive tuner for learning rate and local epochs.

The tuner observes the training loss at each round and applies one of three
responses - loss-increase, stagnation, or healthy-convergence - based on the
relative improvement and a patience counter. This mirrors the three-regime
policy described in Section 3.4 of the thesis.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TunerConfig:
    initial_lr: float = 1e-3
    min_lr: float = 1e-5
    max_lr: float = 1e-2
    initial_local_epochs: int = 1
    min_local_epochs: int = 1
    max_local_epochs: int = 5
    patience: int = 2
    min_improve: float = 0.01
    enabled: bool = True


class AdaptiveTuner:
    def __init__(self, cfg: TunerConfig):
        self.cfg = cfg
        self.lr = cfg.initial_lr
        self.local_epochs = cfg.initial_local_epochs
        self._prev_loss: float | None = None
        self._no_improve = 0

    def step(self, train_loss: float, noise_lock: bool = False) -> tuple[float, int]:
        """Return updated (lr, local_epochs) for the next round.

        ``noise_lock``: when True, disable the healthy-regime LR acceleration.
        Used under heavy DP noise (low epsilon) where train-loss decreases are
        partly spurious: a 1.05x growth then triggers late-training drift
        because the optimiser is following noise rather than signal.
        """
        if not self.cfg.enabled:
            return self.lr, self.local_epochs

        if self._prev_loss is None:
            self._prev_loss = train_loss
            return self.lr, self.local_epochs

        rel_improve = (self._prev_loss - train_loss) / max(self._prev_loss, 1e-8)
        if rel_improve < self.cfg.min_improve:
            self._no_improve += 1
        else:
            self._no_improve = 0

        if train_loss > self._prev_loss:
            # Regime 1: loss increased - contract aggressively.
            self.lr = max(self.cfg.min_lr, self.lr * 0.5)
            self.local_epochs = max(self.cfg.min_local_epochs, self.local_epochs - 1)
        elif self._no_improve >= self.cfg.patience:
            # Regime 2: stagnation - shrink LR gently, extract more signal per round.
            self.lr = max(self.cfg.min_lr, self.lr * 0.7)
            self.local_epochs = min(self.cfg.max_local_epochs, self.local_epochs + 1)
            self._no_improve = 0
        elif not noise_lock:
            # Regime 3: healthy convergence - mildly accelerate.
            self.lr = min(self.cfg.max_lr, self.lr * 1.05)
        # else: healthy but noise-locked - hold LR steady to prevent drift.

        self._prev_loss = train_loss
        return self.lr, self.local_epochs
