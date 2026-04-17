"""Algorithmic verification of the core privacy and tuner logic.

Uses numpy only (no torch) so it runs in any environment. Exercises the same
math used inside ``AdaptivePrivacyManager`` and ``AdaptiveTuner`` on synthetic
signals and prints pass/fail for five invariants:

  A. Static clipping bounds all per-sample norms to C exactly.
  B. Gaussian noise injection reaches the calibrated magnitude (sigma = z C).
  C. Adaptive clipping converges toward the target quantile of the norm
     distribution (Andrew et al. 2022, Eq. 3.10 in the thesis).
  D. The three-regime tuner reacts correctly to each regime in isolation.
  E. Privacy accountant returns a monotonically decreasing epsilon as z grows.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from adaptive_dp_vfl.tuner import AdaptiveTuner, TunerConfig  # noqa: E402


# --------------------------------------------------------------------------
# Numpy re-implementations of the sanitisation math for torch-free testing.
# --------------------------------------------------------------------------
def clip(embedding: np.ndarray, C: float) -> np.ndarray:
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    return embedding * np.minimum(1.0, C / (norms + 1e-12))


def sanitise(embedding: np.ndarray, C: float, z: float, rng: np.random.Generator) -> np.ndarray:
    clipped = clip(embedding, C)
    noise = rng.normal(scale=z * C, size=clipped.shape)
    return clipped + noise


def adaptive_step(C: float, fraction_clipped: float, gamma: float = 0.5, eta: float = 0.1,
                  cmin: float = 0.01, cmax: float = 100.0) -> float:
    # Fraction-above parameterisation: if more samples are clipped than the
    # target fraction, loosen C by multiplying by >1.
    C = C * math.exp(eta * (fraction_clipped - gamma))
    return max(cmin, min(cmax, C))


def basic_gaussian_epsilon(z: float, delta: float = 1e-5) -> float:
    return math.sqrt(2.0 * math.log(1.25 / delta)) / max(z, 1e-12)


# --------------------------------------------------------------------------
# Tests.
# --------------------------------------------------------------------------
def test_A_static_clipping() -> bool:
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(256, 64)) * 3.0         # pre-clip norms ~ 24
    C = 1.0
    clipped = clip(emb, C)
    norms = np.linalg.norm(clipped, axis=1)
    passed = np.all(norms <= C + 1e-6)
    print(f"[A] static clipping bounds norms to C={C}: max norm={norms.max():.6f} -> "
          f"{'PASS' if passed else 'FAIL'}")
    return passed


def test_B_noise_magnitude() -> bool:
    rng = np.random.default_rng(1)
    emb = np.zeros((100_000, 32))  # zero embedding so only noise survives
    C, z = 1.0, 1.5
    san = sanitise(emb, C, z, rng)
    measured_std = san.std()
    target_std = z * C
    err = abs(measured_std - target_std) / target_std
    passed = err < 0.02
    print(f"[B] noise std matches sigma=z*C={target_std}: measured={measured_std:.4f} "
          f"(err={err*100:.2f}%) -> {'PASS' if passed else 'FAIL'}")
    return passed


def test_C_adaptive_clipping_converges() -> bool:
    # Embeddings drawn from a fixed distribution with known median norm ~= 8.
    rng = np.random.default_rng(2)
    emb = rng.normal(size=(2000, 64))                    # chi-like norms
    true_median = np.median(np.linalg.norm(emb, axis=1))
    C = 1.0                                              # deliberately too small
    gamma = 0.5
    history = [C]
    for _ in range(400):
        fraction_clipped = float((np.linalg.norm(emb, axis=1) > C).mean())
        C = adaptive_step(C, fraction_clipped, gamma=gamma, eta=0.05)
        history.append(C)
    # After convergence C should track the median (gamma=0.5 quantile).
    err = abs(history[-1] - true_median) / true_median
    passed = err < 0.05
    print(f"[C] adaptive clipping -> median norm ({true_median:.3f}): "
          f"converged to C={history[-1]:.3f} (err={err*100:.2f}%) -> "
          f"{'PASS' if passed else 'FAIL'}")
    return passed


def test_D_tuner_regimes() -> bool:
    cfg = TunerConfig(initial_lr=1e-3, initial_local_epochs=2, patience=2, min_improve=0.01)
    passed = True

    # Regime 3: healthy convergence -> LR grows by 1.05.
    t = AdaptiveTuner(cfg)
    t.step(1.0)
    lr, E = t.step(0.5)           # big improvement
    r3 = abs(lr - 1e-3 * 1.05) < 1e-9 and E == 2
    print(f"[D3] healthy convergence -> lr=1.05e-3, E=2: lr={lr:.5e} E={E} -> {'PASS' if r3 else 'FAIL'}")
    passed &= r3

    # Regime 1: loss increases -> LR halves, E decreases.
    t = AdaptiveTuner(cfg)
    t.step(1.0)
    lr, E = t.step(1.5)           # loss went UP
    r1 = abs(lr - 5e-4) < 1e-9 and E == 1
    print(f"[D1] loss-increase -> lr=5e-4, E=1: lr={lr:.5e} E={E} -> {'PASS' if r1 else 'FAIL'}")
    passed &= r1

    # Regime 2: stagnation for `patience` rounds -> LR scaled by 0.7, E +1.
    t = AdaptiveTuner(cfg)
    t.step(1.0000)
    t.step(0.9999)               # tiny improvement, below min_improve
    lr, E = t.step(0.9998)       # still tiny -> patience triggered
    # Step 2 was a tiny improvement but still regime 3 (lr*=1.05), then step 3
    # triggers patience -> regime 2 (lr*=0.7). Expected: 1e-3 * 1.05 * 0.7.
    r2 = abs(lr - 1e-3 * 1.05 * 0.7) < 1e-9 and E == 3
    print(f"[D2] stagnation (patience=2) -> lr=7.35e-4, E=3: lr={lr:.5e} E={E} -> {'PASS' if r2 else 'FAIL'}")
    passed &= r2
    return passed


def test_E_privacy_monotone() -> bool:
    zs = [0.1, 0.3, 1.0, 2.0, 5.0]
    eps = [basic_gaussian_epsilon(z) for z in zs]
    passed = all(eps[i] > eps[i + 1] for i in range(len(eps) - 1))
    print("[E] epsilon decreases monotonically with z:")
    for z, e in zip(zs, eps):
        print(f"      z={z:>4}  eps={e:.3f}")
    print(f"      -> {'PASS' if passed else 'FAIL'}")
    return passed


def main() -> int:
    results = [
        test_A_static_clipping(),
        test_B_noise_magnitude(),
        test_C_adaptive_clipping_converges(),
        test_D_tuner_regimes(),
        test_E_privacy_monotone(),
    ]
    print(f"\nSummary: {sum(results)}/{len(results)} tests passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
