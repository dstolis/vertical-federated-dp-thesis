# Adaptive DP-VFL

**Adaptive Differentially Private Vertical Federated Learning** on
[Flower](https://flower.ai) + PyTorch.

Two parties each hold a *vertical* half of every MNIST / Fashion-MNIST image
(left and right halves of the pixel grid). Each party trains a local
**bottom model** that produces an intermediate representation (IR); the IRs
are clipped, perturbed with Gaussian noise, and forwarded to a server that
hosts the **top model**. Gradients flow back the same way, sanitised again.

The novel piece is the **adaptive** controller, which decides — per round —
how aggressively to clip, what learning rate to use, and how many local
epochs each party should run, all from observed training-loss dynamics. The
goal is a usable privacy–utility trade-off without hand-tuning a separate
configuration for every (dataset, noise multiplier) pair.

## What's inside

| Component | What it does |
|---|---|
| `AdaptivePrivacyManager` (`privacy.py`) | Per-sample L2 clip + isotropic Gaussian noise on embeddings. Adaptive variant updates the clip norm *C* once per round via the quantile-matching geometric rule of [Andrew et al., 2022](https://arxiv.org/abs/1905.03871). |
| `AdaptiveTuner` (`tuner.py`) | Three-regime round-level controller for `(lr, local_epochs)`: aggressive contract on loss-increase, gentle shrink + longer training on stagnation, mild acceleration on healthy convergence. Noise-locks the accelerator under heavy DP. |
| `BottomNet` / `TopNet` (`models.py`) | Two small CNNs split at the embedding layer. |
| `VerticalSplitDataset` (`data.py`) | Splits each image down the middle and serves both halves with a shared sample index, so the two parties stay aligned without exchanging labels. |
| `simulate.py` | Single-process simulator — fastest path to results. |
| `simulate_flower_strategy.py` / `simulate_pure_flower.py` | Two Flower deployments validating that the same code paths produce the same numbers under the real federated protocol. |
| `client_app.py` / `server_app.py` | Flower `ClientApp` / `ServerApp` driving the CONFIGURE → FORWARD → BACKWARD cycle. |
| `experiments.py` | Matrix runner over `{dataset, mode, noise_multiplier}` → one CSV per config + `all_results.csv`. |

## Modes

| Mode | DP noise | Adaptive clipping | Adaptive lr / epochs |
|---|:---:|:---:|:---:|
| `baseline` | – | – | – |
| `dp` | ✓ | – | – |
| `adaptive` | ✓ | ✓ | ✓ |

## Quick start

```bash
pip install -e .

# one config
python -m adaptive_dp_vfl --dataset fashion_mnist --mode adaptive --noise 1.0 --rounds 20

# full 14-config matrix → results/all_results.csv
python -m adaptive_dp_vfl --matrix

# regenerate the figures
python scripts/plot_results.py
```

### Run on the Flower runtime

```bash
flwr run . local-simulation
```

This spins up a `SuperLink` with two simulated `SuperNode`s (the two VFL
parties). Config defaults live in the `[tool.flwr.app.config]` block of
[`pyproject.toml`](pyproject.toml).

## Repository layout

```
adaptive_dp_vfl/
  __main__.py                  CLI entrypoint
  models.py                    BottomNet, TopNet
  data.py                      VerticalSplitDataset + loaders
  privacy.py                   AdaptivePrivacyManager
  tuner.py                     AdaptiveTuner
  task.py                      Shared RunConfig / PartyState / ServerState
  simulate.py                  Standalone simulator
  simulate_flower_strategy.py  Flower deployment (custom strategy)
  simulate_pure_flower.py      Flower deployment (vanilla protocol)
  client_app.py                Flower ClientApp
  server_app.py                Flower ServerApp
  experiments.py               Matrix runner
  privacy_flower.py            DP wiring for the Flower path

scripts/
  plot_results.py              Generate figures from all_results.csv
  verify_algorithms.py         Sanity checks for clipping + noise
  verify_vertical_split.py     Confirms the two halves reconstruct the image
  run_matrix_flower.py         Matrix runner for the Flower strategy path
  run_matrix_pure_flower.py    Matrix runner for the pure-Flower path
  compare_v1_v2.py             Diff two result snapshots
  compare_v5_v6_v7.py          Compare standalone vs. Flower vs. pure-Flower

results/                       Latest matrix results + figures
results_v6/, results_v7/       Cross-runtime parity runs
data/                          MNIST / Fashion-MNIST cache (gitignored)
```

## Reproducing the experiments

The full results matrix is 14 configurations:

- 2 datasets × `{baseline}` (1 config per dataset)
- 2 datasets × `{dp, adaptive}` × `{z ∈ 0.3, 1.0, 2.0}` (6 configs per dataset)

`python -m adaptive_dp_vfl --matrix` runs all 14 sequentially and writes
per-config CSVs plus `all_results.csv` into `results/`. Figures are
regenerated from that CSV with `scripts/plot_results.py`.

## Requirements

- Python ≥ 3.10
- See `pyproject.toml` for the pinned dependency floors (Flower ≥ 1.13,
  PyTorch ≥ 2.2).

## Data

MNIST and Fashion-MNIST are downloaded on first run into `data/` (gitignored).
No external data needs to be staged.
