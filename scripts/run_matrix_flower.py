"""Run the v6 Flower-style server-side adaptive clipping matrix.

Produces one CSV per (dataset, z) cell in results_v6/ using the same
RunConfig defaults as v5b but with the server-side adaptive clipping strategy
from adaptive_dp_vfl.privacy_flower.

Usage:
    python scripts/run_matrix_flower.py
"""
from __future__ import annotations

from pathlib import Path

from adaptive_dp_vfl.simulate_flower_strategy import run_flower_strategy
from adaptive_dp_vfl.task import RunConfig


OUT = Path(__file__).resolve().parent.parent / "results_v6"
DATASETS = ("mnist", "fashion_mnist")
ZS = (0.3, 1.0, 2.0)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ds in DATASETS:
        for z in ZS:
            cfg = RunConfig(
                dataset=ds,
                mode="adaptive",           # keeps tuner enabled
                noise_multiplier=z,
                # Flower defaults align with Andrew et al. 2022 defaults.
                target_quantile=0.5,
                clip_norm_lr=0.2,
                min_clipping_norm=0.01,
                max_clipping_norm=20.0,
                max_sigma=10.0,
                num_rounds=40,
            )
            out = OUT / f"{ds}_flower_z{z}.csv"
            print(f"\n=== {ds} / flower_adaptive / z={z} ===")
            run_flower_strategy(
                cfg, out_csv=out,
                clipped_count_stddev=0.0,      # match Flower default of 0
                initial_clipping_norm=0.1,     # Flower default
            )
    print(f"\nDone. CSVs in {OUT}")


if __name__ == "__main__":
    main()
