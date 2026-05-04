"""Run the v7 pure-Flower matrix (Flower DP adaptive clipping + FedAdam only,
no custom tuner).

Writes per-cell CSVs to results_v7/.
"""
from __future__ import annotations

from pathlib import Path

from adaptive_dp_vfl.simulate_pure_flower import run_pure_flower
from adaptive_dp_vfl.task import RunConfig


OUT = Path(__file__).resolve().parent.parent / "results_v7"
DATASETS = ("mnist", "fashion_mnist")
ZS = (0.3, 1.0, 2.0)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ds in DATASETS:
        for z in ZS:
            cfg = RunConfig(
                dataset=ds,
                mode="adaptive",              # turns on DP; tuner is unused
                noise_multiplier=z,
                target_quantile=0.5,          # Flower default gamma
                clip_norm_lr=0.2,             # Flower default eta_C
                min_clipping_norm=0.01,
                max_clipping_norm=20.0,
                max_sigma=10.0,
                num_rounds=40,
                initial_local_epochs=1,       # fixed E for pure-Flower
            )
            out = OUT / f"{ds}_pure_flower_z{z}.csv"
            print(f"\n=== {ds} / pure_flower / z={z} ===")
            run_pure_flower(
                cfg, out_csv=out,
                clipped_count_stddev=0.0,
                initial_clipping_norm=0.1,
            )
    print(f"\nDone. CSVs in {OUT}")


if __name__ == "__main__":
    main()
