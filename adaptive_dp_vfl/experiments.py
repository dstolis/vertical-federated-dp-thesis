"""Experiment matrix runner.

Runs the 14 configurations described in Chapter 4 (2 datasets x 3 modes x
3 privacy levels, plus 2 baselines) through the standalone simulator and
writes per-configuration CSVs plus a combined ``all_results.csv``.
"""
from __future__ import annotations

import csv
from itertools import product
from pathlib import Path

from .simulate import run
from .task import RunConfig

DATASETS = ["mnist", "fashion_mnist"]
MODES = ["baseline", "dp", "adaptive"]
NOISE_LEVELS = {"low": 0.3, "medium": 1.0, "high": 2.0}

OUT_DIR = Path(__file__).resolve().parent.parent / "results"


def build_matrix() -> list[RunConfig]:
    configs: list[RunConfig] = []
    for dataset in DATASETS:
        configs.append(RunConfig(dataset=dataset, mode="baseline", noise_multiplier=0.0))
        for mode in ("dp", "adaptive"):
            for _name, z in NOISE_LEVELS.items():
                configs.append(RunConfig(dataset=dataset, mode=mode, noise_multiplier=z))
    return configs


def _tag(cfg: RunConfig) -> str:
    if cfg.mode == "baseline":
        return f"{cfg.dataset}_baseline"
    return f"{cfg.dataset}_{cfg.mode}_z{cfg.noise_multiplier}"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    for cfg in build_matrix():
        tag = _tag(cfg)
        print(f"==== running {tag} ====")
        rows = run(cfg, out_csv=OUT_DIR / f"{tag}.csv")
        for r in rows:
            r["tag"] = tag
            all_rows.append(r)
    combined = OUT_DIR / "all_results.csv"
    with combined.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"wrote {combined}")


if __name__ == "__main__":
    main()
