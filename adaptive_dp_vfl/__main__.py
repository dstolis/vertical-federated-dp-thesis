"""Command-line entrypoint.

Usage:
    python -m adaptive_dp_vfl                      # run one config with defaults
    python -m adaptive_dp_vfl --matrix             # run the full experiment matrix
    python -m adaptive_dp_vfl --dataset fashion_mnist --mode adaptive --noise 2.0
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .experiments import main as run_matrix
from .simulate import run
from .task import RunConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", action="store_true", help="Run all 14 configs and write all_results.csv")
    p.add_argument("--dataset", choices=["mnist", "fashion_mnist"], default="mnist")
    p.add_argument("--mode", choices=["baseline", "dp", "adaptive"], default="adaptive")
    p.add_argument("--noise", type=float, default=1.0, help="Gaussian noise multiplier z")
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.matrix:
        run_matrix()
        return
    cfg = RunConfig(
        dataset=args.dataset, mode=args.mode,
        noise_multiplier=args.noise if args.mode != "baseline" else 0.0,
        num_rounds=args.rounds, batch_size=args.batch_size, seed=args.seed,
    )
    out = Path(__file__).resolve().parent.parent / "results" / f"{args.dataset}_{args.mode}_z{args.noise}.csv"
    run(cfg, out_csv=out)


if __name__ == "__main__":
    main()
