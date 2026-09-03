"""Robustness sweep for the z=2 cells.

Two experiments feed the robustness section of the paper:

  seeds     re-run the four decisive z=2 cells (2 datasets x {dp, adaptive})
            at additional seeds, to check that the DP-vs-adaptive ordering is
            not a single-seed artefact.

  noln      re-run the same four cells with the terminal LayerNorm removed
            from both bottom networks. This is the causal test of the paper's
            claim: without normalisation the IR norm distribution regains a
            heavy tail (so quantile matching has a signal again) but its scale
            explodes, so clipping to C annihilates the embedding.

Usage:
    python scripts/run_seed_sweep.py seeds 43 44
    python scripts/run_seed_sweep.py noln
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adaptive_dp_vfl.simulate import run
from adaptive_dp_vfl.task import RunConfig

DATASETS = ["mnist", "fashion_mnist"]
MODES = ["dp", "adaptive"]
Z = 2.0
ROOT = Path(__file__).resolve().parent.parent


def sweep_seeds(seeds: list[int]) -> None:
    out = ROOT / "results_seeds"
    for seed in seeds:
        for dataset in DATASETS:
            for mode in MODES:
                cfg = RunConfig(dataset=dataset, mode=mode, noise_multiplier=Z, seed=seed)
                tag = f"seed{seed}_{dataset}_{mode}_z{Z}"
                print(f"==== {tag} ====", flush=True)
                run(cfg, out_csv=out / f"{tag}.csv")


def sweep_noln() -> None:
    out = ROOT / "results_ablation_noln"
    for dataset in DATASETS:
        for mode in MODES:
            cfg = RunConfig(dataset=dataset, mode=mode, noise_multiplier=Z,
                            seed=42, layer_norm=False)
            tag = f"noln_{dataset}_{mode}_z{Z}"
            print(f"==== {tag} ====", flush=True)
            run(cfg, out_csv=out / f"{tag}.csv")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "seeds"
    if which == "seeds":
        sweep_seeds([int(a) for a in sys.argv[2:]] or [43, 44])
    elif which == "noln":
        sweep_noln()
    else:
        raise SystemExit(__doc__)
