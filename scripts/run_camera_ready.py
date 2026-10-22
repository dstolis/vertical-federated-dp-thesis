"""Camera-ready experiment matrix (AICS 2026).

All runs use the validation protocol: 10,000 samples are held out of the
training split for checkpoint selection and the noise-lock, and the test
split is used once, to report the selected model.

Groups
  main      baseline (seed 42) + fixed DP and adaptive at z in {0.3, 1.0, 2.0},
            both datasets, seeds 42, 43, 44
  control   z = 2: fixed DP with C = 5, and fixed DP with C = 5 and lr / 5,
            both datasets, seeds 42, 43, 44
  flower    v6 (Flower adaptive clipping + our tuner) and v7 (Flower only),
            both datasets, z in {0.3, 1.0, 2.0}, seed 42
  noln      no-LayerNorm ablation, fixed DP and adaptive, z = 2, seed 42

Usage
  python scripts/run_camera_ready.py list
  python scripts/run_camera_ready.py run --worker K --workers N [--group G]
Finished runs are skipped, so the script can be restarted safely.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from adaptive_dp_vfl.simulate import run as run_main
from adaptive_dp_vfl.simulate_flower_strategy import run_flower_strategy
from adaptive_dp_vfl.simulate_pure_flower import run_pure_flower
from adaptive_dp_vfl.task import RunConfig

OUT = Path(__file__).resolve().parent.parent / "results_camera_ready"
DATASETS = ("mnist", "fashion_mnist")
ZS = (0.3, 1.0, 2.0)
SEEDS = (42, 43, 44)


SEEDS_ALL = (42, 43, 44)


def jobs() -> list[tuple[str, str, str, RunConfig]]:
    """(group, tag, runner, cfg) in priority order: the headline table first,
    then the controls, Flower at z = 2, the ablation, extra seeds, the rest."""
    js = []
    A = lambda ds, z, seed, **kw: RunConfig(dataset=ds, mode="adaptive", noise_multiplier=z, seed=seed, **kw)
    D = lambda ds, z, seed, **kw: RunConfig(dataset=ds, mode="dp", noise_multiplier=z, seed=seed, **kw)
    # 1. headline: seed 42, every cell
    for ds in DATASETS:
        for z in (1.0, 2.0, 0.3):
            js.append(("main", f"s42_{ds}_adaptive_z{z}", "main", A(ds, z, 42)))
    for ds in DATASETS:
        for z in ZS:
            js.append(("main", f"s42_{ds}_dp_z{z}", "main", D(ds, z, 42)))
        js.append(("main", f"s42_{ds}_baseline", "main",
                   RunConfig(dataset=ds, mode="baseline", noise_multiplier=0.0, seed=42)))
    # 2. controls at z = 2, seed 42
    for ds in DATASETS:
        js.append(("control", f"s42_{ds}_dpC5_z2.0", "main", D(ds, 2.0, 42, clipping_norm=5.0)))
        js.append(("control", f"s42_{ds}_dpC5lr_z2.0", "main",
                   D(ds, 2.0, 42, clipping_norm=5.0, initial_lr=2e-4)))
    # 3. Flower at z = 2
    for ds in DATASETS:
        js.append(("flower", f"v6_{ds}_z2.0", "v6", A(ds, 2.0, 42)))
        js.append(("flower", f"v7_{ds}_z2.0", "v7", A(ds, 2.0, 42, initial_local_epochs=1)))
    # 4. no-LayerNorm ablation
    for ds in DATASETS:
        for mode in ("adaptive", "dp"):
            js.append(("noln", f"noln_{ds}_{mode}_z2.0", "main",
                       RunConfig(dataset=ds, mode=mode, noise_multiplier=2.0, seed=42, layer_norm=False)))
    # 5. extra seeds, every cell
    for seed in (43, 44):
        for ds in DATASETS:
            for z in (1.0, 2.0, 0.3):
                js.append(("main", f"s{seed}_{ds}_adaptive_z{z}", "main", A(ds, z, seed)))
        for ds in DATASETS:
            for z in ZS:
                js.append(("main", f"s{seed}_{ds}_dp_z{z}", "main", D(ds, z, seed)))
    # 6. controls, extra seeds
    for seed in (43, 44):
        for ds in DATASETS:
            js.append(("control", f"s{seed}_{ds}_dpC5_z2.0", "main", D(ds, 2.0, seed, clipping_norm=5.0)))
            js.append(("control", f"s{seed}_{ds}_dpC5lr_z2.0", "main",
                       D(ds, 2.0, seed, clipping_norm=5.0, initial_lr=2e-4)))
    # 8. fixed C = 5 with the adaptive configuration's data order (added after
    #    review: isolates the data-order effect of the adaptive warm-up pass)
    for seed in SEEDS_ALL:
        for ds in DATASETS:
            js.append(("control", f"s{seed}_{ds}_dpC5m_z2.0", "main",
                       D(ds, 2.0, seed, clipping_norm=5.0, match_warmup_order=True)))
    # 7. Flower at z = 0.3 and 1.0
    for ds in DATASETS:
        for z in (1.0, 0.3):
            js.append(("flower", f"v6_{ds}_z{z}", "v6", A(ds, z, 42)))
            js.append(("flower", f"v7_{ds}_z{z}", "v7", A(ds, z, 42, initial_local_epochs=1)))
    return js


def execute(runner: str, cfg: RunConfig, out_csv: Path) -> None:
    if runner == "main":
        run_main(cfg, out_csv=out_csv)
    elif runner == "v6":
        run_flower_strategy(cfg, out_csv=out_csv, clipped_count_stddev=0.0, initial_clipping_norm=0.1)
    elif runner == "v7":
        run_pure_flower(cfg, out_csv=out_csv, clipped_count_stddev=0.0, initial_clipping_norm=0.1)
    else:
        raise ValueError(runner)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "run"])
    ap.add_argument("--group", default=None, help="comma-separated groups")
    ap.add_argument("--tags", default=None, help="comma-separated tag prefixes to include")
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()

    groups = set(a.group.split(",")) if a.group else None
    prefixes = tuple(a.tags.split(",")) if a.tags else None
    all_jobs = [j for j in jobs()
                if (groups is None or j[0] in groups)
                and (prefixes is None or j[1].startswith(prefixes))]
    if a.cmd == "list":
        for i, (g, tag, r, _c) in enumerate(all_jobs):
            done = (OUT / f"{tag}.csv").exists()
            print(f"{i:3d} {g:8s} {r:5s} {tag:40s} {'done' if done else ''}")
        print(f"{len(all_jobs)} jobs")
        return

    torch.set_num_threads(a.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    import os
    for g, tag, runner, cfg in all_jobs:
        out = OUT / f"{tag}.csv"
        claim = OUT / f".{tag}.claim"
        if out.exists():
            continue
        try:
            fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.uname().nodename}:{os.getpid()}".encode())
            os.close(fd)
        except FileExistsError:
            continue
        t0 = time.time()
        print(f"==== {tag} ({runner}) ====", flush=True)
        tmp = OUT / f".{tag}.csv.partial"
        execute(runner, cfg, tmp)
        tmp.rename(out)
        print(f"==== done {tag} in {time.time() - t0:.0f}s ====", flush=True)


if __name__ == "__main__":
    main()
