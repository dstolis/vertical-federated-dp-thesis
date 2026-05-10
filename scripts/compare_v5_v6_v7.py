"""Side-by-side comparison of v5b (results/), v6 (results_v6/), v7 (results_v7/).

Reads the post-restore row (the sentinel-marked final row if present, otherwise
the last training row) for every cell that exists in each results directory
and prints a comparison table + per-(dataset, z) deltas.

Usage:
    python scripts/compare_v5_v6_v7.py
"""
from __future__ import annotations

import csv
import glob
import math
import os
from collections import defaultdict


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Each cell in v5b is named like "mnist_adaptive_z2.0.csv" (mode = adaptive).
# v6 is "mnist_flower_z2.0.csv". v7 is "mnist_pure_flower_z2.0.csv".
# Also pull the plain DP cells from v5b (results/) for a fixed-C reference.
VERSIONS = [
    ("v5b_dp",       "results",       "{ds}_dp_z{z}.csv"),
    ("v5b_adaptive", "results",       "{ds}_adaptive_z{z}.csv"),
    ("v6",           "results_v6",    "{ds}_flower_z{z}.csv"),
    ("v7",           "results_v7",    "{ds}_pure_flower_z{z}.csv"),
]
DATASETS = ("mnist", "fashion_mnist")
ZS = ("0.3", "1.0", "2.0")


def read_final(fp: str) -> dict | None:
    if not os.path.exists(fp):
        return None
    rows = list(csv.DictReader(open(fp)))
    if not rows:
        return None
    # Prefer the post-restore row (noise_lock == "-1") if present.
    restored = [r for r in rows if r.get("noise_lock") == "-1"]
    final = restored[-1] if restored else rows[-1]
    best = max(float(r["val_acc"]) for r in rows)
    return {
        "val_acc": float(final["val_acc"]),
        "best_acc": best,
        "eps": final.get("epsilon", ""),
        "C_A": float(final.get("clipping_norm_A", 0.0)),
    }


def main() -> None:
    table: dict[tuple[str, str], dict[str, dict | None]] = defaultdict(dict)
    for ds in DATASETS:
        for z in ZS:
            for label, subdir, pat in VERSIONS:
                fp = os.path.join(ROOT, subdir, pat.format(ds=ds, z=z))
                table[(ds, z)][label] = read_final(fp)

    header = (f"{'dataset':14s} {'z':>4s} | "
              f"{'v5b_dp':>8s} {'v5b_adapt':>10s} {'v6':>8s} {'v7':>8s} | "
              f"{'6-5a':>7s} {'7-5a':>7s} {'7-6':>7s} {'v6_CA':>6s}")
    print(header)
    print("-" * len(header))

    def val(cell, key):
        d = cell.get(key)
        return d["val_acc"] if d else float("nan")

    def fmt(x): return f"{x:8.4f}" if not math.isnan(x) else f"{'--':>8s}"
    def fmtd(x): return f"{x*100:+6.2f}pp" if not math.isnan(x) else f"{'--':>7s}"

    for ds in DATASETS:
        for z in ZS:
            cell = table[(ds, z)]
            a = val(cell, "v5b_dp")
            b = val(cell, "v5b_adaptive")
            c = val(cell, "v6")
            d = val(cell, "v7")
            d65 = c - b if not (math.isnan(c) or math.isnan(b)) else float("nan")
            d75 = d - b if not (math.isnan(d) or math.isnan(b)) else float("nan")
            d76 = d - c if not (math.isnan(d) or math.isnan(c)) else float("nan")
            ca = cell.get("v6")["C_A"] if cell.get("v6") else float("nan")
            print(f"{ds:14s} {z:>4s} | "
                  f"{fmt(a):>8s} {fmt(b):>10s} {fmt(c):>8s} {fmt(d):>8s} | "
                  f"{fmtd(d65)} {fmtd(d75)} {fmtd(d76)} "
                  f"{ca:6.2f}" if not math.isnan(ca) else
                  f"{ds:14s} {z:>4s} | "
                  f"{fmt(a):>8s} {fmt(b):>10s} {fmt(c):>8s} {fmt(d):>8s} | "
                  f"{fmtd(d65)} {fmtd(d75)} {fmtd(d76)}    --  ")

    print()
    missing = [(ds, z, lbl) for (ds, z), cell in table.items()
               for lbl, v in cell.items() if v is None]
    if missing:
        print("Missing cells (skip-gracefully):")
        for ds, z, lbl in missing:
            print(f"  {lbl:14s}  {ds:14s}  z={z}")


if __name__ == "__main__":
    main()
