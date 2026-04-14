"""Generate the six figures referenced by Chapter 5.

Reads ``results/all_results.csv`` and writes PDF + PNG plots to ``results/figures/``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "results"
FIG_DIR = RESULTS / "figures"


def _save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)


def _style_for(mode: str) -> dict:
    return {
        "baseline": {"color": "#1f77b4", "linestyle": "-"},
        "dp": {"color": "#ff7f0e", "linestyle": "-"},
        "adaptive": {"color": "#2ca02c", "linestyle": "--"},
    }[mode]


def accuracy_curves(df: pd.DataFrame) -> None:
    for dataset, sub in df.groupby("dataset"):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for tag, g in sub.groupby("tag"):
            ax.plot(g["round"], g["val_acc"], label=tag, **_style_for(g["mode"].iloc[0]))
        ax.set_xlabel("Round"); ax.set_ylabel("Validation accuracy")
        ax.set_title(f"Accuracy over rounds - {dataset}"); ax.legend(fontsize=7)
        _save(fig, f"accuracy_curves_{dataset}")


def loss_curves(df: pd.DataFrame) -> None:
    for dataset, sub in df.groupby("dataset"):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for tag, g in sub.groupby("tag"):
            ax.plot(g["round"], g["train_loss"], label=tag, **_style_for(g["mode"].iloc[0]))
        ax.set_xlabel("Round"); ax.set_ylabel("Training loss")
        ax.set_title(f"Training loss over rounds - {dataset}"); ax.legend(fontsize=7)
        _save(fig, f"loss_curves_{dataset}")


def final_accuracy_bars(df: pd.DataFrame) -> None:
    last = df.sort_values("round").groupby("tag").tail(1)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(last["tag"], last["val_acc"])
    ax.set_ylabel("Final validation accuracy")
    ax.tick_params(axis="x", rotation=60, labelsize=7)
    _save(fig, "final_accuracy_bars")


def privacy_utility(df: pd.DataFrame) -> None:
    last = df.sort_values("round").groupby("tag").tail(1)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for (mode, dataset), sub in last.groupby(["mode", "dataset"]):
        sub = sub.sort_values("noise_multiplier")
        ax.plot(sub["noise_multiplier"], sub["val_acc"], marker="o",
                label=f"{dataset}/{mode}")
    ax.set_xlabel("Noise multiplier z"); ax.set_ylabel("Validation accuracy")
    ax.set_title("Privacy-utility trade-off"); ax.legend(fontsize=8)
    _save(fig, "privacy_utility")


def adaptive_behaviour(df: pd.DataFrame) -> None:
    sub = df[df["mode"] == "adaptive"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for tag, g in sub.groupby("tag"):
        axes[0].plot(g["round"], g["lr"], label=tag)
        axes[1].plot(g["round"], g["local_epochs"], label=tag)
        axes[2].plot(g["round"], g["clipping_norm_A"], label=tag)
    axes[0].set_title("Learning rate"); axes[0].set_xlabel("Round"); axes[0].set_yscale("log")
    axes[1].set_title("Local epochs"); axes[1].set_xlabel("Round")
    axes[2].set_title("Clipping norm (party A)"); axes[2].set_xlabel("Round")
    for a in axes: a.legend(fontsize=6)
    _save(fig, "adaptive_behaviour")


def main() -> None:
    combined = RESULTS / "all_results.csv"
    df = pd.read_csv(combined)
    accuracy_curves(df); loss_curves(df); final_accuracy_bars(df)
    privacy_utility(df); adaptive_behaviour(df)
    print(f"figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
