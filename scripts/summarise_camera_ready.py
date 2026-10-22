"""Turn results_camera_ready/*.csv into the numbers and tables used by the
camera-ready paper, so that no number in the paper is typed by hand.

Writes, into the paper directory given as argv[1]:
  numbers.tex        \\newcommand macros for every number quoted in the text
  tab_main.tex       Table 1: test accuracy (mean +- sd over seeds)
  tab_privacy.tex    Table 2: local epochs and cumulative epsilon
  tab_flower.tex     Table 3: Flower comparison
  tab_controls.tex   Table 4: C = 5 controls and the no-LayerNorm ablation
  summary.json       everything above, for verification
Missing runs appear as "--" so the paper compiles while runs are in flight.
"""
from __future__ import annotations

import csv
import json
import math
import statistics as st
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

RES = Path(__file__).resolve().parent.parent / "results_camera_ready"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
DELTA = 1e-5
SEEDS = (42, 43, 44)
ZS = (0.3, 1.0, 2.0)
DS = ("mnist", "fashion_mnist")
NUM_ROUNDS = 40


def load(tag: str):
    p = RES / f"{tag}.csv"
    if not p.exists():
        return None
    rows = list(csv.DictReader(open(p)))
    train = [r for r in rows if r["round"].isdigit() and int(r["round"]) <= NUM_ROUNDS]
    final = rows[-1]
    assert final.get("noise_lock") == "-1", f"{tag}: last row is not the restored model"
    return {
        "test": float(final["test_acc"]),
        "val": float(final["val_acc"]),
        "best_round": int(final["best_round_so_far"]),
        # Each row logs E *after* the tuner step, i.e. the E used in the NEXT
        # round. Epochs actually run = initial E (1) in round 1, then the
        # logged E of rounds 1..39 for rounds 2..40.
        "epochs": 1 + sum(int(float(r["local_epochs"])) for r in train[:-1]),
        "last_round_test": float(train[-1]["test_acc"]),
        "C": [float(r["clipping_norm_A"]) for r in train],
        "rows": len(train),
    }


def agm_eps(zeff: float, delta: float = DELTA) -> float:
    """Exact epsilon of a Gaussian mechanism with noise multiplier zeff
    (Balle and Wang, 2018), found by solving delta(eps) = delta."""
    def logd(eps):
        t1 = norm.cdf(1 / (2 * zeff) - eps * zeff)
        l2 = eps + norm.logcdf(-1 / (2 * zeff) - eps * zeff)
        v = t1 - (math.exp(l2) if l2 < 700 else float("inf"))
        return math.log(v) if v > 0 else -math.inf
    return brentq(lambda e: logd(e) - math.log(delta), 1e-9, 5e3, xtol=1e-10)


def cum_eps(z: float, k: int) -> float:
    """k releases of a Gaussian mechanism with multiplier z compose to one
    Gaussian mechanism with multiplier z / sqrt(k)."""
    return agm_eps(z / math.sqrt(k))


def fmt(x, nd=3):
    return "--" if x is None else f"{x:.{nd}f}"


def ms(vals, nd=3):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "--", None, None
    m = st.mean(vals)
    s = st.stdev(vals) if len(vals) > 1 else 0.0
    return (f"{m:.{nd}f} $\\pm$ {s:.{nd}f}" if len(vals) > 1 else f"{m:.{nd}f}"), m, s


S: dict = {"cells": {}, "flower": {}, "controls": {}, "noln": {}, "baseline": {}}
macros: dict[str, str] = {}

# ---- main cells -------------------------------------------------------------
for ds in DS:
    b = load(f"s42_{ds}_baseline")
    S["baseline"][ds] = None if b is None else b["test"]
    for z in ZS:
        cell = {}
        for mode in ("dp", "adaptive"):
            runs = {s: load(f"s{s}_{ds}_{mode}_z{z}") for s in SEEDS}
            done = {s: r for s, r in runs.items() if r is not None}
            cell[mode] = {
                "test": {s: r["test"] for s, r in done.items()},
                "best_round": {s: r["best_round"] for s, r in done.items()},
                "epochs": {s: r["epochs"] for s, r in done.items()},
                "eps": {s: cum_eps(z, r["epochs"]) for s, r in done.items()},
                "last_round_test": {s: r["last_round_test"] for s, r in done.items()},
                "C": {s: r["C"] for s, r in done.items()},
            }
        # paired gap per seed (only seeds where both finished)
        paired = [cell["adaptive"]["test"][s] - cell["dp"]["test"][s]
                  for s in SEEDS if s in cell["adaptive"]["test"] and s in cell["dp"]["test"]]
        cell["gap_pp"] = [100 * g for g in paired]
        S["cells"][f"{ds}_{z}"] = cell

# ---- controls, flower, ablation --------------------------------------------
for ds in DS:
    for kind in ("dpC5", "dpC5lr", "dpC5m"):
        runs = {s: load(f"s{s}_{ds}_{kind}_z2.0") for s in SEEDS}
        S["controls"][f"{ds}_{kind}"] = {s: {"test": r["test"], "best_round": r["best_round"], "epochs": r["epochs"]}
                                          for s, r in runs.items() if r}
    for v in ("v6", "v7"):
        for z in ZS:
            r = load(f"{v}_{ds}_z{z}")
            if r:
                S["flower"][f"{v}_{ds}_{z}"] = {"test": r["test"], "best_round": r["best_round"],
                                                "epochs": r["epochs"], "C_best": r["C"][r["best_round"] - 1],
                                                "C_final": r["C"][-1]}
    for mode in ("dp", "adaptive"):
        r = load(f"noln_{ds}_{mode}_z2.0")
        if r:
            S["noln"][f"{ds}_{mode}"] = {"test": r["test"], "best_round": r["best_round"], "epochs": r["epochs"]}

# ---- IR norm statistics (collect_ir_norms.py) ------------------------------
NORMS = RES / "ir_norms"
for tag in ("layernorm", "nolayernorm"):
    for which, fn in (("warm", f"norms_{tag}_warmup.npy"), ("trained", f"norms_{tag}.npy")):
        p = NORMS / fn
        if p.exists():
            v = np.load(p)
            S.setdefault("norms", {})[f"{tag}_{which}"] = {
                "n": int(len(v)), "mean": float(v.mean()), "sd": float(v.std()),
                "cv": float(100 * v.std() / v.mean()), "min": float(v.min()), "max": float(v.max())}

# ---- tables -----------------------------------------------------------------
def acc_cell(ds, z, mode, nd=3):
    t = S["cells"][f"{ds}_{z}"][mode]["test"]
    return ms(list(t.values()), nd)


lines = [r"\begin{tabular}{@{}c cc r cc r@{}}", r"\toprule",
         r"& \multicolumn{3}{c}{MNIST} & \multicolumn{3}{c}{Fashion-MNIST} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
         r"$z$ & Fixed DP & Adaptive & $\Delta$ & Fixed DP & Adaptive & $\Delta$ \\", r"\midrule"]
base = [fmt(S["baseline"].get(ds)) for ds in DS]
lines.append(rf"none & \multicolumn{{3}}{{c}}{{{base[0]}}} & \multicolumn{{3}}{{c}}{{{base[1]}}} \\")
for z in ZS:
    parts = [f"{z}"]
    for ds in DS:
        a_dp, _, _ = acc_cell(ds, z, "dp")
        a_ad, _, _ = acc_cell(ds, z, "adaptive")
        g = S["cells"][f"{ds}_{z}"]["gap_pp"]
        gtxt = "--" if not g else f"${st.mean(g):+.1f}$"
        parts += [a_dp, a_ad, gtxt]
    lines.append(" & ".join(parts) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "tab_main.tex").write_text("\n".join(lines) + "\n")

lines = [r"\begin{tabular}{@{}c cc cc cc@{}}", r"\toprule",
         r"& \multicolumn{2}{c}{Local epochs} & \multicolumn{4}{c}{Cumulative $\varepsilon$ per party ($\delta=10^{-5}$)} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-7}",
         r"& \multicolumn{2}{c}{Adaptive} & \multicolumn{2}{c}{MNIST} & \multicolumn{2}{c}{Fashion-MNIST} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
         r"$z$ & MNIST & F-MNIST & Fixed & Adaptive & Fixed & Adaptive \\", r"\midrule"]
for z in ZS:
    ep = []
    for ds in DS:
        e = list(S["cells"][f"{ds}_{z}"]["adaptive"]["epochs"].values())
        ep.append("--" if not e else f"{st.mean(e):.0f}")
    eps = []
    for ds in DS:
        for mode in ("dp", "adaptive"):
            v = list(S["cells"][f"{ds}_{z}"][mode]["eps"].values())
            eps.append("--" if not v else f"{st.mean(v):.1f}")
    lines.append(" & ".join([f"{z}"] + ep + eps) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "tab_privacy.tex").write_text("\n".join(lines) + "\n")

lines = [r"\begin{tabular}{@{}l ccc ccc@{}}", r"\toprule",
         r"& \multicolumn{3}{c}{MNIST} & \multicolumn{3}{c}{Fashion-MNIST} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
         r"Configuration & $z{=}0.3$ & $z{=}1.0$ & $z{=}2.0$ & $z{=}0.3$ & $z{=}1.0$ & $z{=}2.0$ \\", r"\midrule"]
def s42(ds, z, mode):
    return S["cells"][f"{ds}_{z}"][mode]["test"].get(42)
for label, getter in (("Fixed DP ($C{=}1$)", lambda ds, z: s42(ds, z, "dp")),
                      ("Our adaptive controller", lambda ds, z: s42(ds, z, "adaptive")),
                      ("v6: Flower clipping + our tuner", lambda ds, z: S["flower"].get(f"v6_{ds}_{z}", {}).get("test")),
                      ("v7: Flower clipping only", lambda ds, z: S["flower"].get(f"v7_{ds}_{z}", {}).get("test"))):
    lines.append(" & ".join([label] + [fmt(getter(ds, z)) for ds in DS for z in ZS]) + r" \\")
def eps_fmt(z, ep):
    return "--" if ep is None else f"{cum_eps(z, ep):.1f}"
lines.append(r"\midrule")
for label, getter in (("$\\varepsilon$, our adaptive controller", lambda ds, z: S["cells"][f"{ds}_{z}"]["adaptive"]["epochs"].get(42)),
                      ("$\\varepsilon$, v6", lambda ds, z: S["flower"].get(f"v6_{ds}_{z}", {}).get("epochs"))):
    lines.append(" & ".join([label] + [eps_fmt(z, getter(ds, z)) for ds in DS for z in ZS]) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "tab_flower.tex").write_text("\n".join(lines) + "\n")

lines = [r"\begin{tabular}{@{}l cc cc@{}}", r"\toprule",
         r"& \multicolumn{2}{c}{MNIST} & \multicolumn{2}{c}{Fashion-MNIST} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
         r"Configuration at $z{=}2$ & Accuracy & $\varepsilon$ & Accuracy & $\varepsilon$ \\", r"\midrule"]
def ctl(ds, kind):
    v = [d["test"] for d in S["controls"].get(f"{ds}_{kind}", {}).values()]
    return ms(v)[0]
def ctl_eps(ds, kind):
    v = [cum_eps(2.0, d["epochs"]) for d in S["controls"].get(f"{ds}_{kind}", {}).values()]
    return "--" if not v else f"{st.mean(v):.1f}"
def cell_eps(ds, mode):
    v = list(S["cells"][f"{ds}_2.0"][mode]["eps"].values())
    return "--" if not v else f"{st.mean(v):.1f}"
for label, f, g in (("Fixed DP, $C{=}1$", lambda ds: acc_cell(ds, 2.0, "dp")[0], lambda ds: cell_eps(ds, "dp")),
                    ("Adaptive ($C{=}5$ from round 1)", lambda ds: acc_cell(ds, 2.0, "adaptive")[0], lambda ds: cell_eps(ds, "adaptive")),
                    ("Fixed DP, $C{=}5$", lambda ds: ctl(ds, "dpC5"), lambda ds: ctl_eps(ds, "dpC5")),
                    ("Fixed DP, $C{=}5$, adaptive's order", lambda ds: ctl(ds, "dpC5m"), lambda ds: ctl_eps(ds, "dpC5m")),
                    ("Fixed DP, $C{=}5$, $\\alpha/5$", lambda ds: ctl(ds, "dpC5lr"), lambda ds: ctl_eps(ds, "dpC5lr"))):
    lines.append(" & ".join([label, f("mnist"), g("mnist"), f("fashion_mnist"), g("fashion_mnist")]) + r" \\")
lines.append(r"\midrule")
def noln_eps(ds, mode):
    e = S["noln"].get(f"{ds}_{mode}", {}).get("epochs")
    return "--" if e is None else f"{cum_eps(2.0, e):.1f}"
for mode, label in (("dp", "No LayerNorm, fixed DP"), ("adaptive", "No LayerNorm, adaptive")):
    lines.append(" & ".join([label] + [x for ds in DS for x in (fmt(S["noln"].get(f"{ds}_{mode}", {}).get("test")), noln_eps(ds, mode))]) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "tab_controls.tex").write_text("\n".join(lines) + "\n")

# ---- macros for the text ---------------------------------------------------
def put(name, val):
    macros[name] = val


def mean_or(vals, f, default="--"):
    vals = [v for v in vals if v is not None]
    return f(st.mean(vals)) if vals else default


for ds, D in (("mnist", "Mn"), ("fashion_mnist", "Fm")):
    put(f"base{D}", fmt(S["baseline"].get(ds)))
    for z, Z in zip(ZS, ("Low", "Mid", "High")):
        c = S["cells"][f"{ds}_{z}"]
        put(f"gap{D}{Z}", mean_or(c["gap_pp"], lambda m: f"{m:+.1f}"))
        put(f"gapAbs{D}{Z}", mean_or(c["gap_pp"], lambda m: f"{abs(m):.1f}"))
        put(f"gapMin{D}{Z}", mean_or([min(c['gap_pp'])] if c["gap_pp"] else [], lambda m: f"{m:+.1f}"))
        put(f"gapMax{D}{Z}", mean_or([max(c['gap_pp'])] if c["gap_pp"] else [], lambda m: f"{m:+.1f}"))
        put(f"nNeg{D}{Z}", str(sum(1 for g in c["gap_pp"] if g < 0)))
        put(f"nSeeds{D}{Z}", str(len(c["gap_pp"])))
        for mode, M in (("dp", "Dp"), ("adaptive", "Ad")):
            put(f"acc{M}{D}{Z}", mean_or(list(c[mode]["test"].values()), lambda m: f"{m:.3f}"))
            put(f"eps{M}{D}{Z}", mean_or(list(c[mode]["eps"].values()), lambda m: f"{m:.1f}"))
            put(f"ep{M}{D}{Z}", mean_or(list(c[mode]["epochs"].values()), lambda m: f"{m:.0f}"))
        e_dp = list(c["dp"]["eps"].values()); e_ad = list(c["adaptive"]["eps"].values())
        put(f"epsRatio{D}{Z}", "--" if not (e_dp and e_ad) else f"{st.mean(e_ad) / st.mean(e_dp):.1f}")
        ep_ad = list(c["adaptive"]["epochs"].values())
        put(f"epRatio{D}{Z}", "--" if not ep_ad else f"{st.mean(ep_ad) / NUM_ROUNDS:.1f}")

# round-1 statistics at z = 2
r1 = [(ds, m, s, br) for ds in DS for m in ("dp", "adaptive")
      for s, br in S["cells"][f"{ds}_2.0"][m]["best_round"].items()]
put("roundOneCount", str(sum(1 for *_, br in r1 if br == 1)))
put("roundOneTotal", str(len(r1)))
put("roundOneMaxBest", str(max((br for *_, br in r1), default=0)))
fin = S["cells"]["mnist_2.0"]["dp"]
put("lastRoundMnDpHigh", mean_or(list(fin["last_round_test"].values()), lambda m: f"{100*m:.0f}"))
put("bestRoundMnDpHigh", mean_or(list(fin["test"].values()), lambda m: f"{100*m:.0f}"))

for ds, D in (("mnist", "Mn"), ("fashion_mnist", "Fm")):
    for kind, K in (("dpC5", "Cfive"), ("dpC5lr", "CfiveLr")):
        v = [d["test"] for d in S["controls"].get(f"{ds}_{kind}", {}).values()]
        put(f"acc{K}{D}", mean_or(v, lambda m: f"{m:.3f}"))
    cm = S["controls"].get(f"{ds}_dpC5m", {}); c5 = S["controls"].get(f"{ds}_dpC5", {})
    ad = S["cells"][f"{ds}_2.0"]["adaptive"]["test"]
    put(f"accCfiveMatch{D}", mean_or([d["test"] for d in cm.values()], lambda m: f"{m:.3f}"))
    put(f"nSameAdCfiveMatch{D}", str(sum(1 for sd, d in cm.items() if sd in ad and abs(d["test"] - ad[sd]) < 1e-12)))
    put(f"nCfiveMatch{D}", str(len(cm)))
    dd = [100 * (ad[sd] - d["test"]) for sd, d in c5.items() if sd in ad]
    put(f"diffAdCfive{D}", mean_or(dd, lambda m: f"{m:.1f}"))
    put(f"diffAdCfiveMax{D}", mean_or([max(map(abs, dd))] if dd else [], lambda m: f"{m:.1f}"))
    br = [d["best_round"] for d in S["controls"].get(f"{ds}_dpC5lr", {}).values()]
    put(f"cfiveLrBestMin{D}", str(min(br)) if br else "--")
    put(f"cfiveLrBestMax{D}", str(max(br)) if br else "--")
    for v in ("v6", "v7"):
        f2 = S["flower"].get(f"{v}_{ds}_2.0", {})
        put(f"{'flowerSix' if v == 'v6' else 'flowerSeven'}{D}", fmt(f2.get("test")))
        put(f"{'flowerSix' if v == 'v6' else 'flowerSeven'}Best{D}", str(f2.get("best_round", "--")))
        put(f"{'flowerSix' if v == 'v6' else 'flowerSeven'}C{D}", fmt(f2.get("C_best"), 2))
    dp42 = s42(ds, 2.0, "dp")
    for v, V in (("v6", "Six"), ("v7", "Seven")):
        t = S["flower"].get(f"{v}_{ds}_2.0", {}).get("test")
        # gap from the 3-decimal values shown in Table 4, so a reader gets the same number
        put(f"flower{V}Gap{D}", "--" if (t is None or dp42 is None) else f"{100*(round(t, 3)-round(dp42, 3)):+.1f}")
    for mode, M in (("dp", "Dp"), ("adaptive", "Ad")):
        put(f"noln{M}{D}", fmt(S["noln"].get(f"{ds}_{mode}", {}).get("test")))

for key, M in (("layernorm_warm", "warm"), ("layernorm_trained", "trained"),
               ("nolayernorm_trained", "noln")):
    d = S.get("norms", {}).get(key)
    put(f"{M}Mean", "--" if d is None else f"{d['mean']:.3f}" if M == "warm" else ("--" if d is None else f"{d['mean']:.1f}"))
    put(f"{M}Sd", "--" if d is None else f"{d['sd']:.3f}")
    put(f"{M}Cv", "--" if d is None else f"{d['cv']:.1f}")
    put(f"{M}N", "--" if d is None else f"{d['n']:,}".replace(",", "{,}"))
    put(f"{M}Max", "--" if d is None else f"{d['max']:.1e}")

(OUT / "numbers.tex").write_text(
    "% generated by scripts/summarise_camera_ready.py -- do not edit by hand\n"
    + "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())) + "\n")
S_out = {k: v for k, v in S.items()}
for c in S_out["cells"].values():
    for m in ("dp", "adaptive"):
        c[m].pop("C", None)
(OUT / "summary.json").write_text(json.dumps({"summary": S_out, "macros": macros}, indent=1, default=str))
done = len(list(RES.glob("*.csv")))
print(f"{done} runs summarised; {sum(1 for v in macros.values() if '--' in v)} macros still missing")
