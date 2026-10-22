"""Independent re-check of the camera-ready numbers.

Recomputes, from the raw per-round CSVs and with code that shares nothing
with summarise_camera_ready.py, the quantities printed in the paper, then
checks (a) each generated table/macro against the recomputation and (b) that
every table cell appears in the compiled PDF's text.

Usage: python scripts/verify_camera_ready.py <paper_dir>
"""
from __future__ import annotations

import csv
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parent.parent / "results_camera_ready"
PAPER = Path(sys.argv[1])
problems: list[str] = []


def rows(tag):
    with open(RES / f"{tag}.csv") as f:
        return list(csv.DictReader(f))


def reported(tag):
    r = rows(tag)
    last = r[-1]
    assert last["noise_lock"] == "-1"
    train = r[:-1]
    assert len(train) == 40, (tag, len(train))
    best = int(last["best_round_so_far"])
    # the restored model must be the best validation round, re-evaluated identically
    vals = [float(x["val_acc"]) for x in train]
    if abs(max(vals) - float(last["val_acc"])) > 1e-12:
        problems.append(f"{tag}: restored val {last['val_acc']} != max val {max(vals)}")
    if vals.index(max(vals)) + 1 != best:
        problems.append(f"{tag}: best round {best} is not the first argmax of val")
    # the restored model's test accuracy must equal that round's logged test accuracy
    if abs(float(train[best - 1]["test_acc"]) - float(last["test_acc"])) > 1e-12:
        problems.append(f"{tag}: restored test {last['test_acc']} != round-{best} test {train[best-1]['test_acc']}")
    epochs = 1 + sum(int(float(x["local_epochs"])) for x in train[:-1])
    return float(last["test_acc"]), best, epochs


def eps_rdp(z, k, delta=1e-5):
    """Upper bound via RDP, as a sanity check: must be >= the exact value."""
    a = np.linspace(1.01, 2000, 400000)
    return float(np.min(k * a / (2 * z * z) + math.log(1 / delta) / (a - 1)))


def eps_exact_bisect(z, k, delta=1e-5):
    """Analytic Gaussian mechanism by plain bisection in 60-digit arithmetic
    (independent of scipy; float erf loses all precision in the far tail)."""
    import mpmath as mp
    mp.mp.dps = 60
    s = mp.mpf(z) / mp.sqrt(k)
    d = lambda e: mp.ncdf(1 / (2 * s) - e * s) - mp.e ** e * mp.ncdf(-1 / (2 * s) - e * s)
    lo, hi = mp.mpf(0), mp.mpf(1)
    while d(hi) > delta:
        hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if d(mid) > delta else (lo, mid)
    return float(hi)


summary = json.loads((PAPER / "summary.json").read_text())
S = summary["summary"]

# ---- main cells ----------------------------------------------------------------
for ds in ("mnist", "fashion_mnist"):
    for z in (0.3, 1.0, 2.0):
        for mode in ("dp", "adaptive"):
            for seed in (42, 43, 44):
                tag = f"s{seed}_{ds}_{mode}_z{z}"
                if not (RES / f"{tag}.csv").exists():
                    problems.append(f"missing run {tag}")
                    continue
                test, best, ep = reported(tag)
                rec = S["cells"][f"{ds}_{z}"][mode]
                if abs(rec["test"][str(seed)] - test) > 1e-12:
                    problems.append(f"{tag}: summary test differs")
                if rec["epochs"][str(seed)] != ep:
                    problems.append(f"{tag}: epochs {rec['epochs'][str(seed)]} vs {ep}")
                if mode == "dp" and ep != 40:
                    problems.append(f"{tag}: fixed DP ran {ep} epochs, expected 40")
                e1 = eps_exact_bisect(z, ep)
                if abs(rec["eps"][str(seed)] - e1) > 1e-3 * max(1, e1):
                    problems.append(f"{tag}: eps {rec['eps'][str(seed)]:.4f} vs bisection {e1:.4f}")
                if eps_rdp(z, ep) < e1 - 1e-6:
                    problems.append(f"{tag}: RDP bound below the exact value (impossible)")

# ---- restore consistency for every other run (controls, ablation, Flower, baseline)
for f in sorted(RES.glob("*.csv")):
    try:
        reported(f.stem)
    except Exception as e:  # noqa: BLE001
        problems.append(f"{f.stem}: could not check ({e!r})")

# ---- round-1 claim and control macros, recomputed from the CSVs
macros = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", (PAPER / "numbers.tex").read_text()))
n1 = sum(reported(f"s{sd}_{ds}_{m}_z2.0")[1] == 1
         for sd in (42, 43, 44) for ds in ("mnist", "fashion_mnist") for m in ("dp", "adaptive"))
if str(n1) != macros["roundOneCount"]:
    problems.append(f"roundOneCount {macros['roundOneCount']} vs recomputed {n1}")
for ctl, key in (("dpC5", "accCfive"), ("dpC5lr", "accCfiveLr")):
    for ds, sfx in (("mnist", "Mn"), ("fashion_mnist", "Fm")):
        m = np.mean([reported(f"s{sd}_{ds}_{ctl}_z2.0")[0] for sd in (42, 43, 44)])
        if f"{m:.3f}" != macros[key + sfx]:
            problems.append(f"{key+sfx} {macros[key+sfx]} vs recomputed {m:.3f}")

# ---- matched-order control: identical selected model to the adaptive run
for ds, sfx in (("mnist", "Mn"), ("fashion_mnist", "Fm")):
    same = sum(abs(reported(f"s{sd}_{ds}_dpC5m_z2.0")[0] - reported(f"s{sd}_{ds}_adaptive_z2.0")[0]) < 1e-12
               for sd in (42, 43, 44))
    if str(same) != macros["nSameAdCfiveMatch" + sfx]:
        problems.append(f"nSameAdCfiveMatch{sfx} {macros['nSameAdCfiveMatch'+sfx]} vs {same}")
    d = np.mean([100 * (reported(f"s{sd}_{ds}_adaptive_z2.0")[0] - reported(f"s{sd}_{ds}_dpC5_z2.0")[0]) for sd in (42, 43, 44)])
    if f"{d:.1f}" != macros["diffAdCfive" + sfx]:
        problems.append(f"diffAdCfive{sfx} {macros['diffAdCfive'+sfx]} vs {d:.1f}")

# ---- independent rebuild of every number in the controls and Flower tables
def ms(v):
    return f"{np.mean(v):.3f} $\\pm$ {np.std(v, ddof=1):.3f}"
ctl_expect = []
for tags in (["s{sd}_%s_dp_z2.0"], ["s{sd}_%s_adaptive_z2.0"], ["s{sd}_%s_dpC5_z2.0"],
             ["s{sd}_%s_dpC5m_z2.0"], ["s{sd}_%s_dpC5lr_z2.0"]):
    row = []
    for ds in ("mnist", "fashion_mnist"):
        rr = [reported(tags[0].format(sd=sd) % ds) for sd in (42, 43, 44)]
        row += [ms([r[0] for r in rr]), f"{np.mean([eps_exact_bisect(2.0, r[2]) for r in rr]):.1f}"]
    ctl_expect.append(row)
for mode in ("dp", "adaptive"):
    row = []
    for ds in ("mnist", "fashion_mnist"):
        r = reported(f"noln_{ds}_{mode}_z2.0")
        row += [f"{r[0]:.3f}", f"{eps_exact_bisect(2.0, r[2]):.1f}"]
    ctl_expect.append(row)
body = [l for l in (PAPER / "tab_controls.tex").read_text().splitlines() if l.count("&") == 4 and "Configuration" not in l and "multicolumn" not in l]
for got, exp in zip(body, ctl_expect):
    cells = [c.strip().rstrip("\\").strip() for c in got.split("&")[1:]]
    if cells != exp:
        problems.append(f"tab_controls row '{got[:30]}': {cells} vs recomputed {exp}")
if len(body) != len(ctl_expect):
    problems.append("tab_controls: unexpected row count")

fl = [l for l in (PAPER / "tab_flower.tex").read_text().splitlines() if l.count("&") == 6 and "z{=}" not in l]
exp_rows = []
for getter in (lambda ds, z: reported(f"s42_{ds}_dp_z{z}"), lambda ds, z: reported(f"s42_{ds}_adaptive_z{z}"),
               lambda ds, z: reported(f"v6_{ds}_z{z}"), lambda ds, z: reported(f"v7_{ds}_z{z}")):
    exp_rows.append([f"{getter(ds, z)[0]:.3f}" for ds in ("mnist", "fashion_mnist") for z in (0.3, 1.0, 2.0)])
for getter in (lambda ds, z: reported(f"s42_{ds}_adaptive_z{z}"), lambda ds, z: reported(f"v6_{ds}_z{z}")):
    exp_rows.append([f"{eps_exact_bisect(z, getter(ds, z)[2]):.1f}" for ds in ("mnist", "fashion_mnist") for z in (0.3, 1.0, 2.0)])
for v in ("v7",):
    for ds in ("mnist", "fashion_mnist"):
        for z in (0.3, 1.0, 2.0):
            if reported(f"{v}_{ds}_z{z}")[2] != 40:
                problems.append(f"{v}_{ds}_z{z} did not run 40 epochs (caption says it does)")
for got, exp in zip(fl, exp_rows):
    cells = [c.strip().rstrip("\\").strip() for c in got.split("&")[1:]]
    if cells != exp:
        problems.append(f"tab_flower row '{got[:30]}': {cells} vs recomputed {exp}")
if len(fl) != len(exp_rows):
    problems.append(f"tab_flower: {len(fl)} rows vs {len(exp_rows)} expected")

# ---- text hygiene: no em-dashes in the source or PDF, 12 pages, no Type 3 fonts
src = (PAPER / "main.tex").read_text()
if "---" in src or '\u2014' in src:
    problems.append("em-dash in main.tex")
info = subprocess.run(["pdfinfo", str(PAPER / "main.pdf")], capture_output=True, text=True).stdout
pages = int(re.search(r"Pages:\s+(\d+)", info).group(1))
if pages > 12:
    problems.append(f"{pages} pages (limit 12)")
if "Type 3" in subprocess.run(["pdffonts", str(PAPER / "main.pdf")], capture_output=True, text=True).stdout:
    problems.append("Type 3 font in PDF")

# ---- every table cell must appear in the PDF text --------------------------
pdf_text = subprocess.run(["pdftotext", "-layout", str(PAPER / "main.pdf"), "-"],
                          capture_output=True, text=True).stdout
norm_pdf = re.sub(r"\s+", " ", pdf_text.replace("±", "$\\pm$"))
for tab in ("tab_main.tex", "tab_privacy.tex", "tab_flower.tex", "tab_controls.tex"):
    body = (PAPER / tab).read_text()
    for cell in re.findall(r"(-?\d+\.\d+(?: \$\\pm\$ \d+\.\d+)?)", body):
        if cell not in norm_pdf and cell.replace("-", "−") not in norm_pdf:
            problems.append(f"{tab}: '{cell}' not found in PDF text")
for leftover in ("--", "??", "CONTROLRESULT", "FLOWERRESULT", "CONTROLABSTRACT", "CONTROLCONTRIB", "TBD"):
    if leftover in pdf_text.replace("2026--", ""):
        # '--' may legitimately appear as an en dash rendered as '–'; only flag raw markers
        if leftover != "--" or re.search(r"(?<![\w–-])--(?![\w–-])", pdf_text):
            problems.append(f"placeholder '{leftover}' left in PDF")

print("\n".join(problems) if problems else "ALL CHECKS PASSED")
print(f"{len(problems)} problem(s)")
