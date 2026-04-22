"""Compare v1 (pre-optimisation) and v2 (optimised) matrix results."""
import csv, glob, os, math

def load(dirp):
    out = {}
    for fp in sorted(glob.glob(os.path.join(dirp, "*.csv"))):
        name = os.path.basename(fp).replace(".csv", "")
        if name == "all_results":
            continue
        with open(fp) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        last = rows[-1]
        best_acc = max(float(r['val_acc']) for r in rows)
        out[name] = {
            "rounds": len(rows),
            "val_acc": float(last['val_acc']),
            "best_acc": best_acc,
            "train_loss": float(last['train_loss']),
            "C_A": float(last.get('clipping_norm_A', 0)),
            "epsilon": float(last.get('epsilon', 'inf')) if 'epsilon' in last else None,
        }
    return out

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
v3 = load(os.path.join(ROOT, "results_v3"))
v4 = load(os.path.join(ROOT, "results_v4"))
v5 = load(os.path.join(ROOT, "results"))

header = (
    f"{'config':34s} | {'v3':>6} {'v4':>6} {'v5':>6} | "
    f"{'v5-v4':>7} | {'v5_best':>7} {'v5_CA':>5} {'eps':>5}"
)
print(header)
print("-" * len(header))
for k in sorted(set(v3) | set(v4) | set(v5)):
    c = v3.get(k, {}).get('val_acc', float('nan'))
    d = v4.get(k, {}).get('val_acc', float('nan'))
    e = v5.get(k, {}).get('val_acc', float('nan'))
    best = v5.get(k, {}).get('best_acc', float('nan'))
    ca = v5.get(k, {}).get('C_A', 0)
    eps = v5.get(k, {}).get('epsilon')
    d54 = f"{(e-d)*100:+6.2f}pp" if not (math.isnan(d) or math.isnan(e)) else "   n/a"
    is_inf = eps is None or (eps != eps) or eps == float('inf')
    epsstr = "-" if is_inf else f"{eps:.2f}"
    print(f"{k:34s} | {c:>6.3f} {d:>6.3f} {e:>6.3f} | {d54:>7} | {best:>7.4f} {ca:>5.2f} {epsstr:>5}")

print(f"\nv3:{len(v3)}  v4:{len(v4)}  v5:{len(v5)} configs")
