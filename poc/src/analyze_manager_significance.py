"""Same overlap-corrected significance methodology as
analyze_significance.py, applied to the manager-synthesized stance instead
of the flat archetype average.
"""

import json
import pathlib
import sys

import numpy as np
from scipy.stats import pearsonr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

OUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "outputs"


def safe_pearsonr(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None, None
    r, p = pearsonr(x, y)
    return float(r), float(p)


def fmt(v, digits=3):
    return "n/a" if v is None else f"{v:.{digits}f}"


def block_sample(records, stride):
    by_ticker = {}
    for r in records:
        by_ticker.setdefault(r["ticker"], []).append(r)
    out = []
    for recs in by_ticker.values():
        out.extend(sorted(recs, key=lambda r: r["date"])[::stride])
    return out


def report(records, label, value_key="manager_stance"):
    print(f"\n=== {label} ===")
    print(f"{'':12}{'n':>5}{'r':>9}{'p':>10}")
    tickers = sorted(set(r["ticker"] for r in records))
    vals = [r[value_key] for r in records]
    tgt = [r["target"]["mean"] for r in records]
    r, p = safe_pearsonr(vals, tgt)
    print(f"{'overall':12}{len(records):>5}{fmt(r):>9}{fmt(p, 4):>10}")
    for ticker in tickers:
        idx = [i for i, rec in enumerate(records) if rec["ticker"] == ticker]
        r, p = safe_pearsonr([vals[i] for i in idx], [tgt[i] for i in idx])
        print(f"{ticker:12}{len(idx):>5}{fmt(r):>9}{fmt(p, 4):>10}")


def main():
    records = json.loads((OUT_DIR / "examples_with_manager.json").read_text())

    report(records, "Manager stance -- naive (all 372, overlapping windows)")

    for stride, label in [
        (20, "Manager stance -- block-sampled, stride=20"),
        (40, "Manager stance -- block-sampled, stride=40 (non-overlapping)"),
    ]:
        blocked = block_sample(records, stride)
        report(blocked, label)

    print("\n=== For comparison: flat average aggregate_stance, same records ===")
    report(records, "Aggregate (flat average) -- naive", value_key="aggregate_stance")
    for stride, label in [
        (20, "Aggregate -- block-sampled, stride=20"),
        (40, "Aggregate -- block-sampled, stride=40"),
    ]:
        blocked = block_sample(records, stride)
        report(blocked, label, value_key="aggregate_stance")

    # Does the manager actually differ from a flat average, or just restate it?
    manager = np.array([r["manager_stance"] for r in records])
    flat = np.array([r["aggregate_stance"] for r in records])
    r_vs_flat, _ = safe_pearsonr(manager, flat)
    mean_abs_diff = float(np.mean(np.abs(manager - flat)))
    print(f"\n=== Manager vs. flat average: how different is it? ===")
    print(f"correlation(manager, flat_average) = {r_vs_flat:.3f}")
    print(f"mean |manager - flat_average|       = {mean_abs_diff:.3f}")


if __name__ == "__main__":
    main()
