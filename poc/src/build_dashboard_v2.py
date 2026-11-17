"""Assembles the full dashboard_data_v2.json: Checkpoint 2 results +
overlap-corrected significance findings + the manager/senior agent
comparison, all in one structured file for the updated dashboard.
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


def block_sample(records, stride):
    by_ticker = {}
    for r in records:
        by_ticker.setdefault(r["ticker"], []).append(r)
    out = []
    for recs in by_ticker.values():
        out.extend(sorted(recs, key=lambda r: r["date"])[::stride])
    return out


def correlation_block(records, value_key):
    """{stride_label: {overall: {r,p,n}, per_ticker: {...}}} for naive + two strides."""
    out = {}
    for stride, key in [(1, "naive"), (20, "stride20"), (40, "stride40")]:
        subset = records if stride == 1 else block_sample(records, stride)
        tickers = sorted(set(r["ticker"] for r in subset))
        vals = [r[value_key] for r in subset]
        tgt = [r["target"]["mean"] for r in subset]
        r, p = safe_pearsonr(vals, tgt)
        entry = {"overall": {"r": r, "p": p, "n": len(subset)}, "per_ticker": {}}
        for ticker in tickers:
            idx = [i for i, rec in enumerate(subset) if rec["ticker"] == ticker]
            r, p = safe_pearsonr([vals[i] for i in idx], [tgt[i] for i in idx])
            entry["per_ticker"][ticker] = {"r": r, "p": p, "n": len(idx)}
        out[key] = entry
    return out


def main():
    results = json.loads((OUT_DIR / "results.json").read_text())
    manager_records = json.loads((OUT_DIR / "examples_with_manager.json").read_text())

    aggregate_sig = correlation_block(manager_records, "aggregate_stance")
    manager_sig = correlation_block(manager_records, "manager_stance")

    manager_arr = np.array([r["manager_stance"] for r in manager_records])
    flat_arr = np.array([r["aggregate_stance"] for r in manager_records])
    manager_vs_flat_r, _ = safe_pearsonr(manager_arr, flat_arr)
    mean_abs_diff = float(np.mean(np.abs(manager_arr - flat_arr)))

    # Pick a few sample records showing the manager diverging meaningfully from the flat average
    by_divergence = sorted(manager_records, key=lambda r: -abs(r["manager_stance"] - r["aggregate_stance"]))
    manager_samples = by_divergence[:4]

    v2 = {
        "n_examples": results["n_examples"],
        "tickers": results["tickers"],
        "learned_sensitivity": results["learned_sensitivity"],
        "scatter": results["scatter"],
        "losses": results["losses"][::8] + [results["losses"][-1]],  # downsample
        "sample_reasoning": results["sample_reasoning"],
        "significance": {
            "aggregate": aggregate_sig,
            "manager": manager_sig,
        },
        "manager_vs_flat": {
            "correlation": manager_vs_flat_r,
            "mean_abs_diff": mean_abs_diff,
        },
        "manager_samples": manager_samples,
    }

    out_path = OUT_DIR / "dashboard_data_v2.json"
    out_path.write_text(json.dumps(v2))
    print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
