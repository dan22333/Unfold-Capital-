"""Checkpoint 2's actual deliverable: does the raw archetype signal correlate
with what really happened, across 372 un-curated examples -- not a claim of
a tradeable strategy (no costs, no portfolio construction, no significance
testing), but a real step up from Checkpoint 1's single hand-picked event.
"""

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.train import train
from src import substeps
from config.archetypes import ARCHETYPES

OUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "outputs"
EXAMPLES_PATH = OUT_DIR / "examples.json"


def pearson(x, y) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def main():
    losses, runner, target = train()
    predicted = runner.state["agents"]["market"]["predicted_return"].detach().numpy().ravel()
    target_np = target.numpy().ravel()
    sensitivity = substeps.SENSITIVITY.item()

    records = json.loads(EXAMPLES_PATH.read_text())
    aggregate_stances = np.array([r["aggregate_stance"] for r in records])
    targets = np.array([r["target"]["mean"] for r in records])
    tickers = [r["ticker"] for r in records]

    overall_corr = pearson(aggregate_stances, targets)
    per_ticker_corr = {}
    for ticker in sorted(set(tickers)):
        idx = [i for i, t in enumerate(tickers) if t == ticker]
        per_ticker_corr[ticker] = pearson(aggregate_stances[idx], targets[idx])

    # per-archetype correlation: does any single persona's raw stance alone correlate better?
    per_archetype_corr = {}
    for a_idx, a in enumerate(ARCHETYPES):
        stances = np.array([r["archetypes"][a_idx]["stance"] for r in records])
        per_archetype_corr[a["name"]] = pearson(stances, targets)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Loss curve
    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.xlabel("training step")
    plt.ylabel("MSE loss")
    plt.title(f"Calibration over {len(records)} examples")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "loss_curve.png", dpi=120)
    plt.close()

    # Scatter: predicted vs actual, colored by ticker
    plt.figure(figsize=(6, 6))
    colors = {"AAPL": "tab:blue", "NVDA": "tab:orange", "JPM": "tab:green"}
    for ticker in sorted(set(tickers)):
        idx = [i for i, t in enumerate(tickers) if t == ticker]
        plt.scatter(predicted[idx], target_np[idx], s=14, alpha=0.6,
                    label=ticker, color=colors.get(ticker))
    lim = max(abs(predicted).max(), abs(target_np).max()) * 1.1
    plt.plot([-lim, lim], [-lim, lim], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("predicted return (sensitivity x aggregate stance)")
    plt.ylabel("actual realized return")
    plt.title(f"n={len(records)} examples, overall correlation r={overall_corr:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "predicted_vs_actual.png", dpi=120)
    plt.close()

    # Sample reasoning: pick a few examples spanning different archetypes/tickers/stance extremes
    sorted_by_stance = sorted(records, key=lambda r: r["aggregate_stance"])
    sample_records = [
        sorted_by_stance[0],                              # most bearish aggregate
        sorted_by_stance[len(sorted_by_stance) // 2],      # near-neutral
        sorted_by_stance[-1],                              # most bullish aggregate
    ]
    # one more: the single largest |reasoning vs stance disagreement| isn't tracked here,
    # so just add a couple more spread across tickers for variety.
    for ticker in ["NVDA", "JPM"]:
        for r in records:
            if r["ticker"] == ticker:
                sample_records.append(r)
                break

    results = {
        "n_examples": len(records),
        "tickers": sorted(set(tickers)),
        "learned_sensitivity": sensitivity,
        "overall_correlation": overall_corr,
        "per_ticker_correlation": per_ticker_corr,
        "per_archetype_correlation": per_archetype_corr,
        "losses": losses,
        "scatter": [
            {"ticker": r["ticker"], "date": r["date"], "aggregate_stance": r["aggregate_stance"],
             "predicted": sensitivity * r["aggregate_stance"], "actual": r["target"]["mean"]}
            for r in records
        ],
        "sample_reasoning": sample_records,
    }
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))

    print(f"\n=== Checkpoint 2 results ({len(records)} examples, {len(set(tickers))} tickers) ===")
    print(f"Learned sensitivity: {sensitivity:.4f}")
    print(f"Overall correlation (raw aggregate_stance vs realized return): r = {overall_corr:.4f}")
    print("Per-ticker correlation:", per_ticker_corr)
    print("Per-archetype correlation (does one persona alone do better?):", per_archetype_corr)
    print(f"\nWrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
