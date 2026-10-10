"""Produces the POC's actual deliverable: proof the mechanism runs end to
end on real data, not a claim of predictive validity (n=1 event).
"""

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.train import train
from src import shocks, substeps
from config.archetypes import ARCHETYPES
from config.event_data import SHOCK_ANCHOR_DATE

OUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "outputs"


def main():
    losses, runner = train()
    predicted = runner.state["agents"]["market"]["predicted_return"].item()

    prices = shocks.load_prices()
    realized = shocks.realized_forward_returns(prices)

    cache = json.loads(substeps.CACHE_PATH.read_text())
    stances = next(iter(cache.values()))["stances"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Loss curve
    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.xlabel("training step")
    plt.ylabel("MSE loss")
    plt.title("Calibration of `sensitivity` via AgentTorch's differentiable Runner")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "loss_curve.png", dpi=120)
    plt.close()

    # Predicted vs actual
    rows = []
    for horizon, actual in realized.items():
        rows.append({"horizon": horizon, "predicted": predicted, "actual": actual})

    plt.figure(figsize=(5, 5))
    xs = [r["predicted"] for r in rows]
    ys = [r["actual"] for r in rows]
    lim = max(abs(v) for v in xs + ys) * 1.3 or 1.0
    plt.scatter(xs, ys)
    for r in rows:
        plt.annotate(r["horizon"], (r["predicted"], r["actual"]))
    plt.plot([-lim, lim], [-lim, lim], linestyle="--", color="gray")
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.xlabel("predicted return (sensitivity x aggregate archetype stance)")
    plt.ylabel(f"actual realized return ({SHOCK_ANCHOR_DATE} shock)")
    plt.title("Predicted vs. actual — n=1 event, not a signal")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "predicted_vs_actual.png", dpi=120)
    plt.close()

    with open(OUT_DIR / "predicted_vs_actual.csv", "w") as f:
        f.write("horizon,predicted,actual\n")
        for r in rows:
            f.write(f"{r['horizon']},{r['predicted']:.4f},{r['actual']:.4f}\n")

    print("\n=== Final result ===")
    print(f"Learned sensitivity: {substeps.SENSITIVITY.item():.4f}")
    print(f"Aggregate archetype stance (fixed input): mean of {stances}")
    print("\nPer-archetype stance on the real Jan 2-3 2019 AAPL shock day:")
    for a, s in zip(ARCHETYPES, stances):
        print(f"  {a['name']:>10}: {s:+.2f}")
    print(f"\n{'horizon':<8}{'predicted':>12}{'actual':>12}")
    for r in rows:
        print(f"{r['horizon']:<8}{r['predicted']:>12.4f}{r['actual']:>12.4f}")
    print(
        "\nNote: momentum and contrarian point in opposite directions here because the "
        "real outcome was a sharp rebound after the guidance-cut shock -- contrarian's "
        "call was directionally right, momentum's wasn't. One event, three archetypes: "
        "illustrative of the mechanism, not a validated signal."
    )
    print(f"\nWrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
