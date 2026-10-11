"""Statistical honesty check on Checkpoint 2's correlations, plus per-ticker
regime context.

The naive correlations in outputs/results.json treat all 372 examples as
independent draws, but they aren't: each example's forward-return target
window (20 or 40 trading days) overlaps almost entirely with its neighbors'
-- day t and day t+1 share 19/20 or 39/40 of their target window. That
inflates apparent statistical significance. This recomputes correlations on
non-overlapping blocks (stride >= horizon) to see how much signal survives
once overlap is accounted for, and adds per-ticker trend/volatility context
to check whether the correlation sign-flip by ticker looks regime-driven.
"""

import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from config.archetypes import ARCHETYPES
from config.universe import FORWARD_HORIZONS_TRADING_DAYS, EVAL_START, EVAL_END

OUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "outputs"
PRICES_PATH = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw" / "prices.csv"


def safe_pearsonr(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None, None
    r, p = pearsonr(x, y)
    return float(r), float(p)


def block_sample(records, stride):
    """Every `stride`-th example per ticker, in date order."""
    by_ticker = {}
    for r in records:
        by_ticker.setdefault(r["ticker"], []).append(r)
    out = []
    for recs in by_ticker.values():
        out.extend(sorted(recs, key=lambda r: r["date"])[::stride])
    return out


def report_correlations(records, label):
    print(f"\n=== {label} ===")
    print(f"{'':12}{'n':>5}{'r':>9}{'p':>10}")
    tickers = sorted(set(r["ticker"] for r in records))
    agg = [r["aggregate_stance"] for r in records]
    tgt = [r["target"]["mean"] for r in records]
    r, p = safe_pearsonr(agg, tgt)
    print(f"{'overall':12}{len(records):>5}{fmt(r):>9}{fmt(p,4):>10}")
    for ticker in tickers:
        idx = [i for i, rec in enumerate(records) if rec["ticker"] == ticker]
        r, p = safe_pearsonr([agg[i] for i in idx], [tgt[i] for i in idx])
        print(f"{ticker:12}{len(idx):>5}{fmt(r):>9}{fmt(p,4):>10}")
    return tickers


def fmt(v, digits=3):
    return "n/a" if v is None else f"{v:.{digits}f}"


def main():
    records = json.loads((OUT_DIR / "examples.json").read_text())
    tickers = report_correlations(records, "Naive (all 372 examples, overlapping target windows)")

    print(f"\n=== Per-archetype x per-ticker (naive) ===")
    print(f"{'archetype':12}{'ticker':8}{'n':>5}{'r':>9}{'p':>10}")
    for a_idx, a in enumerate(ARCHETYPES):
        for ticker in tickers:
            idx = [i for i, rec in enumerate(records) if rec["ticker"] == ticker]
            stances = [records[i]["archetypes"][a_idx]["stance"] for i in idx]
            targets = [records[i]["target"]["mean"] for i in idx]
            r, p = safe_pearsonr(stances, targets)
            print(f"{a['name']:12}{ticker:8}{len(idx):>5}{fmt(r):>9}{fmt(p,4):>10}")

    max_horizon = max(FORWARD_HORIZONS_TRADING_DAYS.values())
    for stride, label in [
        (20, "Block-sampled, stride=20 (4wk horizon -- still some 8wk overlap)"),
        (max_horizon, f"Block-sampled, stride={max_horizon} (8wk horizon -- fully non-overlapping targets)"),
    ]:
        blocked = block_sample(records, stride)
        report_correlations(blocked, label)

    print("\n=== Per-ticker regime context (eval window) ===")
    prices_df = pd.read_csv(PRICES_PATH, parse_dates=["date"])
    for ticker in tickers:
        s = prices_df[(prices_df["ticker"] == ticker)].sort_values("date")
        s = s[(s["date"] >= EVAL_START) & (s["date"] <= EVAL_END)]
        total_return = s["close"].iloc[-1] / s["close"].iloc[0] - 1
        daily_ret = s["close"].pct_change().dropna()
        ann_vol = daily_ret.std() * (252 ** 0.5)
        trend_r = float(np.corrcoef(np.arange(len(s)), s["close"].values)[0, 1])
        print(f"{ticker:6} total_return={total_return:+.1%}  annualized_vol={ann_vol:.1%}  "
              f"trend_r(price vs time)={trend_r:+.3f}")


if __name__ == "__main__":
    main()
