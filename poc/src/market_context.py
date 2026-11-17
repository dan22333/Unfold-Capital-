"""Per-(ticker, date) prompt context and training targets.

Pure pandas/numpy -- no AgentTorch dependency. generate_examples() is the
bias fix: it enumerates every eligible trading day for every ticker, rather
than a hand-picked date, so which days "count" is a mechanical rule (enough
lookback history, enough forward-return runway) instead of a human choice.
"""

import pathlib

import pandas as pd

import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from config.universe import (
    TICKERS,
    EVAL_START,
    EVAL_END,
    LOOKBACK_WINDOW_TRADING_DAYS,
    FORWARD_HORIZONS_TRADING_DAYS,
)

PRICES_PATH = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw" / "prices.csv"

MAX_HORIZON_DAYS = max(FORWARD_HORIZONS_TRADING_DAYS.values())


def load_prices() -> dict:
    """{ticker: pd.Series of close, indexed by date, sorted}."""
    df = pd.read_csv(PRICES_PATH, parse_dates=["date"])
    out = {}
    for ticker in TICKERS:
        s = df[df["ticker"] == ticker].sort_values("date").set_index("date")["close"]
        out[ticker] = s
    return out


def generate_examples(prices_by_ticker: dict) -> list:
    """Every (ticker, date) in [EVAL_START, EVAL_END] with enough lookback
    history before it and enough forward-return runway after it. No
    hand-picked dates -- this is the whole fix versus Checkpoint 1.
    """
    eval_start, eval_end = pd.Timestamp(EVAL_START), pd.Timestamp(EVAL_END)
    examples = []
    for ticker, prices in prices_by_ticker.items():
        for idx, date in enumerate(prices.index):
            if date < eval_start or date > eval_end:
                continue
            if idx < LOOKBACK_WINDOW_TRADING_DAYS:
                continue
            if idx + MAX_HORIZON_DAYS >= len(prices):
                continue
            examples.append({"ticker": ticker, "date": str(date.date())})
    return examples


def recent_price_context(prices: pd.Series, date: str) -> dict:
    """Trailing daily % returns over the lookback window, for the prompt."""
    as_of_idx = prices.index.get_loc(pd.Timestamp(date))
    window_prices = prices.iloc[as_of_idx - LOOKBACK_WINDOW_TRADING_DAYS : as_of_idx + 1]
    daily_returns = window_prices.pct_change().dropna() * 100.0
    return {
        "date": date,
        "price": float(prices.iloc[as_of_idx]),
        "returns_series": [round(float(r), 1) for r in daily_returns.tolist()],
        "window": LOOKBACK_WINDOW_TRADING_DAYS,
    }


def format_returns_series(returns: list) -> str:
    return ", ".join(f"{r:+.1f}%" for r in returns)


def realized_target(prices: pd.Series, date: str) -> dict:
    """{horizon_name: realized_return} and their mean, measured from `date`."""
    as_of_idx = prices.index.get_loc(pd.Timestamp(date))
    p_now = prices.iloc[as_of_idx]
    out = {}
    for horizon_name, n_days in FORWARD_HORIZONS_TRADING_DAYS.items():
        p_future = prices.iloc[as_of_idx + n_days]
        out[horizon_name] = float(p_future / p_now - 1.0)
    out["mean"] = sum(out.values()) / len(out)
    return out


if __name__ == "__main__":
    prices_by_ticker = load_prices()
    examples = generate_examples(prices_by_ticker)
    print(f"Generated {len(examples)} examples across {len(prices_by_ticker)} tickers")
    for ex in examples[:3]:
        ctx = recent_price_context(prices_by_ticker[ex["ticker"]], ex["date"])
        target = realized_target(prices_by_ticker[ex["ticker"]], ex["date"])
        print(ex, "->", format_returns_series(ctx["returns_series"]), "target:", target["mean"])
