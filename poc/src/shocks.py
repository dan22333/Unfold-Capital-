"""Shock magnitude and realized forward returns, computed from real AAPL prices.

Pure pandas/numpy — no AgentTorch dependency. Used both to build the archetype
prompt context (recent price action) and to compute the real target the
calibration loop trains against (realized forward return).
"""

import pathlib

import pandas as pd

import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from config.event_data import (
    SHOCK_WINDOW_PRE,
    SHOCK_WINDOW_POST,
    SHOCK_ANCHOR_DATE,
    FORWARD_HORIZONS_TRADING_DAYS,
    LOOKBACK_WINDOW_TRADING_DAYS,
)

PRICES_PATH = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw" / "aapl_prices.csv"


def load_prices() -> pd.Series:
    df = pd.read_csv(PRICES_PATH, index_col="date", parse_dates=True)
    return df["close"].sort_index()


def hub_shock_magnitude(prices: pd.Series) -> float:
    """Cumulative return over the announcement/reaction window."""
    p_pre = prices.loc[:SHOCK_WINDOW_PRE].iloc[-1]
    p_post = prices.loc[SHOCK_WINDOW_POST:].iloc[0]
    return float(p_post / p_pre - 1.0)


def realized_forward_returns(prices: pd.Series) -> dict:
    """{horizon_name: realized_return} measured from the shock anchor date."""
    anchor_idx = prices.index.get_indexer([pd.Timestamp(SHOCK_ANCHOR_DATE)], method="bfill")[0]
    p_anchor = prices.iloc[anchor_idx]
    out = {}
    for horizon_name, n_days in FORWARD_HORIZONS_TRADING_DAYS.items():
        target_idx = anchor_idx + n_days
        if target_idx >= len(prices):
            raise ValueError(
                f"Not enough data for horizon {horizon_name}: need index {target_idx}, "
                f"only have {len(prices)} rows. Extend PULL_END in config/event_data.py."
            )
        p_future = prices.iloc[target_idx]
        out[horizon_name] = float(p_future / p_anchor - 1.0)
    return out


def recent_price_context(prices: pd.Series, as_of_date: str) -> dict:
    """Recent price + pct change over the lookback window, for the archetype prompt."""
    as_of_idx = prices.index.get_indexer([pd.Timestamp(as_of_date)], method="bfill")[0]
    p_now = prices.iloc[as_of_idx]
    lookback_idx = max(0, as_of_idx - LOOKBACK_WINDOW_TRADING_DAYS)
    p_lookback = prices.iloc[lookback_idx]
    pct_change = (p_now / p_lookback - 1.0) * 100.0
    return {
        "date": str(prices.index[as_of_idx].date()),
        "price": float(p_now),
        "pct_change": float(pct_change),
        "window": LOOKBACK_WINDOW_TRADING_DAYS,
    }


if __name__ == "__main__":
    prices = load_prices()
    print("Hub shock magnitude:", hub_shock_magnitude(prices))
    print("Realized forward returns:", realized_forward_returns(prices))
    print("Recent price context at shock anchor:", recent_price_context(prices, SHOCK_ANCHOR_DATE))
