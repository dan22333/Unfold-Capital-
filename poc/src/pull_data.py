"""Stage 0.5 / Stage 3 data pull: AAPL daily adjusted close over a small window.

Deliberately tiny — one ticker, ~5 months of daily data (~100 rows). yfinance
is used here as a quick prototyping data source only, not a rigorous
point-in-time backtest feed (see sim-alpha.html's own caveat on this).
"""

import pathlib

import yfinance as yf

import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from config.event_data import TICKER, PULL_START, PULL_END

OUT_PATH = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw" / "aapl_prices.csv"


def pull_prices() -> None:
    df = yf.download(
        TICKER,
        start=PULL_START,
        end=PULL_END,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if df.empty:
        raise RuntimeError(f"No data returned for {TICKER} in [{PULL_START}, {PULL_END}]")

    close = df["Close"].rename(columns={TICKER: "close"}) if hasattr(df["Close"], "rename") else df["Close"]
    out = df[["Close"]].copy()
    out.columns = ["close"]
    out.index.name = "date"

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH)
    print(f"Wrote {len(out)} rows to {OUT_PATH}")
    print(out.head())
    print("...")
    print(out.tail())


if __name__ == "__main__":
    pull_prices()
