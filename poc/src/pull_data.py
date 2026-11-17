"""Pulls daily adjusted close for all TICKERS over the pull window, written
long-format (date, ticker, close) to data/raw/prices.csv.
"""

import pathlib

import pandas as pd
import yfinance as yf

import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from config.universe import TICKERS, PULL_START, PULL_END

OUT_PATH = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw" / "prices.csv"


def pull_prices() -> None:
    df = yf.download(
        TICKERS,
        start=PULL_START,
        end=PULL_END,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )
    if df.empty:
        raise RuntimeError(f"No data returned for {TICKERS} in [{PULL_START}, {PULL_END}]")

    rows = []
    for ticker in TICKERS:
        closes = df[ticker]["Close"].dropna()
        for date, close in closes.items():
            rows.append({"date": date, "ticker": ticker, "close": float(close)})

    out = pd.DataFrame(rows).sort_values(["ticker", "date"])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(out)} rows ({len(TICKERS)} tickers) to {OUT_PATH}")
    print(out.groupby("ticker").size())


if __name__ == "__main__":
    pull_prices()
