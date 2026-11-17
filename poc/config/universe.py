"""The universe of (ticker, date) examples this checkpoint evaluates.

No hand-picked "events" — every trading day in EVAL_START..EVAL_END, for
every ticker, becomes one example, as long as it has enough lookback history
before it and enough forward-return runway after it. Tickers are still a
human choice (chosen for sector/volatility diversity, not for a known
favorable outcome) -- the fix versus Checkpoint 1 is that no *date* within a
ticker's history is hand-picked.
"""

TICKERS = ["AAPL", "NVDA", "JPM"]  # mega-cap tech, high-momentum AI-rally name, steadier financial

# Ordinary 2024 window -- not chosen around any known shock.
EVAL_START = "2024-01-02"
EVAL_END = "2024-06-28"

# Pull with buffer: room for LOOKBACK_WINDOW_TRADING_DAYS before the first
# eval day and max(FORWARD_HORIZONS_TRADING_DAYS) after the last.
PULL_START = "2023-11-15"
PULL_END = "2024-09-15"

LOOKBACK_WINDOW_TRADING_DAYS = 15  # trading days of returns shown in the prompt
FORWARD_HORIZONS_TRADING_DAYS = {
    "4wk": 20,
    "8wk": 40,
}
