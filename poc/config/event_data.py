"""The one real shock event used to calibrate the POC.

Apple's Jan 2, 2019 guidance cut: FQ1 revenue guidance cut from $89-93B to
~$84B, attributed to China/iPhone weakness. AAPL fell roughly -7% to -10%
across Jan 2-3, 2019 (trading halted mid-session on the news).
"""

TICKER = "AAPL"

PULL_START = "2018-10-01"
PULL_END = "2019-03-15"

SHOCK_WINDOW_PRE = "2018-12-31"   # last close before the announcement
SHOCK_WINDOW_POST = "2019-01-03"  # first full reaction day

SHOCK_ANCHOR_DATE = "2019-01-03"  # forward windows measured from here

FORWARD_HORIZONS_TRADING_DAYS = {
    "4wk": 20,
    "8wk": 40,
}

LOOKBACK_WINDOW_TRADING_DAYS = 5  # "past N days" context fed into archetype prompts
