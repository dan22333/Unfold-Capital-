"""Trader archetype definitions for the AgentTorch POC.

Three persona types, matching AgentTorch's Archetype(n_arch=3) usage: one LLM
call per archetype per observation, not one call per agent. Each archetype
gets the same real recent-price context; only its persona description differs.
"""

ARCHETYPES = [
    {
        "name": "momentum",
        "description": (
            "a momentum trader who believes recent price moves tend to continue "
            "in the same direction"
        ),
    },
    {
        "name": "contrarian",
        "description": (
            "a contrarian trader who believes large recent price moves tend to "
            "overshoot and partially reverse"
        ),
    },
    {
        "name": "noise",
        "description": (
            "a noise trader with no particular view, who reacts to price moves "
            "somewhat randomly and inconsistently"
        ),
    },
]

PROMPT_TEMPLATE = (
    # AgentTorch's PromptManager extracts {variable} names with a plain regex
    # that doesn't understand Python format specs (e.g. "{price:.2f}" gets
    # parsed as a literal key "price:.2f", not "price") — so every value here
    # must be pre-formatted to a display-ready string before being passed in,
    # and placeholders must be bare names with no ":" spec.
    "{ticker} closed at ${price} on {date}, a {pct_change}% move over "
    "the past {window} trading days. You are {persona}. "
    "What is your stance on {ticker} right now? "
    "Answer with a single number from -1 (strongly bearish) to 1 (strongly "
    "bullish), and nothing else — no words, no explanation."
)
