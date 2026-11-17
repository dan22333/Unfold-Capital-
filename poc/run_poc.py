"""Single entry point: pull data if missing -> run the full AgentTorch
archetype calibration pipeline -> write outputs.
"""

import pathlib

from src.pull_data import pull_prices, OUT_PATH as PRICES_PATH
from src.evaluate import main as evaluate_main

if __name__ == "__main__":
    if not PRICES_PATH.exists():
        print("Pulling AAPL price data...")
        pull_prices()
    else:
        print(f"Using existing price data at {PRICES_PATH}")

    evaluate_main()
