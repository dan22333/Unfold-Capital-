"""Builds the 1-agent population for the manager archetype -- same real
AgentTorch LoadPopulation/PromptManager mechanism as config/build_population.py,
just a population of size 1 instead of 4.
"""

import json
import pathlib

import pandas as pd

import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from config.manager_archetype import MANAGER_PERSONA

POP_DIR = pathlib.Path(__file__).resolve().parent / "manager_population"


def build() -> None:
    POP_DIR.mkdir(exist_ok=True)
    (POP_DIR / "__init__.py").touch()

    persona_series = pd.Series([0], name="persona")
    persona_series.to_pickle(POP_DIR / "persona.pickle")

    mapping = {"persona": [MANAGER_PERSONA]}
    with open(POP_DIR / "mapping.json", "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"Built manager population at {POP_DIR} with 1 agent.")


if __name__ == "__main__":
    build()
