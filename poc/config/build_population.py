"""Builds the tiny custom 'population' AgentTorch's real Archetype/Behavior/
LoadPopulation machinery expects: one agent row per trader archetype.

AgentTorch's LoadPopulation reads every *.pickle file in a population folder
as a per-agent tensor column, and PromptManager takes the Cartesian product
of mapping.json's value lists for whichever {variables} appear in the prompt
template to decide how many distinct LLM calls to make (one per combination,
broadcast to every agent sharing it) — this is the actual "archetype" trick,
verified by reading agent_torch/core/llm/{archetype,behavior,prompt_manager}.py
and agent_torch/core/dataloader.py directly (the shipped macro_economics
example turned out to import a stale pre-refactor API and isn't a valid
reference for the currently installed agent_torch==0.6.0).

Here: 3 agents, one per archetype, encoded as a single categorical variable
`style` (values 0, 1, 2) whose mapping.json entries are the full persona
description strings substituted into the prompt template's {style}
placeholder. Population size 3 is intentionally tiny — the mechanism is
identical at any scale.
"""

import json
import pathlib

import pandas as pd

import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from config.archetypes import ARCHETYPES

POP_DIR = pathlib.Path(__file__).resolve().parent / "trader_population"


def build() -> None:
    POP_DIR.mkdir(exist_ok=True)
    (POP_DIR / "__init__.py").touch()

    persona_series = pd.Series(range(len(ARCHETYPES)), name="persona")
    persona_series.to_pickle(POP_DIR / "persona.pickle")

    mapping = {"persona": [a["description"] for a in ARCHETYPES]}
    with open(POP_DIR / "mapping.json", "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"Built population at {POP_DIR} with {len(persona_series)} agents.")
    print("mapping.json:", mapping)


if __name__ == "__main__":
    build()
