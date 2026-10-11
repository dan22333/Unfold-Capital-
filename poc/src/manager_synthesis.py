"""Second-stage synthesis: one manager LLM call per example, reading the 4
already-computed archetype stances + reasoning (from outputs/examples.json)
and forming its own view with real discretion -- not a fixed-weight average.

Reuses the real AgentTorch Archetype/Behavior machinery (population size 1
this time), same as the trader archetypes. Only ~372 new LLM calls (one per
example, not per archetype), since this consumes archetype output rather
than recomputing it.
"""

import hashlib
import json
import pathlib
import sys

import torch

from agent_torch.core.llm.archetype import Archetype
from agent_torch.core.llm.behavior import Behavior

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from config import manager_population
from config.manager_archetype import PROMPT_TEMPLATE, format_archetype_summary
from src.llm_backend import OllamaLLM

EXAMPLES_PATH = pathlib.Path(__file__).resolve().parents[1] / "outputs" / "examples.json"
OUT_PATH = pathlib.Path(__file__).resolve().parents[1] / "outputs" / "examples_with_manager.json"
CACHE_PATH = pathlib.Path(__file__).resolve().parents[1] / "cache" / "manager_responses.json"
MEMORY_DIR = pathlib.Path(__file__).resolve().parents[1] / "cache" / "manager_memory"


def _cache_key(kwargs: dict) -> str:
    return hashlib.sha256(json.dumps(kwargs, sort_keys=True).encode()).hexdigest()[:16]


def _load_cache() -> dict:
    return json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2))


def main():
    records = json.loads(EXAMPLES_PATH.read_text())
    print(f"Synthesizing manager view for {len(records)} examples")

    llm = OllamaLLM()
    archetype = Archetype(n_arch=1).llm(llm=llm, user_prompt=PROMPT_TEMPLATE)
    behavior = Behavior(archetype=archetype, region=manager_population)

    cache = _load_cache()
    out_records = []

    for i, rec in enumerate(records):
        prompt_kwargs = {
            "ticker": rec["ticker"],
            "date": rec["date"],
            "archetype_summary": format_archetype_summary(rec["archetypes"]),
        }
        key = _cache_key(prompt_kwargs)

        if key in cache:
            manager_stance, manager_reasoning = cache[key]["stance"], cache[key]["reasoning"]
        else:
            sample_kwargs = {**prompt_kwargs, "device": "cpu", "current_memory_dir": str(MEMORY_DIR)}
            output = behavior.sample(sample_kwargs)  # shape [1, 1]
            manager_stance = output.squeeze().item()
            manager_reasoning = llm.last_raw_responses[0]
            cache[key] = {"stance": manager_stance, "reasoning": manager_reasoning}
            _save_cache(cache)

        out_records.append({**rec, "manager_stance": manager_stance, "manager_reasoning": manager_reasoning})

        if (i + 1) % 25 == 0 or i == len(records) - 1:
            print(f"  [{i + 1}/{len(records)}] {rec['ticker']} {rec['date']}  manager_stance={manager_stance:+.3f}")

    OUT_PATH.write_text(json.dumps(out_records, indent=2))
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
