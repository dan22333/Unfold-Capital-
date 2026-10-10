"""Stage 2: the real AgentTorch Archetype/Behavior classes, our custom Ollama
backend, and our tiny 3-agent persona population — with a FAKE toy market
description (not real data yet), per the plan's Stage 2 isolation goal.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from agent_torch.core.llm.archetype import Archetype
from agent_torch.core.llm.behavior import Behavior

from config import trader_population
from config.archetypes import PROMPT_TEMPLATE
from src.llm_backend import OllamaLLM


def main() -> None:
    llm = OllamaLLM()
    archetype = Archetype(n_arch=1).llm(llm=llm, user_prompt=PROMPT_TEMPLATE)
    behavior = Behavior(archetype=archetype, region=trader_population)

    kwargs = {
        "device": "cpu",
        "current_memory_dir": str(pathlib.Path(__file__).resolve().parents[1] / "cache" / "memory"),
        "ticker": "AAPL",
        "price": "142.19",
        "date": "2019-01-03 (TOY)",
        "pct_change": "-9.5",
        "window": "2",
    }

    output = behavior.sample(kwargs)
    print("\nAveraged sampled behavior tensor (one row per agent, 3 agents = 3 personas):")
    print(output)


if __name__ == "__main__":
    main()
