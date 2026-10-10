"""The two substeps of the POC's one-step AgentTorch model.

DecideStance (Policy): runs the real AgentTorch Archetype/Behavior machinery
once (cached to disk) to get each persona's stance on the real shock-day
price context, and averages them into one aggregate_stance.

ComputePrediction (Transition): the actual differentiable piece — multiplies
aggregate_stance by a learnable `sensitivity` scalar to produce
predicted_return. aggregate_stance itself has no gradient (it came from an
LLM call, which we are not differentiating through — only the deterministic
scaling step after it), which is exactly the calibration this POC is meant
to demonstrate: not "differentiate through the LLM," but "calibrate a
simulation parameter, via real backprop, against the LLM's otherwise-fixed
output."

The LLM call happens ONCE in __init__ (not per training step) — with 500+
gradient-descent iterations expected, re-calling the LLM every step would be
slow and pointless, since aggregate_stance doesn't depend on `sensitivity`.
Results are cached to disk by the real-data context so repeated runs of the
whole script don't re-call the LLM either.
"""

import hashlib
import json
import pathlib

import torch

from agent_torch.core.llm.archetype import Archetype
from agent_torch.core.llm.behavior import Behavior
from agent_torch.core.registry import Registry
from agent_torch.core.substep import SubstepAction, SubstepTransition

import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from config import trader_population
from config.archetypes import PROMPT_TEMPLATE
from config.event_data import SHOCK_ANCHOR_DATE, TICKER
from src.llm_backend import OllamaLLM
from src import shocks

CACHE_PATH = pathlib.Path(__file__).resolve().parents[1] / "cache" / "archetype_responses.json"

# The one learnable parameter in this POC. A plain global nn.Parameter rather
# than threading it through AgentTorch's config-driven calibrate_* mechanism
# (whose exact parameter-collection plumbing wasn't worth fully
# reverse-engineering for a 1-parameter model) — forward() below still runs
# inside AgentTorch's real Runner.step() loop, so this is genuinely
# differentiated through that forward pass via ordinary torch.autograd.
SENSITIVITY = torch.nn.Parameter(torch.tensor(0.1))


def _context_cache_key(kwargs: dict) -> str:
    raw = json.dumps(kwargs, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2))


def _compute_archetype_stances(kwargs: dict) -> list[float]:
    """Real AgentTorch Archetype/Behavior call, cached by context hash."""
    cache = _load_cache()
    key = _context_cache_key(kwargs)
    if key in cache:
        print(f"[cache hit] archetype stances for {kwargs.get('date')}: {cache[key]['stances']}")
        return cache[key]["stances"]

    llm = OllamaLLM()
    archetype = Archetype(n_arch=1).llm(llm=llm, user_prompt=PROMPT_TEMPLATE)
    behavior = Behavior(archetype=archetype, region=trader_population)

    memory_dir = pathlib.Path(__file__).resolve().parents[1] / "cache" / "memory"
    sample_kwargs = {**kwargs, "device": "cpu", "current_memory_dir": str(memory_dir)}
    output = behavior.sample(sample_kwargs)  # shape [3, 1], one row per persona

    stances = output.squeeze(-1).tolist()
    cache[key] = {"kwargs": kwargs, "stances": stances}
    _save_cache(cache)
    print(f"[cache miss -> computed] archetype stances for {kwargs.get('date')}: {stances}")
    return stances


@Registry.register_substep("decide_stance", "policy")
class DecideStance(SubstepAction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        prices = shocks.load_prices()
        context = shocks.recent_price_context(prices, SHOCK_ANCHOR_DATE)
        prompt_kwargs = {
            "ticker": TICKER,
            "price": f"{context['price']:.2f}",
            "date": context["date"],
            "pct_change": f"{context['pct_change']:.1f}",
            "window": str(context["window"]),
        }

        stances = _compute_archetype_stances(prompt_kwargs)
        self.aggregate_stance = torch.tensor(sum(stances) / len(stances))
        print(f"Archetype stances {stances} -> aggregate_stance={self.aggregate_stance.item():.3f}")

    def forward(self, state, observation):
        return {self.output_variables[0]: self.aggregate_stance}


@Registry.register_substep("compute_prediction", "transition")
class ComputePrediction(SubstepTransition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, state, action):
        aggregate_stance = action["market"]["aggregate_stance"]
        predicted_return = (SENSITIVITY * aggregate_stance).reshape(1, 1)
        return {self.output_variables[0]: predicted_return}
