"""The two substeps of the POC's one-step AgentTorch model.

DecideStance (Policy): runs the real AgentTorch Archetype/Behavior machinery
ONCE (not per example) to build the 4-persona population, then loops over
every generated (ticker, date) example calling behavior.sample() -- cached
to disk per example -- collecting each archetype's stance AND reasoning
text. Produces an [N, 1] aggregate_stance tensor, one row per example; N is
not hand-picked, it's however many examples market_context.generate_examples()
produces.

ComputePrediction (Transition): the actual differentiable piece --
multiplies aggregate_stance by a learnable `sensitivity` scalar to produce
predicted_return, broadcasting across the whole [N, 1] population in one
vectorized AgentTorch Runner.step() call. aggregate_stance itself has no
gradient (it came from an LLM call, which we are not differentiating
through -- only the deterministic scaling step after it).
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
from config.archetypes import ARCHETYPES, PROMPT_TEMPLATE
from src.llm_backend import OllamaLLM
from src import market_context

CACHE_PATH = pathlib.Path(__file__).resolve().parents[1] / "cache" / "archetype_responses.json"
EXAMPLES_OUT_PATH = pathlib.Path(__file__).resolve().parents[1] / "outputs" / "examples.json"
MEMORY_DIR = pathlib.Path(__file__).resolve().parents[1] / "cache" / "memory"

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


def _prompt_kwargs_for(prices_by_ticker: dict, ticker: str, date: str) -> dict:
    context = market_context.recent_price_context(prices_by_ticker[ticker], date)
    return {
        "ticker": ticker,
        "price": f"{context['price']:.2f}",
        "date": date,
        "returns_series": market_context.format_returns_series(context["returns_series"]),
        "window": str(context["window"]),
    }


def _compute_all_examples() -> tuple:
    """Returns (aggregate_stance_list, per_example_records). One real
    AgentTorch Archetype/Behavior call per example, cached by context hash
    so interrupted/resumed runs never re-call the LLM for a finished example.
    """
    prices_by_ticker = market_context.load_prices()
    examples = market_context.generate_examples(prices_by_ticker)
    print(f"Evaluating {len(examples)} examples across {len(prices_by_ticker)} tickers")

    llm = OllamaLLM()
    archetype = Archetype(n_arch=1).llm(llm=llm, user_prompt=PROMPT_TEMPLATE)
    behavior = Behavior(archetype=archetype, region=trader_population)

    cache = _load_cache()
    aggregates, records = [], []

    for i, ex in enumerate(examples):
        ticker, date = ex["ticker"], ex["date"]
        prompt_kwargs = _prompt_kwargs_for(prices_by_ticker, ticker, date)
        key = _context_cache_key(prompt_kwargs)

        if key in cache:
            stances, reasonings = cache[key]["stances"], cache[key]["reasonings"]
        else:
            sample_kwargs = {**prompt_kwargs, "device": "cpu", "current_memory_dir": str(MEMORY_DIR)}
            output = behavior.sample(sample_kwargs)  # shape [4, 1], one row per archetype
            stances = output.squeeze(-1).tolist()
            reasonings = list(llm.last_raw_responses)
            cache[key] = {"ticker": ticker, "date": date, "stances": stances, "reasonings": reasonings}
            _save_cache(cache)  # incremental save: a long background run is resumable

        aggregate = sum(stances) / len(stances)
        aggregates.append(aggregate)
        records.append({
            "ticker": ticker,
            "date": date,
            "aggregate_stance": aggregate,
            "archetypes": [
                {"name": a["name"], "stance": s, "reasoning": r}
                for a, s, r in zip(ARCHETYPES, stances, reasonings)
            ],
            "target": market_context.realized_target(prices_by_ticker[ticker], date),
        })

        if (i + 1) % 25 == 0 or i == len(examples) - 1:
            print(f"  [{i + 1}/{len(examples)}] {ticker} {date}  aggregate_stance={aggregate:+.3f}")

    EXAMPLES_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    EXAMPLES_OUT_PATH.write_text(json.dumps(records, indent=2))
    return aggregates, records


@Registry.register_substep("decide_stance", "policy")
class DecideStance(SubstepAction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        aggregates, self.records = _compute_all_examples()
        self.aggregate_stance = torch.tensor(aggregates).reshape(-1, 1)

    def forward(self, state, observation):
        return {self.output_variables[0]: self.aggregate_stance}


@Registry.register_substep("compute_prediction", "transition")
class ComputePrediction(SubstepTransition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, state, action):
        aggregate_stance = action["market"]["aggregate_stance"]
        predicted_return = (SENSITIVITY * aggregate_stance).reshape(-1, 1)
        return {self.output_variables[0]: predicted_return}
