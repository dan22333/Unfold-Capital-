# AgentTorch Technology POC — trader archetypes

**This is a technology demo, not a trading signal.** One ticker (AAPL), one
real historical shock event, three hand-written trader personas, and a single
calibrated parameter. It proves AgentTorch's `Config`/`Runner`/`Substep` loop
plus its LLM-archetype feature (`Archetype`/`Behavior`) run end to end against
real market data with a local, open-weights model (Ollama). Nothing here is a
claim of predictive validity — n=1 event, by design. See
`../agenttorch-explained.html` and `../sim-alpha.html` for the actual business
thesis this is a small step toward.

## What it does

1. Pulls AAPL's real daily price history around its Jan 2, 2019 guidance-cut
   shock (FQ1 revenue guidance cut from $89-93B to ~$84B; AAPL fell ~9.9%).
2. Defines three LLM-driven trader archetypes (momentum, contrarian, noise) as
   a tiny 3-agent AgentTorch population, using the real `Archetype`/`Behavior`
   classes (not a lookalike), backed by a local Ollama model — this is the
   actual "one LLM call per persona, broadcast to every agent sharing it"
   mechanism the full AgentTorch population models use, just at population
   size 3 instead of millions.
3. Each archetype states a stance (-1 to 1) given the real recent AAPL price
   move. On the real shock day: momentum -0.70, contrarian +0.50, noise +0.25.
4. A single learnable `sensitivity` parameter converts the archetypes'
   aggregate stance into a predicted forward return, calibrated via real
   `torch.autograd` backprop through AgentTorch's `Runner.step()` loop against
   AAPL's real realized forward return (mean of its 4wk/8wk returns, +20.7%).
5. Loss converges from ~0.042 to ~0 over 800 steps; learned `sensitivity`
   settles at ~12.4. Interesting (and true) wrinkle: the real outcome was a
   sharp rebound, so contrarian's call was directionally right and momentum's
   wasn't — the aggregate nearly cancels the two out. One event isn't enough
   to mean anything statistically, but it's a legible, honest demo of the
   mechanism.

## Real rough edges hit building this (useful to know before extending it)

- The shipped `models/macro_economics` / `models/covid` examples in
  `agent_torch==0.6.0` import a stale, pre-refactor `AgentTorch.*` namespace
  that doesn't exist in the current package — don't use them as a reference.
- `agent-torch`'s own pip dependencies pull the *latest* LangChain (1.4.x),
  which removed the legacy `langchain.memory`/`langchain.chains` APIs
  AgentTorch's `Archetype`/backend code needs. Pinned to `langchain==0.3.27`
  (+ matching `langchain-openai==0.3.28`) — see `requirements.txt`, which is a
  full freeze of the exact working set, not just top-level packages, because
  of exactly this kind of landmine.
- `PromptManager`'s `{variable}` extraction is a naive regex that doesn't
  understand Python format specs — `{price:.2f}` gets parsed as a literal key
  `"price:.2f"`, not `"price"`. Use bare `{price}` placeholders and
  pre-format numeric values to strings before passing them as kwargs.
- **A real bug**: `LangchainMemoryHandler.save_memory` (in
  `agent_torch/core/llm/agent_memory.py`) unconditionally does
  `output["text"]`, but every `LLMBackend.prompt()` implementation — including
  the *shipped* `LangchainLLM` backend, not just our custom Ollama one —
  returns a plain list of strings. Unpatched, this throws on every call,
  `Behavior.sample()`'s retry loop silently swallows the exception 10 times,
  and you get a NaN with no visible error (empty output list -> divide by
  zero). Patched with a one-line monkeypatch in `src/llm_backend.py`, clearly
  commented.
- AgentTorch's config-driven `calibrate_*`/learnable-args auto-wiring for
  parameters wasn't worth fully reverse-engineering for a 1-parameter model —
  `sensitivity` is a plain global `torch.nn.Parameter` referenced directly by
  the substep and the training loop, which is still genuinely differentiated
  through `Runner.step()`'s real forward pass via ordinary `torch.autograd`.

## Setup

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
# or: source .venv/bin/activate && pip install -r requirements.txt

ollama pull llama3.1:8b   # or llama3.2:3b if that's too slow locally
```

## Run

```bash
python src/stage0_verify_ollama.py   # confirm the local model responds at all
python src/stage2_smoke_test.py      # confirm the real Archetype/Behavior path works
python run_poc.py                    # full pipeline: data -> archetypes -> train -> evaluate
```

Outputs land in `outputs/`: `loss_curve.png`, `predicted_vs_actual.png`,
`predicted_vs_actual.csv`. Archetype LLM calls are cached to
`cache/archetype_responses.json` by context hash, so re-running `run_poc.py`
doesn't re-call the model.

## Explicitly out of scope for this pass

The multi-supplier ripple network from the business thesis; the full
point-in-time event panel, leakage gauntlet, and baseline ladder; portfolio
construction; more than one real shock event; statistical significance
testing. See `C:\Users\User\.claude\plans\jazzy-riding-hippo.md` for the full
plan, staged build order, and the alternatives considered (OASIS, Concordia)
before settling on AgentTorch.
