# References

Source material behind the Unfold Capital thesis. Each entry explains *what it is*, its
*key claims*, and *why it matters for an AI-native fund* that reasons causally about ripple
effects rather than curve-fitting on price history.

Two of the six are archived locally as PDFs; the rest are external links.

| # | Source | Local copy |
|---|--------|-----------|
| 1 | Harvard Business School — *"If AI Knows Your Next Trade, What Happens to Money Managers?"* | link only |
| 2 | AgentTorch (MIT Media Lab) — open-source LPM framework | link only |
| 3 | Ayush Chopra thesis — *Population AI: Modeling Societies of Interacting Agents* | [`chopra-thesis-population-ai.pdf`](chopra-thesis-population-ai.pdf) |
| 4 | MIT Technology Review — *Ayush Chopra simulates complex events* | link only |
| 5 | Meta FAIR — *CWM: Code World Model* | link only |
| 6 | MIT Media Lab — *What Can We Learn From a Billion Agents?* (perspective) | [`lpm-perspective-billion-agents.pdf`](lpm-perspective-billion-agents.pdf) |

---

## 1. "If AI Knows Your Next Trade, What Happens to Money Managers?"

**Harvard Business School, Working Knowledge** ·
<https://www.library.hbs.edu/working-knowledge/if-ai-knows-your-next-trade-what-happens-to-money-managers>
Underlying paper: *"Mimicking Finance"* (Feb 2026), Lauren H. Cohen (HBS), Yiwen Lu
(Wharton), Quoc H. Nguyen (DePaul).

**What it is.** Research showing that AI can predict most mutual-fund managers' quarterly
buy/sell/hold decisions, and that the *predictable* managers are the ones who underperform.

**Key findings.**
- ~**71%** of managers' quarterly decisions are anticipatable by AI; mid-cap blend funds
  hit **75%**.
- More experienced managers, larger/older funds, and multi-fund managers are *more*
  predictable.
- Performance splits sharply by predictability. Least-predictable managers earned ~**+0.14%**
  risk-adjusted in Q1 rising to **+0.4%** by Q4; most-predictable went from **−0.09%** to
  **−0.42%**. Cohen: *"even though prices aren't predictable, human behavior is."*

**Why it matters for Unfold.** This is the market case for the whole project. If formulaic,
pattern-matching behavior is both predictable and unrewarded, then edge has to come from a
genuinely *different* information source — causal, mechanism-level reasoning about how
events propagate — not from a faster version of what every fund already does. It defines
the bar: be the unpredictable manager, by reasoning where others pattern-match.

## 2. AgentTorch

**MIT Media Lab (Ayush Chopra et al.)** · <https://github.com/AgentTorch/AgentTorch> ·
docs: <https://agenttorch.github.io/AgentTorch/>

**What it is.** Open-source framework for building and running **Large Population Models
(LPMs)** — simulations of millions of interacting autonomous agents ("digital societies")
on hardware from a laptop to a supercomputer.

**Four design principles.** *Scalability* (millions of agents in seconds), *Differentiability*
(gradients flow through the stochastic simulation, enabling gradient-based calibration and
optimization), *Composition* (plug in LLMs, neural nets, physics sims as agent behaviors),
*Generalization* (geospatial human worlds, cellular systems, digital avatars). PyTorch-style
API, GPU-optimized, Python ≥3.9, AGPL-3.0.

**Why it matters for Unfold.** This is the concrete engine for the "population of interacting
entities" idea. The **differentiability** is the non-obvious unlock: it lets you *calibrate*
agent behavior to observed data and *optimize* over interventions instead of only running
forward what-ifs. The **archetype** trick (one LLM query per unique agent profile, not per
agent) is what makes LLM-driven populations affordable at scale — directly relevant to
simulating a universe of companies without an LLM call per firm per step.

## 3. Thesis — *Population AI: Modeling Societies of Interacting Agents*

**Ayush Chopra, MIT PhD (May 2026), advisor Ramesh Raskar.** Local PDF (188 pp).
Readers: Milind Tambe (Harvard), Joel Leibo (DeepMind). Mirror: <https://iceberg.mit.edu/thesis.pdf>

**What it is.** The full academic foundation for LPMs and AgentTorch. Core premise:
*"intelligence in agentic populations emerges not just from how smart each agent is, but
from how they interact."*

**Structure worth knowing.**
- **LPM formal definition** — each agent carries a parameterized, learnable behavioral model
  with two parts: a **Cognition module** (state → decision) and a **Coordination module**
  (exchange sensitivity messages with network neighbors). Population outcomes emerge from the
  joint update across the interaction network.
- **Ch. 4 — differentiable ABMs (`flame`)**: how gradients are pushed through a full
  simulation; hybrid forward/reverse-mode AD to keep memory flat over long horizons.
- **Ch. 5 — *The Limits of Agency* / Lucas's Critique** (most relevant to markets):
  studies when LLM-agent behavior stays valid as scale and regime change — i.e. whether a
  calibrated simulation survives the thing being simulated reacting to it. The central
  hazard for any market simulator.
- **Ch. 9 — Project Iceberg / Iceberg Index**: national-scale labor-market model of the
  human-AI workforce (151M workers, 3,000 counties); argues aggregate statistics miss
  structural, below-the-surface risk that a population model surfaces.

**Why it matters for Unfold.** It is the rigorous version of the pitch, and it names the
traps. Chapter 5 (Lucas's critique) is effectively a pre-registration of the fund's biggest
failure mode: a model whose own predictions change the world it predicted. The
cognition/coordination split is a clean template for an "event → firm response → network
propagation" pipeline.

## 4. MIT Technology Review — Ayush Chopra, Innovators

<https://www.technologyreview.com/innovator/ayush-chopra-simulates-complex-events/>

**What it is.** Accessible profile of Chopra's LPM work and its real-world hits.

**Key points.** Agent *swarms* (hundreds of millions of simple agents) produce emergent
intelligence rather than one smart assistant. **Spring 2024**: an LPM predicted avian-flu
spread by simulating poultry/cattle supply chains, bird migration, farm economics, and
consumer behavior — flagged **100 hotspots, 87 confirmed** by year end. Also modeling
AI's impact on 151M US workers. Thesis line: *"Central orchestrators don't scale to 100
billion agents"* → coordination must be decentralized.

**Why it matters for Unfold.** The avian-flu result is the strongest external evidence that
mechanism-level population simulation makes *falsifiable, accurate* forward predictions
about a messy real economy — supply chains included. That is the exact shape of claim the
fund needs to make about equities.

## 5. Meta FAIR — *CWM: Code World Model*

<https://ai.meta.com/research/publications/cwm-an-open-weights-llm-for-research-on-code-generation-with-world-models/>

**What it is.** An open-weights **32B** dense decoder LLM (up to 131k context) mid-trained
on *observation–action trajectories* from Python interpreters and agentic Docker
environments — so it learns how code execution *unfolds*, not just static code text.

**World-model value vs. a plain LLM.** CWM can mentally *simulate* execution step by step,
supporting more deliberate planning and agentic coding. Results: **65.8%** SWE-bench
Verified, **68.6%** LiveCodeBench, **96.6%** Math-500, **76.0%** AIME 2024.

**Why it matters for Unfold.** The cleanest proof-of-concept that an LLM **with a world
model** beats an LLM that only pattern-matches surface text — in a domain (code) with
crisp ground truth. It generalizes the fund's core bet to a single model: predicting the
*next state of a system* from a model of its dynamics beats predicting the *next token*
from correlations. Markets are the same argument with noisier ground truth.

## 6. *What Can We Learn From a Billion Agents?* (perspective)

**MIT Media Lab** · Local PDF · <https://lpm.media.mit.edu/perspective.pdf>

**What it is.** The vision/manifesto behind LPMs: within a decade ~60B AI agents will
augment society, and the bottleneck won't be smarter individual agents but **coordination
at a scale beyond human cognition** (Dunbar's number). Proposes **protocol-centric
intelligence**: LLMs enhance individuals; LPMs shape collective behavior.

**Three claimed LPM innovations.** *Differentiable population simulation* (trace how a
small change ripples through the whole population, end-to-end gradients revealing
non-obvious leverage points), *adaptive protocol fields* (rules that evolve with feedback),
*digital-physical bridge* (discover protocols in simulation, deploy on edge devices, feed
outcomes back).

**Why it matters for Unfold.** The phrase *"trace how small changes ripple through an
entire population"* is the ripple-effect hypothesis in one line. It frames the differentiable
population as a tool for finding **non-obvious leverage points** — exactly what an alpha
signal is: where a shock produces an outsized, under-priced fundamental consequence.

---

## The through-line

1. **(1) HBS** — predictable, pattern-matching managers lose; edge must come from a
   different *kind* of reasoning.
2. **(5) CWM** — in a clean-ground-truth domain, an LLM *with a world model* beats one
   without. The bet generalizes.
3. **(2,3,4,6) LPMs / AgentTorch / Chopra** — the method, engine, formal theory, and
   real-world validation (avian flu, labor markets) for simulating a population of
   interacting economic entities and **tracing ripple effects** — while naming the central
   trap (Lucas's critique: the market reacting to your own prediction).

Unfold Capital's wager, stated against these sources: build the *world-model* manager —
causal, mechanism-level, population-scale — and be the unpredictable 29% that still earns
its fee.
