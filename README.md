# Unfold Capital

Research toward an **AI-native hedge fund** — closer in spirit to **Bridgewater than to a
statistical quant**. The bet is not on finding faint statistical patterns in price data;
it is on **reasoning about the future**: building a causal understanding of *how the
economic machine works* — how a shock propagates from one company to its suppliers,
customers, and competitors — and using that to anticipate *what will happen* before it
shows up in consensus estimates and prices.

Where a classic quant asks *"what has statistically predicted returns?"*, this asks
*"given this event, how will the affected companies actually respond, and where does that
leave fundamentals versus what the market expects?"* — cause-and-effect reasoning about
the world, not curve-fitting on history. Modern LLM/agent systems make that kind of
population-scale causal simulation newly tractable.

Concretely: can a structured economic graph and a population of interacting agent
"entities" predict fundamental **ripple effects** through supply chains *before* they are
reflected in prices? The question is not whether agents can read filings or write research
code — that is becoming table stakes at every fund. It is whether a **causal,
mechanism-level simulation** of how companies respond to shocks contains *incremental*
predictive information beyond the compressed signal already captured by customer stock
returns.

## The differentiated hypothesis

Customer-momentum signals (e.g. FactSet's supply-chain + network-centrality work)
compress everything into one number: the customer's stock return. That is powerful but
**lossy** — a +20% customer move could come from higher demand, layoffs, a buyback,
multiple expansion, or lower costs, each implying very different consequences for its
suppliers.

The proposed system tries to recover the *mechanism*:

```
Event → entity state change → behavioral response → network propagation
      → ΔRevenue / ΔMargin / ΔEPS → gap vs. consensus → position
```

The key empirical test is incremental value:

```
r(i,t+1) = α + β1·CustomerMomentum(i,t) + β2·RippleSignal(i,t) + controls + ε(i,t)
```

A useful `RippleSignal` stays predictive out of sample, with an economically meaningful
and stable `β2`, and improves net portfolio performance **after costs**.

## What's in this repo

| File | What it is |
|------|-----------|
| `AI_Native_Hedge_Fund_Research_Memo.docx` | Working research memo (Sept 2026): competitive landscape, the differentiated hypothesis, methodology, and go/no-go criteria. |
| `sim-alpha.html` | Study notes — *LLMs as World-Simulators for Market Alpha*: papers explained from the ground up, a blunt build-vs-buy verdict, strategy families, data survey, system design, and a 12-week MVP plan. |
| `agenttorch-explained.html` | Explainer on AgentTorch / Large Population Models and the Ripple Effect Protocol (architecture inspiration). |
| `simulating-the-future.html` | Deep-dive: Chopra/AgentTorch, convexity search, CS329A, AI trends, and the MVP test. |
| `references/` | Source material behind the thesis — two archived PDFs plus a cataloged, annotated reading list. See [`references/README.md`](references/README.md). |

## Evidence & foundations

The thesis rests on a chain of outside results. Full summaries and local PDFs are in
[`references/`](references/README.md); the short version of why each one is here:

- **The market case — HBS, *"Mimicking Finance"* (Cohen, Lu, Nguyen, 2026).** AI can
  predict ~**71%** of fund managers' quarterly trades, and the *predictable* managers are the
  ones who *lose* (most-predictable: −0.42% by Q4; least-predictable: +0.4%). *"Prices
  aren't predictable, but human behavior is."* Edge can't come from a faster version of what
  every fund already does — it has to come from a different *kind* of reasoning. This sets
  the bar: be the unpredictable manager.

- **The proof of concept — Meta FAIR, *CWM: Code World Model*.** A 32B LLM mid-trained on
  execution *trajectories* learns to simulate how code runs, and beats plain code LLMs
  (65.8% SWE-bench Verified). In a domain with clean ground truth, an LLM **with a world
  model** beats one that only pattern-matches text. Markets are the same argument with
  noisier ground truth.

- **The method — Large Population Models (Chopra / MIT Media Lab).** Simulate *populations*
  of interacting agents, not smarter individual agents; outcomes emerge from the interaction.
  - **AgentTorch** — the open-source engine. Differentiable (gradients flow through the
    simulation, so you can *calibrate* to data and *optimize* over interventions) and
    scalable (millions of agents via LLM "archetypes," one query per unique profile).
  - **Chopra's PhD thesis** — the formal theory (cognition + coordination modules) and,
    critically, **Chapter 5 / Lucas's critique**: when does a calibrated simulation survive
    the world reacting to it? That is this fund's biggest failure mode, named in advance.
  - **Real-world validation** — an LPM predicted avian-flu spread through poultry/cattle
    *supply chains* and migration: 100 hotspots flagged, 87 confirmed (MIT Tech Review). The
    perspective paper frames the goal exactly: *"trace how small changes ripple through an
    entire population"* to find non-obvious leverage points — which is what an alpha signal is.

**The through-line:** predictable managers lose (HBS) → world-models beat pattern-matchers
where truth is checkable (CWM) → population simulation can trace real ripple effects through
supply chains (LPMs). Unfold's wager is to build the *world-model manager* and be the
unpredictable minority that still earns its fee.

## Research plan (highlights)

- **Point-in-time event panel.** Unit of observation is `(event, company, timestamp)`;
  freeze the information set at the decision time to prevent look-ahead leakage. Store
  graph, state, evidence, estimated response, confidence, consensus, and realized outcomes.
- **Cross-sectional learning, not one model per company.** Shrink sparse-history firms
  toward peer/industry priors: `estimate = λ·company + (1-λ)·prior`.
- **Time split, not random split.** Train early → validate later → one untouched recent
  test period.
- **Baselines that must be beaten.** Customer momentum, centrality-weighted momentum,
  standard factor models, single-LLM+RAG, graph-without-response-functions, direct-effects-only.
- **Alpha and risk kept separate.** Optimizer solves position weights jointly:
  `max_w  α̂ᵀw − λ·wᵀΣw − γ·Turnover(w)` under dollar/beta/sector neutrality.

## Main failure modes to guard against

Narrative hallucination · look-ahead contamination · common factors masquerading as alpha ·
overfitting the event taxonomy · signal redundancy vs. customer returns ·
propagation explosion at later hops · capacity/implementation gap after real trading costs.

## Go / no-go criterion

A strong result is **not** "the agents give plausible answers." It is: the ripple model
beats customer momentum + standard factors + a single-LLM baseline on an untouched period,
with meaningfully better fundamental-forecast accuracy and/or abnormal-return prediction,
and a higher **net** Sharpe after costs. If the simpler customer-return signal does just
as well, use the simpler system.

---

*Working research — 2026. See [`references/`](references/README.md) for the annotated source
list and archived papers, and the memo and study notes for the wider bibliography (FactSet,
Man Group, Point72, State Street, HBS, Meta FAIR, AgentTorch / Population AI).*
