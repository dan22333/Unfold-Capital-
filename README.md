# Unfold Capital

Research toward an **AI-native hedge fund**: can a structured economic graph and a
population of interacting LLM/agent "entities" predict fundamental **ripple effects**
through supply chains *before* they are reflected in consensus estimates and prices?

The core question is not whether agents can read filings or write research code — that
is becoming table stakes at every major fund. It is whether a **causal, mechanism-level
simulation** of how companies respond to shocks contains *incremental* predictive
information beyond the compressed signal already captured by customer stock returns.

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

*Working research — September 2026. See the memo and study notes for full references
(FactSet, Man Group, Point72, State Street, AgentTorch / Population AI).*
