# AI/LLM Research for Fundamental Credit Analysis — Notes

Research notes from a survey of recent (2025-2026) published work on using LLMs and AI
agents to do the work of a fundamental analyst — compiled with a bias toward fixed
income / credit research (issuer surveillance, covenant analysis, coverage scaling)
rather than equity stock-picking or systematic/quant strategies.

Working thesis: this is not a replacement for fundamental analysts, but a way to scale
analyst capacity — wider coverage breadth, continuous monitoring, covenant/filings
analysis, and first-draft research — while judgment on idiosyncratic credit decisions
stays human.

---

## Academic / agent architectures

### FundaPod
*"FundaPod: A Multi-Persona Agent Pod Architecture with Knowledge Graph Memory for
AI-Assisted Fundamental Investment Research"* (arXiv, 2026)
[arxiv.org/pdf/2605.27864](https://arxiv.org/pdf/2605.27864)

Proposes a "pod" of LLM agents, each assigned a distinct analyst persona within a
research process, backed by a shared knowledge-graph memory so findings persist and
compound across sessions instead of being re-derived from scratch each time a question
is asked. Explicitly framed around *fundamental* investment research rather than
trading signals — the closest academic analog to a scaled research-team workflow
rather than a single-shot chatbot.

### Large Language Model Agents in Finance: A Survey
*"Large Language Model Agents in Finance: A Survey Bridging Research, Practice, and
Real-World Deployment"* (Findings of ACL / EMNLP 2025, pp. 17889-17907)
[aclanthology.org/2025.findings-emnlp.972](https://aclanthology.org/2025.findings-emnlp.972/)

A systematic, dual-perspective (academic + industry) survey of LLM agents in finance.
Introduces a functional taxonomy spanning five domains: data analysis, investment
research, trading, investment management, and risk management. Most useful as an
orienting map of the field and a jumping-off point for deeper citations — it's a
synthesis, not a single empirical result to lean on.

---

## Credit-specific tooling and research

### 9fin
[9fin.com](https://www.9fin.com/) — AI credit tools:
[launch announcement](https://www.9fin.com/insights/9fin-launches-next-generation-of-ai-tools-for-debt-capital-markets) ·
[coverage](https://uk.finance.yahoo.com/news/debt-research-firm-bets-ai-090108773.html)

Debt-market intelligence platform (founded 2016) covering leveraged finance, private
credit, and distressed debt, used by 350+ institutions (banks, asset managers, hedge
funds, law firms, private credit funds). Recently shipped two AI products: **AI Chat**,
which answers covenant/cap-structure/filings questions with responses linked back to
source documents, and **Research Grid**, which generates structured, templated
cross-issuer comparisons. Notable because the model is grounded in 9fin's own curated,
proprietary credit dataset rather than being a generic LLM wrapper — the closest
existing commercial product to a junior credit-analyst tool.

### CovenantAI
Saunders et al., *"CovenantAI — New Insights into Covenant Violations"* (SBFC 2025,
University of Sydney)
[sbfc.sydney.edu.au/2025/papers/SBFC2025_6B2_P261.pdf](https://sbfc.sydney.edu.au/2025/papers/SBFC2025_6B2_P261.pdf)

Academic paper using GPT-4o-mini to identify covenants and covenant thresholds directly
from SEC filings, and to classify complex renegotiation outcomes (amendments, waivers,
technical defaults) — outperforming traditional keyword- and database-driven detection
approaches. Directly relevant to the covenant-monitoring slice of a credit analyst's
job rather than valuation or thesis-writing.

### AI in Private Credit: Underwriting, Covenant Surveillance, and the Monitoring Gap
Leigh Coney (SSRN working paper)
[papers.ssrn.com/sol3/papers.cfm?abstract_id=6898658](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6898658)

Focuses specifically on the *ongoing surveillance* half of private credit work, as
distinct from initial underwriting — i.e., the continuous monitoring function most
analogous to what a credit analyst does across an existing coverage list after a
bond/loan is already held. Frames the "monitoring gap" (positions that go unwatched
between formal reviews) as the area AI is best suited to close.

---

## Industry practice — who's actually building this

### Hedgeweek: Avala Global founder on "AI agent fleets"
[hedgeweek.com — AI agent 'fleets' could transform hedge fund research](https://www.hedgeweek.com/ai-agent-fleets-could-transform-hedge-fund-research-says-avala-global-founder/)

Divya Nettimi (founder, Avala Global, ~$2bn AUM) predicts that within 3-5 years an
analyst who currently covers ~20 names could instead manage a fleet of AI agents
covering ~200 — an "order of magnitude" productivity change rather than an incremental
one. Same piece references Viking Global's internal "VikingGPT," built to let
portfolio managers stress-test trade ideas. Both executives quoted in the piece stress
that human judgment stays central — interpreting management teams, synthesizing
qualitative and quantitative signals, and pattern recognition built over years are
explicitly called out as not being automated away.

### Lone Pine Capital co-CIO on commoditizing the "base layer"
Same Hedgeweek piece as above; Kelly Granat (co-CIO, Lone Pine Capital, ~$19.5bn AUM)

Notable framing: AI could **"commoditise the base layer" of fundamental analysis** —
i.e., standardize and systematize the mechanical, repeatable parts of research across
analysts and sectors. Worth sitting with: if the mechanical component of fundamental
work becomes systematized and standardized, the line between "fundamental" and
"systematic" analysis starts to blur, which has implications for where differentiated
alpha in fundamental credit research can still come from.

---

## Open questions / possible next steps

- What would a covenant-surveillance agent pipeline need architecturally to run
  continuously across a ~100-200 issuer coverage list (data sources, grounding/RAG
  design, review cadence, escalation triggers for human review)?
- How do 9fin's AI Chat and Research Grid actually perform in practice vs. the
  marketing claims — worth a hands-on trial if accessible.
- Closer look at what BlackRock's Aladdin Copilot and S&P's Credit Companion have
  actually shipped vs. announced, since both are relevant reference points for what a
  large asset manager's internal tool could look like.
