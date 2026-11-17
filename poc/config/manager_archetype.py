"""The senior/manager agent: a second-stage synthesis persona that reads the
4 trader archetypes' stances and reasoning for an example and forms its own
view -- with real discretion to weight, discount, or set aside a view it
finds unconvincing, rather than a fixed-weight average.

Single-persona population (n_arch via PromptManager's grouping = 1), same
real AgentTorch Archetype/Behavior/LoadPopulation machinery as the trader
archetypes, just a population of size 1 instead of 4.
"""

MANAGER_PERSONA = (
    "a senior portfolio manager reviewing your team's views before deciding "
    "your own position"
)

PROMPT_TEMPLATE = (
    "You are {persona}.\n\n"
    "Your team's analysts gave you the following views on {ticker} as of "
    "{date}:\n{archetype_summary}\n\n"
    "Use your own judgment -- you don't have to weight every view equally. "
    "Some analysts' reasoning may be more convincing than others for this "
    "specific situation, and you may discount or set aside a view you find "
    "unconvincing. In 1-2 sentences, explain your synthesis. Then on a new "
    "line write exactly 'Stance: ' followed by a single number from -1 "
    "(strongly bearish) to 1 (strongly bullish)."
)


def format_archetype_summary(archetypes: list) -> str:
    lines = []
    for a in archetypes:
        lines.append(f"- {a['name']} (stance {a['stance']:+.2f}): {a['reasoning'].strip()}")
    return "\n".join(lines)
