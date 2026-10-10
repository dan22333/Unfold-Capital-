"""The LLM call behind each archetype's stance.

Stage 2 of the plan: wire a local Ollama model into something AgentTorch's
Archetype/Behavior layer can call. This module provides the actual query +
parsing logic independent of how it's wired in (AgentTorch's LangchainLLM if
it accepts an Ollama-backed Langchain model, or a small custom LLMBackend
subclass otherwise — see poc/README.md for which path Stage 2 landed on).

Small open-weights models are not perfectly reliable about "output just a
number," so this includes a tolerant parser with one retry before falling
back to a neutral 0.0 (logged, never silently wrong).
"""

import re

import ollama

from agent_torch.core.llm.backend import LLMBackend
import agent_torch.core.llm.agent_memory as agent_memory

MODEL = "llama3.1:8b"


def _patched_langchain_save_memory(self, query, output, agent_id):
    """Monkeypatch for a confirmed bug in agent_torch==0.6.0.

    LangchainMemoryHandler.save_memory unconditionally does output["text"],
    but every LLMBackend.prompt() implementation — including the SHIPPED
    LangchainLLM backend, not just ours — returns a plain list of strings.
    Unpatched, this raises on every call, the exception is swallowed by
    Behavior.sample()'s retry loop, and after 10 silent retries you get an
    empty agent_outputs list, divided by zero, producing NaN with no visible
    error. Verified by reading agent_torch/core/llm/{archetype,behavior,
    agent_memory,backend}.py directly against the installed 0.6.0 package.
    """
    text = output["text"] if isinstance(output, dict) else output
    self.agent_memory[agent_id].save_context({"input": query["agent_query"]}, {"output": text})


agent_memory.LangchainMemoryHandler.save_memory = _patched_langchain_save_memory


def _extract_float_in_range(text: str, lo: float = -1.0, hi: float = 1.0) -> float | None:
    matches = re.findall(r"-?\d*\.?\d+", text)
    for m in matches:
        try:
            value = float(m)
        except ValueError:
            continue
        if lo <= value <= hi:
            return value
    return None


def query_stance(prompt: str, model: str = MODEL, retries: int = 1) -> float:
    """Query the local model for a stance in [-1, 1]. Falls back to 0.0 (neutral)."""
    last_raw = None
    for attempt in range(retries + 1):
        response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
        raw = response["message"]["content"]
        last_raw = raw
        value = _extract_float_in_range(raw)
        if value is not None:
            return value
    print(f"WARNING: could not parse a stance in [-1, 1] from model output: {last_raw!r}. Using 0.0.")
    return 0.0


class OllamaLLM(LLMBackend):
    """AgentTorch's shipped LangchainLLM backend hardcodes ChatOpenAI; this is
    the confirmed fallback the docs call out — a small LLMBackend subclass
    that routes through our local Ollama model instead. `backend = "langchain"`
    so AgentTorch's LLMArchetype.initialize_memory() uses its (LLM-agnostic)
    LangchainMemoryHandler.
    """

    def __init__(self, model: str = MODEL):
        super().__init__()
        self.backend = "langchain"
        self.model = model

    def initialize_llm(self):
        return self  # no separate predictor object needed; prompt() calls Ollama directly

    def prompt(self, prompt_list):
        # Must return plain strings, not floats: AgentTorch's memory-saving
        # path (LLMArchetype.__call__ -> save_memory) expects a
        # string/subscriptable output like the other backends return, and
        # does its own float(...) conversion downstream in Behavior.
        outputs = []
        for item in prompt_list:
            query = item["agent_query"] if isinstance(item, dict) else item
            outputs.append(str(query_stance(query, model=self.model)))
        return outputs


if __name__ == "__main__":
    test_prompt = (
        "AAPL closed at $142.19 on 2019-01-03, a -9.5% move over the past 2 "
        "trading days. You are a momentum trader who believes recent price "
        "moves tend to continue in the same direction. What is your stance on "
        "AAPL right now? Answer with a single number from -1 (strongly "
        "bearish) to 1 (strongly bullish), and nothing else."
    )
    print("Stance:", query_stance(test_prompt))
