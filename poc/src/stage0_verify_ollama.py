"""Stage 0 verification: a raw Ollama call with zero AgentTorch involved.

Run this after `ollama pull <model>` to confirm the local model responds
before wiring anything into AgentTorch's Archetype/Behavior/LLMBackend layer.
"""

import ollama

MODEL = "llama3.1:8b"


def main() -> None:
    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": (
                    "AAPL closed at $142.19 on 2019-01-03, a -9.5% move over the "
                    "past 2 trading days. You are a momentum trader who believes "
                    "recent price moves tend to continue in the same direction. "
                    "What is your stance on AAPL right now? Answer with a single "
                    "number from -1 (strongly bearish) to 1 (strongly bullish), "
                    "and nothing else."
                ),
            }
        ],
    )
    print("Raw model response:")
    print(response["message"]["content"])


if __name__ == "__main__":
    main()
