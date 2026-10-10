"""The calibration loop: real AgentTorch Runner.step() forward passes,
vectorized over the whole [N, 1] example population, backprop into the
single `sensitivity` parameter against each example's real realized
forward return.
"""

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from agent_torch.core import Registry, Runner
from agent_torch.core.helpers import read_config

from src import substeps  # noqa: F401 - import triggers @Registry.register_substep
from src import market_context

CONFIG_PATH = pathlib.Path(__file__).resolve().parents[1] / "config" / "model_config.yaml"


def real_targets() -> torch.Tensor:
    """[N, 1] realized forward returns (mean of 4wk/8wk), one row per
    example, in the SAME order substeps.generate_examples() produces --
    both call market_context.generate_examples() with the same deterministic
    inputs, so the ordering matches without needing to thread data through
    AgentTorch's state.
    """
    prices_by_ticker = market_context.load_prices()
    examples = market_context.generate_examples(prices_by_ticker)
    targets = [
        market_context.realized_target(prices_by_ticker[ex["ticker"]], ex["date"])["mean"]
        for ex in examples
    ]
    return torch.tensor(targets).reshape(-1, 1), len(examples)


def build_runner(num_examples: int) -> Runner:
    config = read_config(str(CONFIG_PATH))
    config["state"]["agents"]["market"]["number"] = num_examples
    registry = Registry()
    runner = Runner(config, registry)
    runner.init()
    return runner


def train(num_steps: int = 800, lr: float = 0.05):
    target, num_examples = real_targets()
    print(f"{num_examples} examples, target shape {tuple(target.shape)}, "
          f"mean target {target.mean().item():+.4f}")

    runner = build_runner(num_examples)

    optimizer = torch.optim.Adam([substeps.SENSITIVITY], lr=lr)
    losses = []

    for i in range(num_steps):
        runner.step(1)
        predicted = runner.state["agents"]["market"]["predicted_return"]

        loss = torch.nn.functional.mse_loss(predicted, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if i % 25 == 0 or i == num_steps - 1:
            print(
                f"step {i:4d}  loss={loss.item():.6f}  "
                f"sensitivity={substeps.SENSITIVITY.item():.4f}"
            )

    return losses, runner, target


if __name__ == "__main__":
    train()
