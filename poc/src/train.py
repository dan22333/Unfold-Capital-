"""The calibration loop: real AgentTorch Runner.step() forward passes,
backprop into the single `sensitivity` parameter, against AAPL's real
realized forward returns.
"""

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from agent_torch.core import Registry, Runner
from agent_torch.core.helpers import read_config

from src import substeps  # noqa: F401 - import triggers @Registry.register_substep
from src import shocks
from config.event_data import SHOCK_ANCHOR_DATE

CONFIG_PATH = pathlib.Path(__file__).resolve().parents[1] / "config" / "model_config.yaml"


def build_runner() -> Runner:
    config = read_config(str(CONFIG_PATH))
    registry = Registry()
    runner = Runner(config, registry)
    runner.init()
    return runner


def real_targets() -> torch.Tensor:
    """Realized forward returns at the shock anchor date, both horizons, as the training target.

    predicted_return is a single scalar (one aggregate stance x sensitivity),
    so we calibrate it against the mean of the two horizons' realized returns
    -- the simplest honest reduction for a 1-parameter model.
    """
    prices = shocks.load_prices()
    realized = shocks.realized_forward_returns(prices)
    mean_realized = sum(realized.values()) / len(realized)
    print(f"Realized forward returns at {SHOCK_ANCHOR_DATE}: {realized} (mean={mean_realized:.4f})")
    return torch.tensor(mean_realized).reshape(1, 1)


def train(num_steps: int = 800, lr: float = 0.05):
    runner = build_runner()
    target = real_targets()

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
                f"sensitivity={substeps.SENSITIVITY.item():.4f}  "
                f"predicted={predicted.item():.4f}  target={target.item():.4f}"
            )

    return losses, runner


if __name__ == "__main__":
    train()
