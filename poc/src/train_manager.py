"""Calibrates SENSITIVITY_MANAGER against the manager-synthesized stance,
via the same real AgentTorch Runner.step() differentiable loop as
src/train.py -- just fed by manager_stance instead of the flat archetype
average.
"""

import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from agent_torch.core import Registry, Runner
from agent_torch.core.helpers import read_config

from src import manager_substeps  # noqa: F401 - import triggers @Registry.register_substep

CONFIG_PATH = pathlib.Path(__file__).resolve().parents[1] / "config" / "model_config_manager.yaml"
EXAMPLES_WITH_MANAGER_PATH = (
    pathlib.Path(__file__).resolve().parents[1] / "outputs" / "examples_with_manager.json"
)


def real_targets():
    records = json.loads(EXAMPLES_WITH_MANAGER_PATH.read_text())
    targets = [r["target"]["mean"] for r in records]
    return torch.tensor(targets).reshape(-1, 1), len(records)


def build_runner(num_examples: int) -> Runner:
    config = read_config(str(CONFIG_PATH))
    config["state"]["agents"]["market"]["number"] = num_examples
    registry = Registry()
    runner = Runner(config, registry)
    runner.init()
    return runner


def train(num_steps: int = 800, lr: float = 0.05):
    target, num_examples = real_targets()
    print(f"{num_examples} examples, target shape {tuple(target.shape)}")

    runner = build_runner(num_examples)
    optimizer = torch.optim.Adam([manager_substeps.SENSITIVITY_MANAGER], lr=lr)
    losses = []

    for i in range(num_steps):
        runner.step(1)
        predicted = runner.state["agents"]["market"]["predicted_return"]
        loss = torch.nn.functional.mse_loss(predicted, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if i % 100 == 0 or i == num_steps - 1:
            print(f"step {i:4d}  loss={loss.item():.6f}  "
                  f"sensitivity_manager={manager_substeps.SENSITIVITY_MANAGER.item():.4f}")

    return losses, runner, target


if __name__ == "__main__":
    train()
