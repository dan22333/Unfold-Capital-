"""The manager-stage substeps: same shape as src/substeps.py's
DecideStance/ComputePrediction, but the Policy stage loads the already-
synthesized manager_stance per example (from manager_synthesis.py's output)
instead of calling the LLM itself -- the LLM call already happened in
Stage 1 of this checkpoint's build. A separate SENSITIVITY_MANAGER
parameter keeps this calibration independent of the trader-archetype one
in substeps.py, so both can be compared side by side.
"""

import json
import pathlib
import sys

import torch

from agent_torch.core.registry import Registry
from agent_torch.core.substep import SubstepAction, SubstepTransition

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

EXAMPLES_WITH_MANAGER_PATH = (
    pathlib.Path(__file__).resolve().parents[1] / "outputs" / "examples_with_manager.json"
)

SENSITIVITY_MANAGER = torch.nn.Parameter(torch.tensor(0.1))


@Registry.register_substep("manager_stance", "policy")
class ManagerStance(SubstepAction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        records = json.loads(EXAMPLES_WITH_MANAGER_PATH.read_text())
        self.records = records
        self.manager_stance = torch.tensor(
            [r["manager_stance"] for r in records]
        ).reshape(-1, 1)

    def forward(self, state, observation):
        return {self.output_variables[0]: self.manager_stance}


@Registry.register_substep("compute_prediction_manager", "transition")
class ComputePredictionManager(SubstepTransition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, state, action):
        manager_stance = action["market"]["manager_stance"]
        predicted_return = (SENSITIVITY_MANAGER * manager_stance).reshape(-1, 1)
        return {self.output_variables[0]: predicted_return}
