"""
Policy network skeleton for GoGame.

Follows SystemSpec Section 3.4: lightweight CNN that predicts move probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PolicyConfig:
    board_size: int = 19
    input_planes: int = 17
    channels: int = 128


class PolicyNetwork(nn.Module):
    """
    Lightweight CNN policy network.

    Inputs:
        features: Tensor[batch, planes, board, board]
    Outputs:
        Tensor[batch, board * board] representing move probabilities.
    """

    def __init__(self, config: PolicyConfig = PolicyConfig()):
        super().__init__()
        self.config = config
        c = config.channels
        p = config.input_planes

        self.conv1 = nn.Conv2d(p, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, c, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(c, c, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(c, c, kernel_size=3, padding=1)
        self.head = nn.Conv2d(c, 1, kernel_size=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(c)
        self.bn3 = nn.BatchNorm2d(c)
        self.bn4 = nn.BatchNorm2d(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.head(x)
        batch, _, board, _ = x.shape
        x = x.view(batch, board * board)
        return F.log_softmax(x, dim=1)

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(features)
        return torch.exp(logits)


def load_policy_model(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
) -> PolicyNetwork:
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config_dict = state.get("config", {})
    model = PolicyNetwork(PolicyConfig(**config_dict))
    model.load_state_dict(state["model_state"])
    model.to(device or torch.device("cpu"))
    model.eval()
    return model

