"""
Value network skeleton for GoGame.

Predicts win probability for the current player.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ValueConfig:
    board_size: int = 19
    input_planes: int = 17
    channels: int = 128


class ValueNetwork(nn.Module):
    """
    CNN-based value network returning scalar win probability.

    Output range: (0, 1) after sigmoid.
    """

    def __init__(self, config: ValueConfig = ValueConfig()):
        super().__init__()
        self.config = config
        c = config.channels
        p = config.input_planes

        self.conv1 = nn.Conv2d(p, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, c, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(c, c, kernel_size=3, padding=1)

        board = config.board_size
        self.fc1 = nn.Linear(c * board * board, 256)
        self.fc2 = nn.Linear(256, 1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(c)
        self.bn3 = nn.BatchNorm2d(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            value = self.forward(features)
        return value


def load_value_model(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
) -> ValueNetwork:
    state = torch.load(checkpoint_path, map_location=device)
    config_dict = state.get("config", {})
    model = ValueNetwork(ValueConfig(**config_dict))
    model.load_state_dict(state["model_state"])
    model.to(device or torch.device("cpu"))
    model.eval()
    return model

