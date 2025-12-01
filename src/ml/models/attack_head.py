"""
Attack opportunity detection head for multi-task model.

Outputs a heatmap indicating attack opportunities across the board.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttackHead(nn.Module):
    """
    Head for attack opportunity detection.

    Outputs a heatmap of shape (batch, board_size, board_size)
    where values range from 0 (no opportunity) to 1 (high opportunity).
    """

    def __init__(self, input_channels: int, board_size: int = 19):
        super().__init__()
        self.board_size = board_size

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)

        # Output layer: single channel heatmap
        self.conv_out = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: Feature tensor from backbone
                     Shape: (batch, input_channels, board_size, board_size)

        Returns:
            Attack heatmap of shape (batch, board_size, board_size)
            Values in range [0, 1] after sigmoid
        """
        x = F.relu(self.bn1(self.conv1(features)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv_out(x)

        # Squeeze channel dimension and apply sigmoid
        x = x.squeeze(1)  # (batch, board_size, board_size)
        x = torch.sigmoid(x)  # Normalize to [0, 1]

        return x


def test_attack_head():
    """Test the attack head with dummy data."""
    input_channels = 64
    board_size = 9
    batch_size = 2

    head = AttackHead(input_channels=input_channels, board_size=board_size)

    # Dummy features from backbone
    dummy_features = torch.randn(batch_size, input_channels, board_size, board_size)

    with torch.no_grad():
        output = head(dummy_features)

    print(f"Input features shape: {dummy_features.shape}")
    print(f"Output attack map shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")

    # Verify output shape
    assert output.shape == (batch_size, board_size, board_size), \
        f"Expected ({batch_size}, {board_size}, {board_size}), got {output.shape}"

    # Verify output range
    assert output.min() >= 0.0 and output.max() <= 1.0, \
        f"Output should be in [0, 1], got [{output.min():.3f}, {output.max():.3f}]"

    print("✅ Attack head test passed!")


if __name__ == "__main__":
    test_attack_head()

