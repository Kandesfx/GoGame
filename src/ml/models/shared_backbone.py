"""
Shared CNN backbone for multi-task learning.

Lightweight ResNet-like architecture optimized for Go board analysis.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with batch normalization."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection
        return F.relu(out)


class SharedBackbone(nn.Module):
    """
    Shared feature extractor for multi-task learning.

    Architecture:
    - Input: 17 planes (board features)
    - Output: Feature maps for task-specific heads
    """

    def __init__(self, input_planes: int = 17, base_channels: int = 64, num_res_blocks: int = 4):
        super().__init__()
        self.input_planes = input_planes
        self.base_channels = base_channels

        # Initial convolution
        self.conv1 = nn.Conv2d(input_planes, base_channels, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(base_channels)

        # Residual blocks
        self.res_blocks = nn.ModuleList([ResidualBlock(base_channels) for _ in range(num_res_blocks)])

        # Output projection
        self.conv_out = nn.Conv2d(base_channels, base_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, input_planes, board_size, board_size)

        Returns:
            Feature tensor of shape (batch, base_channels, board_size, board_size)
        """
        x = F.relu(self.bn1(self.conv1(x)))

        # Apply residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)

        # Output projection
        x = self.conv_out(x)

        return x

    def get_output_channels(self) -> int:
        """Return number of output channels."""
        return self.base_channels


def test_backbone():
    """Test the backbone with dummy data."""
    backbone = SharedBackbone(input_planes=17, base_channels=64, num_res_blocks=4)

    # Dummy input: batch=2, planes=17, board=9x9
    dummy_input = torch.randn(2, 17, 9, 9)

    with torch.no_grad():
        output = backbone(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output channels: {backbone.get_output_channels()}")

    # Verify output shape
    assert output.shape == (2, 64, 9, 9), f"Expected (2, 64, 9, 9), got {output.shape}"

    print("✅ Backbone test passed!")


if __name__ == "__main__":
    test_backbone()

