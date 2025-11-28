"""
Intent recognition head for multi-task model.

Outputs both intent classification and spatial heatmap.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class IntentHead(nn.Module):
    """
    Head for intent recognition.

    Outputs:
    1. Intent classification (5 classes: territory, attack, defense, connection, cut)
    2. Intent heatmap (spatial regions related to intent)
    """

    INTENT_CLASSES = ["territory", "attack", "defense", "connection", "cut"]
    NUM_INTENTS = len(INTENT_CLASSES)

    def __init__(self, input_channels: int, board_size: int = 19):
        super().__init__()
        self.board_size = board_size

        # Shared feature extraction
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)

        # Intent classification branch
        self.fc1 = nn.Linear(16 * board_size * board_size, 128)
        self.fc2 = nn.Linear(128, self.NUM_INTENTS)

        # Intent heatmap branch
        self.conv_heatmap = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            features: Feature tensor from backbone
                     Shape: (batch, input_channels, board_size, board_size)

        Returns:
            tuple of:
            - intent_logits: (batch, NUM_INTENTS) - classification logits
            - intent_heatmap: (batch, board_size, board_size) - spatial heatmap
        """
        # Shared features
        x = F.relu(self.bn1(self.conv1(features)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Classification branch
        x_flat = torch.flatten(x, start_dim=1)  # (batch, 16 * board_size * board_size)
        intent_logits = self.fc2(F.relu(self.fc1(x_flat)))  # (batch, NUM_INTENTS)

        # Heatmap branch
        intent_heatmap = self.conv_heatmap(x)  # (batch, 1, board_size, board_size)
        intent_heatmap = intent_heatmap.squeeze(1)  # (batch, board_size, board_size)
        intent_heatmap = torch.sigmoid(intent_heatmap)  # Normalize to [0, 1]

        return intent_logits, intent_heatmap

    def predict_intent(self, features: torch.Tensor) -> tuple[list[str], torch.Tensor, torch.Tensor]:
        """
        Predict intent with class names.

        Returns:
            tuple of:
            - intent_names: List of predicted intent class names
            - intent_probs: (batch, NUM_INTENTS) - class probabilities
            - intent_heatmap: (batch, board_size, board_size) - spatial heatmap
        """
        intent_logits, intent_heatmap = self.forward(features)
        intent_probs = F.softmax(intent_logits, dim=1)
        intent_indices = torch.argmax(intent_probs, dim=1)

        intent_names = [self.INTENT_CLASSES[idx.item()] for idx in intent_indices]

        return intent_names, intent_probs, intent_heatmap


def test_intent_head():
    """Test the intent head with dummy data."""
    input_channels = 64
    board_size = 9
    batch_size = 2

    head = IntentHead(input_channels=input_channels, board_size=board_size)

    # Dummy features from backbone
    dummy_features = torch.randn(batch_size, input_channels, board_size, board_size)

    with torch.no_grad():
        intent_logits, intent_heatmap = head(dummy_features)
        intent_names, intent_probs, _ = head.predict_intent(dummy_features)

    print(f"Input features shape: {dummy_features.shape}")
    print(f"Intent logits shape: {intent_logits.shape}")
    print(f"Intent heatmap shape: {intent_heatmap.shape}")
    print(f"Predicted intents: {intent_names}")
    print(f"Intent probabilities shape: {intent_probs.shape}")

    # Verify output shapes
    assert intent_logits.shape == (batch_size, head.NUM_INTENTS), \
        f"Expected ({batch_size}, {head.NUM_INTENTS}), got {intent_logits.shape}"
    assert intent_heatmap.shape == (batch_size, board_size, board_size), \
        f"Expected ({batch_size}, {board_size}, {board_size}), got {intent_heatmap.shape}"

    print("✅ Intent head test passed!")


if __name__ == "__main__":
    test_intent_head()

