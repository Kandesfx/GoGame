"""
Multi-task learning model for Go position analysis.

Combines shared backbone with task-specific heads for:
- Threat detection
- Attack opportunity detection
- Intent recognition
- Position evaluation
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .attack_head import AttackHead
from .intent_head import IntentHead
from .shared_backbone import SharedBackbone
from .threat_head import ThreatHead


@dataclass
class MultiTaskConfig:
    """Configuration for multi-task model."""

    input_planes: int = 17
    board_size: int = 19
    base_channels: int = 64
    num_res_blocks: int = 4


class MultiTaskModel(nn.Module):
    """
    Multi-task model for Go position analysis.

    Architecture:
    - Shared backbone extracts features
    - Task-specific heads produce outputs
    """

    def __init__(self, config: MultiTaskConfig = MultiTaskConfig()):
        super().__init__()
        self.config = config

        # Shared backbone
        self.backbone = SharedBackbone(
            input_planes=config.input_planes,
            base_channels=config.base_channels,
            num_res_blocks=config.num_res_blocks,
        )

        # Task-specific heads
        backbone_channels = self.backbone.get_output_channels()
        self.threat_head = ThreatHead(backbone_channels, config.board_size)
        self.attack_head = AttackHead(backbone_channels, config.board_size)
        self.intent_head = IntentHead(backbone_channels, config.board_size)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass through all tasks.

        Args:
            x: Input tensor of shape (batch, input_planes, board_size, board_size)

        Returns:
            Dictionary with outputs from all heads:
            - "threat_map": (batch, board_size, board_size)
            - "attack_map": (batch, board_size, board_size)
            - "intent_logits": (batch, num_intents)
            - "intent_heatmap": (batch, board_size, board_size)
        """
        # Extract shared features
        features = self.backbone(x)

        # Task-specific outputs
        threat_map = self.threat_head(features)
        attack_map = self.attack_head(features)
        intent_logits, intent_heatmap = self.intent_head(features)

        return {
            "threat_map": threat_map,
            "attack_map": attack_map,
            "intent_logits": intent_logits,
            "intent_heatmap": intent_heatmap,
        }

    def predict(self, x: torch.Tensor) -> dict:
        """
        Predict with human-readable outputs.

        Args:
            x: Input tensor

        Returns:
            Dictionary with predictions including class names
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            intent_names, intent_probs, _ = self.intent_head.predict_intent(
                self.backbone(x)
            )

        return {
            "threat_map": outputs["threat_map"],
            "attack_map": outputs["attack_map"],
            "intent": {
                "names": intent_names,
                "probabilities": intent_probs,
                "heatmap": outputs["intent_heatmap"],
            },
        }


def test_multi_task_model():
    """Test the multi-task model with dummy data."""
    config = MultiTaskConfig(input_planes=17, board_size=9, base_channels=64, num_res_blocks=4)
    model = MultiTaskModel(config)

    # Dummy input
    batch_size = 2
    dummy_input = torch.randn(batch_size, config.input_planes, config.board_size, config.board_size)

    # Forward pass
    with torch.no_grad():
        outputs = model(dummy_input)
        predictions = model.predict(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print("\nOutput shapes:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")

    print("\nPredictions:")
    print(f"  Intent names: {predictions['intent']['names']}")
    print(f"  Intent probabilities shape: {predictions['intent']['probabilities'].shape}")

    # Verify output shapes
    assert outputs["threat_map"].shape == (batch_size, config.board_size, config.board_size)
    assert outputs["attack_map"].shape == (batch_size, config.board_size, config.board_size)
    assert outputs["intent_logits"].shape == (batch_size, IntentHead.NUM_INTENTS)
    assert outputs["intent_heatmap"].shape == (batch_size, config.board_size, config.board_size)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size (MB): {total_params * 4 / 1024 / 1024:.2f}")  # Assuming float32

    print("\n✅ Multi-task model test passed!")


if __name__ == "__main__":
    test_multi_task_model()

