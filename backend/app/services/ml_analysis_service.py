"""
ML Analysis Service for position analysis.

Provides detailed position analysis using trained ML models.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch

try:
    from src.ml.models.multi_task_model import MultiTaskModel, MultiTaskConfig
    from src.ml.features import board_to_tensor
except ImportError:
    MultiTaskModel = None
    board_to_tensor = None
    logging.warning("ML models not found. ML analysis will not work.")

try:
    import gogame_py as go
except ImportError:
    go = None
    logging.warning("gogame_py not found. ML analysis will not work.")

logger = logging.getLogger(__name__)


class MLAnalysisService:
    """Service for ML-powered position analysis."""

    def __init__(self, model_path: Optional[Path] = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = None
        self.model_loaded = False

        if MultiTaskModel is None or board_to_tensor is None:
            logger.warning("ML models not available. Analysis will use fallback methods.")
            return

        if model_path and model_path.exists():
            self.load_model(model_path)
        else:
            logger.warning(f"Model not found at {model_path}. Using fallback methods.")

    def load_model(self, model_path: Path):
        """Load trained model from checkpoint."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            config_dict = checkpoint.get("config", {})
            config = MultiTaskConfig(**config_dict)

            self.model = MultiTaskModel(config)
            self.model.load_state_dict(checkpoint["model_state"])
            self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True

            logger.info(f"Loaded ML model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            self.model_loaded = False

    async def analyze_position(
        self, board: "go.Board", current_player: "go.Color"
    ) -> Dict[str, Any]:
        """
        Analyze current position using ML models.

        Returns:
            Dictionary with analysis results:
            - threats: Threat detection results
            - attacks: Attack opportunity results
            - intent: Intent recognition results
            - evaluation: Position evaluation
        """
        if not self.model_loaded or self.model is None:
            return self._fallback_analysis(board, current_player)

        try:
            # Convert board to tensor
            board_tensor = board_to_tensor(board, current_player)
            board_tensor = board_tensor.unsqueeze(0).to(self.device)  # Add batch dimension

            # Run inference
            with torch.no_grad():
                predictions = self.model.predict(board_tensor)

            # Process outputs
            threat_map = predictions["threat_map"][0].cpu().numpy().tolist()
            attack_map = predictions["attack_map"][0].cpu().numpy().tolist()
            intent_info = predictions["intent"]

            # Extract threat regions
            threat_regions = self._extract_regions(threat_map, threshold=0.6, board_size=board.size())

            # Extract attack opportunities
            attack_opportunities = self._extract_regions(attack_map, threshold=0.6, board_size=board.size())

            return {
                "threats": {
                    "heatmap": threat_map,
                    "regions": threat_regions,
                },
                "attacks": {
                    "heatmap": attack_map,
                    "opportunities": attack_opportunities,
                },
                "intent": {
                    "primary_intent": intent_info["names"][0],
                    "confidence": float(intent_info["probabilities"][0].max()),
                    "all_intents": [
                        {
                            "type": name,
                            "probability": float(prob),
                        }
                        for name, prob in zip(intent_info["names"], intent_info["probabilities"][0])
                    ],
                    "heatmap": intent_info["heatmap"][0].cpu().numpy().tolist(),
                },
            }

        except Exception as e:
            logger.error(f"Error in ML analysis: {e}", exc_info=True)
            return self._fallback_analysis(board, current_player)

    def _extract_regions(
        self, heatmap: list[list[float]], threshold: float, board_size: int
    ) -> list[Dict[str, Any]]:
        """
        Extract regions from heatmap above threshold.

        Simple implementation: find connected components.
        """
        regions = []
        visited = [[False] * board_size for _ in range(board_size)]

        for y in range(board_size):
            for x in range(board_size):
                if not visited[y][x] and heatmap[y][x] >= threshold:
                    # BFS to find connected region
                    region_positions = []
                    queue = [(x, y)]

                    while queue:
                        cx, cy = queue.pop(0)
                        if visited[cy][cx] or heatmap[cy][cx] < threshold:
                            continue

                        visited[cy][cx] = True
                        region_positions.append([cx, cy])

                        # Check neighbors
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nx, ny = cx + dx, cy + dy
                            if 0 <= nx < board_size and 0 <= ny < board_size:
                                if not visited[ny][nx] and heatmap[ny][nx] >= threshold:
                                    queue.append((nx, ny))

                    if region_positions:
                        avg_value = sum(heatmap[cy][cx] for cx, cy in region_positions) / len(region_positions)
                        regions.append({
                            "positions": region_positions,
                            "severity": float(avg_value),
                            "size": len(region_positions),
                        })

        return regions

    def _fallback_analysis(self, board: "go.Board", current_player: "go.Color") -> Dict[str, Any]:
        """Fallback analysis using rule-based methods."""
        board_size = board.size()

        # Simple fallback: return empty analysis
        return {
            "threats": {
                "heatmap": [[0.0] * board_size for _ in range(board_size)],
                "regions": [],
            },
            "attacks": {
                "heatmap": [[0.0] * board_size for _ in range(board_size)],
                "opportunities": [],
            },
            "intent": {
                "primary_intent": "unknown",
                "confidence": 0.0,
                "all_intents": [],
                "heatmap": [[0.0] * board_size for _ in range(board_size)],
            },
            "fallback": True,
        }


def get_ml_analysis_service(model_path: Optional[Path] = None) -> MLAnalysisService:
    """Factory function to create ML analysis service."""
    if model_path is None:
        # Default model path
        model_path = Path("models/multi_task_model.pt")

    return MLAnalysisService(model_path=model_path, device="cpu")

