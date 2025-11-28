"""
Data collection script for ML training.

Generates self-play games and extracts training samples.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List

try:
    import gogame_py as go
except ImportError:
    go = None
    logging.warning("gogame_py not found. Data collection will not work.")

from ..features import board_to_tensor

logger = logging.getLogger(__name__)


class DataCollector:
    """Collects training data from self-play games."""

    def __init__(self, board_size: int = 9, output_dir: Path = Path("data/training")):
        self.board_size = board_size
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if go is None:
            raise ImportError("gogame_py module required for data collection")

        self.ai_player = go.AIPlayer()

    async def collect_game(self, game_id: int) -> List[Dict[str, Any]]:
        """
        Collect one self-play game and extract training samples.

        Returns:
            List of training samples, each containing:
            - board_state: Tensor
            - move: Move object
            - metadata: Dict
        """
        board = go.Board(self.board_size)
        current_player = board.current_player()
        samples = []
        move_number = 0
        pass_count = 0
        max_moves = self.board_size * self.board_size * 2

        while pass_count < 2 and move_number < max_moves:
            # Extract board state
            board_tensor = board_to_tensor(board, current_player)

            # Get AI move (using MCTS level 3)
            mcts_result = self.ai_player.mcts_result(board, level=3)
            if mcts_result and mcts_result.best_move.is_valid():
                move = mcts_result.best_move
            else:
                # Fallback to first legal move
                legal_moves = board.get_legal_moves(current_player)
                move = legal_moves[0] if legal_moves else go.Move.pass_move(current_player)

            # Store sample
            samples.append({
                "board_state": board_tensor,
                "move": move,
                "move_number": move_number,
                "current_player": current_player,
                "board_hash": board.zobrist_hash(),
            })

            # Apply move
            if board.is_legal_move(move):
                board.make_move(move)
                pass_count = pass_count + 1 if move.is_pass() else 0
            else:
                break

            current_player = board.current_player()
            move_number += 1

            if board.is_game_over():
                break

        # Calculate game outcome
        black_prisoners = board.get_prisoners(go.Color.Black)
        white_prisoners = board.get_prisoners(go.Color.White)
        outcome = 1.0 if black_prisoners > white_prisoners else 0.0 if white_prisoners > black_prisoners else 0.5

        # Add outcome to all samples
        for sample in samples:
            sample["outcome"] = outcome if sample["current_player"] == go.Color.Black else 1.0 - outcome

        logger.info(f"Collected game {game_id}: {len(samples)} samples, outcome={outcome:.2f}")

        return samples

    async def collect_games(self, num_games: int = 100) -> List[Dict[str, Any]]:
        """
        Collect multiple games.

        Args:
            num_games: Number of games to collect

        Returns:
            List of all training samples from all games
        """
        all_samples = []

        logger.info(f"Starting data collection: {num_games} games")

        for game_id in range(num_games):
            try:
                samples = await self.collect_game(game_id)
                all_samples.extend(samples)

                if (game_id + 1) % 10 == 0:
                    logger.info(f"Progress: {game_id + 1}/{num_games} games, {len(all_samples)} total samples")

            except Exception as e:
                logger.error(f"Error collecting game {game_id}: {e}", exc_info=True)
                continue

        logger.info(f"Data collection complete: {len(all_samples)} total samples")

        return all_samples

    def save_samples(self, samples: List[Dict[str, Any]], filename: str = "training_data.pt"):
        """Save samples to disk."""
        import torch

        output_path = self.output_dir / filename
        torch.save(samples, output_path)
        logger.info(f"Saved {len(samples)} samples to {output_path}")


async def main():
    """Main function for data collection."""
    logging.basicConfig(level=logging.INFO)

    collector = DataCollector(board_size=9, output_dir=Path("data/training"))

    # Collect games
    samples = await collector.collect_games(num_games=50)

    # Save samples
    collector.save_samples(samples, "self_play_9x9_50games.pt")

    print(f"\n✅ Collected {len(samples)} training samples")


if __name__ == "__main__":
    asyncio.run(main())

