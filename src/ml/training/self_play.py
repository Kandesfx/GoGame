"""
Self-play training skeleton for GoGame.

Implements high-level loop described in SystemSpec:
1. Generate games via current policy/value networks.
2. Store experience.
3. Update networks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from ..policy_network import PolicyNetwork, PolicyConfig
from ..value_network import ValueNetwork, ValueConfig
from ..features import board_to_tensor, policy_from_visits

try:
    import gogame_py as go  # type: ignore
except ImportError as exc:  # pragma: no cover - raised when bindings missing
    go = None  # type: ignore
    _BINDING_ERROR = exc
else:
    _BINDING_ERROR = None


@dataclass
class SelfPlayConfig:
    games_per_iteration: int = 50
    board_size: int = 9
    temperature: float = 1.0
    learning_rate: float = 1e-3
    batch_size: int = 128
    device: str = "cpu"
    replay_buffer_size: int = 10_000
    save_dir: Path = Path("checkpoints")


class ReplayBuffer:
    """Simple in-memory buffer for training data."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.states: List[torch.Tensor] = []
        self.policy_targets: List[torch.Tensor] = []
        self.value_targets: List[torch.Tensor] = []

    def add(self, state: torch.Tensor, policy: torch.Tensor, value: torch.Tensor) -> None:
        if len(self.states) >= self.capacity:
            self.states.pop(0)
            self.policy_targets.pop(0)
            self.value_targets.pop(0)
        self.states.append(state.detach().clone())
        self.policy_targets.append(policy.detach().clone())
        self.value_targets.append(value.detach().clone())

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = torch.randint(0, len(self.states), (batch_size,))
        states = torch.stack([self.states[i] for i in indices])
        policy = torch.stack([self.policy_targets[i] for i in indices])
        value = torch.stack([self.value_targets[i] for i in indices])
        return states, policy, value

    def __len__(self) -> int:
        return len(self.states)


class SelfPlayTrainer:
    def __init__(self, config: SelfPlayConfig):
        if go is None:
            raise ImportError(
                "gogame_py module not found. Build the C++ bindings before running self-play training."
            ) from _BINDING_ERROR

        self.config = config
        self.device = torch.device(config.device)
        self.policy = PolicyNetwork(PolicyConfig(board_size=config.board_size)).to(self.device)
        self.value = ValueNetwork(ValueConfig(board_size=config.board_size)).to(self.device)
        self.replay = ReplayBuffer(config.replay_buffer_size)
        self.optimizer_policy = torch.optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        self.optimizer_value = torch.optim.Adam(self.value.parameters(), lr=config.learning_rate)
        self.ai_player = go.AIPlayer()

    def generate_self_play_games(self) -> None:
        """
        Generate self-play games and populate replay buffer.

        Uses AIPlayer with default configs (Level 3 MCTS fallback to Minimax).
        """
        for _ in range(self.config.games_per_iteration):
            board = go.Board(self.config.board_size)
            current_player = board.current_player()
            pass_count = 0
            max_moves = self.config.board_size * self.config.board_size * 2
            move_counter = 0

            state_tensors: List[torch.Tensor] = []
            policy_targets: List[torch.Tensor] = []
            players: List[go.Color] = []

            while pass_count < 2 and move_counter < max_moves:
                features = board_to_tensor(board, current_player)
                policy_vec, move = self._select_move_with_policy(board, current_player)

                state_tensors.append(features)
                policy_targets.append(policy_vec)
                players.append(current_player)

                board.make_move(move)
                pass_count = pass_count + 1 if move.is_pass() else 0
                current_player = board.current_player()
                move_counter += 1

                if board.is_game_over():
                    break

            outcome = self._estimate_outcome(board)
            for state, policy_vec, player in zip(state_tensors, policy_targets, players):
                value = torch.tensor(
                    [outcome if player == go.Color.Black else 1.0 - outcome],
                    dtype=torch.float32,
                )
                self.replay.add(state, policy_vec, value)

    def update_networks(self) -> None:
        """Run one gradient step if enough data."""
        if len(self.replay) < self.config.batch_size:
            return

        states, policy_targets, value_targets = self.replay.sample(self.config.batch_size)
        states = states.to(self.device)
        policy_targets = policy_targets.to(self.device)
        value_targets = value_targets.to(self.device)

        # Policy loss (cross-entropy)
        policy_logits = self.policy(states)
        policy_loss = torch.mean(torch.sum(-policy_targets * policy_logits, dim=1))

        # Value loss (MSE)
        value_preds = self.value(states)
        value_loss = torch.mean((value_preds - value_targets) ** 2)

        loss = policy_loss + value_loss
        self.optimizer_policy.zero_grad()
        self.optimizer_value.zero_grad()
        loss.backward()
        self.optimizer_policy.step()
        self.optimizer_value.step()

    def save_checkpoint(self, iteration: int) -> None:
        self.config.save_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "config": self.config,
            "policy_config": self.policy.config.__dict__,
            "value_config": self.value.config.__dict__,
            "policy_state": self.policy.state_dict(),
            "value_state": self.value.state_dict(),
        }
        torch.save(checkpoint, self.config.save_dir / f"self_play_{iteration:04d}.pt")

    # Internal helpers -----------------------------------------------------

    def _select_move_with_policy(
        self,
        board,
        player,
    ) -> Tuple[torch.Tensor, any]:
        """Return (policy_vector, move) using MCTS result when available."""
        mcts_result = self.ai_player.mcts_result(board, 3)
        board_size = board.size()

        if mcts_result is not None and mcts_result.total_visits > 0:
            policy_vec = policy_from_visits(mcts_result.top_moves, board_size)
            move = mcts_result.best_move
            if not move.is_valid():
                move = self._fallback_move(board, player)
                policy_vec = self._one_hot_policy(move, board_size)
        else:
            move = self._fallback_move(board, player)
            policy_vec = self._one_hot_policy(move, board_size)

        return policy_vec, move

    def _fallback_move(self, board, player):
        """Fallback using Minimax or first legal move."""
        minimax_result = self.ai_player.minimax_result(board, 2)
        if minimax_result is not None and minimax_result.best_move.is_valid():
            return minimax_result.best_move

        legal_moves = board.get_legal_moves(player)
        if not legal_moves:
            return go.Move.pass_move(player)
        return legal_moves[0]

    def _one_hot_policy(self, move, board_size: int) -> torch.Tensor:
        policy = torch.zeros(board_size * board_size, dtype=torch.float32)
        if move.is_valid() and not move.is_pass():
            idx = move.y() * board_size + move.x()
            policy[idx] = 1.0
        return policy

    def _estimate_outcome(self, board) -> float:
        """Simple outcome estimation using prisoner difference."""
        black_score = board.get_prisoners(go.Color.Black)
        white_score = board.get_prisoners(go.Color.White)
        if black_score > white_score:
            return 1.0
        if white_score > black_score:
            return 0.0
        return 0.5

