"""
Feature extraction utilities for GoGame ML pipeline.

Currently implements a basic 3-plane representation:
    plane 0: stones of the current player
    plane 1: stones of the opponent
    plane 2: indicator of current player (all ones if black, zeros if white)
"""

from __future__ import annotations

from typing import Iterable

import torch

try:
    import gogame_py as go  # type: ignore
except ImportError as exc:  # pragma: no cover - raised when bindings missing
    go = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _ensure_binding() -> None:
    if go is None:
        raise ImportError(
            "gogame_py module not found. Build the C++ bindings before using ML pipeline."
        ) from _IMPORT_ERROR


def board_to_tensor(board, to_move) -> torch.Tensor:
    """
    Convert a C++ Board (via binding) into a tensor of shape (3, board, board).

    Args:
        board: gogame_py.Board instance
        to_move: gogame_py.Color indicating current player
    """
    _ensure_binding()

    size = board.size()
    features = torch.zeros((3, size, size), dtype=torch.float32)

    for y in range(size):
        for x in range(size):
            stone = board.at(x, y)
            if stone == go.Stone.Empty:
                continue
            if stone == go.Stone.Black:
                if to_move == go.Color.Black:
                    features[0, y, x] = 1.0
                else:
                    features[1, y, x] = 1.0
            elif stone == go.Stone.White:
                if to_move == go.Color.White:
                    features[0, y, x] = 1.0
                else:
                    features[1, y, x] = 1.0

    features[2].fill_(1.0 if to_move == go.Color.Black else 0.0)
    return features


def policy_from_visits(
    stats: Iterable,
    board_size: int,
) -> torch.Tensor:
    """
    Convert visit counts into a policy vector.

    Args:
        stats: iterable of gogame_py.MCTSMoveStats objects
        board_size: board dimension N
    """
    _ensure_binding()

    policy = torch.zeros(board_size * board_size, dtype=torch.float32)
    total_visits = 0
    indexed_stats = []
    for move_stat in stats:
        move = move_stat.move
        idx = move.y() * board_size + move.x()
        indexed_stats.append((idx, move_stat.visits))
        total_visits += move_stat.visits

    if total_visits == 0:
        return policy

    for idx, visits in indexed_stats:
        policy[idx] = visits / total_visits

    return policy

