"""
Script để generate 17-plane features từ board state trên Colab.

Features bao gồm:
- Plane 0-1: Current player stones, opponent stones
- Plane 2-7: Liberty counts (1, 2, 3+ liberties)
- Plane 8-15: Move history (last 4 moves)
- Plane 16: Turn indicator
"""

import numpy as np
import torch
from typing import Tuple, Optional, Union
import numpy as np


def get_liberties_simple(board_state: np.ndarray, x: int, y: int, board_size: int) -> int:
    """
    Tính số liberties của một quân cờ tại (x, y).
    Simplified version - chỉ đếm empty neighbors.
    """
    if board_state[y, x] == 0:
        return 0
    
    color = board_state[y, x]
    liberties = 0
    
    # Check 4 neighbors
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < board_size and 0 <= ny < board_size:
            if board_state[ny, nx] == 0:
                liberties += 1
    
    return liberties


def board_to_features_17_planes(
    board_state: np.ndarray,
    current_player: str,
    move_history: Optional[list] = None,
    board_size: int = 9
) -> torch.Tensor:
    """
    Convert board state thành 17-plane tensor.
    
    Args:
        board_state: numpy array [board_size, board_size]
                    0 = empty, 1 = black, 2 = white
        current_player: 'B' hoặc 'W'
        move_history: List of (x, y) tuples cho last 4 moves
        board_size: Kích thước bàn cờ
    
    Returns:
        Tensor [17, board_size, board_size]
    """
    features = torch.zeros((17, board_size, board_size), dtype=torch.float32)
    
    # Plane 0: Current player stones
    # Plane 1: Opponent stones
    if current_player == 'B':
        features[0] = torch.from_numpy((board_state == 1).astype(np.float32))
        features[1] = torch.from_numpy((board_state == 2).astype(np.float32))
    else:  # White
        features[0] = torch.from_numpy((board_state == 2).astype(np.float32))
        features[1] = torch.from_numpy((board_state == 1).astype(np.float32))
    
    # Plane 2-3: Current player stones with 1 liberty
    # Plane 4-5: Current player stones with 2 liberties
    # Plane 6-7: Current player stones with 3+ liberties
    for y in range(board_size):
        for x in range(board_size):
            if board_state[y, x] == 0:
                continue
            
            # Determine if this is current player's stone
            is_current = (
                (current_player == 'B' and board_state[y, x] == 1) or
                (current_player == 'W' and board_state[y, x] == 2)
            )
            
            if is_current:
                liberties = get_liberties_simple(board_state, x, y, board_size)
                if liberties == 1:
                    features[2, y, x] = 1.0
                elif liberties == 2:
                    features[4, y, x] = 1.0
                elif liberties >= 3:
                    features[6, y, x] = 1.0
            else:
                # Opponent stones
                liberties = get_liberties_simple(board_state, x, y, board_size)
                if liberties == 1:
                    features[3, y, x] = 1.0
                elif liberties == 2:
                    features[5, y, x] = 1.0
                elif liberties >= 3:
                    features[7, y, x] = 1.0
    
    # Plane 8-15: Move history (last 4 moves)
    # Mỗi move = 2 planes (x và y coordinates)
    if move_history:
        for i, (mx, my) in enumerate(move_history[-4:]):  # Last 4 moves only
            if 0 <= mx < board_size and 0 <= my < board_size:
                features[8 + i * 2, my, mx] = 1.0  # X coordinate
                features[8 + i * 2 + 1, my, mx] = 1.0  # Y coordinate
    
    # Plane 16: Turn indicator (1 = Black, 0 = White)
    features[16].fill_(1.0 if current_player == 'B' else 0.0)
    
    return features


def generate_policy_label(
    move: Optional[Tuple[int, int]],
    board_size: int
) -> torch.Tensor:
    """
    Generate policy label từ move.
    
    Args:
        move: (x, y) tuple hoặc None cho pass move, hoặc (-1, -1) cho pass
        board_size: Kích thước bàn cờ
    
    Returns:
        Tensor [board_size * board_size + 1] với one-hot tại move position
        Index cuối cùng (board_size * board_size) dành cho pass move
    
    Raises:
        ValueError: Nếu move không hợp lệ (ngoài board và không phải pass)
    """
    # Policy vector: board positions + 1 pass move
    policy = torch.zeros(board_size * board_size + 1, dtype=torch.float32)
    
    # Handle pass moves
    if move is None or move == (-1, -1) or (isinstance(move, tuple) and len(move) == 2 and move[0] == -1 and move[1] == -1):
        # Pass move → index cuối cùng
        policy[-1] = 1.0
        return policy
    
    # Validate move format
    if not isinstance(move, (tuple, list)) or len(move) != 2:
        raise ValueError(f"Invalid move format: {move}. Expected (x, y) tuple or None for pass.")
    
    x, y = move
    
    # Validate coordinates
    if not isinstance(x, (int, np.integer)) or not isinstance(y, (int, np.integer)):
        raise ValueError(f"Move coordinates must be integers: got ({type(x).__name__}, {type(y).__name__})")
    
    # Check if valid board position
    if 0 <= x < board_size and 0 <= y < board_size:
        idx = y * board_size + x
        policy[idx] = 1.0
    else:
        # Invalid coordinates (outside board) → treat as pass
        # Log warning but don't crash
        import warnings
        warnings.warn(
            f"Move ({x}, {y}) is outside board size {board_size}. "
            f"Treating as pass move.",
            UserWarning
        )
        policy[-1] = 1.0
    
    return policy


def generate_value_label(
    winner: Optional[str],
    current_player: str,
    game_result: Optional[str] = None
) -> float:
    """
    Generate value label (win probability) với validation chặt chẽ.
    
    Args:
        winner: 'B', 'W', 'DRAW', hoặc None
        current_player: 'B' hoặc 'W' (phải khớp với người chơi ở position)
        game_result: String như "B+12.5" (optional, để tính chính xác hơn)
    
    Returns:
        float: 0.0 - 1.0 (1.0 = current player wins, 0.5 = draw hoặc unknown)
    
    Raises:
        ValueError: Nếu current_player không hợp lệ
    """
    # Validate current_player
    if current_player not in ('B', 'W', 'b', 'w'):
        raise ValueError(
            f"Invalid current_player: '{current_player}'. "
            f"Must be 'B', 'W', 'b', or 'w'."
        )
    
    # Normalize to uppercase
    current_player = current_player.upper()
    
    # Handle None winner
    if winner is None:
        return 0.5  # Unknown result
    
    # Normalize winner
    if isinstance(winner, str):
        winner = winner.upper()
    
    # Handle DRAW
    if winner == 'DRAW' or winner == '0':
        return 0.5  # Draw game - both players get 0.5
    
    # Validate winner format
    if winner not in ('B', 'W'):
        # Try to parse from game_result if provided
        if game_result:
            game_result_upper = str(game_result).upper().strip()
            if game_result_upper.startswith('B+') or game_result_upper == 'B':
                winner = 'B'
            elif game_result_upper.startswith('W+') or game_result_upper == 'W':
                winner = 'W'
            else:
                # Cannot determine winner
                return 0.5
        else:
            # Invalid winner format and no game_result
            return 0.5
    
    # Calculate value: 1.0 if current player wins, 0.0 otherwise
    if winner == current_player:
        return 1.0
    else:
        return 0.0


if __name__ == "__main__":
    # Test
    board_size = 9
    board_state = np.zeros((board_size, board_size), dtype=np.int8)
    board_state[4, 4] = 1  # Black stone at center
    board_state[3, 4] = 2  # White stone
    
    features = board_to_features_17_planes(
        board_state,
        current_player='B',
        move_history=[(4, 4), (3, 4)],
        board_size=board_size
    )
    
    print(f"Features shape: {features.shape}")
    print(f"Plane 0 (Black stones) sum: {features[0].sum()}")
    print(f"Plane 1 (White stones) sum: {features[1].sum()}")
    
    # Test policy label
    policy = generate_policy_label((4, 4), board_size)
    print(f"Policy shape: {policy.shape}")
    print(f"Policy at (4,4): {policy[4 * board_size + 4]}")
    
    # Test pass move
    policy_pass = generate_policy_label(None, board_size)
    print(f"Policy pass shape: {policy_pass.shape}")
    print(f"Policy pass (last index): {policy_pass[-1]}")
    
    # Test value label
    value = generate_value_label('B', 'B')
    print(f"Value (Black wins, Black to move): {value}")
    
    # Test value label with validation
    try:
        value_invalid = generate_value_label('B', 'X')  # Should raise ValueError
    except ValueError as e:
        print(f"✅ Caught expected ValueError: {e}")

