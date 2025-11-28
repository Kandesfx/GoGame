"""Wrapper để gọi gogame_py AI từ MSYS2 Python (tránh DLL conflicts)."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Path to MSYS2 Python
MSYS2_PYTHON = Path("C:/msys64/mingw64/bin/python3.exe")
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def _check_msys2_python() -> bool:
    """Kiểm tra MSYS2 Python có sẵn không."""
    exists = MSYS2_PYTHON.exists()
    if not exists:
        logger.warning(f"MSYS2 Python not found at {MSYS2_PYTHON}")
    return exists


def call_ai_select_move(board_state: Dict[str, Any], level: int) -> Optional[Dict[str, Any]]:
    """Gọi AI để chọn move qua MSYS2 Python subprocess.
    
    Args:
        board_state: Board state dict với moves, current_player, etc.
        level: AI level (1-4)
    
    Returns:
        Dict với move info hoặc None nếu fail
    """
    if not _check_msys2_python():
        logger.error(f"❌ MSYS2 Python not found at {MSYS2_PYTHON}, AI features disabled")
        logger.error(f"❌ To enable AI features, install MSYS2 and build gogame_py module")
        return None
    
    # Tính timeout động dựa trên level và board size
    board_size = board_state.get('board_size', 9)
    
    # Timeout calculation:
    # Level 1-2 (Minimax): Nhanh, 10-15 giây đủ
    # Level 3 (Minimax depth 4): Cần 20-40 giây (tùy board size)
    # Level 4 (Minimax depth 5): Cần 40-80 giây (tùy board size)
    # Board size lớn hơn cần thêm thời gian
    timeout_map = {
        1: 15,   # Minimax depth 1 - rất nhanh
        2: 20,   # Minimax depth 2 - nhanh
        3: 60 if board_size >= 19 else 40 if board_size >= 13 else 20,   # Minimax depth 4 (tự động điều chỉnh)
        4: 120 if board_size >= 19 else 80 if board_size >= 13 else 40,  # Minimax depth 5 (tự động điều chỉnh)
    }
    
    timeout = timeout_map.get(level, 30)
    logger.debug(f"AI level {level}, board size {board_size}x{board_size}, timeout: {timeout}s")
    
    # Tạo script để chạy AI
    # Chuyển đổi JSON null thành None trong Python
    moves_json = json.dumps(board_state.get('moves', []))
    moves_json = moves_json.replace('null', 'None')  # Chuyển null thành None
    
    script_content = f"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(r"{PROJECT_ROOT}") / "build"))

import gogame_py
import json

# Reconstruct board
board = gogame_py.Board({board_state.get('board_size', 9)})

# Replay moves
moves = {moves_json}
for move_doc in moves:
    if not move_doc:
        continue
    color = gogame_py.Color.Black if move_doc.get('color') == 'B' else gogame_py.Color.White
    position = move_doc.get('position')
    if position and position is not None and isinstance(position, list) and len(position) == 2:
        x, y = position
        move = gogame_py.Move(x, y, color)
    else:
        move = gogame_py.Move.pass_move(color)
    
    if board.is_legal_move(move):
        board.make_move(move)

# Select AI move
ai = gogame_py.AIPlayer()
ai_move = ai.select_move(board, {level})

# Return result
result = {{
    'x': ai_move.x if not ai_move.is_pass else None,
    'y': ai_move.y if not ai_move.is_pass else None,
    'is_pass': ai_move.is_pass,
    'color': 'W' if ai_move.color == gogame_py.Color.White else 'B',
}}

print(json.dumps(result))
"""
    
    try:
        # Chạy script với MSYS2 Python với timeout động
        result = subprocess.run(
            [str(MSYS2_PYTHON), "-c", script_content],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(PROJECT_ROOT),
        )
        
        if result.returncode == 0:
            move_data = json.loads(result.stdout.strip())
            return move_data
        else:
            logger.error(f"AI subprocess failed: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        logger.error(f"AI subprocess timeout after {timeout}s (level {level}, board {board_size}x{board_size})")
        logger.error(f"Level {level} với board {board_size}x{board_size} cần nhiều thời gian hơn. "
                    f"Xem xét giảm playouts hoặc tăng timeout.")
        return None
    except Exception as e:
        logger.error(f"Error calling AI: {e}", exc_info=True)
        return None


def call_ai_evaluate(board_state: Dict[str, Any]) -> Optional[float]:
    """Gọi AI để evaluate position.
    
    Args:
        board_state: Board state dict
    
    Returns:
        Evaluation score hoặc None
    """
    if not _check_msys2_python():
        return None
    
    script_content = f"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(r"{PROJECT_ROOT}") / "build"))

import gogame_py
import json

# Reconstruct board
board = gogame_py.Board({board_state.get('board_size', 9)})
moves = {json.dumps(board_state.get('moves', []))}

for move_doc in moves:
    color = gogame_py.Color.Black if move_doc['color'] == 'B' else gogame_py.Color.White
    if move_doc.get('position'):
        x, y = move_doc['position']
        move = gogame_py.Move(x, y, color)
    else:
        move = gogame_py.Move.pass_move(color)
    
    if board.is_legal_move(move):
        board.make_move(move)

# Evaluate
from gogame_py import MinimaxConfig, MinimaxEngine
config = MinimaxConfig()
config.max_depth = 3
config.use_alpha_beta = True
engine = MinimaxEngine(config)

current_player = board.current_player()
result = engine.search(board, current_player)

print(result.evaluation)
"""
    
    try:
        result = subprocess.run(
            [str(MSYS2_PYTHON), "-c", script_content],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(PROJECT_ROOT),
        )
        
        if result.returncode == 0:
            return float(result.stdout.strip())
        else:
            logger.error(f"Evaluation subprocess failed: {result.stderr}")
            return None
    except Exception as e:
        logger.error(f"Error calling evaluation: {e}", exc_info=True)
        return None

