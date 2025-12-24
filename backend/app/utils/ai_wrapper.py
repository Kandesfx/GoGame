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


def _check_gogame_py_module(build_dir: Path) -> bool:
    """Kiểm tra xem gogame_py module có tồn tại trong build directory không."""
    # Tìm tất cả file .pyd trong build directory
    pyd_files = list(build_dir.glob("gogame_py*.pyd"))
    if pyd_files:
        logger.debug(f"Found gogame_py module in build directory: {pyd_files[0]}")
        return True
    
    logger.warning(f"gogame_py module not found in build directory: {build_dir}")
    return False


def call_ai_select_move(board_state: Dict[str, Any], level: int) -> Optional[Dict[str, Any]]:
    """Gọi AI để chọn move qua MSYS2 Python subprocess.
    
    Args:
        board_state: Board state dict với moves, current_player, etc.
        level: AI level (1-6)
    
    Returns:
        Dict với move info hoặc None nếu fail
    """
    # Level 3-4: Dùng Minimax depth 3-4 (nhanh hơn MCTS)
    # Nếu ML available, có thể dùng PolicyNet trực tiếp (nhanh nhất)
    if level in [3, 4]:
        try:
            from .mcts_ml_wrapper import select_move_mcts_ml
            result = select_move_mcts_ml(board_state, level)
            if result:
                return result
        except ImportError as e:
            logger.warning(f"MCTS+ML wrapper not available: {e}, falling back to C++ MCTS")
        except Exception as e:
            logger.error(f"Error in MCTS+ML search: {e}", exc_info=True)
            # Fallback to C++ MCTS
            logger.info("Falling back to C++ MCTS")
    
    # Level 1-2: Dùng C++ AI (Minimax)
    if not _check_msys2_python():
        logger.error(f"❌ MSYS2 Python not found at {MSYS2_PYTHON}, AI features disabled")
        logger.error(f"❌ To enable AI features, install MSYS2 and build gogame_py module")
        return None
    
    # QUAN TRỌNG: Kiểm tra xem gogame_py module có tồn tại trong build directory không
    build_dir = PROJECT_ROOT / "build"
    if not _check_gogame_py_module(build_dir):
        logger.error(f"❌ gogame_py module not found in build directory: {build_dir}")
        logger.error(f"❌ Please build the module first: cmake --build build --target gogame_py")
        return None
    
    # Tính timeout động dựa trên level, board size và số nước đã đánh
    board_size = board_state.get('board_size', 9)
    num_moves = len(board_state.get('moves', []))
    
    # Timeout calculation (tối ưu để AI nghĩ nhanh):
    # Level 1-4 (Minimax): Dùng Minimax cho tất cả levels (nhanh hơn MCTS)
    # Board size lớn hơn và nhiều nước hơn cần thêm thời gian
    base_timeout_map = {
        1: 5,    # Minimax depth 1 - rất nhanh
        2: 8,    # Minimax depth 2 - nhanh
        3: 15 if board_size >= 19 else 10 if board_size >= 13 else 6,  # Minimax depth 2 - tăng timeout cho board lớn
        4: 20 if board_size >= 19 else 15 if board_size >= 13 else 10,  # Minimax depth 3 - tăng timeout cho board lớn
    }
    
    base_timeout = base_timeout_map.get(level, 30)
    
    # Điều chỉnh timeout dựa trên số nước đã đánh
    # Nhiều nước hơn = nhiều legal moves hơn = cần nhiều thời gian hơn
    if num_moves > 20:
        # Sau 20 nước, tăng timeout thêm 50%
        timeout = int(base_timeout * 1.5)
    elif num_moves > 10:
        # Sau 10 nước, tăng timeout thêm 25%
        timeout = int(base_timeout * 1.25)
    else:
        timeout = base_timeout
    logger.info(f"🤖 [WRAPPER] AI level {level}, board size {board_size}x{board_size}, timeout: {timeout}s")
    
    # Tạo script để chạy AI
    # Chuyển đổi JSON null thành None trong Python
    moves_json = json.dumps(board_state.get('moves', []))
    moves_json = moves_json.replace('null', 'None')  # Chuyển null thành None
    
    script_content = f"""
import sys
import os
import json
from pathlib import Path

# QUAN TRỌNG: Chỉ import từ build directory, không import từ root hoặc venv
build_dir = Path(r"{build_dir}")

# Kiểm tra xem module có tồn tại trong build directory không
pyd_files = [f for f in os.listdir(str(build_dir)) if f.startswith('gogame_py') and f.endswith('.pyd')]
if not pyd_files:
    error_msg = json.dumps({{"error": "gogame_py module not found in build directory", "build_dir": str(build_dir)}})
    print(error_msg, file=sys.stderr)
    sys.exit(1)

# Chỉ thêm build directory vào path, không thêm root
# Xóa các path khác có thể chứa gogame_py cũ
original_path = sys.path.copy()
sys.path = [str(build_dir)]  # Chỉ dùng build directory

try:
    import gogame_py
    # Kiểm tra xem module có thực sự từ build directory không
    module_file = getattr(gogame_py, '__file__', None)
    if module_file:
        module_path = Path(module_file).resolve()
        build_path = build_dir.resolve()
        if build_path not in module_path.parents and module_path.parent != build_path:
            error_msg = json.dumps({{"error": f"gogame_py module loaded from wrong location: {{module_path}} (expected in {{build_path}})"}})
            print(error_msg, file=sys.stderr)
            sys.exit(1)
except ImportError as e:
    error_msg = json.dumps({{"error": f"Failed to import gogame_py from build directory: {{e}}"}})
    print(error_msg, file=sys.stderr)
    sys.exit(1)

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

# Select AI move (SAU khi replay xong tất cả moves)
ai = gogame_py.AIPlayer()
# Level 3-4 dùng Minimax depth 2-3 (đã được config trong C++)
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
            try:
                move_data = json.loads(result.stdout.strip())
                # Kiểm tra xem có error không
                if "error" in move_data:
                    logger.error(f"AI subprocess returned error: {move_data['error']}")
                    return None
                return move_data
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse AI response as JSON: {result.stdout}")
                return None
        else:
            logger.error(f"AI subprocess failed (returncode={result.returncode}): {result.stderr}")
            if "gogame_py module not found" in result.stderr:
                logger.error(f"❌ gogame_py module not found in build directory. Please build it first.")
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
    
    # Kiểm tra xem gogame_py module có tồn tại không
    build_dir = PROJECT_ROOT / "build"
    if not _check_gogame_py_module(build_dir):
        logger.error(f"❌ gogame_py module not found in build directory: {build_dir}")
        return None
    
    script_content = f"""
import sys
from pathlib import Path
build_dir = Path(r"{build_dir}")
sys.path.insert(0, str(build_dir))

# Kiểm tra xem module có tồn tại không
import os
pyd_files = [f for f in os.listdir(str(build_dir)) if f.startswith('gogame_py') and f.endswith('.pyd')]
if not pyd_files:
    print("ERROR: gogame_py module not found", file=sys.stderr)
    sys.exit(1)

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
