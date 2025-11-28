"""Test gogame_py với Python từ MSYS2 (tránh DLL conflicts)."""

import sys
import os
from pathlib import Path

# Fix encoding for MSYS2
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"

# Add build directory to path
PROJECT_ROOT = Path(__file__).parent.parent
BUILD_DIR = PROJECT_ROOT / "build"
sys.path.insert(0, str(BUILD_DIR))

print("=" * 60)
print("Testing gogame_py module")
print("=" * 60)
print(f"Python: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Build dir: {BUILD_DIR}")
print()

try:
    import gogame_py
    print("✅ Import thành công!")
    
    # Test Board
    print("\n--- Testing Board ---")
    board = gogame_py.Board(9)
    print(f"✅ Board created: size={board.size()}")
    print(f"   Current player: {board.current_player()}")
    
    # Test Move
    print("\n--- Testing Move ---")
    move = gogame_py.Move(3, 3, gogame_py.Color.Black)
    # x, y are properties, not methods
    print(f"✅ Move created: ({move.x}, {move.y})")
    print(f"   Is pass: {move.is_pass}")
    print(f"   Is valid: {move.is_valid}")
    
    # Test legal moves
    legal_moves = board.get_legal_moves(board.current_player())
    print(f"✅ Legal moves: {len(legal_moves)} moves available")
    
    # Test making a move
    if board.is_legal_move(move):
        board.make_move(move)
        print(f"✅ Move applied successfully")
        print(f"   Current player after move: {board.current_player()}")
        print(f"   Prisoners Black: {board.get_prisoners(gogame_py.Color.Black)}")
    
    # Test AIPlayer
    print("\n--- Testing AIPlayer ---")
    ai = gogame_py.AIPlayer()
    print("✅ AIPlayer created")
    
    # Test AI move selection
    print("\n--- Testing AI Move Selection ---")
    ai_move = ai.select_move(board, 1)  # level 1
    print(f"✅ AI selected move: ({ai_move.x}, {ai_move.y})")
    print(f"   Is pass: {ai_move.is_pass}")
    
    # Test MinimaxEngine
    print("\n--- Testing MinimaxEngine ---")
    from gogame_py import MinimaxConfig, MinimaxEngine
    
    config = MinimaxConfig()
    config.max_depth = 2
    config.use_alpha_beta = True
    engine = MinimaxEngine(config)
    
    result = engine.search(board, board.current_player())
    print(f"✅ Minimax search completed")
    print(f"   Best move: ({result.best_move.x}, {result.best_move.y})")
    print(f"   Evaluation: {result.evaluation}")
    print(f"   Nodes searched: {result.nodes_searched}")
    
    # Test MCTSEngine
    print("\n--- Testing MCTSEngine ---")
    from gogame_py import MCTSConfig, MCTSEngine
    
    mcts_config = MCTSConfig(num_playouts=100, time_limit_seconds=1.0)
    mcts_engine = MCTSEngine(mcts_config)
    
    mcts_result = mcts_engine.search(board, board.current_player())
    print(f"✅ MCTS search completed")
    print(f"   Best move: ({mcts_result.best_move.x}, {mcts_result.best_move.y})")
    print(f"   Win rate: {mcts_result.win_rate:.2%}")
    print(f"   Total visits: {mcts_result.total_visits}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
    print("\nModule sẵn sàng để sử dụng trong backend!")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("\n💡 Solutions:")
    print("  1. Đảm bảo đang chạy trong MSYS2 MinGW 64-bit shell")
    print("  2. Hoặc chạy: /c/msys64/mingw64/bin/python3 scripts/test_gogame_py_msys2.py")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

