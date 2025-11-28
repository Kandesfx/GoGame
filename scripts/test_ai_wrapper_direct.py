"""Test AI wrapper trực tiếp để debug."""

import sys
from pathlib import Path

# Add backend to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.utils.ai_wrapper import call_ai_select_move

print("=" * 60)
print("Testing AI Wrapper Directly")
print("=" * 60)

# Test board state
board_state = {
    "board_size": 9,
    "moves": [
        {"number": 1, "color": "B", "position": [3, 3]},
        {"number": 2, "color": "B", "position": [3, 4]},
    ],
    "current_player": "W",  # Should be White's turn
}

print(f"Board state: {board_state}")
print(f"Number of moves: {len(board_state['moves'])}")
print(f"Current player: {board_state['current_player']}")

print("\nCalling AI wrapper...")
result = call_ai_select_move(board_state, level=1)

if result:
    print(f"✅ AI move result: {result}")
else:
    print("❌ AI wrapper returned None")

