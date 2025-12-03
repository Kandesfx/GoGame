"""
Test script để verify các sửa đổi labeling.

Chạy: python scripts/test_labeling_fixes.py
"""

import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
from generate_features_colab import (
    generate_policy_label,
    generate_value_label,
    board_to_features_17_planes
)


def test_policy_label_normal_move():
    """Test policy label cho normal move"""
    print("🧪 Test 1: Policy label cho normal move")
    board_size = 19
    move = (5, 5)
    
    policy = generate_policy_label(move, board_size)
    
    assert policy.shape == (board_size * board_size + 1,), \
        f"Expected shape {(board_size * board_size + 1,)}, got {policy.shape}"
    
    expected_idx = 5 * board_size + 5
    assert policy[expected_idx] == 1.0, \
        f"Expected policy[{expected_idx}] = 1.0, got {policy[expected_idx]}"
    
    assert policy[-1] == 0.0, \
        f"Pass move index should be 0.0 for normal move, got {policy[-1]}"
    
    assert policy.sum() == 1.0, \
        f"Policy should sum to 1.0, got {policy.sum()}"
    
    print("   ✅ PASS: Normal move policy label correct")


def test_policy_label_pass_move():
    """Test policy label cho pass move"""
    print("🧪 Test 2: Policy label cho pass move")
    board_size = 19
    
    # Test None
    policy1 = generate_policy_label(None, board_size)
    assert policy1.shape == (board_size * board_size + 1,)
    assert policy1[-1] == 1.0, "Pass move (None) should set last index to 1.0"
    assert policy1[:-1].sum() == 0.0, "All board positions should be 0"
    
    # Test (-1, -1)
    policy2 = generate_policy_label((-1, -1), board_size)
    assert policy2[-1] == 1.0, "Pass move (-1, -1) should set last index to 1.0"
    
    print("   ✅ PASS: Pass move policy label correct")


def test_policy_label_invalid_move():
    """Test policy label cho invalid move (outside board)"""
    print("🧪 Test 3: Policy label cho invalid move (outside board)")
    board_size = 19
    move = (20, 20)  # Outside board
    
    # Should treat as pass move (with warning)
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        policy = generate_policy_label(move, board_size)
        
        # Should have warning
        assert len(w) > 0, "Should have warning for invalid move"
        assert "outside board" in str(w[0].message).lower()
    
    # Should be treated as pass
    assert policy[-1] == 1.0, "Invalid move should be treated as pass"
    
    print("   ✅ PASS: Invalid move handled correctly (treated as pass)")


def test_value_label_validation():
    """Test value label validation"""
    print("🧪 Test 4: Value label validation")
    
    # Test valid cases
    assert generate_value_label('B', 'B') == 1.0, "Black wins, Black to move = 1.0"
    assert generate_value_label('W', 'W') == 1.0, "White wins, White to move = 1.0"
    assert generate_value_label('B', 'W') == 0.0, "Black wins, White to move = 0.0"
    assert generate_value_label('W', 'B') == 0.0, "White wins, Black to move = 0.0"
    assert generate_value_label('DRAW', 'B') == 0.5, "Draw = 0.5"
    assert generate_value_label(None, 'B') == 0.5, "Unknown = 0.5"
    
    # Test case insensitive
    assert generate_value_label('b', 'b') == 1.0, "Case insensitive should work"
    assert generate_value_label('w', 'W') == 1.0, "Case insensitive should work"
    
    # Test invalid current_player
    try:
        generate_value_label('B', 'X')
        assert False, "Should raise ValueError for invalid current_player"
    except ValueError as e:
        assert 'current_player' in str(e).lower()
        print("   ✅ PASS: Invalid current_player raises ValueError")
    
    # Test with game_result
    value = generate_value_label('INVALID', 'B', game_result='B+12.5')
    assert value == 1.0, "Should parse from game_result"
    
    print("   ✅ PASS: Value label validation correct")


def test_value_label_range():
    """Test value label luôn trong range [0.0, 1.0]"""
    print("🧪 Test 5: Value label range validation")
    
    test_cases = [
        ('B', 'B', None),
        ('W', 'W', None),
        ('B', 'W', None),
        ('W', 'B', None),
        ('DRAW', 'B', None),
        (None, 'B', None),
        ('INVALID', 'B', 'B+12.5'),
        ('INVALID', 'B', 'W+5.5'),
    ]
    
    for winner, current_player, game_result in test_cases:
        try:
            value = generate_value_label(winner, current_player, game_result)
            assert 0.0 <= value <= 1.0, \
                f"Value {value} not in range [0.0, 1.0] for winner={winner}, player={current_player}"
        except ValueError:
            # Expected for invalid current_player
            pass
    
    print("   ✅ PASS: All value labels in range [0.0, 1.0]")


def test_policy_label_shape_consistency():
    """Test policy label shape consistency cho các board sizes"""
    print("🧪 Test 6: Policy label shape consistency")
    
    board_sizes = [9, 13, 19]
    
    for board_size in board_sizes:
        # Normal move
        policy_normal = generate_policy_label((0, 0), board_size)
        assert policy_normal.shape == (board_size * board_size + 1,), \
            f"Board size {board_size}: Expected shape {(board_size * board_size + 1,)}, got {policy_normal.shape}"
        
        # Pass move
        policy_pass = generate_policy_label(None, board_size)
        assert policy_pass.shape == (board_size * board_size + 1,), \
            f"Board size {board_size}: Pass move shape incorrect"
    
    print("   ✅ PASS: Policy label shapes consistent across board sizes")


def test_integration():
    """Test integration với board_to_features"""
    print("🧪 Test 7: Integration test")
    
    board_size = 19
    board_state = np.zeros((board_size, board_size), dtype=np.int8)
    board_state[5, 5] = 1  # Black stone
    
    # Test với normal move
    features = board_to_features_17_planes(
        board_state,
        current_player='B',
        move_history=[(5, 5)],
        board_size=board_size
    )
    
    policy = generate_policy_label((5, 5), board_size)
    value = generate_value_label('B', 'B')
    
    assert features.shape == (17, board_size, board_size)
    assert policy.shape == (board_size * board_size + 1,)
    assert 0.0 <= value <= 1.0
    
    print("   ✅ PASS: Integration test passed")


def main():
    """Run all tests"""
    print("=" * 60)
    print("🧪 TESTING LABELING FIXES")
    print("=" * 60)
    print()
    
    tests = [
        test_policy_label_normal_move,
        test_policy_label_pass_move,
        test_policy_label_invalid_move,
        test_value_label_validation,
        test_value_label_range,
        test_policy_label_shape_consistency,
        test_integration,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"   ❌ FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()
    
    print("=" * 60)
    print(f"📊 RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())

