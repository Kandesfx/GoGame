"""
Test script để verify labels cho Multi-task Model đúng format tài liệu.

Chạy: python scripts/test_multi_task_labels.py
"""

import sys
from pathlib import Path
import numpy as np

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import torch
except ImportError:
    print("⚠️  torch not installed. Install with: pip install torch")
    sys.exit(1)

from label_generators import (
    ThreatLabelGenerator,
    AttackLabelGenerator,
    IntentLabelGenerator,
    EvaluationLabelGenerator
)
from generate_features_colab import board_to_features_17_planes


def create_test_board(board_size=19):
    """Tạo test board với một số quân cờ"""
    board = np.zeros((board_size, board_size), dtype=np.int8)
    
    # Đặt một số quân cờ
    board[3, 3] = 1  # Black
    board[3, 4] = 1  # Black
    board[4, 3] = 1  # Black
    board[4, 4] = 2  # White (bị bao quanh - atari)
    
    board[10, 10] = 2  # White
    board[10, 11] = 2  # White
    board[11, 10] = 1  # Black (có thể cắt)
    
    return board


def test_threat_label_generator():
    """Test ThreatLabelGenerator"""
    print("🧪 Test 1: ThreatLabelGenerator")
    
    board_size = 19
    board = create_test_board(board_size)
    current_player = 'B'
    
    gen = ThreatLabelGenerator(board_size)
    threat_map = gen.generate_threat_map(board, current_player)
    
    # Verify format
    assert isinstance(threat_map, torch.Tensor), "threat_map must be Tensor"
    assert threat_map.shape == (board_size, board_size), \
        f"Expected shape ({board_size}, {board_size}), got {threat_map.shape}"
    assert threat_map.dtype == torch.float32, "Must be float32"
    assert (threat_map >= 0.0).all() and (threat_map <= 1.0).all(), \
        "Values must be in range [0.0, 1.0]"
    
    print(f"   ✅ PASS: Threat map shape {threat_map.shape}, range [{threat_map.min():.2f}, {threat_map.max():.2f}]")


def test_attack_label_generator():
    """Test AttackLabelGenerator"""
    print("🧪 Test 2: AttackLabelGenerator")
    
    board_size = 19
    board = create_test_board(board_size)
    current_player = 'B'
    
    gen = AttackLabelGenerator(board_size)
    attack_map = gen.generate_attack_map(board, current_player)
    
    # Verify format
    assert isinstance(attack_map, torch.Tensor), "attack_map must be Tensor"
    assert attack_map.shape == (board_size, board_size), \
        f"Expected shape ({board_size}, {board_size}), got {attack_map.shape}"
    assert attack_map.dtype == torch.float32, "Must be float32"
    assert (attack_map >= 0.0).all() and (attack_map <= 1.0).all(), \
        "Values must be in range [0.0, 1.0]"
    
    print(f"   ✅ PASS: Attack map shape {attack_map.shape}, range [{attack_map.min():.2f}, {attack_map.max():.2f}]")


def test_intent_label_generator():
    """Test IntentLabelGenerator"""
    print("🧪 Test 3: IntentLabelGenerator")
    
    board_size = 19
    board = create_test_board(board_size)
    current_player = 'B'
    move = (5, 5)
    prev_moves = [(3, 3), (4, 4)]
    
    gen = IntentLabelGenerator(board_size)
    intent_label = gen.generate_intent_label(board, move, prev_moves, current_player)
    
    # Verify format
    assert isinstance(intent_label, dict), "intent_label must be dict"
    assert 'type' in intent_label, "Must have 'type' key"
    assert 'confidence' in intent_label, "Must have 'confidence' key"
    assert 'region' in intent_label, "Must have 'region' key"
    
    assert intent_label['type'] in ['territory', 'attack', 'defense', 'connection', 'cut'], \
        f"Invalid intent type: {intent_label['type']}"
    assert 0.0 <= intent_label['confidence'] <= 1.0, \
        f"Confidence must be in [0.0, 1.0], got {intent_label['confidence']}"
    assert isinstance(intent_label['region'], list), "region must be list"
    
    print(f"   ✅ PASS: Intent label type={intent_label['type']}, confidence={intent_label['confidence']:.2f}")


def test_evaluation_label_generator():
    """Test EvaluationLabelGenerator"""
    print("🧪 Test 4: EvaluationLabelGenerator")
    
    board_size = 19
    board = create_test_board(board_size)
    current_player = 'B'
    winner = 'B'
    game_result = 'B+12.5'
    
    gen = EvaluationLabelGenerator(board_size)
    eval_label = gen.generate_evaluation(board, current_player, winner, game_result)
    
    # Verify format
    assert isinstance(eval_label, dict), "evaluation_label must be dict"
    assert 'win_probability' in eval_label, "Must have 'win_probability' key"
    assert 'territory_map' in eval_label, "Must have 'territory_map' key"
    assert 'influence_map' in eval_label, "Must have 'influence_map' key"
    
    assert 0.0 <= eval_label['win_probability'] <= 1.0, \
        f"win_probability must be in [0.0, 1.0], got {eval_label['win_probability']}"
    
    assert isinstance(eval_label['territory_map'], torch.Tensor), \
        "territory_map must be Tensor"
    assert eval_label['territory_map'].shape == (board_size, board_size), \
        f"territory_map shape incorrect: {eval_label['territory_map'].shape}"
    
    assert isinstance(eval_label['influence_map'], torch.Tensor), \
        "influence_map must be Tensor"
    assert eval_label['influence_map'].shape == (board_size, board_size), \
        f"influence_map shape incorrect: {eval_label['influence_map'].shape}"
    
    print(f"   ✅ PASS: Evaluation label win_prob={eval_label['win_probability']:.2f}")


def test_full_label_format():
    """Test format đầy đủ theo tài liệu"""
    print("🧪 Test 5: Full label format (theo ML_COMPREHENSIVE_GUIDE.md)")
    
    board_size = 19
    board = create_test_board(board_size)
    current_player = 'B'
    move = (5, 5)
    winner = 'B'
    game_result = 'B+12.5'
    move_history = [(3, 3), (4, 4)]
    
    # Generate features
    features = board_to_features_17_planes(
        board, current_player, move_history=move_history, board_size=board_size
    )
    
    # Generate all labels
    threat_gen = ThreatLabelGenerator(board_size)
    attack_gen = AttackLabelGenerator(board_size)
    intent_gen = IntentLabelGenerator(board_size)
    eval_gen = EvaluationLabelGenerator(board_size)
    
    threat_map = threat_gen.generate_threat_map(board, current_player)
    attack_map = attack_gen.generate_attack_map(board, current_player)
    intent_label = intent_gen.generate_intent_label(board, move, move_history, current_player)
    evaluation_label = eval_gen.generate_evaluation(board, current_player, winner, game_result)
    
    # Create labeled sample theo format tài liệu
    labeled_sample = {
        'features': features,
        'labels': {
            'threat_map': threat_map,
            'attack_map': attack_map,
            'intent': intent_label,
            'evaluation': evaluation_label
        },
        'metadata': {
            'board_size': board_size,
            'current_player': current_player,
            'move_number': 10,
            'game_result': game_result,
            'winner': winner
        }
    }
    
    # Verify format theo tài liệu (dòng 379-410)
    assert 'features' in labeled_sample, "Must have 'features'"
    assert labeled_sample['features'].shape == (17, board_size, board_size), \
        f"Features shape incorrect: {labeled_sample['features'].shape}"
    
    assert 'labels' in labeled_sample, "Must have 'labels'"
    labels = labeled_sample['labels']
    
    assert 'threat_map' in labels, "Must have 'threat_map' in labels"
    assert labels['threat_map'].shape == (board_size, board_size), \
        f"threat_map shape incorrect: {labels['threat_map'].shape}"
    
    assert 'attack_map' in labels, "Must have 'attack_map' in labels"
    assert labels['attack_map'].shape == (board_size, board_size), \
        f"attack_map shape incorrect: {labels['attack_map'].shape}"
    
    assert 'intent' in labels, "Must have 'intent' in labels"
    assert isinstance(labels['intent'], dict), "intent must be dict"
    assert 'type' in labels['intent'], "intent must have 'type'"
    assert 'confidence' in labels['intent'], "intent must have 'confidence'"
    assert 'region' in labels['intent'], "intent must have 'region'"
    
    assert 'evaluation' in labels, "Must have 'evaluation' in labels"
    assert isinstance(labels['evaluation'], dict), "evaluation must be dict"
    assert 'win_probability' in labels['evaluation'], "evaluation must have 'win_probability'"
    assert 'territory_map' in labels['evaluation'], "evaluation must have 'territory_map'"
    assert 'influence_map' in labels['evaluation'], "evaluation must have 'influence_map'"
    
    assert 'metadata' in labeled_sample, "Must have 'metadata'"
    
    print("   ✅ PASS: Full label format đúng với tài liệu ML_COMPREHENSIVE_GUIDE.md")
    print(f"      - Features: {labeled_sample['features'].shape}")
    print(f"      - Threat map: {labels['threat_map'].shape}")
    print(f"      - Attack map: {labels['attack_map'].shape}")
    print(f"      - Intent: {labels['intent']['type']} (confidence: {labels['intent']['confidence']:.2f})")
    print(f"      - Evaluation: win_prob={labels['evaluation']['win_probability']:.2f}")


def test_pass_move_handling():
    """Test xử lý pass moves"""
    print("🧪 Test 6: Pass move handling")
    
    board_size = 19
    board = create_test_board(board_size)
    current_player = 'B'
    move = None  # Pass move
    prev_moves = []
    
    intent_gen = IntentLabelGenerator(board_size)
    intent_label = intent_gen.generate_intent_label(board, move, prev_moves, current_player)
    
    # Pass move thường là defense
    assert intent_label['type'] in ['defense', 'territory'], \
        f"Pass move intent should be defense/territory, got {intent_label['type']}"
    
    print(f"   ✅ PASS: Pass move handled correctly (intent: {intent_label['type']})")


def main():
    """Run all tests"""
    print("=" * 70)
    print("🧪 TESTING MULTI-TASK MODEL LABELS")
    print("=" * 70)
    print()
    
    tests = [
        test_threat_label_generator,
        test_attack_label_generator,
        test_intent_label_generator,
        test_evaluation_label_generator,
        test_full_label_format,
        test_pass_move_handling,
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
    
    print("=" * 70)
    print(f"📊 RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    
    if failed == 0:
        print("✅ All tests passed! Labels format đúng với tài liệu.")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())

