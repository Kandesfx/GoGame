"""
Script test để kiểm tra ML model hoạt động trong game.

Test các chức năng:
1. Load model
2. Predict move từ board state
3. Test với nhiều board states khác nhau
"""

import sys
from pathlib import Path

# Thêm paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "backend" / "app" / "services"))
sys.path.insert(0, str(project_root / "src" / "ml"))

print("=" * 70)
print("🎮 TEST ML MODEL TRONG GAME")
print("=" * 70)

try:
    from ml_model_service import MLModelService, get_ml_model_service
    print("✅ ML model service imported successfully\n")
except ImportError as e:
    print(f"❌ Failed to import ML model service: {e}")
    sys.exit(1)

# Test 1: Load model
print("📦 TEST 1: Load Model")
print("-" * 70)
checkpoint_path = project_root / "checkpoints" / "final_model.pt"

if not checkpoint_path.exists():
    print(f"❌ Checkpoint not found: {checkpoint_path}")
    print(f"   Hãy đảm bảo file final_model.pt đã được đặt trong thư mục checkpoints/")
    sys.exit(1)

print(f"✅ Checkpoint found: {checkpoint_path}")

try:
    ml_service = MLModelService(str(checkpoint_path), device='cpu')
    
    if ml_service.is_loaded():
        print(f"✅ Model loaded successfully!")
        print(f"   Board size: {ml_service.board_size}")
        print(f"   Device: {ml_service.device}")
    else:
        print("❌ Model not loaded")
        sys.exit(1)
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 2: Predict với board state đơn giản
print("🎯 TEST 2: Predict Move - Board State Đơn Giản")
print("-" * 70)

board_position_1 = {
    "4,4": "B",  # Black ở center
    "3,4": "W",  # White bên cạnh
    "4,3": "W",  # White bên cạnh
}
current_player_1 = "B"
move_history_1 = [(4, 4), (3, 4)]

print(f"Board state:")
print(f"  Black: {[k for k, v in board_position_1.items() if v == 'B']}")
print(f"  White: {[k for k, v in board_position_1.items() if v == 'W']}")
print(f"Current player: {current_player_1}")
print(f"Move history: {move_history_1}")

try:
    best_move, policy_prob, win_prob = ml_service.predict_move(
        board_position_1, current_player_1, move_history_1
    )
    
    if best_move:
        x, y = best_move
        print(f"\n✅ Prediction successful!")
        print(f"   Best move: ({x}, {y})")
        print(f"   Policy probability: {policy_prob:.4f} ({policy_prob*100:.2f}%)")
        print(f"   Win probability: {win_prob:.4f} ({win_prob*100:.2f}%)")
    else:
        print("\n⚠️  No move predicted")
except Exception as e:
    print(f"\n❌ Prediction failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 3: Predict với board state phức tạp hơn
print("🎯 TEST 3: Predict Move - Board State Phức Tạp")
print("-" * 70)

board_position_2 = {
    # Một số quân cờ đã đặt
    "3,3": "B", "4,3": "B", "5,3": "B",
    "3,4": "W", "4,4": "W", "5,4": "W",
    "3,5": "B", "4,5": "B",
    "6,6": "W",
}
current_player_2 = "B"
move_history_2 = [(4, 4), (3, 3), (5, 4), (4, 5)]

print(f"Board state (phức tạp hơn):")
print(f"  Black: {[k for k, v in board_position_2.items() if v == 'B']}")
print(f"  White: {[k for k, v in board_position_2.items() if v == 'W']}")
print(f"Current player: {current_player_2}")
print(f"Move history: {move_history_2}")

try:
    best_move, policy_prob, win_prob = ml_service.predict_move(
        board_position_2, current_player_2, move_history_2
    )
    
    if best_move:
        x, y = best_move
        print(f"\n✅ Prediction successful!")
        print(f"   Best move: ({x}, {y})")
        print(f"   Policy probability: {policy_prob:.4f} ({policy_prob*100:.2f}%)")
        print(f"   Win probability: {win_prob:.4f} ({win_prob*100:.2f}%)")
    else:
        print("\n⚠️  No move predicted")
except Exception as e:
    print(f"\n❌ Prediction failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 4: Predict với board trống (đầu game)
print("🎯 TEST 4: Predict Move - Board Trống (Đầu Game)")
print("-" * 70)

board_position_3 = {}  # Board trống
current_player_3 = "B"  # Black đi trước
move_history_3 = []

print(f"Board state: Trống (đầu game)")
print(f"Current player: {current_player_3} (Black đi trước)")

try:
    best_move, policy_prob, win_prob = ml_service.predict_move(
        board_position_3, current_player_3, move_history_3
    )
    
    if best_move:
        x, y = best_move
        print(f"\n✅ Prediction successful!")
        print(f"   Best move: ({x}, {y})")
        print(f"   Policy probability: {policy_prob:.4f} ({policy_prob*100:.2f}%)")
        print(f"   Win probability: {win_prob:.4f} ({win_prob*100:.2f}%)")
        print(f"\n💡 Model khuyến nghị nước đi đầu tiên ở ({x}, {y})")
    else:
        print("\n⚠️  No move predicted")
except Exception as e:
    print(f"\n❌ Prediction failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 5: Test với White player
print("🎯 TEST 5: Predict Move - White Player")
print("-" * 70)

board_position_4 = {
    "4,4": "B",  # Black đã đánh
}
current_player_4 = "W"  # White turn
move_history_4 = [(4, 4)]

print(f"Board state:")
print(f"  Black: {[k for k, v in board_position_4.items() if v == 'B']}")
print(f"Current player: {current_player_4} (White turn)")

try:
    best_move, policy_prob, win_prob = ml_service.predict_move(
        board_position_4, current_player_4, move_history_4
    )
    
    if best_move:
        x, y = best_move
        print(f"\n✅ Prediction successful!")
        print(f"   Best move: ({x}, {y})")
        print(f"   Policy probability: {policy_prob:.4f} ({policy_prob*100:.2f}%)")
        print(f"   Win probability: {win_prob:.4f} ({win_prob*100:.2f}%)")
    else:
        print("\n⚠️  No move predicted")
except Exception as e:
    print(f"\n❌ Prediction failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 6: Performance test
print("⚡ TEST 6: Performance Test")
print("-" * 70)

import time

board_position_perf = {
    "4,4": "B",
    "3,4": "W",
    "4,3": "W",
}
current_player_perf = "B"

print("Running 10 predictions to test performance...")
start_time = time.time()

for i in range(10):
    try:
        best_move, _, _ = ml_service.predict_move(
            board_position_perf, current_player_perf, None
        )
    except Exception as e:
        print(f"Error in prediction {i+1}: {e}")
        break

end_time = time.time()
avg_time = (end_time - start_time) / 10

print(f"\n✅ Performance test completed!")
print(f"   Average time per prediction: {avg_time*1000:.2f} ms")
print(f"   Total time for 10 predictions: {(end_time - start_time)*1000:.2f} ms")

if avg_time < 0.5:
    print(f"   ⚡ Performance: Tốt (< 500ms)")
elif avg_time < 1.0:
    print(f"   ⚡ Performance: Chấp nhận được (< 1s)")
else:
    print(f"   ⚠️  Performance: Hơi chậm (> 1s)")

print()

# Summary
print("=" * 70)
print("📊 TÓM TẮT")
print("=" * 70)
print(f"✅ Model đã được load thành công")
print(f"✅ Board size: {ml_service.board_size}")
print(f"✅ Device: {ml_service.device}")
print(f"✅ Model sẵn sàng sử dụng trong game!")
print()
print("💡 Để test trong game thực tế:")
print("   1. Khởi động backend server")
print("   2. Tạo AI match mới")
print("   3. AI sẽ tự động sử dụng ML model để đánh")
print("   4. Kiểm tra logs để xem ML model có được sử dụng không")
print("=" * 70)

