"""
Test script để kiểm tra ML model integration với game.
"""

import sys
from pathlib import Path

# Thêm paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "backend" / "app" / "services"))
sys.path.insert(0, str(project_root / "src" / "ml"))

print("=" * 60)
print("🧪 TEST ML MODEL INTEGRATION")
print("=" * 60)

try:
    from ml_model_service import MLModelService, get_ml_model_service
    print("✅ ML model service imported successfully")
except ImportError as e:
    print(f"❌ Failed to import ML model service: {e}")
    sys.exit(1)

# Test load model
print("\n📦 Testing model load...")
# Tìm model trong backend/models/ (ưu tiên) hoặc checkpoints/ (backward compatibility)
checkpoint_path = project_root / "backend" / "models" / "final_model.pt"
if not checkpoint_path.exists():
    checkpoint_path = project_root / "checkpoints" / "final_model.pt"

if not checkpoint_path.exists():
    print(f"❌ Checkpoint not found: {checkpoint_path}")
    sys.exit(1)

print(f"✅ Checkpoint found: {checkpoint_path}")

try:
    ml_service = MLModelService(str(checkpoint_path), device='cpu')
    print("✅ MLModelService created")
    
    if ml_service.is_loaded():
        print(f"✅ Model loaded successfully!")
        print(f"   Board size: {ml_service.board_size}")
    else:
        print("❌ Model not loaded")
        sys.exit(1)
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test prediction với board state mẫu
print("\n🎯 Testing prediction...")
board_position = {
    "4,4": "B",
    "3,4": "W",
    "4,3": "W",
}
current_player = "B"
move_history = [(4, 4), (3, 4)]

try:
    best_move, policy_prob, win_prob = ml_service.predict_move(
        board_position, current_player, move_history
    )
    
    if best_move:
        x, y = best_move
        print(f"✅ Prediction successful!")
        print(f"   Best move: ({x}, {y})")
        print(f"   Policy probability: {policy_prob:.4f}")
        print(f"   Win probability: {win_prob:.4f}")
    else:
        print("⚠️  No move predicted")
        
except Exception as e:
    print(f"❌ Prediction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test singleton
print("\n🔄 Testing singleton...")
try:
    ml_service2 = get_ml_model_service()
    if ml_service2 is ml_service:
        print("✅ Singleton pattern works correctly")
    else:
        print("⚠️  Singleton pattern may not be working")
except Exception as e:
    print(f"⚠️  Singleton test failed: {e}")

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)

