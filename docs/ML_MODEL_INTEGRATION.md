# 🤖 TÍCH HỢP ML MODEL VÀO GAME

## ✅ Đã Hoàn Thành

Model đã được tích hợp thành công vào game! Bây giờ game sẽ sử dụng ML model để chơi AI thay vì chỉ dùng MCTS/minimax.

## 📁 Cấu Trúc Tích Hợp

```
backend/app/services/
├── ml_model_service.py      ← Service mới để load và sử dụng ML model
└── match_service.py         ← Đã được cập nhật để sử dụng ML model

checkpoints/
└── final_model.pt           ← Model đã train (đặt ở đây)
```

## 🔄 Cách Hoạt Động

1. **Khi AI cần đánh**: `MatchService._make_ai_move()` sẽ:
   - Thử sử dụng ML model trước (nếu có)
   - Fallback về MCTS/minimax nếu ML model không available

2. **ML Model Service**:
   - Tự động load model từ `checkpoints/final_model.pt`
   - Convert board state sang 17-plane features
   - Predict move và win probability
   - Trả về move tốt nhất

## 🎮 Sử Dụng

### Tự động

Model sẽ tự động được sử dụng khi:
- Tạo AI match mới
- AI cần đánh nước đi

Không cần cấu hình gì thêm!

### Kiểm tra Model

Chạy script test để kiểm tra model:

```bash
python scripts/test_ml_integration.py
```

### Load Model Thủ Công

Nếu muốn load model với checkpoint khác:

```python
from backend.app.services.ml_model_service import MLModelService

# Load model
ml_service = MLModelService(
    checkpoint_path='checkpoints/best_model.pt',  # hoặc đường dẫn khác
    device='cpu'  # hoặc 'cuda' nếu có GPU
)

# Predict move
best_move, policy_prob, win_prob = ml_service.predict_move(
    board_position={'4,4': 'B', '3,4': 'W'},
    current_player='B',
    move_history=[(4, 4), (3, 4)]
)
```

## 🔧 Cấu Hình

### Thay Đổi Checkpoint

Mặc định, service sẽ tìm `checkpoints/final_model.pt`. Nếu muốn dùng checkpoint khác:

1. Đặt file vào `checkpoints/`
2. Hoặc chỉ định đường dẫn khi tạo service:

```python
ml_service = MLModelService(
    checkpoint_path='checkpoints/best_model.pt'
)
```

### Device (CPU/GPU)

Mặc định dùng CPU. Nếu có GPU:

```python
ml_service = MLModelService(device='cuda')
```

## 📊 Logs

Khi ML model được sử dụng, bạn sẽ thấy logs:

```
✅ ML model AI move successful
🤖 ML model AI move: (4, 5), prob=0.4196, win_prob=0.5000
```

## ⚠️ Lưu Ý

1. **Model format**: Model đã được compile với `torch.compile()` nên có prefix `_orig_mod.` - code đã xử lý tự động

2. **Board size**: Model hiện tại được train cho board size 19. Nếu dùng board size khác, cần train model mới.

3. **Fallback**: Nếu ML model không available hoặc có lỗi, game sẽ tự động fallback về MCTS/minimax.

4. **Performance**: 
   - CPU: ~100-500ms per move
   - GPU: ~10-50ms per move (nếu có)

## 🐛 Troubleshooting

### Model không load được

```python
# Kiểm tra file có tồn tại không
from pathlib import Path
checkpoint_path = Path('checkpoints/final_model.pt')
print(f"Exists: {checkpoint_path.exists()}")
```

### Lỗi import

Đảm bảo đã cài đặt:
```bash
pip install torch torchvision torchaudio
```

### Model trả về move không hợp lệ

Model có thể trả về move không hợp lệ (ví dụ: đã có quân cờ). Code đã xử lý bằng cách:
- Validate move trước khi apply
- Fallback về pass nếu move không hợp lệ

## 📚 Xem Thêm

- **Hướng dẫn sử dụng model**: `docs/HUONG_DAN_SU_DUNG_MODEL.md`
- **Training guide**: `scripts/README_COLAB_TRAINING.md`
- **ML model service code**: `backend/app/services/ml_model_service.py`

---

**Model đã sẵn sàng sử dụng! 🎉**

