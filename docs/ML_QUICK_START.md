# 🚀 ML Training Quick Start Guide

Hướng dẫn nhanh để bắt đầu với ML training cho GoGame.

## 📋 Prerequisites

1. **Python 3.10+** với các packages:
   ```bash
   pip install torch torchvision numpy
   ```

2. **gogame_py module** đã được build (C++ bindings)

3. **Dữ liệu training** (sẽ được generate tự động)

## 🎯 Bước 1: Test Model Architecture

Kiểm tra xem các model components có hoạt động không:

```bash
# Test shared backbone
python src/ml/models/shared_backbone.py

# Test threat head
python src/ml/models/threat_head.py

# Test attack head
python src/ml/models/attack_head.py

# Test intent head
python src/ml/models/intent_head.py

# Test full multi-task model
python src/ml/models/multi_task_model.py
```

Tất cả các tests nên pass và in ra thông tin về model (số parameters, output shapes, etc.)

## 📊 Bước 2: Collect Training Data

Generate self-play games để tạo training data:

```bash
python src/ml/training/data_collector.py
```

Script này sẽ:
- Generate 50 self-play games (có thể chỉnh trong code)
- Extract training samples từ mỗi game
- Lưu vào `data/training/self_play_9x9_50games.pt`

**Lưu ý**: Có thể mất vài phút đến vài giờ tùy vào số lượng games.

## 🏋️ Bước 3: Training (Coming Soon)

Training script sẽ được implement trong Phase 3 của roadmap. Hiện tại bạn có thể:

1. Review architecture trong `src/ml/models/`
2. Collect more data với `data_collector.py`
3. Experiment với model architecture

## 🔍 Bước 4: Test Inference (Khi có trained model)

Khi đã có trained model, test inference:

```python
from backend.app.services.ml_analysis_service import MLAnalysisService
import gogame_py as go

# Load model
service = MLAnalysisService(model_path=Path("models/multi_task_model.pt"))

# Create test board
board = go.Board(9)
current_player = board.current_player()

# Analyze position
analysis = await service.analyze_position(board, current_player)

print(analysis)
```

## 📁 Cấu trúc Files

```
src/ml/
├── models/
│   ├── __init__.py
│   ├── shared_backbone.py      # ✅ Ready
│   ├── threat_head.py          # ✅ Ready
│   ├── attack_head.py          # ✅ Ready
│   ├── intent_head.py          # ✅ Ready
│   └── multi_task_model.py     # ✅ Ready
├── training/
│   ├── __init__.py
│   ├── data_collector.py        # ✅ Ready
│   ├── label_generator.py       # ⏳ TODO
│   ├── dataset.py               # ⏳ TODO
│   └── train_multi_task.py      # ⏳ TODO
└── inference/
    ├── analyzer.py              # ⏳ TODO
    └── postprocessor.py         # ⏳ TODO
```

## 🐛 Troubleshooting

### Lỗi: `gogame_py not found`
- Đảm bảo đã build C++ bindings
- Check `gogame_py.pyd` hoặc `.so` file tồn tại

### Lỗi: `CUDA out of memory`
- Model hiện tại chạy trên CPU
- Nếu có GPU, có thể set `device="cuda"` trong code

### Lỗi: Import errors
- Đảm bảo đang chạy từ project root
- Check Python path includes `src/`

## 📚 Next Steps

1. **Đọc chi tiết**: Xem `docs/ML_TRAINING_ROADMAP.md` để hiểu đầy đủ
2. **Collect more data**: Tăng số lượng games để có dataset lớn hơn
3. **Implement training**: Theo Phase 3 trong roadmap
4. **Experiment**: Thử nghiệm với architecture, hyperparameters

## 💡 Tips

- Bắt đầu với board size nhỏ (9x9) để test nhanh
- Collect ít nhất 1000 games để có dataset đủ lớn
- Monitor GPU/CPU usage khi training
- Save checkpoints thường xuyên

---

**Happy Training! 🎉**

