# 🚀 HƯỚNG DẪN TRAINING TRÊN COLAB

## 📋 Tổng Quan

Các script này được thiết kế để chạy trên Google Colab với GPU miễn phí.

## 📁 Cấu Trúc Scripts

```
scripts/
├── parse_sgf_colab.py              # Parse SGF files → positions
├── generate_features_colab.py      # Generate 17-plane features
├── generate_labels_colab.py        # Generate policy + value labels
├── train_colab.py                  # Training script hoàn chỉnh
└── colab_notebook_template.py      # Template notebook với tất cả cells
```

## 🎯 Workflow

### Bước 1: Parse SGF Files

```python
from parse_sgf_colab import process_sgf_directory
from pathlib import Path

WORK_DIR = Path('/content/drive/MyDrive/GoGame_ML')

process_sgf_directory(
    sgf_dir=WORK_DIR / 'raw_sgf',
    output_dir=WORK_DIR / 'processed',
    board_sizes=[9, 13, 19]
)
```

**Output:** `processed/positions_9x9.pt`, `processed/positions_13x13.pt`, ...

### Bước 2: Generate Labels

```python
from generate_labels_colab import process_dataset_file

process_dataset_file(
    input_path=WORK_DIR / 'processed' / 'positions_9x9.pt',
    output_path=WORK_DIR / 'datasets' / 'labeled_9x9.pt',
    filter_handicap=True
)
```

**Output:** `datasets/labeled_9x9.pt` với:
- `features`: Tensor [17, 9, 9]
- `policy`: Tensor [81]
- `value`: float

### Bước 3: Training

```python
from train_colab import train_model

train_model(
    train_dataset_path=str(WORK_DIR / 'datasets' / 'labeled_9x9.pt'),
    val_dataset_path=None,  # Auto-split
    board_size=9,
    batch_size=32,
    num_epochs=10,
    learning_rate=0.001,
    checkpoint_dir=str(WORK_DIR / 'checkpoints')
)
```

**Output:** 
- `checkpoints/best_model.pt` - Model tốt nhất
- `checkpoints/final_model.pt` - Model cuối cùng
- `checkpoints/checkpoint_epoch_X.pt` - Checkpoints định kỳ

## 📝 Sử Dụng Template Notebook

1. Mở `scripts/colab_notebook_template.py`
2. Copy từng cell vào Colab notebook
3. Chạy theo thứ tự từ Cell 1 → Cell 14

## 🔧 Requirements

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas tqdm sgf
```

## 📊 Dataset Format

### Input (từ parse_sgf_colab.py):

```python
{
    'positions': [
        {
            'board_state': np.ndarray,  # [board_size, board_size]
            'move': (x, y),
            'current_player': 'B' or 'W',
            'winner': 'B' or 'W' or None,
            'game_result': 'B+12.5',
            ...
        },
        ...
    ],
    'board_size': 9,
    'total': 80000
}
```

### Output (từ generate_labels_colab.py):

```python
{
    'labeled_data': [
        {
            'features': torch.Tensor,  # [17, board_size, board_size]
            'policy': torch.Tensor,    # [board_size * board_size]
            'value': float,            # 0.0 - 1.0
            'metadata': {...}
        },
        ...
    ],
    'board_size': 9,
    'total': 80000
}
```

## 🎓 Model Output

Sau khi training, bạn sẽ có:

1. **best_model.pt**: Model với validation loss thấp nhất
   ```python
   {
       'policy_net_state_dict': {...},
       'value_net_state_dict': {...},
       'policy_config': {...},
       'value_config': {...},
       'board_size': 9,
       'val_policy_loss': 0.5234,
       'val_value_loss': 0.1234
   }
   ```

2. **final_model.pt**: Model sau epoch cuối cùng

3. **checkpoint_epoch_X.pt**: Checkpoints để resume training

## 💡 Tips

1. **GPU Memory**: Nếu hết memory, giảm `batch_size` (32 → 16 → 8)
2. **Training Time**: 10 epochs cho 80K samples ≈ 2-3 giờ trên Colab T4
3. **Save Checkpoints**: Lưu thường xuyên để tránh mất dữ liệu khi session timeout
4. **Data Augmentation**: Đã được tích hợp trong `GoDataset` class

## 🐛 Troubleshooting

### Lỗi: "CUDA out of memory"
- Giảm `batch_size`
- Giảm `channels` trong model config

### Lỗi: "Module not found"
- Upload code files vào Drive
- Hoặc copy code trực tiếp vào cells

### Lỗi: "File not found"
- Kiểm tra đường dẫn
- Đảm bảo đã mount Drive

## 📚 Tài Liệu Tham Khảo

- Chi tiết: `docs/ML_TRAINING_COLAB_GUIDE.md`
- Kaggle guide: `docs/ML_TRAINING_KAGGLE_GUIDE.md`
- Comprehensive guide: `docs/ML_COMPREHENSIVE_GUIDE.md`

