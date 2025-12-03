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

#### Training cơ bản (cho GPU nhỏ hoặc test nhanh):
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

#### Training tối ưu GPU RAM (cho L4 24GB hoặc GPU lớn):
```python
from train_colab import train_model

train_model(
    train_dataset_path=str(WORK_DIR / 'datasets' / 'labeled_19x19.pt'),
    val_dataset_path=None,
    board_size=19,
    batch_size=4096,  # Tăng từ 1024 để tận dụng GPU RAM
    num_epochs=10,
    learning_rate=0.001,
    checkpoint_dir=str(WORK_DIR / 'checkpoints'),
    use_chunks=True,  # Dùng chunks để load dataset lớn
    model_channels=256,  # Tăng từ 128 để model lớn hơn
    max_train_samples=None,  # None = dùng tất cả samples
    gradient_accumulation_steps=1,  # Có thể tăng để effective batch size lớn hơn
    enable_pin_memory=True,  # Tăng tốc data loading
    checkpoint_prefix=None  # Auto-detect từ dataset path, hoặc set thủ công như "dataset_2019"
)
```

**⚠️ Quan trọng - Tránh ghi đè checkpoint:**
- Khi train nhiều dataset khác nhau, script sẽ tự động tạo prefix từ tên dataset
- Ví dụ: train từ `/content/split19` → prefix = `split19`
- Checkpoint sẽ được lưu: `split19_checkpoint_epoch_1.pt`, `split19_best_model.pt`, etc.
- Nếu muốn set thủ công: `checkpoint_prefix="dataset_2019"`

**Tối ưu GPU RAM:**
- **Batch size**: Tăng từ 1024 → 4096 hoặc 8192 (tùy GPU RAM)
- **Model channels**: Tăng từ 128 → 256 hoặc 512 (model lớn hơn, tốt hơn)
- **Training samples**: Bỏ giới hạn 200k, dùng tất cả 600k samples
- **Gradient accumulation**: Nếu muốn effective batch size = 8192, dùng batch_size=4096 + accumulation_steps=2

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

## 🔄 Loading Model for Inference (Sử dụng trong App)

Sau khi training xong, bạn cần load model để sử dụng trong app. Dưới đây là hướng dẫn chi tiết:

### Cách 1: Load từ `best_model.pt` (Khuyến nghị)

```python
import torch
from policy_network import PolicyNetwork, PolicyConfig
from value_network import ValueNetwork, ValueConfig

# Đường dẫn đến checkpoint
checkpoint_path = 'checkpoints/best_model.pt'  # hoặc 'final_model.pt'

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Khởi tạo model với config từ checkpoint
policy_config = PolicyConfig(**checkpoint['policy_config'])
value_config = ValueConfig(**checkpoint['value_config'])

policy_net = PolicyNetwork(policy_config)
value_net = ValueNetwork(value_config)

# Load weights vào model
policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
value_net.load_state_dict(checkpoint['value_net_state_dict'])

# Set model sang eval mode (quan trọng cho inference)
policy_net.eval()
value_net.eval()

# Bây giờ có thể dùng để predict
# Ví dụ: predict move từ board state
with torch.no_grad():
    # features: Tensor [1, 17, board_size, board_size]
    policy_logits = policy_net(features)
    value_pred = value_net(features)
```

### Cách 2: Load từ `final_model.pt`

```python
# Tương tự như trên, chỉ đổi đường dẫn
checkpoint_path = 'checkpoints/final_model.pt'
checkpoint = torch.load(checkpoint_path, map_location='cpu')
# ... (phần còn lại giống Cách 1)
```

### Cách 3: Load từ checkpoint epoch cụ thể

```python
# Load từ checkpoint epoch 3
checkpoint_path = 'checkpoints/checkpoint_epoch_3.pt'
checkpoint = torch.load(checkpoint_path, map_location='cpu')
# ... (phần còn lại giống Cách 1)
```

### Lưu ý quan trọng:

1. **Nên dùng `best_model.pt`**: Model này có validation loss thấp nhất, thường là model tốt nhất
2. **Luôn set `eval()` mode**: Quan trọng để tắt dropout và batch normalization trong inference
3. **Dùng `torch.no_grad()`**: Tắt gradient computation để tiết kiệm memory và tăng tốc
4. **`map_location='cpu'`**: Nếu load trên CPU, hoặc `'cuda:0'` nếu load trên GPU

### Ví dụ đầy đủ: Load và sử dụng trong app

```python
import torch
from policy_network import PolicyNetwork, PolicyConfig
from value_network import ValueNetwork, ValueConfig

class GoAIModel:
    def __init__(self, checkpoint_path='checkpoints/best_model.pt'):
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Khởi tạo models
        policy_config = PolicyConfig(**checkpoint['policy_config'])
        value_config = ValueConfig(**checkpoint['value_config'])
        
        self.policy_net = PolicyNetwork(policy_config)
        self.value_net = ValueNetwork(value_config)
        
        # Load weights
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        
        # Set eval mode
        self.policy_net.eval()
        self.value_net.eval()
        
        self.board_size = checkpoint['board_size']
    
    def predict_move(self, features):
        """
        Predict move từ board features.
        
        Args:
            features: Tensor [1, 17, board_size, board_size]
        
        Returns:
            policy: Tensor [board_size * board_size] - xác suất cho mỗi move
            value: float - giá trị vị trí (0-1)
        """
        with torch.no_grad():
            policy_logits = self.policy_net(features)
            value_pred = self.value_net(features)
        
        # Convert logits to probabilities
        policy_probs = torch.softmax(policy_logits, dim=1)
        
        return policy_probs[0], value_pred.item()

# Sử dụng
model = GoAIModel('checkpoints/best_model.pt')
# ... dùng model.predict_move(features) trong app
```

## 💡 Tips

### Tối ưu GPU RAM (L4 24GB):
1. **Batch size**: Tăng lên 4096-8192 để tận dụng GPU RAM
2. **Model channels**: Tăng từ 128 → 256 hoặc 512
3. **Training samples**: Bỏ giới hạn, dùng tất cả samples có sẵn
4. **Gradient accumulation**: Dùng để tăng effective batch size mà không cần tăng batch_size
5. **Pin memory**: Bật `enable_pin_memory=True` để tăng tốc data loading

### GPU Memory nhỏ:
1. **GPU Memory**: Nếu hết memory, giảm `batch_size` (4096 → 2048 → 1024 → 512)
2. **Model channels**: Giảm từ 256 → 128 → 64 nếu cần
3. **Training samples**: Giới hạn số samples với `max_train_samples`

### Khác:
1. **Training Time**: 10 epochs cho 200K samples ≈ 1-2 giờ trên Colab L4 với batch_size=4096
2. **Save Checkpoints**: Lưu thường xuyên để tránh mất dữ liệu khi session timeout
3. **Data Augmentation**: Đã được tích hợp trong `GoDataset` class
4. **Monitor GPU**: Script sẽ tự động in GPU RAM usage trong quá trình training

## 🐛 Troubleshooting

### Lỗi: "CUDA out of memory"
- Giảm `batch_size` (4096 → 2048 → 1024)
- Giảm `model_channels` (256 → 128 → 64)
- Giảm `max_train_samples` nếu đang dùng quá nhiều
- Tăng `gradient_accumulation_steps` và giảm `batch_size` để giữ effective batch size

### GPU RAM chưa được tận dụng tối đa
- Tăng `batch_size` lên 4096 hoặc 8192
- Tăng `model_channels` lên 256 hoặc 512
- Bỏ `max_train_samples` để dùng tất cả data
- Kiểm tra GPU memory usage trong log để xem còn bao nhiêu RAM trống

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

