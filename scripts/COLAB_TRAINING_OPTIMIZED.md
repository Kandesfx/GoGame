# 🚀 HƯỚNG DẪN TRAINING TỐI ƯU TRÊN COLAB PRO VỚI CHUNKS

## 📋 Tổng quan

Script `train_colab_optimized.py` được tối ưu đặc biệt cho Colab Pro với:
- ✅ **Tối ưu GPU**: Auto-detect batch size, mixed precision training
- ✅ **Progress bars chi tiết**: Hiển thị loss, GPU memory, thời gian
- ✅ **Memory management**: Tự động clear cache, optimize DataLoader
- ✅ **Support chunks**: Tự động detect pattern `labeled_19x19_YYYY_XXXX.pt`
- ✅ **System monitoring**: Hiển thị RAM, GPU info

## 🎯 Cấu trúc chunks

Script hỗ trợ các pattern:
- `labeled_19x19_2012_0001.pt`, `labeled_19x19_2012_0002.pt`, ...
- `chunk_0001.pt`, `chunk_0002.pt`, ...
- Hoặc bất kỳ `*.pt` files trong directory

## 📦 Setup trên Colab

### Bước 1: Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Bước 2: Tạo cấu trúc thư mục

```python
from pathlib import Path

WORK_DIR = Path('/content/drive/MyDrive/GoGame_ML')

# Tạo các thư mục cần thiết
(WORK_DIR / 'code').mkdir(parents=True, exist_ok=True)
(WORK_DIR / 'datasets').mkdir(parents=True, exist_ok=True)
(WORK_DIR / 'checkpoints').mkdir(parents=True, exist_ok=True)
```

### Bước 3: Upload files cần thiết

Upload vào `GoGame_ML/code/`:
- `train_colab_optimized.py` ⭐ (Script mới)
- `chunk_dataset_optimized.py` ⭐ (File mới với pattern detection)
- `policy_network.py`
- `value_network.py`

### Bước 4: Import và chạy

```python
import sys
sys.path.insert(0, str(WORK_DIR / 'code'))

from train_colab_optimized import train_model_optimized
from pathlib import Path

WORK_DIR = Path('/content/drive/MyDrive/GoGame_ML')

# Training với chunks (cấu trúc: labeled_19x19_2012_0001.pt)
train_model_optimized(
    train_dataset_path=str(WORK_DIR / 'datasets' / 'labeled_19x19_2012_chunks'),
    val_dataset_path=None,  # Có thể dùng chunks riêng
    board_size=None,  # Auto-detect
    batch_size=None,  # ⭐ Auto-detect optimal batch size
    num_epochs=10,
    learning_rate=0.001,
    checkpoint_dir=str(WORK_DIR / 'checkpoints'),
    save_every=2,
    use_chunks=True,  # ⭐ Enable chunks
    use_mixed_precision=True,  # ⭐ Mixed precision (nhanh hơn)
    chunk_pattern=None,  # ⭐ Auto-detect pattern
    pin_memory=True,  # ⭐ Tối ưu GPU transfer
    prefetch_factor=2  # Prefetch batches
)
```

## ⚙️ Các tham số quan trọng

### `batch_size=None` (Auto-detect)
- Script sẽ tự động tìm batch size tối ưu dựa trên GPU memory
- Bắt đầu từ 32, tăng dần đến khi gần hết memory
- Khuyến nghị: Để `None` để auto-detect

### `use_mixed_precision=True`
- Sử dụng bfloat16/float16 để tăng tốc độ training
- Giảm memory usage ~50%
- Tăng tốc độ ~1.5-2x
- Chỉ hoạt động nếu GPU hỗ trợ

### `pin_memory=True`
- Pin memory trong RAM để transfer nhanh hơn lên GPU
- Chỉ dùng khi có GPU
- Tăng tốc độ ~10-20%

### `chunk_pattern=None` (Auto-detect)
- Tự động detect pattern:
  - Ưu tiên: `labeled_*_*.pt`
  - Sau đó: `chunk_*.pt`
  - Cuối cùng: `*.pt`
- Hoặc chỉ định pattern cụ thể: `"labeled_19x19_2012_*.pt"`

## 📊 Output và Monitoring

Script sẽ hiển thị:

### System Information
```
🖥️  SYSTEM INFORMATION
💾 RAM: 32.0 GB total, 28.5 GB available
🎮 GPU: Tesla T4
   Memory: 16.0 GB
   ✅ bfloat16 supported (mixed precision)
```

### Training Progress
```
Epoch 1/10: 100%|████████| 5000/5000 [10:23<00:00, 8.0it/s, p_loss=2.3456, v_loss=0.1234, mem=GPU: 8.2GB (peak: 9.1GB)]
   ⏱️  Time: 623.4s | GPU Memory: 9.12 GB
```

### Epoch Summary
```
📊 Epoch 1/10 Summary:
   Train - Policy: 2.3456, Value: 0.1234
   Val   - Policy: 2.4012, Value: 0.1345
   ⏱️  Time: 623.4s (10.4 min)
   💾 Saved checkpoint: checkpoint_epoch_2.pt
   ⭐ Saved best model: best_model.pt (val_loss: 2.5357)
```

## 🎯 Best Practices

### 1. Batch Size
- **Để `None`**: Script tự động tìm optimal
- **Manual**: Bắt đầu với 32, tăng/giảm dựa trên GPU memory

### 2. Mixed Precision
- **Luôn bật** nếu GPU hỗ trợ (Colab Pro T4/V100: ✅)
- Tăng tốc độ đáng kể, giảm memory

### 3. Pin Memory
- **Bật** khi có GPU
- **Tắt** nếu gặp RAM issues

### 4. Chunk Pattern
- **Để `None`**: Auto-detect (khuyến nghị)
- **Chỉ định** nếu có nhiều loại .pt files trong cùng directory

### 5. Save Every
- **2 epochs**: Cân bằng giữa storage và safety
- **1 epoch**: An toàn hơn, tốn storage hơn
- **5 epochs**: Tiết kiệm storage, rủi ro cao hơn

## 🔧 Troubleshooting

### GPU Memory Error
```python
# Giảm batch size
train_model_optimized(
    ...
    batch_size=16,  # Giảm từ 32 xuống 16
    ...
)
```

### RAM Issues
```python
# Tắt pin_memory
train_model_optimized(
    ...
    pin_memory=False,
    prefetch_factor=1,  # Giảm prefetch
    ...
)
```

### Slow Training
```python
# Kiểm tra GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA: {torch.cuda.is_available()}")

# Bật mixed precision
train_model_optimized(
    ...
    use_mixed_precision=True,
    pin_memory=True,
    ...
)
```

### Chunks Not Found
```python
# Kiểm tra pattern
from pathlib import Path
chunks_dir = Path('/content/drive/MyDrive/GoGame_ML/datasets/labeled_19x19_2012_chunks')
files = list(chunks_dir.glob("*.pt"))
print(f"Found {len(files)} files:")
for f in files[:5]:
    print(f"  {f.name}")

# Chỉ định pattern cụ thể
train_model_optimized(
    ...
    chunk_pattern="labeled_19x19_2012_*.pt",
    ...
)
```

### Import Error
```python
# Nếu gặp lỗi import
import sys
sys.path.insert(0, str(WORK_DIR / 'code'))

# Đảm bảo file tên đúng
from chunk_dataset_optimized import ChunkDataset, create_chunk_dataset
```

## 📈 Performance Tips

1. **Restart runtime** trước khi train để có RAM/GPU sạch
2. **Monitor GPU memory** trong progress bar
3. **Clear cache** sau mỗi epoch (tự động nếu dùng chunks)
4. **Save checkpoints** thường xuyên để tránh mất progress
5. **Download checkpoints** định kỳ về local

## 🎉 Kết quả

Sau khi training xong, bạn sẽ có:
- `checkpoint_epoch_N.pt`: Checkpoints mỗi N epochs
- `best_model.pt`: Model tốt nhất (lowest validation loss)
- `final_model.pt`: Model cuối cùng

Download về local:
```python
from google.colab import files

# Download best model
files.download(str(WORK_DIR / 'checkpoints' / 'best_model.pt'))
```

---

**Chúc bạn training thành công! 🚀**

