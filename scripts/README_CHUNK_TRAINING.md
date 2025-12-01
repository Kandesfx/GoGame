# 📦 HƯỚNG DẪN TRAIN VỚI CHUNKS

## 🎯 Tại sao dùng Chunks?

Khi dataset lớn (>500K samples), merge tất cả chunks vào 1 file sẽ gây **MemoryError** trên Colab (RAM limit ~12GB).

**Giải pháp**: Train trực tiếp từ chunks mà không cần merge!

## ✅ Ưu điểm

- ✅ **Không cần merge**: Tiết kiệm RAM và thời gian
- ✅ **Memory-efficient**: Chỉ cache 1 chunk tại một thời điểm
- ✅ **Tự động**: Auto-detect board_size từ dataset
- ✅ **Tương thích**: Có thể dùng với merged file hoặc chunks

## 🚀 Cách sử dụng

### 1. Upload Files

Upload vào `GoGame_ML/code/`:
- `chunk_dataset.py` ⭐ (File mới)
- `train_colab.py` (Đã cập nhật)

### 2. Import và Train

```python
from pathlib import Path
from train_colab import train_model

WORK_DIR = Path('/content/drive/MyDrive/GoGame_ML')

# Train từ chunks
train_model(
    train_dataset_path=str(WORK_DIR / 'datasets' / 'labeled_19x19_2019_chunks'),  # Chunks directory
    val_dataset_path=None,
    board_size=None,  # Auto-detect
    batch_size=16,  # ⭐ Giảm nếu gặp RAM issues
    num_epochs=10,
    learning_rate=0.001,
    checkpoint_dir=str(WORK_DIR / 'checkpoints'),
    use_chunks=True  # ⭐ Enable chunks mode
)
```

### 3. Hoặc dùng trực tiếp ChunkDataset

```python
from chunk_dataset import create_chunk_dataset
from torch.utils.data import DataLoader

# Tạo dataset từ chunks
chunks_dir = WORK_DIR / 'datasets' / 'labeled_19x19_2019_chunks'
train_dataset = create_chunk_dataset(chunks_dir, augment=True)

# Tạo DataLoader (tối ưu memory)
train_loader = DataLoader(
    train_dataset,
    batch_size=16,  # Giảm nếu cần
    shuffle=True,
    num_workers=0,
    pin_memory=False,  # Tắt để giảm RAM
    prefetch_factor=2
)

# Train như bình thường
# ...
```

## ⚙️ Tối ưu Memory

### Batch Size

| RAM Available | Batch Size | Chunks |
|--------------|------------|--------|
| < 8GB | 8-12 | ✅ |
| 8-12GB | 12-16 | ✅ |
| > 12GB | 16-32 | ✅ |

### DataLoader Settings

```python
DataLoader(
    dataset,
    batch_size=16,  # ⭐ Giảm nếu RAM hết
    shuffle=True,
    num_workers=0,  # Colab không support multiprocessing
    pin_memory=False,  # ⭐ Tắt để giảm RAM
    prefetch_factor=2,  # ⭐ Giảm prefetch
    persistent_workers=False
)
```

## 🔧 Troubleshooting

### Vẫn bị Full RAM?

1. **Giảm batch_size**: 16 → 8 hoặc 4
2. **Clear cache định kỳ**:
   ```python
   # Sau mỗi epoch
   if hasattr(train_dataset, 'clear_cache'):
       train_dataset.clear_cache()
       import gc
       gc.collect()
   ```
3. **Giảm chunk size khi tạo**:
   ```python
   # Khi gán nhãn
   save_chunk_size=30000  # Thay vì 50000
   ```

### ChunkDataset not found?

```python
# Đảm bảo đã upload chunk_dataset.py vào code/
import sys
sys.path.insert(0, str(WORK_DIR / 'code'))
from chunk_dataset import ChunkDataset
```

## 📊 So sánh

| Method | RAM Usage | Speed | Shuffle |
|--------|-----------|-------|---------|
| Merged File | ~15GB+ | Fast | ✅ |
| Chunks (1 cache) | ~2-3GB | Medium | ✅ |
| Chunks (no cache) | ~1GB | Slow | ⚠️ |

## 💡 Best Practices

1. **Dataset nhỏ (<200K)**: Dùng merged file
2. **Dataset lớn (>200K)**: Dùng chunks
3. **Batch size**: Bắt đầu với 16, giảm nếu cần
4. **Clear cache**: Sau mỗi epoch nếu RAM cao
5. **Monitor RAM**: Theo dõi trong Colab resource monitor

---

**Chúc bạn train thành công! 🎉**

