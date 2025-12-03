# 🚀 TRAINING VỚI AUTO-COPY CHUNKS

## ✨ Tính năng mới

Script `train_colab_optimized.py` đã được cập nhật với:
- ✅ **Tự động copy chunks** từ Google Drive vào local disk
- ✅ **Bỏ scan** - chỉ copy và train trực tiếp
- ✅ **Progress bars chi tiết** cho mọi bước
- ✅ **Smart caching** - không copy lại nếu đã có

## 🎯 Cách sử dụng

### 1. Training tự động (khuyến nghị)

Chỉ cần chỉ định đường dẫn Google Drive, script sẽ tự động:
1. Copy chunks vào local disk (với progress bar)
2. Train từ local chunks (nhanh hơn 10-20x)

```python
from train_colab_optimized import train_model_optimized
from pathlib import Path

WORK_DIR = Path('/content/drive/MyDrive/GoGame_ML')

# Chỉ cần chỉ định đường dẫn Google Drive
train_model_optimized(
    train_dataset_path=str(WORK_DIR / 'datasets' / 'labeled_19x19_2012_chunks'),
    # Script sẽ tự động:
    # 1. Copy chunks vào /content/chunks_labeled_19x19_2012_chunks
    # 2. Train từ local chunks
    batch_size=None,  # Auto-detect
    num_epochs=10,
    use_chunks=True,
    use_mixed_precision=True
)
```

### 2. Đổi sang chunk khác

Sau khi train xong, chỉ cần đổi đường dẫn:

```python
# Train chunk 2012
train_model_optimized(
    train_dataset_path=str(WORK_DIR / 'datasets' / 'labeled_19x19_2012_chunks'),
    ...
)

# Train chunk 2013 (sẽ tự động copy vào local)
train_model_optimized(
    train_dataset_path=str(WORK_DIR / 'datasets' / 'labeled_19x19_2013_chunks'),
    ...
)
```

### 3. Sử dụng local path trực tiếp

Nếu chunks đã ở local disk, chỉ định trực tiếp:

```python
# Chunks đã ở local
train_model_optimized(
    train_dataset_path='/content/chunks_labeled_19x19_2012_chunks',
    # Script sẽ không copy lại
    ...
)
```

## 📊 Progress Bars

Script hiển thị progress bars cho:

1. **Copy chunks**: 
   ```
   Copying chunks: 100%|████████| 32.0GB/32.0GB [15:23<00:00, 34.2MB/s]
   ```

2. **Loading chunk metadata**:
   ```
   Loading chunk metadata: 100%|████████| 32/32 [00:30<00:00, 1.07file/s]
   ```

3. **Training**:
   ```
   Epoch 1/10: 100%|████████| 5000/5000 [10:23<00:00, 8.0it/s]
   ```

## ⚙️ Tùy chọn

### Force copy lại

Nếu muốn copy lại ngay cả khi đã có:

```python
from train_colab_optimized import copy_chunks_to_local

local_path = copy_chunks_to_local(
    source_dir='/content/drive/MyDrive/GoGame_ML/datasets/labeled_19x19_2012_chunks',
    force_copy=True  # Copy lại
)
```

### Chỉ định local directory

```python
local_path = copy_chunks_to_local(
    source_dir='/content/drive/MyDrive/GoGame_ML/datasets/labeled_19x19_2012_chunks',
    local_dir='/content/my_custom_chunks',  # Custom local path
    chunk_pattern='labeled_19x19_2012_*.pt'  # Pattern cụ thể
)
```

## 🎯 Workflow đề xuất

### Lần đầu tiên:
1. Chạy training với đường dẫn Google Drive
2. Script tự động copy chunks (15-25 phút)
3. Training bắt đầu từ local chunks

### Lần sau:
1. Chỉ cần đổi đường dẫn sang chunk khác
2. Script tự động copy chunk mới (nếu chưa có)
3. Training bắt đầu ngay

### Nếu muốn train lại chunk cũ:
- Script sẽ phát hiện chunks đã có trên local
- Bỏ qua bước copy, train ngay

## 📈 So sánh

| Phương pháp | Thời gian setup | Thời gian train |
|-------------|----------------|-----------------|
| **Cũ (scan từ Drive)** | 5-10 phút scan | Chậm (I/O từ Drive) |
| **Mới (auto-copy)** | 15-25 phút copy (1 lần) | Nhanh (I/O từ local) |

**Lợi ích**: Sau lần copy đầu tiên, các lần train sau sẽ nhanh hơn 10-20x!

## 💡 Tips

1. **Copy một lần, train nhiều lần**: Copy chunks vào local một lần, sau đó train nhiều epochs
2. **Kiểm tra local disk**: Đảm bảo có đủ dung lượng (~50GB cho 32 chunks)
3. **Cleanup**: Xóa local chunks sau khi train xong để giải phóng disk:
   ```python
   import shutil
   shutil.rmtree('/content/chunks_labeled_19x19_2012_chunks')
   ```

---

**Chúc bạn training thành công! 🚀**

