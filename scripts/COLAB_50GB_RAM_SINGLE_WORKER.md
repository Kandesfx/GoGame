# 🚀 Tối Ưu Cho Colab 50GB RAM - Single Worker

## ✅ Cấu Hình

- **Workers**: 1 (single-threaded)
- **RAM**: 50GB (tận dụng với chunk size lớn)
- **Multiprocessing**: Disabled (default)

## 🎯 Tối Ưu

### 1. **Chunk Size Lớn** (Tận Dụng 50GB RAM)
- ✅ Default: 100K samples (~5GB)
- ✅ Tự động điều chỉnh lên đến 300K samples (~15GB)
- ✅ Giảm I/O overhead, tận dụng RAM

### 2. **Single-Threaded Optimized**
- ✅ Reuse label generators (giảm overhead)
- ✅ Tốc độ: ~100-150 pos/s
- ✅ Ổn định, không bị đứng

### 3. **Memory Management**
- ✅ Chunk size lớn = ít I/O hơn
- ✅ GC ít thường xuyên hơn (mỗi 20K)
- ✅ Tận dụng RAM để cache nhiều data

## 📋 Cách Sử Dụng

### Basic (Recommended)

```python
from pathlib import Path
from generate_labels_colab import process_dataset_file

WORK_DIR = Path('/content/drive/MyDrive/GoGame_ML')

# Single-threaded với 1 worker, tận dụng 50GB RAM
process_dataset_file(
    input_path=WORK_DIR / 'processed' / 'positions_19x19_2019.pt',
    output_path=WORK_DIR / 'datasets' / 'labeled_19x19_2019.pt',
    filter_handicap=True
    # num_workers=1 (default)
    # use_multiprocessing=False (default)
    # save_chunk_size tự động = 100K-300K
)
```

### Manual Tuning

```python
# Tối ưu tối đa với chunk size lớn
process_dataset_file(
    input_path=...,
    output_path=...,
    save_chunk_size=200000,  # 200K samples (~10GB)
    num_workers=1,  # Single-threaded
    use_multiprocessing=False
)
```

## 📊 Performance

### Single-Threaded (1 Worker):
- **Speed**: ~100-150 pos/s
- **Memory**: 5-15GB (tùy chunk size)
- **Stability**: ✅ Rất ổn định
- **Dataset**: Không giới hạn (với incremental save)

## ⚙️ Tuning Chunk Size

### Dataset Nhỏ (<200K positions):
```python
save_chunk_size=300000  # 300K samples (~15GB) - tận dụng RAM tối đa
```

### Dataset Trung Bình (200K-500K positions):
```python
save_chunk_size=200000  # 200K samples (~10GB)
```

### Dataset Lớn (>500K positions):
```python
save_chunk_size=100000  # 100K samples (~5GB)
```

## 📈 Expected Performance

Với dataset 624K positions trên 50GB RAM (1 worker):
- **Thời gian**: ~1.5-2 giờ (100-150 pos/s)
- **Memory**: 5-15GB (tùy chunk size)
- **Stability**: ✅ Rất ổn định, không bị đứng

## 🎯 Best Practices

1. **Chunk Size**: Dùng lớn (100K-300K) để tận dụng RAM
2. **Single-Threaded**: 1 worker = ổn định, không bị đứng
3. **Monitor**: Vẫn nên monitor memory (dù có 50GB)
4. **I/O**: Chunk size lớn = ít I/O hơn = nhanh hơn

## ⚠️ Lưu Ý

1. **Chunk Size**: Không nên quá 300K (tránh overhead khi save)
2. **Memory**: Dù có 50GB, vẫn nên dùng incremental save
3. **Speed**: Single-threaded chậm hơn multiprocessing nhưng ổn định hơn

## 🔗 Liên Quan

- `scripts/generate_labels_colab.py` - Script đã tối ưu
- `scripts/SINGLE_THREADED_OPTIMIZATION.md` - Chi tiết tối ưu single-threaded
- `scripts/COLAB_50GB_RAM_OPTIMIZATION.md` - Nếu muốn enable multiprocessing

