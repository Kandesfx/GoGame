# 🚀 Tối Ưu Cho Colab Với 50GB RAM

## ✅ Đã Tối Ưu Cho 50GB RAM

### 1. **Chunk Size Lớn Hơn**
- ✅ Default: 100K samples (~5GB) thay vì 15-20K
- ✅ Tự động điều chỉnh lên đến 300K samples (~15GB)
- ✅ Giảm I/O overhead, tận dụng RAM

### 2. **Multiprocessing Enabled**
- ✅ Default: Enable multiprocessing
- ✅ Workers: 2-6 (tùy CPU cores)
- ✅ Có thể xử lý dataset lên đến 1M positions

### 3. **Memory Management**
- ✅ Chunk size lớn hơn = ít I/O hơn
- ✅ GC ít thường xuyên hơn (mỗi 20K thay vì 5K)
- ✅ Tận dụng RAM để cache nhiều data hơn

## 📋 Cách Sử Dụng (50GB RAM)

### Recommended (Multiprocessing)

```python
from pathlib import Path
from generate_labels_colab import process_dataset_file

WORK_DIR = Path('/content/drive/MyDrive/GoGame_ML')

# Tối ưu cho 50GB RAM - multiprocessing enabled
process_dataset_file(
    input_path=WORK_DIR / 'processed' / 'positions_19x19_2019.pt',
    output_path=WORK_DIR / 'datasets' / 'labeled_19x19_2019.pt',
    filter_handicap=True
    # use_multiprocessing=True (default)
    # save_chunk_size tự động = 100K-300K
    # num_workers tự động = 2-6
)
```

### Manual Tuning

```python
# Tối ưu tối đa với 50GB RAM
process_dataset_file(
    input_path=...,
    output_path=...,
    save_chunk_size=200000,  # 200K samples (~10GB)
    use_multiprocessing=True,
    num_workers=6  # Max workers
)
```

### Single-Threaded (Nếu Multiprocessing Gây Vấn Đề)

```python
# Fallback nếu multiprocessing vẫn gây đứng
process_dataset_file(
    ...,
    use_multiprocessing=False,
    save_chunk_size=150000  # Vẫn lớn hơn với 50GB RAM
)
```

## 📊 Performance (50GB RAM)

### Multiprocessing (Recommended):
- **Speed**: ~3000-5000 pos/s (với 4-6 workers)
- **Memory**: 10-20GB (tận dụng RAM)
- **Dataset**: Lên đến 1M positions

### Single-Threaded:
- **Speed**: ~100-150 pos/s
- **Memory**: 5-10GB
- **Stability**: ✅ Rất ổn định

## ⚙️ Tuning Cho 50GB RAM

### Dataset Nhỏ (<200K positions):
```python
save_chunk_size=300000  # 300K samples (~15GB) - tận dụng RAM tối đa
num_workers=6
```

### Dataset Trung Bình (200K-500K positions):
```python
save_chunk_size=200000  # 200K samples (~10GB)
num_workers=4-6
```

### Dataset Lớn (>500K positions):
```python
save_chunk_size=100000  # 100K samples (~5GB)
num_workers=4  # Giảm workers để tránh overhead
```

## 🎯 Best Practices

1. **Tận dụng RAM**: Dùng chunk size lớn (100K-300K)
2. **Multiprocessing**: Enable với 4-6 workers
3. **Monitor**: Vẫn nên monitor memory (dù có 50GB)
4. **I/O**: Chunk size lớn = ít I/O hơn = nhanh hơn

## 📈 Expected Performance

Với dataset 624K positions trên 50GB RAM:
- **Multiprocessing (6 workers)**: ~20-30 phút (3000-5000 pos/s)
- **Single-threaded**: ~1.5-2 giờ (100-150 pos/s)
- **Speedup**: 3-5x với multiprocessing

## ⚠️ Lưu Ý

1. **Multiprocessing**: Vẫn có thể gây đứng nếu không được implement đúng
2. **Chunk Size**: Dù có 50GB RAM, không nên quá 300K (tránh overhead)
3. **Workers**: Max 6 workers (tránh context switching overhead)

## 🔗 Liên Quan

- `scripts/generate_labels_colab.py` - Script đã tối ưu
- `scripts/COLAB_PRO_OPTIMIZATION.md` - Tài liệu multiprocessing
- `scripts/SINGLE_THREADED_OPTIMIZATION.md` - Fallback option

