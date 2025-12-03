# 🎓 Tối Ưu Cho Colab Student

## ⚠️ Lưu Ý Về Colab Student

Colab Student có resources hạn chế hơn Colab Pro:
- **RAM**: 12-15GB (thấp hơn Pro)
- **CPU**: 2-4 cores (ít hơn Pro)
- **GPU**: Không có hoặc T4 (hạn chế)
- **Timeout**: Có thể bị giới hạn

## ✅ Đã Tối Ưu Cho Colab Student

### 1. **Single-Threaded Default**
- ✅ Multiprocessing **TẮT** mặc định (tránh đứng máy)
- ✅ Tối ưu single-threaded với reused generators
- ✅ Speed: ~100-150 pos/s (đủ nhanh)

### 2. **Chunk Size Nhỏ Hơn**
- ✅ Default: 15K samples (~750MB) thay vì 20-50K
- ✅ Tự động điều chỉnh theo dataset size
- ✅ Giảm memory usage từ 30GB → <3GB

### 3. **Memory Management Tối Ưu**
- ✅ Reuse label generators (giảm overhead)
- ✅ Clear memory ngay sau mỗi chunk
- ✅ GC thường xuyên

### 4. **Auto-Disable Multiprocessing**
- ✅ Tự động disable nếu dataset > 200K positions
- ✅ Tránh memory overflow

## 📋 Cách Sử Dụng (Colab Student)

### Recommended (Single-Threaded)

```python
from pathlib import Path
from generate_labels_colab import process_dataset_file

WORK_DIR = Path('/content/drive/MyDrive/GoGame_ML')

# Tối ưu cho Colab Student - single-threaded
process_dataset_file(
    input_path=WORK_DIR / 'processed' / 'positions_19x19_2019.pt',
    output_path=WORK_DIR / 'datasets' / 'labeled_19x19_2019.pt',
    filter_handicap=True
    # use_multiprocessing=False (default)
    # save_chunk_size tự động = 15K-20K
)
```

### Nếu Muốn Thử Multiprocessing (Cẩn Thận!)

```python
# CHỈ dùng nếu dataset < 200K positions
process_dataset_file(
    input_path=...,
    output_path=...,
    use_multiprocessing=True,
    num_workers=2,  # Tối đa 2 workers cho Colab Student
    save_chunk_size=10000  # Chunk size nhỏ hơn
)
```

## 📊 Performance (Colab Student)

### Single-Threaded (Recommended):
- **Speed**: ~100-150 pos/s
- **Memory**: <3GB (với incremental save)
- **Stability**: ✅ Ổn định, không bị đứng

### Multiprocessing (Not Recommended):
- **Speed**: Có thể nhanh hơn nhưng...
- **Memory**: Có thể >10GB → **BỊ ĐỨNG**
- **Stability**: ❌ Có thể gây đứng máy

## ⚙️ Tuning Cho Colab Student

### Dataset Nhỏ (<100K positions):
```python
process_dataset_file(
    ...,
    save_chunk_size=20000  # Có thể lớn hơn một chút
)
```

### Dataset Lớn (>500K positions):
```python
process_dataset_file(
    ...,
    save_chunk_size=10000  # Nhỏ hơn để an toàn
)
```

### Nếu Bị Memory Error:
```python
process_dataset_file(
    ...,
    save_chunk_size=5000,  # Rất nhỏ
    use_multiprocessing=False  # Bắt buộc single-threaded
)
```

## 🔍 Monitoring

Script tự động log:
- Memory usage
- Processing speed
- Chunk save progress

**Nếu thấy memory > 10GB**: Giảm `save_chunk_size` ngay!

## ⚠️ Troubleshooting

### Vấn Đề: Máy Bị Đứng
**Giải pháp**:
```python
# Force single-threaded
process_dataset_file(
    ...,
    use_multiprocessing=False,
    num_workers=1,
    save_chunk_size=10000  # Giảm chunk size
)
```

### Vấn Đề: Memory Error
**Giải pháp**:
```python
# Giảm chunk size
process_dataset_file(
    ...,
    save_chunk_size=5000  # Rất nhỏ
)
```

### Vấn Đề: Chạy Quá Chậm
**Giải pháp**:
- Đảm bảo đang dùng single-threaded (reuse generators)
- Kiểm tra xem có process khác đang chạy không
- Restart runtime nếu cần

## 🎯 Best Practices Cho Colab Student

1. **Luôn dùng single-threaded** (default)
2. **Chunk size**: 10K-20K (tùy dataset)
3. **Monitor memory**: Nếu > 10GB, giảm chunk size
4. **Restart runtime** nếu memory bị leak
5. **Process từng năm** thay vì tất cả cùng lúc

## 📈 Expected Performance

Với dataset 624K positions trên Colab Student:
- **Single-threaded**: ~1.5-2 giờ (100-150 pos/s)
- **Memory**: <3GB (với incremental save)
- **Stability**: ✅ Ổn định

## 🔗 Liên Quan

- `scripts/generate_labels_colab.py` - Script đã tối ưu
- `scripts/SINGLE_THREADED_OPTIMIZATION.md` - Chi tiết tối ưu
- `scripts/COLAB_PRO_OPTIMIZATION.md` - Nếu upgrade lên Pro

