# 🔄 Cập Nhật: Gán Nhãn Trên Colab

## ✅ Đã Cập Nhật

Script `generate_labels_colab.py` đã được cập nhật để sử dụng **Multi-task Labels** giống như local script.

### Các Thay Đổi

1. ✅ **Multi-task Label Generators**: Đã tích hợp đầy đủ
   - `ThreatLabelGenerator` - Threat detection map
   - `AttackLabelGenerator` - Attack opportunity map
   - `IntentLabelGenerator` - Intent classification (5 classes)
   - `EvaluationLabelGenerator` - Position evaluation

2. ✅ **Format Labels**: Đúng theo tài liệu `MULTI_TASK_LABELS_IMPLEMENTATION.md`
   ```python
   {
       'features': Tensor[17, board_size, board_size],
       'labels': {
           'threat_map': Tensor[board_size, board_size],
           'attack_map': Tensor[board_size, board_size],
           'intent': {
               'type': str,  # 'territory', 'attack', 'defense', 'connection', 'cut'
               'confidence': float,
               'region': List[Tuple[int, int]]
           },
           'evaluation': {
               'win_probability': float,
               'territory_map': Tensor[board_size, board_size],
               'influence_map': Tensor[board_size, board_size]
           }
       },
       'policy': Tensor[board_size * board_size + 1],  # Backward compat
       'value': float,  # Backward compat
       'metadata': {...}
   }
   ```

3. ✅ **Metadata**: Đã thêm `date_processed` và `errors` count

## 📋 Cách Sử Dụng Trên Colab

### 1. Upload Files Lên Colab

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Upload các files cần thiết:
# - scripts/generate_labels_colab.py
# - scripts/label_generators.py
# - scripts/generate_features_colab.py
# - data/processed/positions_*.pt (từ local)
```

### 2. Generate Labels

```python
from pathlib import Path
from generate_labels_colab import process_dataset_file

WORK_DIR = Path('/content/drive/MyDrive/GoGame_ML')

# Process một năm
process_dataset_file(
    input_path=WORK_DIR / 'processed' / 'positions_19x19_2019.pt',
    output_path=WORK_DIR / 'datasets' / 'labeled_19x19_2019.pt',
    filter_handicap=True,
    save_chunk_size=50000  # Quan trọng cho Colab RAM limit
)
```

### 3. Process Nhiều Năm

```python
years = [2019, 2018, 2017]
board_sizes = [19, 13, 9]

for year in years:
    for board_size in board_sizes:
        input_file = WORK_DIR / 'processed' / f'positions_{board_size}x{board_size}_{year}.pt'
        output_file = WORK_DIR / 'datasets' / f'labeled_{board_size}x{board_size}_{year}.pt'
        
        if input_file.exists():
            print(f"Processing {year} - {board_size}x{board_size}...")
            process_dataset_file(
                input_path=input_file,
                output_path=output_file,
                filter_handicap=True,
                save_chunk_size=50000
            )
        else:
            print(f"Skipping {year} - {board_size}x{board_size} (file not found)")
```

## ⚙️ Tùy Chọn

### Incremental Save (Khuyến Nghị cho Colab)

Colab có RAM limit (~12-15GB), nên nên dùng incremental save:

```python
process_dataset_file(
    input_path=...,
    output_path=...,
    save_chunk_size=50000,  # Save mỗi 50K samples (~1.2GB)
    skip_merge=False  # True nếu muốn giữ chunks riêng
)
```

### Auto-enable Incremental Save

Script tự động enable nếu estimated memory > 4GB:

```python
process_dataset_file(
    input_path=...,
    output_path=...,
    auto_enable_incremental=True  # Default: True
)
```

### Giữ Chunks Riêng (Nếu RAM quá thấp)

```python
process_dataset_file(
    input_path=...,
    output_path=...,
    save_chunk_size=50000,
    skip_merge=True  # Giữ chunks riêng, merge sau
)

# Merge sau (khi có đủ RAM)
from generate_labels_colab import merge_chunks
chunk_files = sorted(WORK_DIR / 'datasets' / 'labeled_19x19_2019_chunks' / '*.pt')
merge_chunks(chunk_files, WORK_DIR / 'datasets' / 'labeled_19x19_2019.pt')
```

## 📊 So Sánh: Colab vs Local

| Feature | Colab Script | Local Script |
|---------|-------------|--------------|
| Multi-task Labels | ✅ | ✅ |
| Incremental Save | ✅ (quan trọng) | ✅ (optional) |
| Multiprocessing | ❌ (single-threaded) | ✅ (multiprocessing) |
| Memory Management | ✅ (auto) | ✅ (auto) |
| Error Handling | ✅ | ✅ |
| Progress Tracking | ✅ (tqdm) | ✅ (tqdm) |

**Lý do không dùng multiprocessing trên Colab:**
- Colab có giới hạn số processes
- Single-threaded đủ nhanh với Colab CPU
- Tránh overhead của multiprocessing

## ⚠️ Lưu Ý

1. **RAM Limit**: Colab free có ~12GB RAM. Với dataset lớn, bắt buộc dùng `save_chunk_size`.

2. **Timeout**: Colab free có timeout ~12 giờ. Với dataset rất lớn, có thể cần chạy nhiều lần.

3. **Google Drive**: Đảm bảo có đủ dung lượng trên Drive cho output files.

4. **Upload Files**: Cần upload đầy đủ:
   - `generate_labels_colab.py`
   - `label_generators.py`
   - `generate_features_colab.py`
   - `positions_*.pt` files

## 🔗 Liên Quan

- `scripts/MULTI_TASK_LABELS_IMPLEMENTATION.md` - Tài liệu về multi-task labels
- `scripts/generate_labels_local.py` - Script local (tương tự)
- `scripts/label_generators.py` - Label generators

## 📝 Example Notebook

```python
# Cell 1: Setup
from google.colab import drive
drive.mount('/content/drive')

import sys
sys.path.append('/content/drive/MyDrive/GoGame_ML/scripts')

# Cell 2: Import
from pathlib import Path
from generate_labels_colab import process_dataset_file

# Cell 3: Process
WORK_DIR = Path('/content/drive/MyDrive/GoGame_ML')

process_dataset_file(
    input_path=WORK_DIR / 'processed' / 'positions_19x19_2019.pt',
    output_path=WORK_DIR / 'datasets' / 'labeled_19x19_2019.pt',
    filter_handicap=True,
    save_chunk_size=50000
)
```

