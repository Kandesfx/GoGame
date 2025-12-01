# 🏷️ QUICK START: GÁN NHÃN TRÊN COLAB

## 🚀 Setup Nhanh

### 1. Mount Drive & Setup

```python
from google.colab import drive
from pathlib import Path

drive.mount('/content/drive')
WORK_DIR = Path('/content/drive/MyDrive/GoGame_ML')
(WORK_DIR / 'processed').mkdir(parents=True, exist_ok=True)
(WORK_DIR / 'datasets').mkdir(parents=True, exist_ok=True)
```

### 2. Upload Scripts

Upload vào `GoGame_ML/code/`:
- `generate_labels_colab.py`
- `generate_features_colab.py`

### 3. Import & Run

```python
import sys
sys.path.insert(0, str(WORK_DIR / 'code'))

from generate_labels_colab import process_dataset_file

# Process một file
process_dataset_file(
    input_path=WORK_DIR / 'processed' / 'positions_19x19_2019.pt',
    output_path=WORK_DIR / 'datasets' / 'labeled_19x19_2019.pt',
    filter_handicap=True,
    save_chunk_size=50000,  # Save mỗi 50K samples
    auto_enable_incremental=True
)
```

## 📊 Tính Năng

- ✅ **Auto Incremental Save**: Tự động save chunks khi memory > 4GB
- ✅ **Memory Management**: Tránh MemoryError với Colab RAM limit
- ✅ **Progress Tracking**: Real-time progress với tqdm
- ✅ **Error Handling**: Logging chi tiết và skip lỗi

## ⚙️ Parameters

| Parameter | Mô tả | Mặc định |
|-----------|-------|----------|
| `save_chunk_size` | Save mỗi N samples | `50000` |
| `auto_enable_incremental` | Tự động enable nếu memory > 4GB | `True` |
| `filter_handicap` | Bỏ qua handicap games | `True` |

## 🔧 Troubleshooting

### MemoryError
```python
# Giảm chunk size
save_chunk_size=30000  # Thay vì 50000
```

### Session Timeout
- Chunks đã được save, có thể merge lại:
```python
from generate_labels_colab import merge_chunks
chunks_dir = WORK_DIR / 'datasets' / 'labeled_19x19_2019_chunks'
chunk_files = sorted(chunks_dir.glob('chunk_*.pt'))
merge_chunks(chunk_files, WORK_DIR / 'datasets' / 'labeled_19x19_2019.pt')
```

## 📚 Xem Thêm

- **Chi tiết**: `docs/COLAB_LABELING_GUIDE.md`
- **Training**: `docs/ML_TRAINING_COLAB_GUIDE.md`
- **Template**: `scripts/colab_notebook_template.py`

---

**Lưu ý**: Với dataset lớn (>500K positions), nên xử lý trên local với `generate_labels_local.py` (có multiprocessing).

