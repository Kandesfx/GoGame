# 🏷️ HƯỚNG DẪN GÁN NHÃN TRÊN COLAB

## 📑 MỤC LỤC

1. [Tổng quan](#1-tổng-quan)
2. [Setup Colab](#2-setup-colab)
3. [Upload Data](#3-upload-data)
4. [Gán Nhãn](#4-gán-nhãn)
5. [Download Kết Quả](#5-download-kết-quả)
6. [Troubleshooting](#6-troubleshooting)

---

## 1. TỔNG QUAN

Script `generate_labels_colab.py` được tối ưu cho Google Colab với:

- ✅ **Incremental Save**: Tự động save chunks định kỳ để tránh MemoryError
- ✅ **Memory Management**: Tự động detect và enable incremental save khi cần
- ✅ **Error Handling**: Logging chi tiết và skip lỗi
- ✅ **Progress Tracking**: Real-time progress với tqdm
- ✅ **Google Drive Integration**: Lưu trực tiếp vào Drive

### So sánh Local vs Colab

| Tính năng | Local | Colab |
|-----------|-------|-------|
| Multiprocessing | ✅ (8+ workers) | ⚠️ (Hạn chế) |
| Incremental Save | ✅ | ✅ (Quan trọng hơn) |
| RAM Limit | ~16GB | ~12-15GB |
| Session Timeout | ❌ | ⚠️ (90 phút free) |
| GPU | ❌ | ✅ (Không cần cho labeling) |

**Khuyến nghị**: 
- **Local**: Xử lý dataset lớn (>500K positions) với multiprocessing
- **Colab**: Xử lý dataset vừa (<500K positions) hoặc test workflow

---

## 2. SETUP COLAB

### 2.1. Tạo Notebook Mới

1. Mở [Google Colab](https://colab.research.google.com/)
2. Tạo notebook mới: `File` → `New notebook`
3. Đặt tên: `GoGame_Labeling.ipynb`

### 2.2. Mount Google Drive

```python
# Cell 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
```

Chọn account và cho phép truy cập Drive.

### 2.3. Tạo Cấu Trúc Thư Mục

```python
# Cell 2: Setup directories
from pathlib import Path

WORK_DIR = Path('/content/drive/MyDrive/GoGame_ML')

# Tạo các thư mục cần thiết
(WORK_DIR / 'processed').mkdir(parents=True, exist_ok=True)
(WORK_DIR / 'datasets').mkdir(parents=True, exist_ok=True)
(WORK_DIR / 'code').mkdir(parents=True, exist_ok=True)

print(f"✅ Working directory: {WORK_DIR}")
```

### 2.4. Upload Scripts

**Cách 1: Upload trực tiếp**

1. Upload các file vào `GoGame_ML/code/`:
   - `generate_labels_colab.py`
   - `generate_features_colab.py`

**Cách 2: Copy code vào notebook**

Copy nội dung từ `scripts/generate_labels_colab.py` vào một cell.

### 2.5. Install Dependencies

```python
# Cell 3: Install packages
!pip install torch numpy tqdm
```

---

## 3. UPLOAD DATA

### 3.1. Upload Positions File

**Cách 1: Upload từ Local**

1. Upload file `.pt` vào `GoGame_ML/processed/`:
   ```
   /content/drive/MyDrive/GoGame_ML/processed/
   ├── positions_19x19_2019.pt
   ├── positions_19x19_2020.pt
   └── ...
   ```

**Cách 2: Download từ URL (nếu có)**

```python
# Cell 4: Download data (nếu cần)
import urllib.request

url = "https://your-url.com/positions_19x19_2019.pt"
output_path = WORK_DIR / 'processed' / 'positions_19x19_2019.pt'

urllib.request.urlretrieve(url, output_path)
print(f"✅ Downloaded to {output_path}")
```

### 3.2. Verify Data

```python
# Cell 5: Verify positions file
import torch

data_path = WORK_DIR / 'processed' / 'positions_19x19_2019.pt'
data = torch.load(data_path, map_location='cpu', weights_only=False)

print(f"📊 Data info:")
print(f"   Board size: {data['board_size']}x{data['board_size']}")
print(f"   Total positions: {len(data['positions']):,}")
print(f"   Year: {data.get('year', 'N/A')}")

# Estimate memory
estimated_mb = len(data['positions']) * 50 / 1024
print(f"   Estimated memory: ~{estimated_mb:.0f}MB")
```

---

## 4. GÁN NHÃN

### 4.1. Import Script

```python
# Cell 6: Import labeling script
import sys
sys.path.insert(0, str(WORK_DIR / 'code'))

from generate_labels_colab import process_dataset_file
```

### 4.2. Process Một File

```python
# Cell 7: Generate labels cho một file
process_dataset_file(
    input_path=str(WORK_DIR / 'processed' / 'positions_19x19_2019.pt'),
    output_path=str(WORK_DIR / 'datasets' / 'labeled_19x19_2019.pt'),
    filter_handicap=True,
    save_chunk_size=50000,  # Save mỗi 50K samples (~1.2GB)
    auto_enable_incremental=True  # Tự động enable nếu estimated memory > 4GB
)
```

### 4.3. Process Nhiều Files (Theo Năm)

```python
# Cell 8: Process nhiều năm
for year in [2019, 2020, 2021]:
    input_file = WORK_DIR / 'processed' / f'positions_19x19_{year}.pt'
    output_file = WORK_DIR / 'datasets' / f'labeled_19x19_{year}.pt'
    
    if input_file.exists():
        print(f"\n🔄 Processing year {year}...")
        process_dataset_file(
            input_path=str(input_file),
            output_path=str(output_file),
            filter_handicap=True,
            save_chunk_size=50000,
            auto_enable_incremental=True
        )
    else:
        print(f"⚠️  Skipping year {year} (file not found)")

print("\n✅ All years processed!")
```

### 4.4. Monitor Progress

Script sẽ hiển thị:
- Progress bar với tốc độ xử lý (pos/s)
- Memory usage warnings
- Chunk save notifications
- Error summary

**Ví dụ output:**
```
💡 Auto-enabling incremental save (chunk size: 50,000) to prevent MemoryError (estimated: ~15,000MB)
📁 Incremental save enabled: chunks will be saved to /content/drive/MyDrive/GoGame_ML/datasets/labeled_19x19_2019_chunks
Generating labels: 100%|████████| 622k/622k [2:15:30<00:00, 76.5pos/s]
💾 Saving chunk 1 (50,000 samples) to chunk_0001.pt
✅ Chunk 1 saved. Memory cleared.
...
📦 Merging 13 chunks...
✅ Saved merged dataset to labeled_19x19_2019.pt
```

---

## 5. DOWNLOAD KẾT QUẢ

### 5.1. Verify Output

```python
# Cell 9: Verify labeled dataset
import torch

dataset_path = WORK_DIR / 'datasets' / 'labeled_19x19_2019.pt'
data = torch.load(dataset_path, map_location='cpu', weights_only=False)

print(f"📊 Labeled dataset info:")
print(f"   Board size: {data['board_size']}x{data['board_size']}")
print(f"   Total samples: {data['total']:,}")

# Xem một sample
sample = data['labeled_data'][0]
print(f"\n📝 Sample structure:")
print(f"   Features shape: {sample['features'].shape}")
print(f"   Policy shape: {sample['policy'].shape}")
print(f"   Value: {sample['value']}")
```

### 5.2. Download về Local

**Cách 1: Download từ Colab**

```python
# Cell 10: Download file
from google.colab import files

# Download labeled dataset
files.download(str(WORK_DIR / 'datasets' / 'labeled_19x19_2019.pt'))
```

**Cách 2: Copy từ Drive**

Files đã được lưu vào Google Drive, bạn có thể:
1. Mở Google Drive
2. Tìm file trong `GoGame_ML/datasets/`
3. Download về máy

---

## 6. TROUBLESHOOTING

### 6.1. MemoryError khi Merge Chunks

**Triệu chứng:**
- Gán nhãn xong (100%) nhưng merge chunks bị dừng ở 50%
- RAM hết (12.4/12.7 GB)
- Process bị kill

**Giải pháp:**

**Option 1: Skip merge và merge sau (Khuyến nghị)**

```python
# Gán nhãn với skip_merge=True
process_dataset_file(
    input_path=WORK_DIR / 'processed' / 'positions_19x19_2019.pt',
    output_path=WORK_DIR / 'datasets' / 'labeled_19x19_2019.pt',
    filter_handicap=True,
    save_chunk_size=50000,
    skip_merge=True  # ⭐ Bỏ qua merge, giữ chunks riêng
)

# Sau đó, restart runtime và merge riêng
from merge_chunks_colab import merge_chunks_from_directory

chunks_dir = WORK_DIR / 'datasets' / 'labeled_19x19_2019_chunks'
output_path = WORK_DIR / 'datasets' / 'labeled_19x19_2019.pt'
merge_chunks_from_directory(chunks_dir, output_path)
```

**Option 2: Giảm chunk size**

```python
# Giảm chunk size để có ít chunks hơn
process_dataset_file(
    input_path=...,
    output_path=...,
    save_chunk_size=30000  # Giảm từ 50000 → ít chunks hơn
)
```

**Option 3: Restart và merge**

1. Restart runtime: `Runtime` → `Restart runtime`
2. Mount Drive lại
3. Merge chunks:

```python
from generate_labels_colab import merge_chunks
from pathlib import Path

chunks_dir = Path('/content/drive/MyDrive/GoGame_ML/datasets/labeled_19x19_2019_chunks')
chunk_files = sorted(chunks_dir.glob('chunk_*.pt'))

output_path = Path('/content/drive/MyDrive/GoGame_ML/datasets/labeled_19x19_2019.pt')
merge_chunks(chunk_files, output_path)
```

### 6.2. Session Timeout

**Triệu chứng:**
- Runtime bị disconnect sau 90 phút (free tier)

**Giải pháp:**
1. **Incremental save đã tự động xử lý**: Nếu crash, chunks đã được save
2. **Resume từ chunks**: Load và merge chunks còn lại

```python
# Merge chunks còn lại (nếu crash)
from generate_labels_colab import merge_chunks
from pathlib import Path

chunks_dir = WORK_DIR / 'datasets' / 'labeled_19x19_2019_chunks'
chunk_files = sorted(chunks_dir.glob('chunk_*.pt'))

output_path = WORK_DIR / 'datasets' / 'labeled_19x19_2019.pt'
merge_chunks(chunk_files, output_path)
```

### 6.3. Import Error

**Triệu chứng:**
```
ModuleNotFoundError: No module named 'generate_features_colab'
```

**Giải pháp:**
1. Đảm bảo đã upload `generate_features_colab.py` vào `code/`
2. Hoặc copy code vào notebook

```python
# Copy code trực tiếp vào notebook
# (Xem scripts/generate_features_colab.py)
```

### 6.4. Slow Processing

**Triệu chứng:**
- Tốc độ < 50 pos/s

**Giải pháp:**
1. Colab free tier có giới hạn CPU
2. Xử lý trên local với multiprocessing sẽ nhanh hơn
3. Hoặc upgrade Colab Pro

### 6.5. Chunks Không Merge

**Triệu chứng:**
- Có chunks nhưng không có file merged

**Giải pháp:**
```python
# Merge thủ công
from generate_labels_colab import merge_chunks
from pathlib import Path

chunks_dir = WORK_DIR / 'datasets' / 'labeled_19x19_2019_chunks'
chunk_files = sorted(chunks_dir.glob('chunk_*.pt'))

if chunk_files:
    output_path = WORK_DIR / 'datasets' / 'labeled_19x19_2019.pt'
    merge_chunks(chunk_files, output_path)
    print(f"✅ Merged {len(chunk_files)} chunks")
else:
    print("⚠️  No chunks found")
```

---

## 7. BEST PRACTICES

### 7.1. Chunk Size

- **Small dataset (<100K)**: Không cần incremental save
- **Medium (100K-500K)**: `save_chunk_size=50000`
- **Large (>500K)**: `save_chunk_size=30000` hoặc xử lý trên local

### 7.2. Batch Processing

Xử lý từng năm/file riêng biệt để:
- Dễ monitor progress
- Tránh timeout
- Dễ resume nếu crash

### 7.3. Backup

- Files đã được lưu vào Google Drive (tự động backup)
- Giữ chunks để có thể merge lại nếu cần

### 7.4. Monitoring

Theo dõi:
- Memory usage warnings
- Processing speed
- Chunk save frequency

---

## 8. QUICK REFERENCE

### Command Template

```python
from generate_labels_colab import process_dataset_file
from pathlib import Path

WORK_DIR = Path('/content/drive/MyDrive/GoGame_ML')

process_dataset_file(
    input_path=WORK_DIR / 'processed' / 'positions_19x19_2019.pt',
    output_path=WORK_DIR / 'datasets' / 'labeled_19x19_2019.pt',
    filter_handicap=True,
    save_chunk_size=50000,
    auto_enable_incremental=True
)
```

### Check Progress

```python
# Xem chunks đã save
chunks_dir = WORK_DIR / 'datasets' / 'labeled_19x19_2019_chunks'
chunks = sorted(chunks_dir.glob('chunk_*.pt'))
print(f"📦 Saved {len(chunks)} chunks")
for chunk in chunks:
    data = torch.load(chunk, map_location='cpu', weights_only=False)
    print(f"   {chunk.name}: {data['total_samples']:,} samples")
```

---

## 9. NEXT STEPS

Sau khi gán nhãn xong:

1. **Verify Dataset**: Kiểm tra số lượng samples và format
2. **Merge Years** (nếu cần): Dùng `merge_datasets.py` để gộp nhiều năm
3. **Train Model**: Sử dụng `train_colab.py` để train model

Xem thêm:
- `docs/ML_TRAINING_COLAB_GUIDE.md` - Hướng dẫn training
- `scripts/README_COLAB_TRAINING.md` - Quick start

---

**Chúc bạn thành công! 🎉**

