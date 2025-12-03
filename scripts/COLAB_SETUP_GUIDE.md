# 🚀 Hướng Dẫn Setup Colab Cho Label Generation

## 📋 Bước 1: Upload Files Lên Google Drive

### Cách 1: Upload Thủ Công

1. Mở Google Drive: https://drive.google.com
2. Tạo thư mục: `GoGame_ML/scripts/`
3. Upload các files sau vào `GoGame_ML/scripts/`:
   - `generate_labels_colab.py`
   - `label_generators.py`
   - `generate_features_colab.py`
   - `setup_colab_labels.py` (optional, để setup dễ hơn)

### Cách 2: Upload Từ Colab

```python
# Cell 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 2: Upload files (chạy lệnh này trong terminal Colab)
# !mkdir -p /content/drive/MyDrive/GoGame_ML/scripts
# Sau đó upload files qua UI hoặc dùng wget/git clone
```

### Cách 3: Clone từ GitHub (nếu có repo)

```python
# Nếu code đã push lên GitHub
!git clone https://github.com/your-repo/GoGame.git /content/drive/MyDrive/GoGame_ML
```

## 📋 Bước 2: Setup Python Path

### Cách 1: Dùng Setup Script (Khuyến Nghị)

```python
# Cell 1: Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 2: Upload setup script (hoặc copy-paste nội dung)
# Upload setup_colab_labels.py lên Colab, sau đó:
exec(open('/content/drive/MyDrive/GoGame_ML/scripts/setup_colab_labels.py').read())
```

### Cách 2: Setup Thủ Công

```python
# Cell 1: Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 2: Add to Python path
import sys
from pathlib import Path

SCRIPTS_DIR = Path('/content/drive/MyDrive/GoGame_ML/scripts')
sys.path.insert(0, str(SCRIPTS_DIR))

# Verify
print(f"Scripts directory: {SCRIPTS_DIR}")
print(f"Exists: {SCRIPTS_DIR.exists()}")
```

### Cách 3: Copy Files Trực Tiếp Vào Colab

```python
# Cell 1: Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 2: Copy files vào /content (temporary)
import shutil
from pathlib import Path

drive_scripts = Path('/content/drive/MyDrive/GoGame_ML/scripts')
local_scripts = Path('/content/scripts')
local_scripts.mkdir(exist_ok=True)

# Copy files
for file in ['generate_labels_colab.py', 'label_generators.py', 'generate_features_colab.py']:
    src = drive_scripts / file
    dst = local_scripts / file
    if src.exists():
        shutil.copy(src, dst)
        print(f"✅ Copied {file}")

# Add to path
import sys
sys.path.insert(0, str(local_scripts))
```

## 📋 Bước 3: Verify Import

```python
# Test import
try:
    from generate_labels_colab import process_dataset_file
    from label_generators import ThreatLabelGenerator
    from generate_features_colab import board_to_features_17_planes
    print("✅ All modules imported successfully!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nPlease check:")
    print("1. Files are uploaded to correct location")
    print("2. Python path is set correctly")
```

## 📋 Bước 4: Sử Dụng

```python
from pathlib import Path
from generate_labels_colab import process_dataset_file

WORK_DIR = Path('/content/drive/MyDrive/GoGame_ML')

# Process một file
process_dataset_file(
    input_path=WORK_DIR / 'processed' / 'positions_19x19_2019.pt',
    output_path=WORK_DIR / 'datasets' / 'labeled_19x19_2019.pt',
    filter_handicap=True,
    save_chunk_size=50000
)
```

## 🔧 Troubleshooting

### Lỗi: `ModuleNotFoundError: No module named 'generate_labels_colab'`

**Nguyên nhân**: File chưa được upload hoặc path chưa đúng.

**Giải pháp**:
1. Kiểm tra file có tồn tại:
   ```python
   from pathlib import Path
   file_path = Path('/content/drive/MyDrive/GoGame_ML/scripts/generate_labels_colab.py')
   print(f"File exists: {file_path.exists()}")
   ```

2. Kiểm tra Python path:
   ```python
   import sys
   print("Python paths:")
   for p in sys.path:
       print(f"  {p}")
   ```

3. Thêm path thủ công:
   ```python
   import sys
   sys.path.insert(0, '/content/drive/MyDrive/GoGame_ML/scripts')
   ```

### Lỗi: `ModuleNotFoundError: No module named 'label_generators'`

**Nguyên nhân**: File `label_generators.py` chưa được upload.

**Giải pháp**: Upload file `label_generators.py` vào cùng thư mục.

### Lỗi: `ModuleNotFoundError: No module named 'generate_features_colab'`

**Nguyên nhân**: File `generate_features_colab.py` chưa được upload.

**Giải pháp**: Upload file `generate_features_colab.py` vào cùng thư mục.

## 📁 Cấu Trúc Thư Mục Khuyến Nghị

```
/content/drive/MyDrive/GoGame_ML/
├── scripts/
│   ├── generate_labels_colab.py
│   ├── label_generators.py
│   ├── generate_features_colab.py
│   └── setup_colab_labels.py
├── processed/
│   ├── positions_19x19_2019.pt
│   ├── positions_19x19_2018.pt
│   └── ...
└── datasets/
    ├── labeled_19x19_2019.pt
    └── ...
```

## 🚀 Quick Start (Copy-Paste Ready)

```python
# === CELL 1: Setup ===
from google.colab import drive
drive.mount('/content/drive')

import sys
from pathlib import Path

# Add scripts to path
SCRIPTS_DIR = Path('/content/drive/MyDrive/GoGame_ML/scripts')
sys.path.insert(0, str(SCRIPTS_DIR))

# Verify
print(f"✅ Scripts directory: {SCRIPTS_DIR}")
print(f"✅ Exists: {SCRIPTS_DIR.exists()}")

# === CELL 2: Import ===
try:
    from generate_labels_colab import process_dataset_file
    print("✅ generate_labels_colab imported")
except ImportError as e:
    print(f"❌ Error: {e}")
    print("Please upload generate_labels_colab.py to scripts directory")

# === CELL 3: Use ===
from pathlib import Path
WORK_DIR = Path('/content/drive/MyDrive/GoGame_ML')

process_dataset_file(
    input_path=WORK_DIR / 'processed' / 'positions_19x19_2019.pt',
    output_path=WORK_DIR / 'datasets' / 'labeled_19x19_2019.pt',
    filter_handicap=True,
    save_chunk_size=50000
)
```

## 📝 Checklist

Trước khi chạy, đảm bảo:

- [ ] Google Drive đã được mount
- [ ] Thư mục `GoGame_ML/scripts/` đã được tạo
- [ ] Các files đã được upload:
  - [ ] `generate_labels_colab.py`
  - [ ] `label_generators.py`
  - [ ] `generate_features_colab.py`
- [ ] Python path đã được thêm
- [ ] Import test đã pass

## 🔗 Liên Quan

- `scripts/UPDATE_COLAB_LABELS.md` - Tài liệu về multi-task labels trên Colab
- `scripts/generate_labels_colab.py` - Script chính
- `scripts/label_generators.py` - Label generators

