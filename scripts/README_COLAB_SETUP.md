# 🚀 Hướng dẫn Setup Colab nhanh

## Bước 1: Chuẩn bị trên Local

Chạy script helper để tạo files cần thiết:

```bash
python scripts/setup_colab_helper.py
```

Script này sẽ tạo:
- `gogame_ml_code.zip` - File ZIP chứa code model
- `GoGame_ML_Training_Template.ipynb` - Notebook template (optional)

## Bước 2: Setup trên Colab

### 2.1. Tạo Notebook mới
- Vào https://colab.research.google.com
- File → New Notebook
- Enable GPU: Runtime → Change runtime type → GPU (T4)

### 2.2. Mount Drive và Setup thư mục

```python
# Cell 1: Mount Drive
from google.colab import drive
from pathlib import Path
import os

drive.mount('/content/drive')

WORK_DIR = Path('/content/drive/MyDrive/GoGame_ML')
WORK_DIR.mkdir(exist_ok=True)

# Tạo cấu trúc thư mục
(WORK_DIR / 'datasets').mkdir(exist_ok=True)
(WORK_DIR / 'code').mkdir(exist_ok=True)
(WORK_DIR / 'checkpoints').mkdir(exist_ok=True)
(WORK_DIR / 'logs').mkdir(exist_ok=True)
(WORK_DIR / 'outputs').mkdir(exist_ok=True)

os.chdir(WORK_DIR)
print(f"✅ Working directory: {WORK_DIR}")
```

### 2.3. Upload Code

```python
# Cell 2: Upload code ZIP
from google.colab import files
import zipfile

uploaded = files.upload()  # Chọn file gogame_ml_code.zip

for filename in uploaded.keys():
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(WORK_DIR / 'code')
        print(f"✅ Extracted {filename}")
```

### 2.4. Upload Dataset

```python
# Cell 3: Upload dataset
from google.colab import files
import shutil

uploaded = files.upload()  # Chọn file .pt

for filename in uploaded.keys():
    if filename.endswith('.pt'):
        shutil.move(filename, WORK_DIR / 'datasets' / filename)
        print(f"✅ Moved {filename} to datasets/")
```

### 2.5. Install Dependencies

```python
# Cell 4: Install packages
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install numpy pandas tqdm tensorboard scikit-learn
!pip install sgf

import sys
sys.path.insert(0, str(WORK_DIR / 'code'))
sys.path.insert(0, str(WORK_DIR / 'code' / 'models'))

print("✅ Dependencies installed")
```

### 2.6. Verify Setup

```python
# Cell 5: Verify
import torch
from pathlib import Path

print("🔍 Verifying setup...")

# Check GPU
print(f"GPU: {torch.cuda.is_available()}")

# Check code
code_files = list((WORK_DIR / 'code' / 'models').glob('*.py'))
print(f"Code files: {len(code_files)}")

# Check dataset
dataset_files = list((WORK_DIR / 'datasets').glob('*.pt'))
print(f"Dataset files: {len(dataset_files)}")

if torch.cuda.is_available() and code_files and dataset_files:
    print("✅ Setup complete! Ready to train!")
else:
    print("⚠️  Please check missing items above")
```

## Bước 3: Bắt đầu Training

Xem chi tiết trong `docs/ML_TRAINING_COLAB_GUIDE.md` phần 5 (Quy trình Training).

## Cấu trúc Thư mục

```
Google Drive/MyDrive/GoGame_ML/
├── datasets/          ← Upload dataset .pt vào đây
├── code/              ← Upload code ZIP vào đây
│   └── models/
├── checkpoints/       (tự động tạo)
├── logs/              (tự động tạo)
└── outputs/           (tự động tạo)
```

## Lưu ý

- Dataset phải là file `.pt` (PyTorch) với format:
  ```python
  {
      'positions' hoặc 'labeled_data': [...],
      'board_size': 9,
      'total': 10000
  }
  ```
- Code model phải có đầy đủ các file trong `code/models/`
- Nếu dataset lớn (>1GB), nên upload lên Google Drive trước, rồi copy vào Colab

