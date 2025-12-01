# 📦 HƯỚNG DẪN CÀI ĐẶT CHO MSYS2

## ✅ Đã Cài Đặt

Bạn đã cài thành công:
- ✅ `mingw-w64-x86_64-python-numpy`
- ✅ `mingw-w64-x86_64-python-tqdm`

## ⚠️ Còn Thiếu

Cần cài thêm:
- ❌ `sgf` (không có trong MSYS2 repos)
- ❌ `torch` (không có trong MSYS2 repos)
- ❌ `pip` (để cài sgf và torch)

## 🔧 Bước Tiếp Theo

### Bước 1: Cài pip cho MSYS2 Python

```bash
pacman -S mingw-w64-x86_64-python-pip
```

### Bước 2: Cài sgf và torch qua pip

```bash
# Cài sgf
python -m pip install sgf

# Cài torch (CPU version - nhẹ hơn)
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# Hoặc torch với CUDA (nếu có GPU)
python -m pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Bước 3: Kiểm Tra

```bash
python -c "import sgf; import numpy; import torch; import tqdm; print('✅ All packages installed!')"
```

### Bước 4: Test Script

```bash
python scripts/parse_sgf_local.py --help
```

## 📝 Tóm Tắt Lệnh

```bash
# Cài pip
pacman -S mingw-w64-x86_64-python-pip

# Cài sgf và torch
python -m pip install sgf
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# Kiểm tra
python -c "import sgf; import numpy; import torch; import tqdm; print('OK')"
```

## 🎯 Sau Khi Cài Xong

Bạn có thể chạy:

```bash
python scripts/parse_sgf_local.py \
    --input data/raw_sgf \
    --output data/processed \
    --year 2019
```

