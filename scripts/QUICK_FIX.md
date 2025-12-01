# 🚀 QUICK FIX - Cài Đặt Dependencies

## ⚠️ Vấn Đề Hiện Tại

Bạn đang ở **Git Bash**, không phải **MSYS2 terminal**, nên:
- ❌ `pacman` không hoạt động
- ❌ Python MSYS2 không có `pip`

## ✅ Giải Pháp Nhanh

### Option 1: Dùng Python Windows (Khuyến Nghị - Dễ Nhất)

**Nếu có `py` command:**
```bash
# Trong Git Bash hoặc bất kỳ terminal nào
py -m pip install sgf numpy torch tqdm

# Kiểm tra
py -c "import sgf; import numpy; import torch; import tqdm; print('OK')"

# Chạy script
py scripts/parse_sgf_local.py --input data/raw_sgf --output data/processed --year 2019
```

**Nếu không có `py` command (dùng đường dẫn trực tiếp):**
```bash
# Cài đặt
/c/Users/HAI/AppData/Local/Programs/Python/Python312/python.exe -m pip install sgf numpy torch tqdm

# Kiểm tra
/c/Users/HAI/AppData/Local/Programs/Python/Python312/python.exe -c "import sgf; import numpy; import torch; import tqdm; print('OK')"

# Chạy script (dùng helper script)
bash scripts/run_with_python_windows.sh scripts/parse_sgf_local.py --input data/raw_sgf --output data/processed --year 2019
```

### Option 2: Mở MSYS2 Terminal

1. **Mở MSYS2 MinGW64 terminal:**
   - Từ Start Menu: `MSYS2 MinGW 64-bit`
   - Hoặc chạy: `C:\msys64\mingw64.exe`

2. **Chạy các lệnh:**
   ```bash
   cd /d/Hai/study/TTNT/GoGame
   
   # Cài pip
   pacman -S --noconfirm mingw-w64-x86_64-python-pip
   
   # Cài sgf và torch
   python -m pip install sgf
   python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
   
   # Kiểm tra
   python -c "import sgf; import numpy; import torch; import tqdm; print('OK')"
   ```

### Option 3: Dùng Script Helper

```bash
# Chạy script tự động (sẽ detect environment)
bash scripts/install_dependencies.sh
```

## 🎯 Khuyến Nghị

**Dùng Python Windows** (`py` command) vì:
- ✅ Không cần MSYS2
- ✅ Có sẵn pre-built wheels
- ✅ Dễ cài đặt
- ✅ Hoạt động từ mọi terminal

## 📝 Sau Khi Cài Xong

```bash
# Test script (dùng Python Windows)
/c/Users/HAI/AppData/Local/Programs/Python/Python312/python.exe scripts/parse_sgf_local.py --help

# Hoặc dùng helper script
bash scripts/run_with_python_windows.sh scripts/parse_sgf_local.py --help

# Parse một năm
bash scripts/run_with_python_windows.sh scripts/parse_sgf_local.py \
    --input data/raw_sgf \
    --output data/processed \
    --year 2019
```

## 🎯 Tạo Alias (Tùy Chọn)

Để dễ dùng hơn, thêm vào `~/.bashrc`:

```bash
# Thêm vào ~/.bashrc
alias pythonw='/c/Users/HAI/AppData/Local/Programs/Python/Python312/python.exe'

# Sau đó reload
source ~/.bashrc

# Dùng như bình thường
pythonw scripts/parse_sgf_local.py --input data/raw_sgf --output data/processed --year 2019
```

