# 🔧 HƯỚNG DẪN SETUP CHO MSYS2/GIT BASH

## ⚠️ Vấn Đề

MSYS2 Python không có pre-built wheels cho `numpy` và `torch`, cần compile từ source (phức tạp).

## ✅ Giải Pháp

### Option 1: Dùng Python từ Windows (Khuyến Nghị)

1. **Kiểm tra Python Windows:**
   ```bash
   # Trong Git Bash hoặc PowerShell
   py --version
   # hoặc
   python --version  # Nếu đã add vào PATH
   ```

2. **Cài đặt packages:**
   ```bash
   py -m pip install sgf numpy torch tqdm
   ```

3. **Chạy script:**
   ```bash
   py scripts/parse_sgf_local.py --input data/raw_sgf --output data/processed --year 2019
   ```

### Option 2: Cài qua MSYS2 Packages + pip

```bash
# Cài đặt packages có sẵn qua pacman
pacman -S mingw-w64-x86_64-python-numpy
pacman -S mingw-w64-x86_64-python-tqdm

# Cài đặt packages không có trong MSYS2 qua pip
python -m pip install sgf
python -m pip install torch --index-url https://download.pytorch.org/whl/cu118
# Hoặc CPU-only:
# python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Lưu ý:** `sgf` và `torch` không có trong MSYS2 repos, phải cài qua pip.

### Option 3: Dùng Virtual Environment với Python Windows

1. **Tạo venv với Python Windows:**
   ```bash
   # Tìm Python Windows
   where python  # Windows
   which python  # Linux/Mac
   
   # Tạo venv (dùng Python Windows, không phải MSYS2)
   py -m venv venv_windows
   ```

2. **Activate và cài đặt:**
   ```bash
   # Windows CMD
   venv_windows\Scripts\activate.bat
   pip install sgf numpy torch tqdm
   
   # Git Bash/MSYS2
   source venv_windows/Scripts/activate
   pip install sgf numpy torch tqdm
   ```

3. **Chạy script:**
   ```bash
   source venv_windows/Scripts/activate
   python scripts/parse_sgf_local.py --input data/raw_sgf --output data/processed --year 2019
   ```

## 🎯 Quick Fix cho Lỗi Hiện Tại

Nếu bạn đang dùng MSYS2 Python và gặp lỗi, hãy:

1. **Tìm Python Windows:**
   ```bash
   # Trong Git Bash
   /c/Users/$USER/AppData/Local/Programs/Python/Python*/python.exe --version
   ```

2. **Dùng Python đó để cài đặt:**
   ```bash
   /c/Users/$USER/AppData/Local/Programs/Python/Python*/python.exe -m pip install sgf numpy torch tqdm
   ```

3. **Tạo alias để dễ dùng:**
   ```bash
   # Thêm vào ~/.bashrc
   alias pythonw='/c/Users/$USER/AppData/Local/Programs/Python/Python*/python.exe'
   
   # Sau đó dùng:
   pythonw scripts/parse_sgf_local.py --input data/raw_sgf --output data/processed --year 2019
   ```

## 📝 Kiểm Tra

```bash
# Kiểm tra Python nào đang được dùng
which python
python --version

# Kiểm tra packages đã cài
python -c "import sgf; import numpy; import torch; import tqdm; print('OK')"
```

## 💡 Tips

- **Nếu có nhiều Python:** Dùng `py` launcher trên Windows (tự động chọn đúng version)
- **Nếu vẫn lỗi:** Thử dùng Python từ Anaconda/Miniconda
- **Virtual Environment:** Luôn dùng venv để tránh conflict

