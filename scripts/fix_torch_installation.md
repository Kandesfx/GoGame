# 🔧 Fix Torch Installation Error

## ❌ Vấn Đề

Bạn đang gặp lỗi:
```
ERROR: Could not find a version that satisfies the requirement torch (from versions: none)
ERROR: No matching distribution found for torch
```

**Nguyên nhân:** Bạn đang dùng MSYS2 Python (GCC-compiled), nhưng PyTorch không có pre-built wheels cho MSYS2 Python trên Windows.

## ✅ Giải Pháp

### Option 1: Cài Windows Python (Khuyến Nghị)

1. **Tải và cài Python từ Windows:**
   - Truy cập: https://www.python.org/downloads/
   - Tải Python 3.10+ (Windows installer)
   - Khi cài, **chọn "Add Python to PATH"**

2. **Tạo venv mới với Windows Python:**
   ```bash
   # Trong Git Bash hoặc PowerShell
   /c/Users/$USER/AppData/Local/Programs/Python/Python*/python.exe -m venv venv_windows
   ```

3. **Activate và cài đặt:**
   ```bash
   # Git Bash
   source venv_windows/Scripts/activate
   pip install sgf numpy torch tqdm
   
   # Hoặc Windows CMD
   venv_windows\Scripts\activate.bat
   pip install sgf numpy torch tqdm
   ```

### Option 2: Dùng Python Launcher (nếu đã có Windows Python)

Nếu bạn đã có Windows Python nhưng chưa trong PATH:

```bash
# Tìm Python Windows
cmd.exe /c "where python"

# Hoặc tìm thủ công
ls /c/Users/$USER/AppData/Local/Programs/Python/Python*/python.exe

# Dùng Python đó để cài đặt
/c/Users/$USER/AppData/Local/Programs/Python/Python*/python.exe -m pip install sgf numpy torch tqdm
```

### Option 3: Dùng Anaconda/Miniconda

Nếu bạn có Anaconda hoặc Miniconda:

```bash
# Tạo environment mới
conda create -n gogame python=3.10
conda activate gogame

# Cài đặt packages
conda install pytorch cpuonly -c pytorch
pip install sgf numpy tqdm
```

### Option 4: Build PyTorch từ Source (Không khuyến nghị)

Nếu bạn muốn tiếp tục dùng MSYS2 Python, bạn cần build PyTorch từ source, điều này rất phức tạp và tốn thời gian.

## 🎯 Quick Check

Sau khi cài đặt, kiểm tra:

```bash
python --version  # Nên là CPython, không phải GCC
python -c "import torch; print(torch.__version__)"
```

## 📝 Lưu Ý

- **MSYS2 Python** (GCC-compiled) không tương thích với PyTorch wheels
- **Windows Python** (CPython) là lựa chọn tốt nhất
- Luôn dùng **virtual environment** để tránh conflict

