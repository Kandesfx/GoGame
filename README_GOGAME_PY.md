# gogame_py Module - Build & Setup Guide

## Tổng quan

Module `gogame_py` là Python binding cho C++ AI engine (Minimax, MCTS). Module được build với **MinGW** và yêu cầu **MSYS2 Python** để chạy (tránh DLL conflicts với MSVC Python).

**⚠️ QUAN TRỌNG**: Module phải được build và đặt trong thư mục `build/` để backend nhận diện đúng. Backend sẽ **chỉ** sử dụng module nếu nó nằm trong `build/`.

## Build Module

### 1. Yêu cầu

- **MSYS2** đã cài đặt (tải từ: https://www.msys2.org/)
- **MSYS2 MinGW 64-bit shell** (mở từ Start Menu: `MSYS2 MinGW 64-bit`)
- **CMake** (cài qua `pacman -S mingw-w64-x86_64-cmake`)
- **pybind11** (cài qua `pacman -S mingw-w64-x86_64-pybind11`)
- **Ninja** (tùy chọn, nhanh hơn: `pacman -S mingw-w64-x86_64-ninja`)

### 2. Cài đặt Dependencies (nếu chưa có)

```bash
# Trong MSYS2 MinGW 64-bit shell
pacman -Syu  # Update package database
pacman -S mingw-w64-x86_64-cmake
pacman -S mingw-w64-x86_64-pybind11
pacman -S mingw-w64-x86_64-ninja  # Tùy chọn nhưng khuyến nghị
```

### 3. Build Module

#### Cách 1: Sử dụng Script Tự Động (Khuyến nghị)

**Windows (CMD/PowerShell):**
```bash
# Từ thư mục root của project
scripts\build_gogame_py_simple.bat
```

**Linux/Mac/MSYS2:**
```bash
# Từ thư mục root của project
chmod +x scripts/build_gogame_py_simple.sh
./scripts/build_gogame_py_simple.sh
```

#### Cách 2: Build Thủ Công

**Trong MSYS2 MinGW 64-bit shell:**

```bash
# Di chuyển đến thư mục project
cd /d/Hai/study/TTNT/GoGame

# Tạo thư mục build (nếu chưa có)
mkdir -p build
cd build

# Configure CMake (chỉ cần chạy 1 lần, hoặc khi thay đổi CMakeLists.txt)
cmake .. -G "Ninja" -DCMAKE_BUILD_TYPE=Release
# Hoặc không dùng Ninja:
# cmake .. -DCMAKE_BUILD_TYPE=Release

# Build module
cmake --build . --target gogame_py
```

**Lưu ý:**
- Nếu dùng `-G "Ninja"`, cần cài Ninja trước
- Nếu không dùng Ninja, CMake sẽ dùng Makefiles mặc định
- File output sẽ có tên dạng: `gogame_py.cp312-mingw_x86_64_msvcrt_gnu.pyd` (số version Python có thể khác)
- Nếu thấy `ninja: no work to do` → Module đã được build và không có thay đổi, không cần rebuild

**Force Rebuild (nếu cần rebuild lại từ đầu):**
```bash
# Xóa file cũ và rebuild
rm gogame_py*.pyd  # Xóa file cũ
cmake --build . --target gogame_py  # Build lại

# Hoặc xóa toàn bộ build directory và build lại
cd ..
rm -rf build
mkdir build
cd build
cmake .. -G "Ninja" -DCMAKE_BUILD_TYPE=Release
cmake --build . --target gogame_py
```

### 4. Kiểm tra Build Thành Công

Sau khi build, kiểm tra file đã được tạo:

```bash
# Trong MSYS2 shell
ls -lh build/gogame_py*.pyd

# Hoặc trong Windows CMD
dir build\gogame_py*.pyd
```

File sẽ có dạng: `gogame_py.cp312-mingw_x86_64_msvcrt_gnu.pyd` (hoặc tương tự)

**⚠️ QUAN TRỌNG**: File phải nằm trong thư mục `build/` (không phải root hoặc thư mục khác). Backend sẽ chỉ nhận diện module nếu nó ở đúng vị trí này.

### 5. Test Module

#### Test với MSYS2 Python:

```bash
# Trong MSYS2 MinGW 64-bit shell
/c/msys64/mingw64/bin/python3 scripts/test_gogame_py_msys2.py
```

Hoặc test trực tiếp:

```bash
/c/msys64/mingw64/bin/python3 -c "
import sys
sys.path.insert(0, 'build')
import gogame_py
print('✅ Import thành công!')
board = gogame_py.Board(9)
print(f'✅ Board created: size={board.size()}')
ai = gogame_py.AIPlayer()
print('✅ AIPlayer created')
print('✅ Module hoạt động!')
"
```

## Sử dụng trong Backend

Backend có **3 modes** để sử dụng AI (theo thứ tự ưu tiên):

### Mode 1: Direct Import (Ưu tiên cao nhất)

Backend sẽ tự động kiểm tra xem `gogame_py` có thể import được và có nằm trong `build/` không:

- ✅ Nếu module có trong `build/` → Sử dụng direct import (nhanh nhất)
- ❌ Nếu module không có trong `build/` → Bỏ qua, chuyển sang mode khác

**Lưu ý**: Direct import chỉ hoạt động nếu:
1. Module được build và nằm trong `build/`
2. Backend Python có thể import được (thường chỉ với MSYS2 Python)

### Mode 2: Subprocess Wrapper (Fallback)

Nếu direct import không khả dụng, backend sẽ dùng subprocess wrapper:

- Wrapper gọi AI qua MSYS2 Python subprocess
- Yêu cầu: MSYS2 Python tại `C:/msys64/mingw64/bin/python3.exe`
- Module vẫn phải có trong `build/` để wrapper hoạt động

### Mode 3: ML Model (Fallback cuối cùng)

Nếu cả direct import và wrapper đều không khả dụng, backend sẽ dùng ML model (nếu có).

## Setup cho Backend

### Option A: Dùng MSYS2 Python cho Backend (Đơn giản nhất)

```bash
# Tạo venv với MSYS2 Python
/c/msys64/mingw64/bin/python3 -m venv venv_msys2
source venv_msys2/bin/activate  # Windows: venv_msys2\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt

# Run server
cd backend
uvicorn app.main:app --reload
```

**Ưu điểm**: Direct import hoạt động, không cần wrapper.

### Option B: Dùng venv Python + Wrapper (Khuyến nghị cho Production)

Backend đã được config để tự động dùng wrapper nếu direct import fail. Chỉ cần đảm bảo:

1. **MSYS2 Python** có sẵn tại: `C:/msys64/mingw64/bin/python3.exe`
2. **Module đã build** tại: `build/gogame_py*.pyd` (⚠️ phải trong `build/`)

Backend sẽ tự động detect và dùng wrapper.

## Kiểm tra Backend Nhận Diện Module

### 1. Kiểm tra nhanh (không cần start backend)

Chạy script kiểm tra:
```bash
python scripts/check_ai_availability.py
```

Script sẽ kiểm tra:
- ✅ File có trong `build/` không
- ✅ Direct import có hoạt động không
- ✅ Wrapper có sẵn không
- ✅ Backend config

### 2. Kiểm tra khi start backend

Khi start backend, kiểm tra logs:

**✅ Thành công:**
```
✅ gogame_py module loaded successfully from build directory: D:\Hai\study\TTNT\GoGame\build\gogame_py.cp312-mingw_x86_64_msvcrt_gnu.pyd
```

**⚠️ Cảnh báo (module ở sai vị trí):**
```
⚠️ gogame_py module found but NOT in build directory!
⚠️ Module path: D:\Hai\study\TTNT\GoGame\gogame_py.pyd
⚠️ Expected in: D:\Hai\study\TTNT\GoGame\build
```

→ Module đang ở sai vị trí. Cần xóa file ở root và đảm bảo chỉ có file trong `build/`.

**❌ Lỗi (module không tìm thấy):**
```
gogame_py module not found. AI features will be disabled.
✅ AI wrapper loaded successfully
```

→ Direct import không hoạt động, nhưng wrapper sẽ được dùng (vẫn OK).

### 3. Kiểm tra AI hoạt động

Sau khi start backend, thử tạo match với AI:
- Nếu AI đánh được → ✅ OK
- Nếu AI không đánh → Kiểm tra logs để xem lỗi gì

## Test

### Test Module trực tiếp

```bash
/c/msys64/mingw64/bin/python3 scripts/test_gogame_py_msys2.py
```

### Test Backend với AI

```bash
# Start server
cd backend
uvicorn app.main:app --reload

# Trong terminal khác
python scripts/test_backend_with_ai.py
```

## Troubleshooting

### "ninja: no work to do" khi build

**Nguyên nhân**: Module đã được build và không có thay đổi trong source code, nên Ninja không cần rebuild.

**Giải pháp**:
- ✅ **Bình thường**: Nếu file `gogame_py*.pyd` đã tồn tại trong `build/`, không cần làm gì thêm
- 🔄 **Muốn rebuild**: Nếu muốn force rebuild (ví dụ sau khi sửa code C++):
  ```bash
  # Cách 1: Xóa file cũ và build lại
  cd build
  rm gogame_py*.pyd
  cmake --build . --target gogame_py
  
  # Cách 2: Clean và rebuild toàn bộ
  cd build
  cmake --build . --target clean
  cmake --build . --target gogame_py
  
  # Cách 3: Xóa build directory và build lại từ đầu
  cd ..
  rm -rf build
  mkdir build
  cd build
  cmake .. -G "Ninja" -DCMAKE_BUILD_TYPE=Release
  cmake --build . --target gogame_py
  ```

### Module not found trong build/

**Nguyên nhân**: Module chưa build hoặc build ở vị trí sai.

**Giải pháp**:
```bash
# 1. Xóa thư mục build cũ (nếu cần)
rm -rf build  # MSYS2/Linux
# hoặc
rmdir /s /q build  # Windows CMD

# 2. Build lại
mkdir build
cd build
cmake .. -G "Ninja" -DCMAKE_BUILD_TYPE=Release
cmake --build . --target gogame_py

# 3. Kiểm tra file đã được tạo
ls -lh gogame_py*.pyd
```

### Module found nhưng không trong build/

**Nguyên nhân**: Có file `gogame_py.pyd` ở root hoặc venv, nhưng không có trong `build/`.

**Giải pháp**:
```bash
# 1. Xóa file ở root (nếu có)
rm gogame_py.pyd  # MSYS2/Linux
# hoặc
del gogame_py.pyd  # Windows CMD

# 2. Build lại module vào build/
cd build
cmake --build . --target gogame_py

# 3. Kiểm tra chỉ có file trong build/
ls -lh build/gogame_py*.pyd
```

### DLL load failed

**Nguyên nhân**: Module build với MinGW nhưng Python đang dùng MSVC runtime.

**Giải pháp**:
1. Dùng MSYS2 Python: `/c/msys64/mingw64/bin/python3`
2. Hoặc dùng wrapper (backend tự động)

### CMake không tìm thấy pybind11

**Nguyên nhân**: pybind11 chưa cài hoặc CMake không tìm thấy.

**Giải pháp**:
```bash
# Cài pybind11 qua MSYS2
pacman -S mingw-w64-x86_64-pybind11

# Hoặc cài qua pip (cho Python hiện tại)
pip install pybind11

# Sau đó rebuild
cd build
cmake .. -G "Ninja" -DCMAKE_BUILD_TYPE=Release
cmake --build . --target gogame_py
```

### Wrapper subprocess failed

**Nguyên nhân**: MSYS2 Python không tìm thấy hoặc path sai.

**Giải pháp**: 
1. Kiểm tra `C:/msys64/mingw64/bin/python3.exe` tồn tại
2. Nếu path khác, sửa trong `backend/app/utils/ai_wrapper.py`:
   ```python
   MSYS2_PYTHON = Path("C:/msys64/mingw64/bin/python3.exe")
   ```

### Build thành công nhưng backend vẫn báo "AI not available"

**Nguyên nhân**: 
1. Module không nằm trong `build/`
2. Backend chưa restart sau khi build
3. Có file cũ ở vị trí khác đang được import

**Giải pháp**:
1. Kiểm tra file có trong `build/`:
   ```bash
   ls -lh build/gogame_py*.pyd
   ```
2. Xóa file ở root (nếu có):
   ```bash
   rm gogame_py.pyd  # hoặc del gogame_py.pyd trên Windows
   ```
3. Restart backend
4. Kiểm tra logs khi start backend để xem module có được load không

## API Reference

### Board

```python
board = gogame_py.Board(size=9)
board.size()  # Get board size
board.current_player()  # Get current player (Color.Black or Color.White)
board.get_legal_moves(color)  # Get legal moves for color
board.is_legal_move(move)  # Check if move is legal
board.make_move(move)  # Apply move
board.undo_move()  # Undo last move
board.get_prisoners(color)  # Get prisoners count
```

### AIPlayer

```python
ai = gogame_py.AIPlayer()
move = ai.select_move(board, level=1)  # level: 1-4
# move.x, move.y, move.is_pass, move.color
```

### MinimaxEngine

```python
from gogame_py import MinimaxConfig, MinimaxEngine

config = MinimaxConfig()
config.max_depth = 3
config.use_alpha_beta = True
engine = MinimaxEngine(config)

result = engine.search(board, color)
# result.best_move, result.evaluation, result.nodes_searched
```

### MCTSEngine

```python
from gogame_py import MCTSConfig, MCTSEngine

config = MCTSConfig(num_playouts=1000, time_limit_seconds=5.0)
engine = MCTSEngine(config)

result = engine.search(board, color)
# result.best_move, result.win_rate, result.total_visits
```

## Tóm tắt Build Process

1. **Cài đặt MSYS2** và mở MinGW 64-bit shell
2. **Cài dependencies**: `pacman -S mingw-w64-x86_64-cmake mingw-w64-x86_64-pybind11 mingw-w64-x86_64-ninja`
3. **Build module**: 
   ```bash
   cd build
   cmake .. -G "Ninja" -DCMAKE_BUILD_TYPE=Release
   cmake --build . --target gogame_py
   ```
4. **Kiểm tra**: File `gogame_py*.pyd` phải có trong `build/`
5. **Test**: Dùng MSYS2 Python để test module
6. **Restart backend**: Backend sẽ tự động nhận diện module từ `build/`
