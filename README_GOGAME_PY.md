# gogame_py Module - Build & Setup Guide

## Tổng quan

Module `gogame_py` là Python binding cho C++ AI engine (Minimax, MCTS). Module được build với **MinGW** và yêu cầu **MSYS2 Python** để chạy (tránh DLL conflicts với MSVC Python).

## Build Module

### 1. Yêu cầu

- **MSYS2 MinGW 64-bit shell**
- **CMake** (cài qua `pacman -S mingw-w64-x86_64-cmake`)
- **pybind11** (cài qua `pacman -S mingw-w64-x86_64-pybind11`)

### 2. Build

```bash
# Trong MSYS2 MinGW 64-bit shell
cd /d/Hai/study/TTNT/GoGame
mkdir -p build
cd build
cmake .. -G "Ninja" -DCMAKE_BUILD_TYPE=Release
cmake --build . --target gogame_py
```

Module sẽ được tạo tại: `build/gogame_py.cp312-mingw_x86_64_msvcrt_gnu.pyd`

### 3. Test Module

```bash
# Dùng MSYS2 Python
/c/msys64/mingw64/bin/python3 scripts/test_gogame_py_msys2.py
```

## Sử dụng trong Backend

Backend có **2 modes** để sử dụng AI:

### Mode 1: Direct Import (không khả dụng với venv Python)

Nếu backend chạy với **MSYS2 Python**, có thể import trực tiếp:

```python
import gogame_py
ai = gogame_py.AIPlayer()
move = ai.select_move(board, level=1)
```

### Mode 2: Subprocess Wrapper (khuyến nghị)

Backend tự động fallback sang **subprocess wrapper** nếu direct import fail. Wrapper gọi AI qua MSYS2 Python subprocess:

```python
# Backend tự động detect và dùng wrapper
from app.utils.ai_wrapper import call_ai_select_move

move_data = call_ai_select_move(board_state, level=1)
```

## Setup cho Backend

### Option A: Dùng MSYS2 Python cho Backend (đơn giản nhất)

```bash
# Tạo venv với MSYS2 Python
/c/msys64/mingw64/bin/python3 -m venv venv_msys2
source venv_msys2/bin/activate

# Install dependencies
pip install -r backend/requirements.txt

# Run server
cd backend
uvicorn app.main:app --reload
```

### Option B: Dùng venv Python + Wrapper (hiện tại)

Backend đã được config để tự động dùng wrapper nếu direct import fail. Chỉ cần đảm bảo:

1. **MSYS2 Python** có sẵn tại: `C:/msys64/mingw64/bin/python3.exe`
2. **Module đã build** tại: `build/gogame_py*.pyd`

Backend sẽ tự động detect và dùng wrapper.

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

### DLL load failed

**Nguyên nhân**: Module build với MinGW nhưng Python đang dùng MSVC runtime.

**Giải pháp**:
1. Dùng MSYS2 Python: `/c/msys64/mingw64/bin/python3`
2. Hoặc dùng wrapper (backend tự động)

### Module not found

**Nguyên nhân**: Module chưa build hoặc không trong Python path.

**Giải pháp**:
```bash
# Build module
cmake --build build --target gogame_py

# Test với MSYS2 Python
/c/msys64/mingw64/bin/python3 -c "import sys; sys.path.insert(0, 'build'); import gogame_py"
```

### Wrapper subprocess failed

**Nguyên nhân**: MSYS2 Python không tìm thấy hoặc path sai.

**Giải pháp**: Kiểm tra `C:/msys64/mingw64/bin/python3.exe` tồn tại.

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

