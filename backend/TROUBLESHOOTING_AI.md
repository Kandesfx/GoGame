# Troubleshooting AI Features

## Vấn đề: AI không đánh sau khi user đánh

### Triệu chứng
- User đánh nước cờ
- AI không tự động đánh lại
- Có thể đánh liên tiếp như PvP offline

### Nguyên nhân

Có 3 nguyên nhân chính:

#### 1. Không có `gogame_py` module

**Triệu chứng:**
- Backend log hiển thị: `WARNING:root:gogame_py module not found. AI features will be disabled.`
- AI không đánh sau khi user đánh

**Nguyên nhân:**
- Chưa build C++ AI engine
- Module `gogame_py.pyd` (Windows) hoặc `gogame_py.so` (Linux) không có trong Python path

**Giải pháp:**
1. Build C++ AI engine (xem [INSTALLATION.md](../INSTALLATION.md))
2. Copy `gogame_py.pyd` vào thư mục backend hoặc thêm vào PYTHONPATH
3. Restart backend server

#### 2. Không có MSYS2 Python (cho wrapper)

**Triệu chứng:**
- Backend log hiển thị: `WARNING:root:AI wrapper not available either.`
- AI không đánh sau khi user đánh

**Nguyên nhân:**
- Wrapper cần MSYS2 Python tại `C:/msys64/mingw64/bin/python3.exe`
- Trên máy mới không có MSYS2

**Giải pháp:**
1. Cài đặt MSYS2 (xem [INSTALLATION.md](../INSTALLATION.md))
2. Build gogame_py module với MSYS2 Python
3. Hoặc build trực tiếp và dùng direct import (không cần wrapper)

#### 3. Logic AI move không được trigger

**Triệu chứng:**
- Không có warning về gogame_py
- Nhưng AI vẫn không đánh

**Nguyên nhân:**
- Logic kiểm tra `current_player` có thể sai
- AI move được gọi nhưng fail silently

**Giải pháp:**
1. Kiểm tra backend logs để xem có log về AI move không
2. Tìm các log bắt đầu với `🤖 [WRAPPER]` hoặc `🤖 [FALLBACK]`
3. Kiểm tra MongoDB để xem `current_player` có đúng không

## Cách Debug

### 1. Kiểm tra Backend Logs

Tìm các log sau trong backend console:

```
🤖 [FALLBACK] AI turn after user move
🤖 [WRAPPER] Starting AI move wrapper
✅ AI wrapper returned move
⚠️ [WRAPPER] AI wrapper returned no move
❌ [WRAPPER] AI wrapper not available
```

### 2. Kiểm tra MongoDB

```bash
# Kết nối MongoDB
mongosh

# Chọn database
use gogame

# Kiểm tra game state
db.games.findOne({"match_id": "your-match-id"})

# Kiểm tra:
# - current_player: Phải là "W" sau khi user (Black) đánh
# - moves: Phải có move của user và (nếu AI đã đánh) move của AI
```

### 3. Kiểm tra gogame_py Module

```python
# Trong Python shell
import gogame_py
print(gogame_py.__file__)  # Xem module được load từ đâu
```

### 4. Test AI Wrapper Trực tiếp

```python
# Test wrapper
from backend.app.utils.ai_wrapper import call_ai_select_move

board_state = {
    "board_size": 9,
    "moves": [{"number": 1, "color": "B", "position": [4, 4]}],
    "current_player": "W"
}

result = call_ai_select_move(board_state, level=1)
print(result)  # Phải có move hoặc None
```

## Giải pháp Tạm thời

Nếu không thể build C++ engine ngay, bạn có thể:

1. **Chỉ test PvP matches** - PvP không cần AI
2. **Sử dụng frontend để test** - Frontend có thể hoạt động mà không cần AI
3. **Build trên máy khác** - Build gogame_py trên máy có toolchain, copy sang máy mới

## Giải pháp Lâu dài

Để có đầy đủ tính năng AI:

1. **Cài đặt MSYS2** (Windows) hoặc build tools (Linux)
2. **Build C++ engine:**
   ```bash
   mkdir -p build
   cd build
   cmake ..
   cmake --build .
   ```
3. **Copy module vào Python path:**
   ```bash
   # Windows
   copy build\gogame_py.pyd backend\
   
   # Linux
   cp build/gogame_py.so backend/
   ```
4. **Restart backend server**

## Logs Quan trọng

Khi chạy backend, bạn sẽ thấy:

### Nếu không có gogame_py:
```
WARNING:root:gogame_py module not found. AI features will be disabled.
WARNING:root:AI wrapper not available either.
```

### Nếu có gogame_py nhưng wrapper không hoạt động:
```
🤖 [FALLBACK] AI turn after user move
🤖 [WRAPPER] Starting AI move wrapper
❌ [WRAPPER] AI wrapper not available
```

### Nếu AI move thành công:
```
🤖 [FALLBACK] AI turn after user move
🤖 [WRAPPER] Starting AI move wrapper
✅ AI wrapper returned move: {'x': 3, 'y': 3, 'is_pass': False, 'color': 'W'}
✅ [FALLBACK] AI move successful
```

## Liên hệ

Nếu vẫn gặp vấn đề sau khi thử các giải pháp trên, vui lòng:
1. Gửi backend logs đầy đủ
2. Gửi MongoDB game state
3. Gửi thông tin về môi trường (OS, Python version, có MSYS2 không)

