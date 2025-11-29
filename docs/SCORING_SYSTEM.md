# Hệ Thống Tính Điểm Cờ Vây - Chi Tiết

## 📋 Tổng Quan

Trong cờ vây, điểm cuối trận được tính theo **luật Trung Quốc (Chinese Rules)**:
**Điểm = Số quân còn trên bàn + Lãnh thổ + Komi (Điểm bù)**

**Lưu ý quan trọng:** Số quân bị ăn (captured stones/prisoners) **KHÔNG** được tính vào điểm và **KHÔNG** được hiển thị trong giao diện người dùng. Chỉ có số quân còn trên bàn cờ và lãnh thổ được tính điểm.

## 🎯 Các Thành Phần Tính Điểm

### 1. Territory (Lãnh thổ) 🏔️

**Territory** là các giao điểm trống được bao quanh hoàn toàn bởi quân của một màu.

#### Cách tính (Flood-Fill Algorithm):
- Duyệt qua tất cả các giao điểm trống trên bàn cờ
- Với mỗi ô trống chưa được xử lý, dùng **flood-fill** để tìm tất cả các ô trống liên thông (kết nối với nhau)
- Kiểm tra xem vùng trống đó có được bao quanh hoàn toàn bởi một màu không:
  - Nếu chỉ có quân của một màu bao quanh → tính tất cả các ô trong vùng là territory của màu đó
  - Nếu có cả 2 màu bao quanh hoặc ra ngoài bàn cờ → không tính là territory (vùng tranh chấp)

#### Ví dụ:
```
Bảng 5×5:
. . . . .     . = trống
. B B . .     B = Black
. B . . .     W = White
. . W W .
. . W . .
```

**Phân tích bằng flood-fill:**

1. **Vùng trống ở góc trên trái (0,0)**: 
   - Flood-fill từ (0,0) → tìm các ô trống liên thông
   - Kiểm tra biên: có cả Black và White → **KHÔNG tính** (tranh chấp)

2. **Vùng trống giữa Black (1,3), (2,3), (2,4)**:
   - Flood-fill từ (1,3) → tìm vùng {(1,3), (2,3), (2,4)}
   - Kiểm tra biên: chỉ có Black bao quanh → **Tính là territory_black = 3**

3. **Vùng trống giữa White (3,4), (4,4)**:
   - Flood-fill từ (3,4) → tìm vùng {(3,4), (4,4)}
   - Kiểm tra biên: chỉ có White bao quanh → **Tính là territory_white = 2**

**Kết quả:**
- `territory_black = 3` (vùng {(1,3), (2,3), (2,4)})
- `territory_white = 2` (vùng {(3,4), (4,4)})

### 2. Số Quân Còn Trên Bàn ⚫⚪

**Số quân còn trên bàn** là số quân của mỗi màu còn lại trên bàn cờ khi kết thúc ván.

#### Quy tắc:
- Đếm tất cả các quân Black còn trên bàn → `stones_black`
- Đếm tất cả các quân White còn trên bàn → `stones_white`
- Mỗi quân còn trên bàn = 1 điểm

#### Ví dụ:
- Black có 25 quân còn trên bàn → `stones_black = 25` → Black được 25 điểm
- White có 23 quân còn trên bàn → `stones_white = 23` → White được 23 điểm

### 3. Komi (Điểm bù) ⚖️

**Komi** là điểm bù cho White vì White đi sau (Black đi trước có lợi thế).

#### Giá trị Komi theo luật Trung Quốc:
- **Luôn là 7.5 điểm** (không phụ thuộc vào kích thước bàn cờ)

#### Lý do:
- Black đi trước có lợi thế nhỏ
- Komi bù đắp lợi thế này
- Số lẻ (0.5) để tránh hòa
- Theo luật Trung Quốc, komi luôn là 7.5

## 📊 Công Thức Tính Điểm (Luật Trung Quốc)

### Công thức đầy đủ:

```python
# Đếm số quân còn trên bàn
stones_black = count_black_stones_on_board()
stones_white = count_white_stones_on_board()

# Tính lãnh thổ
territory_black = count_territory_black()
territory_white = count_territory_white()

# Komi (luật Trung Quốc: luôn là 7.5)
komi = 7.5

# Black điểm = Số quân trên bàn + Lãnh thổ
black_score = stones_black + territory_black

# White điểm = Số quân trên bàn + Lãnh thổ + Komi
white_score = stones_white + territory_white + komi

# So sánh điểm
score_diff = black_score - white_score

if abs(score_diff) < 0.1:  # Hòa (chênh lệch < 0.1)
    result = "DRAW"
elif score_diff > 0:  # Black thắng
    result = f"B+{score_diff:.1f}"
else:  # White thắng
    result = f"W+{abs(score_diff):.1f}"
```

## 📝 Ví Dụ Cụ Thể

### Ví dụ 1: Trận đấu 9×9

**Tình huống:**
- Số quân Black còn trên bàn: 25 quân
- Số quân White còn trên bàn: 23 quân
- Territory Black: 15 điểm
- Territory White: 12 điểm
- Komi: 7.5 điểm (luật Trung Quốc)

**Tính điểm:**
```python
# Black điểm = Số quân trên bàn + Lãnh thổ
black_score = stones_black + territory_black
black_score = 25 + 15 = 40 điểm

# White điểm = Số quân trên bàn + Lãnh thổ + Komi
white_score = stones_white + territory_white + komi
white_score = 23 + 12 + 7.5 = 42.5 điểm

# Kết quả
score_diff = 40 - 42.5 = -2.5
result = "W+2.5"  # White thắng 2.5 điểm
```

### Ví dụ 2: Fallback Mode (không có board_position)

**Tình huống:**
- Không có thông tin về số quân trên bàn và lãnh thổ
- Chỉ có thông tin về prisoners (quân bị bắt)
- Fallback: dùng prisoners để ước tính

**Lưu ý:** Đây là cách tính đơn giản, không chính xác 100% nhưng đủ cho fallback mode khi không có đầy đủ thông tin.

## 🔍 Implementation trong Code

### 1. gogame_py Mode (Chính xác)

**File**: `backend/app/services/match_service.py` - `_calculate_game_result()`

```python
def _calculate_game_result(self, board: "go.Board", match: match_model.Match) -> str:
    # Đếm số quân còn trên bàn
    stones_black = 0
    stones_white = 0
    
    for x in range(match.board_size):
        for y in range(match.board_size):
            stone = board.at(x, y)
            if stone == go.Stone.Black:
                stones_black += 1
            elif stone == go.Stone.White:
                stones_white += 1
    
    # Tính territory bằng flood-fill: tìm các vùng trống được bao quanh hoàn toàn bởi một màu
    territory_black, territory_white = self._calculate_territory_flood_fill(board, match.board_size)
    
    # Komi (luật Trung Quốc: luôn là 7.5)
    komi = 7.5
    
    # Tính điểm theo luật Trung Quốc: Số quân trên bàn + Lãnh thổ + Komi
    black_score = stones_black + territory_black
    white_score = stones_white + territory_white + komi
    
    # So sánh
    score_diff = black_score - white_score
    if abs(score_diff) < 0.1:
        return "DRAW"
    elif score_diff > 0:
        return f"B+{score_diff:.1f}"
    else:
        return f"W+{abs(score_diff):.1f}"
```

### 2. Fallback Mode (Đơn giản)

**File**: `backend/app/services/match_service.py` - `_calculate_game_result_fallback()`

```python
def _calculate_game_result_fallback(self, board_position: dict, match: match_model.Match) -> str:
    # Đếm số quân còn trên bàn từ board_position
    stones_black = 0
    stones_white = 0
    
    for x in range(match.board_size):
        for y in range(match.board_size):
            key = f"{x},{y}"
            stone_color = board_position.get(key)
            if stone_color == "B":
                stones_black += 1
            elif stone_color == "W":
                stones_white += 1
    
    # Tính territory bằng flood-fill: tìm các vùng trống được bao quanh hoàn toàn bởi một màu
    territory_black, territory_white = self._calculate_territory_flood_fill_fallback(board_position, match.board_size)
    
    # Komi (luật Trung Quốc: luôn là 7.5)
    komi = 7.5
    
    # Tính điểm theo luật Trung Quốc
    black_score = stones_black + territory_black
    white_score = stones_white + territory_white + komi
    
    # So sánh và trả về kết quả
    score_diff = black_score - white_score
    if abs(score_diff) < 0.1:
        return "DRAW"
    elif score_diff > 0:
        return f"B+{score_diff:.1f}"
    else:
        return f"W+{abs(score_diff):.1f}"
```

## ⚠️ Lưu Ý Quan Trọng

### 1. Công Thức Tính Điểm (Luật Trung Quốc)
- **Điểm = Số quân còn trên bàn + Lãnh thổ + Komi**
- Mỗi quân còn trên bàn = 1 điểm
- Lãnh thổ = các giao điểm trống được bao quanh bởi quân của một màu
- Komi chỉ được cộng vào điểm của White

### 2. Territory Calculation
- Sử dụng **Flood-Fill Algorithm** để tìm các vùng trống được bao quanh hoàn toàn bởi một màu
- Thuật toán chính xác: tìm tất cả các ô trống liên thông và kiểm tra xem vùng đó có được bao quanh bởi một màu duy nhất không
- Các vùng trống liên thông với biên bàn cờ hoặc có cả 2 màu bao quanh → không tính là territory

### 3. Komi
- Luôn được cộng vào điểm của White
- Giá trị cố định: 7.5 điểm (theo luật Trung Quốc)
- Số lẻ (0.5) để tránh hòa

### 4. Fallback Mode
- Sử dụng `board_position` từ MongoDB để tính điểm
- Cần `gogame_py` hoặc `board_position` đầy đủ để tính điểm chính xác
- **Lưu ý:** Số quân bị ăn (prisoners) không được sử dụng trong tính điểm

## 🎮 Khi Nào Tính Điểm?

Điểm được tính khi:
1. **Cả 2 bên đều pass** (2 lần pass liên tiếp)
2. **Một bên đầu hàng** (resign)
3. **Game kết thúc** (timeout hoặc các điều kiện khác)

## 📊 Format Kết Quả

- `"B+X"` - Black thắng X điểm
- `"W+X"` - White thắng X điểm
- `"DRAW"` - Hòa
- `"B+R"` - Black thắng (đối thủ đầu hàng)
- `"W+R"` - White thắng (đối thủ đầu hàng)

## 🖥️ Frontend và Backend

### Frontend (React)
- **KHÔNG tự tính điểm**
- Chỉ lấy kết quả từ backend qua API `/matches/{match_id}`
- Hiển thị `result` từ backend (format: "B+X", "W+X", "DRAW", "B+R", "W+R")
- Có hàm `formatGameResult()` để format string hiển thị, nhưng không tính điểm

### Backend (FastAPI)
- **Tự động tính điểm** khi game kết thúc
- Sử dụng `_calculate_game_result()` để tính điểm đầy đủ
- Lưu `result` vào database (PostgreSQL)
- Trả về `result` trong response API

### Flow:
```
1. Game kết thúc (2 passes hoặc resign)
   ↓
2. Backend tự động tính điểm
   - Đếm số quân còn trên bàn
   - Tính territory (lãnh thổ)
   - Cộng komi (7.5 cho White)
   - So sánh điểm
   ↓
3. Backend lưu result vào database
   ↓
4. Frontend gọi API `/matches/{match_id}`
   ↓
5. Frontend nhận result từ backend
   ↓
6. Frontend hiển thị result (format string)
```

**Lưu ý:** Số quân bị ăn (captured stones/prisoners) không được hiển thị trong UI và không được sử dụng trong tính điểm.

## 🔧 Cải Thiện Tương Lai

1. **Territory Calculation**:
   - Implement flood-fill algorithm
   - Xử lý dead stones (quân chết)
   - Xử lý seki (vùng tranh chấp)

2. **Scoring Accuracy**:
   - Sử dụng `gogame_py` board scoring
   - Xử lý life-and-death situations
   - Tính điểm theo luật Trung Quốc chính xác

3. **UI Display**:
   - Hiển thị territory trên bàn cờ
   - Hiển thị điểm chi tiết (số quân trên bàn + territory + komi)
   - Animation khi tính điểm
   - Hiển thị breakdown điểm (số quân trên bàn, territory, komi riêng biệt)

