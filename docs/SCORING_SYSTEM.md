# Hệ Thống Tính Điểm Cờ Vây - Chi Tiết

## 📋 Tổng Quan

Trong cờ vây, điểm cuối trận được tính bằng:
**Điểm = Territory (Lãnh thổ) + Prisoners (Quân bị bắt) + Komi (Điểm bù)**

## 🎯 Các Thành Phần Tính Điểm

### 1. Territory (Lãnh thổ) 🏔️

**Territory** là các giao điểm trống được bao quanh hoàn toàn bởi quân của một màu.

#### Cách tính (đơn giản hóa):
- Duyệt qua tất cả các giao điểm trống trên bàn cờ
- Kiểm tra các quân kề bên (4 hướng: trên, dưới, trái, phải)
- Nếu chỉ có quân của một màu kề bên → tính là territory của màu đó
- Nếu có cả 2 màu kề bên → không tính là territory (vùng tranh chấp)

#### Ví dụ:
```
. . . . .     . = trống
. B B . .     B = Black
. B . . .     W = White
. . W W .
. . W . .
```

- Giao điểm (1,1) có quân Black kề bên → `territory_black += 1`
- Giao điểm (3,3) có quân White kề bên → `territory_white += 1`
- Giao điểm (2,2) có cả Black và White kề bên → không tính

### 2. Prisoners (Quân bị bắt) ⚫⚪

**Prisoners** là số quân đối phương bị bắt trong suốt ván cờ.

#### Quy tắc quan trọng:
- `prisoners_black` = Số quân **Black** bị bắt = **Điểm của White**
- `prisoners_white` = Số quân **White** bị bắt = **Điểm của Black**

#### Ví dụ:
- Black bắt 5 quân White → `prisoners_white = 5` → Black được 5 điểm
- White bắt 3 quân Black → `prisoners_black = 3` → White được 3 điểm

### 3. Komi (Điểm bù) ⚖️

**Komi** là điểm bù cho White vì White đi sau (Black đi trước có lợi thế).

#### Giá trị Komi chuẩn:
- **9×9**: 6.5 điểm
- **13×13**: 7.5 điểm
- **19×19**: 7.5 điểm

#### Lý do:
- Black đi trước có lợi thế nhỏ
- Komi bù đắp lợi thế này
- Số lẻ (0.5) để tránh hòa

## 📊 Công Thức Tính Điểm

### Công thức đầy đủ:

```python
# Black điểm = Territory + Prisoners (quân White bị bắt)
black_score = territory_black + prisoners_white

# White điểm = Territory + Prisoners (quân Black bị bắt) + Komi
white_score = territory_white + prisoners_black + komi

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
- Territory Black: 15 điểm
- Territory White: 12 điểm
- Prisoners Black (quân Black bị bắt): 3 quân
- Prisoners White (quân White bị bắt): 5 quân
- Komi: 6.5 điểm

**Tính điểm:**
```python
# Black điểm
black_score = territory_black + prisoners_white
black_score = 15 + 5 = 20 điểm

# White điểm
white_score = territory_white + prisoners_black + komi
white_score = 12 + 3 + 6.5 = 21.5 điểm

# Kết quả
score_diff = 20 - 21.5 = -1.5
result = "W+1.5"  # White thắng 1.5 điểm
```

### Ví dụ 2: Chỉ dùng Prisoners (Fallback mode)

**Tình huống:**
- Prisoners Black: 2 quân
- Prisoners White: 4 quân
- Không tính territory (fallback mode)

**Tính điểm:**
```python
# Black điểm (chỉ prisoners)
black_score = prisoners_white = 4 điểm

# White điểm (chỉ prisoners)
white_score = prisoners_black = 2 điểm

# Kết quả
score_diff = 4 - 2 = 2
result = "B+2"  # Black thắng 2 điểm
```

## 🔍 Implementation trong Code

### 1. gogame_py Mode (Chính xác)

**File**: `backend/app/services/match_service.py` - `_calculate_game_result()`

```python
def _calculate_game_result(self, board: "go.Board", match: match_model.Match) -> str:
    # Lấy prisoners từ board
    prisoners_black = board.get_prisoners(go.Color.Black)
    prisoners_white = board.get_prisoners(go.Color.White)
    
    # Tính territory (đơn giản hóa)
    territory_black = 0
    territory_white = 0
    for x in range(match.board_size):
        for y in range(match.board_size):
            if board.at(x, y) == go.Stone.Empty:
                # Kiểm tra neighbors
                has_black_neighbor = False
                has_white_neighbor = False
                # ... logic kiểm tra ...
                
                if has_black_neighbor and not has_white_neighbor:
                    territory_black += 1
                elif has_white_neighbor and not has_black_neighbor:
                    territory_white += 1
    
    # Komi
    komi = 6.5 if match.board_size == 9 else 7.5
    
    # Tính điểm
    black_score = territory_black + prisoners_white
    white_score = territory_white + prisoners_black + komi
    
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

**File**: `backend/app/services/match_service.py` - Các chỗ tính điểm trong fallback mode

```python
# Chỉ dùng prisoners (không có territory)
prisoners_black = game_doc.get("prisoners_black", 0)
prisoners_white = game_doc.get("prisoners_white", 0)

# Tính điểm
black_score = prisoners_white  # Black điểm = quân White bị bắt
white_score = prisoners_black  # White điểm = quân Black bị bắt

# So sánh
if black_score > white_score:
    result = f"B+{black_score - white_score}"
elif white_score > black_score:
    result = f"W+{white_score - black_score}"
else:
    result = "DRAW"
```

## ⚠️ Lưu Ý Quan Trọng

### 1. Prisoners Logic
- **KHÔNG BAO GIỜ** dùng `prisoners_black` cho điểm của Black
- **LUÔN NHỚ**: Prisoners của đối phương = Điểm của mình

### 2. Territory Calculation
- Logic hiện tại là đơn giản hóa (chỉ kiểm tra neighbors trực tiếp)
- Logic đầy đủ cần flood-fill để tìm tất cả các giao điểm trong vùng
- Có thể cải thiện trong tương lai

### 3. Komi
- Luôn được cộng vào điểm của White
- Giá trị phụ thuộc vào kích thước bàn cờ
- Số lẻ (0.5) để tránh hòa

### 4. Fallback Mode
- Chỉ dùng prisoners (không có territory và komi)
- Không chính xác 100% nhưng đủ để xác định thắng thua
- Cần `gogame_py` để tính điểm chính xác

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
   - Tính territory
   - Tính prisoners
   - Cộng komi
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
   - Hiển thị điểm chi tiết (territory + prisoners + komi)
   - Animation khi tính điểm
   - Hiển thị breakdown điểm (territory, prisoners, komi riêng biệt)

