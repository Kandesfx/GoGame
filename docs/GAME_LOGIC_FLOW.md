# 🔄 LOGIC GAME KHI ĐÁNH CỜ - GIẢI THÍCH DỄ HIỂU

## 📋 TỔNG QUAN

Khi bạn click đánh cờ trên bàn cờ, hệ thống sẽ xử lý theo các bước sau:

```
User Click → Frontend → Backend API → Game Engine → Validation → Apply Move → Capture Check → Save State → AI Move (nếu có) → Response
```

---

## 🎯 FLOW CHI TIẾT

### **BƯỚC 1: User Click trên Frontend** 🖱️

**Vị trí:** `frontend-web/src/components/Board.jsx` hoặc `frontend/app/widgets/board_widget.py`

**Chuyện gì xảy ra:**
- Bạn click vào một giao điểm (intersection) trên bàn cờ
- Frontend ghi nhận tọa độ (x, y) và màu quân (Black/White)
- Frontend gửi HTTP request đến backend:

```javascript
// Ví dụ request
POST /matches/{match_id}/move
{
  "x": 3,
  "y": 4,
  "color": "B",  // Black
  "move_number": 5
}
```

---

### **BƯỚC 2: Backend Nhận Request** 📥

**Vị trí:** `backend/app/routers/matches.py` - hàm `submit_move()`

**Chuyện gì xảy ra:**
1. **Kiểm tra quyền:** Xác minh bạn có trong match không
2. **Gọi MatchService:** Chuyển request sang `match_service.record_move()`

```python
# Line 97-111 trong matches.py
@router.post("/{match_id}/move")
async def submit_move(match_id, payload, current_user, match_service):
    match = match_service.get_match(match_id)
    # Kiểm tra user có trong match không
    if current_user.id not in {match.black_player_id, match.white_player_id}:
        raise HTTPException(403, "Not in match")
    
    # Gọi service xử lý logic
    result = await match_service.record_move(match, payload, current_user.id)
    return result
```

---

### **BƯỚC 3: MatchService Xử Lý Logic** ⚙️

**Vị trí:** `backend/app/services/match_service.py` - hàm `record_move()`

**Chuyện gì xảy ra:**

#### 3.1. **Kiểm tra đối thủ disconnect (PvP)**
```python
# Nếu đối thủ đã tạo match mới → auto-resign
if self.check_opponent_disconnected(match, current_user_id):
    # Đối thủ thua, bạn thắng
    raise ValueError("Đối thủ đã rời khỏi trận đấu. Bạn thắng!")
```

#### 3.2. **Load Board State từ MongoDB**
```python
# Lấy game state hiện tại từ MongoDB
board = await self._get_or_create_board(match)
# Board này chứa:
# - Tất cả quân cờ đã đánh
# - Prisoners (quân bị bắt)
# - Current player (lượt ai)
# - Ko index (vị trí ko nếu có)
```

**Cách load board:**
- Lấy tất cả moves từ MongoDB
- Replay từng move để xây dựng lại board state hiện tại
- Giống như xem lại ván cờ từ đầu đến hiện tại

#### 3.3. **Tạo Move Object**
```python
# Convert từ request sang Move object của C++ engine
color = go.Color.Black if move.color == "B" else go.Color.White
go_move = go.Move(move.x, move.y, color)
```

---

### **BƯỚC 4: Validate Move (Kiểm Tra Hợp Lệ)** ✅

**Vị trí:** `src/game/board.cpp` - hàm `is_legal_move()`

**Các kiểm tra:**

#### 4.1. **Kiểm tra bounds (biên bàn cờ)**
```cpp
// Line 73-75
if (!in_bounds(move.x(), move.y())) {
    return false;  // Ngoài bàn cờ → không hợp lệ
}
```

#### 4.2. **Kiểm tra vị trí trống**
```cpp
// Line 79-81
if (grid_[index] != Stone::Empty) {
    return false;  // Đã có quân → không hợp lệ
}
```

#### 4.3. **Kiểm tra Ko Rule**
```cpp
// Line 83-85
if (ko_index_ == index) {
    return false;  // Vị trí ko → không hợp lệ
}
```

**Ko Rule là gì?**
- Sau khi ăn 1 quân, không được đặt lại đúng vị trí đó ngay lập tức
- Ngăn vòng lặp vô tận (ăn → bị ăn lại → ăn → ...)

#### 4.4. **Kiểm tra Suicide (Tự Sát)**
```cpp
// Line 87-102
// Tạo board tạm và thử apply move
Board temp(*this);
temp.apply_move(move, undo);

// Kiểm tra sau khi đặt quân, nhóm của mình còn khí không
const GroupInfo own_group = temp.collect_group(index);
if (own_group.liberties.empty()) {
    return false;  // Tự sát → không hợp lệ
}
```

**Suicide là gì?**
- Đặt quân vào vị trí khiến nhóm của mình không còn khí
- **TRỪ KHI:** Nước đi đó ăn được quân đối phương (khi đó đối phương bị bắt trước)

---

### **BƯỚC 5: Apply Move (Áp Dụng Nước Đi)** 🎯

**Vị trí:** `src/game/board.cpp` - hàm `apply_move()`

**Chuyện gì xảy ra:**

#### 5.1. **Đặt quân cờ**
```cpp
// Line 220-221
const int index = to_index(move.x(), move.y());
grid_[index] = stone_from_color(move.color());  // Đặt quân vào bàn cờ
```

#### 5.2. **Kiểm tra và Ăn Quân (Capture)**
```cpp
// Line 225-255
// Kiểm tra các nhóm đối phương xung quanh nước đi mới
for (const int neighbor : neighbors(index)) {
    if (grid_[neighbor] == opponent_stone) {
        const GroupInfo opponent_group = collect_group(neighbor);
        
        // Nếu nhóm đối phương không còn khí → BẮT
        if (opponent_group.liberties.empty()) {
            // Capture toàn bộ nhóm
            for (const int stone_index : opponent_group.stones) {
                captured_indices_set.insert(stone_index);
            }
        }
    }
}

// Xóa các quân bị bắt
for (const int captured_index : captured_indices_set) {
    remove_stone(captured_index, undo);
}
```

**Capture Rule:**
- Các quân cùng màu liên kết (ngang/dọc) tạo thành **nhóm**
- Mỗi nhóm có **khí** (liberties) = các giao điểm trống liền kề
- Khi nhóm **không còn khí** → toàn bộ nhóm bị bắt
- Số quân bị bắt được đếm vào **prisoners**

**Ví dụ minh họa:**
```
Trạng thái ban đầu (bàn cờ 4x4):
  y=0  y=1  y=2  y=3
x=0  .    .    .    .
x=1  .    W    W    .  ← Nhóm trắng: (1,1) và (1,2)
x=2  .    B    .    .  ← Quân đen: (2,1)
x=3  .    .    .    .

Khí của nhóm trắng: (0,1), (0,2), (2,1), (2,2), (1,3), (1,0) = 6 khí
(Lưu ý: (2,1) là khí vì có quân đen nhưng vẫn là giao điểm trống liền kề)

Bước 1: Đen đặt quân tại (2,2):
  y=0  y=1  y=2  y=3
x=0  .    .    .    .
x=1  .    W    W    .  ← Nhóm trắng còn 4 khí: (0,1), (0,2), (1,0), (1,3)
x=2  .    B    B    .  ← Đen đặt quân tại (2,2) - chặn khí (2,2)
x=3  .    .    .    .

Bước 2: Đen đặt quân tại (0,1):
  y=0  y=1  y=2  y=3
x=0  .    B    .    .  ← Đen đặt quân tại (0,1) - chặn khí (0,1)
x=1  .    W    W    .  ← Nhóm trắng còn 3 khí: (0,2), (1,0), (1,3)
x=2  .    B    B    .
x=3  .    .    .    .

Bước 3: Đen đặt quân tại (0,2):
  y=0  y=1  y=2  y=3
x=0  .    B    B    .  ← Đen đặt quân tại (0,2) - chặn khí (0,2)
x=1  .    W    W    .  ← Nhóm trắng còn 2 khí: (1,0), (1,3)
x=2  .    B    B    .
x=3  .    .    .    .

Bước 4: Đen đặt quân tại (1,3):
  y=0  y=1  y=2  y=3
x=0  .    B    B    .
x=1  .    W    W    B  ← Đen đặt quân tại (1,3) - chặn khí (1,3)
x=2  .    B    B    .  ← Nhóm trắng còn 1 khí: (1,0)
x=3  .    .    .    .

Bước 5: Đen đặt quân tại (1,0):
  y=0  y=1  y=2  y=3
x=0  .    B    B    .
x=1  B    .    .    B  ← Đen đặt quân tại (1,0) - chặn khí cuối cùng (1,0)
x=2  .    B    B    .  ← Nhóm trắng KHÔNG CÒN KHÍ → BỊ BẮT
x=3  .    .    .    .

Kết quả cuối cùng:
  y=0  y=1  y=2  y=3
x=0  .    B    B    .
x=1  B    .    .    B  ← 2 quân trắng tại (1,1) và (1,2) đã bị nhấc khỏi bàn cờ
x=2  .    B    B    .
x=3  .    .    .    .

Prisoners: Black = 2 (đã bắt 2 quân trắng tại (1,1) và (1,2))
```

**Lưu ý quan trọng:**
- Nhóm 2 quân liền nhau ở biên có **4 khí** (không phải 3)
- Cần chặn hết **TẤT CẢ** khí mới bắt được
- Mỗi nước đi chỉ chặn 1 khí (nếu đặt cạnh nhóm)

#### 5.3. **Kiểm tra lại Suicide (Double Check)**
```cpp
// Line 262-268
const GroupInfo own_group = collect_group(index);
if (own_group.liberties.empty()) {
    // Không nên xảy ra vì đã check ở validation
    // Nhưng nếu có bug → revert và throw error
    undo_move(undo);
    throw std::runtime_error("Suicide move applied unexpectedly");
}
```

#### 5.4. **Cập nhật Ko Index**
```cpp
// Line 270-272
// Nếu chỉ ăn 1 quân và nhóm mình chỉ có 1 quân → Ko
if (captured_indices_set.size() == 1 && own_group.stones.size() == 1) {
    ko_index_ = *captured_indices_set.begin();  // Ghi nhớ vị trí ko
}
```

#### 5.5. **Cập nhật Current Player**
```cpp
// Line 208
to_move_ = opposite_color(move.color());  // Đổi lượt
```

---

### **BƯỚC 6: Lưu State vào MongoDB** 💾

**Vị trí:** `backend/app/services/match_service.py` - Line 336-355

**Chuyện gì xảy ra:**
```python
# Lưu move vào MongoDB
move_doc = {
    "number": move.move_number,
    "color": move.color,
    "position": [move.x, move.y]
}

await collection.update_one(
    {"match_id": match.id},
    {
        "$push": {"moves": move_doc},  # Thêm move mới
        "$set": {
            "current_player": "W" if board.current_player() == go.Color.White else "B",
            "prisoners_black": board.get_prisoners(go.Color.Black),
            "prisoners_white": board.get_prisoners(go.Color.White),
        },
    },
)
```

**Lưu gì:**
- Move mới (số thứ tự, màu, vị trí)
- Current player (lượt ai)
- Prisoners (số quân bị bắt)

---

### **BƯỚC 7: Kiểm Tra Game Over** 🏁

**Vị trí:** `backend/app/services/match_service.py` - Line 357-379

**Chuyện gì xảy ra:**
```python
# Kiểm tra game over
is_game_over = board.is_game_over()

if is_game_over:
    # Game kết thúc → tính điểm
    match.finished_at = datetime.now(timezone.utc)
    result_str = self._calculate_game_result(board, match)
    match.result = result_str  # Ví dụ: "B+3.5" hoặc "W+2.0"
    
    # Update Elo ratings (nếu là PvP)
    if not match.ai_level:
        stats_service.update_elo_ratings(match)
```

**Game Over khi nào?**
- 2 passes liên tiếp (cả 2 bên đều pass)
- Một bên resign (đầu hàng)
- Timeout (nếu có)

**Tính điểm:**
- Territory (đất) + Prisoners (quân bắt được)
- Trắng được cộng thêm Komi (6.5 cho 9x9, 7.5 cho 13x13/19x19)

---

### **BƯỚC 8: AI Move (Nếu là AI Match)** 🤖

**Vị trí:** `backend/app/services/match_service.py` - Line 381-398

**Chuyện gì xảy ra:**
```python
# Nếu là AI match và chưa kết thúc
if match.ai_level and not is_game_over:
    current_player = board.current_player()
    
    # AI là White, user là Black
    if current_player == go.Color.White:
        # AI chọn nước đi
        ai_move_result = await self._make_ai_move(match, board)
```

**AI chọn nước đi như thế nào?**

1. **Load board state hiện tại**
2. **Gọi AI Engine:**
   - Level 1-2: Minimax với Alpha-Beta pruning
   - Level 3-4: MCTS (Monte Carlo Tree Search)
3. **AI tính toán:**
   - Tạo cây tìm kiếm
   - Đánh giá các nước đi có thể
   - Chọn nước đi tốt nhất
4. **Apply AI move:**
   - Validate (giống user move)
   - Apply move
   - Capture check
   - Lưu vào MongoDB

**Timeout:**
- AI có timeout (mặc định 30 giây)
- Nếu quá timeout → retry hoặc pass

---

### **BƯỚC 9: Trả Response về Frontend** 📤

**Vị trí:** `backend/app/services/match_service.py` - Line 400-409

**Response:**
```python
result = {
    "status": "accepted",
    "move": {"x": move.x, "y": move.y, "color": move.color},
    "game_over": is_game_over,
}

# Nếu có AI move
if ai_move_result:
    result["ai_move"] = {
        "x": ai_move_result["x"],
        "y": ai_move_result["y"],
        "is_pass": ai_move_result.get("is_pass", False)
    }

return result
```

---

### **BƯỚC 10: Frontend Cập Nhật UI** 🎨

**Vị trí:** `frontend-web/src/components/Board.jsx`

**Chuyện gì xảy ra:**
1. Nhận response từ backend
2. Cập nhật bàn cờ:
   - Vẽ quân cờ mới
   - Xóa quân bị bắt (nếu có)
   - Cập nhật prisoners count
3. Nếu có AI move:
   - Vẽ quân AI sau một chút delay (để user thấy rõ)
4. Nếu game over:
   - Hiển thị kết quả
   - Disable board (không cho đánh nữa)

---

## 🔄 FLOW TỔNG QUAN (Sơ Đồ)

```
┌─────────────────────────────────────────────────────────────┐
│                    USER CLICK (x, y)                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         FRONTEND: POST /matches/{id}/move                   │
│         {x, y, color, move_number}                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         BACKEND: matches.py → submit_move()                  │
│         - Kiểm tra quyền                                      │
│         - Gọi match_service.record_move()                    │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         MATCH SERVICE: record_move()                         │
│         1. Check opponent disconnect                          │
│         2. Load board từ MongoDB                             │
│         3. Tạo Move object                                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         C++ BOARD ENGINE: is_legal_move()                    │
│         - Check bounds                                        │
│         - Check vị trí trống                                  │
│         - Check Ko rule                                       │
│         - Check Suicide                                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         C++ BOARD ENGINE: apply_move()                       │
│         1. Đặt quân cờ                                       │
│         2. Kiểm tra nhóm đối phương → Capture               │
│         3. Kiểm tra Suicide (double check)                    │
│         4. Cập nhật Ko index                                  │
│         5. Đổi lượt                                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         SAVE TO MONGODB                                       │
│         - Push move mới                                      │
│         - Update current_player                              │
│         - Update prisoners                                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         CHECK GAME OVER                                       │
│         - Nếu game over → Tính điểm → Update Elo            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         AI MOVE (nếu là AI match)                            │
│         - AI chọn nước đi (Minimax/MCTS)                     │
│         - Apply AI move                                       │
│         - Save to MongoDB                                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         RETURN RESPONSE                                       │
│         {status, move, game_over, ai_move?}                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         FRONTEND: Update UI                                   │
│         - Vẽ quân cờ mới                                     │
│         - Xóa quân bị bắt                                    │
│         - Vẽ AI move (nếu có)                                │
│         - Hiển thị kết quả (nếu game over)                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 CÁC LUẬT QUAN TRỌNG

### 1. **Capture Rule (Luật Ăn Quân)**
- Nhóm không còn khí → bị bắt
- Quân bị bắt được đếm vào prisoners

### 2. **Suicide Rule (Luật Tự Sát)**
- Không được đặt quân vào vị trí tự sát
- **TRỪ KHI:** Ăn được quân đối phương

### 3. **Ko Rule (Luật Ko)**
- Không được lặp lại trạng thái bàn cờ ngay lập tức
- Sau khi ăn 1 quân, không được đặt lại đúng vị trí đó

### 4. **Pass Rule**
- Có thể pass (bỏ lượt)
- 2 passes liên tiếp → game over

---

## 💡 TÓM TẮT

**Khi bạn đánh cờ:**
1. ✅ Frontend gửi request
2. ✅ Backend validate quyền
3. ✅ Game engine kiểm tra hợp lệ
4. ✅ Apply move + Capture check
5. ✅ Lưu vào database
6. ✅ Kiểm tra game over
7. ✅ AI move (nếu có)
8. ✅ Trả response
9. ✅ Frontend cập nhật UI

**Tất cả diễn ra trong vài trăm milliseconds!** ⚡

