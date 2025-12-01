# 🎯 TỐI ƯU HÓA BOARD STATE TRANSMISSION

## 📋 VẤN ĐỀ HIỆN TẠI

### 1. **Response không có thông tin captured stones**
- Response hiện tại chỉ có `move`, `game_over`, `ai_move`
- Frontend phải reload toàn bộ state sau mỗi move
- Không có thông tin về quân nào bị bắt

### 2. **Board state không tối ưu**
- Gửi toàn bộ `board_position` mỗi lần (có thể 361 entries cho 19x19)
- Không có diff (chỉ gửi thay đổi)
- Tốn bandwidth và thời gian xử lý

### 3. **Frontend không hiển thị capture animation**
- Quân bị bắt không có animation
- User không thấy rõ quân nào bị bắt

---

## ✅ GIẢI PHÁP ĐỀ XUẤT

### **1. Cải thiện Move Response**

#### Schema mới:
```python
class MoveResponse(BaseModel):
    status: Literal["accepted", "rejected"]
    move: MoveInfo
    captured: list[tuple[int, int]] = []  # Danh sách quân bị bắt: [(x1,y1), (x2,y2), ...]
    board_diff: BoardDiff  # Chỉ gửi thay đổi
    prisoners_black: int = 0
    prisoners_white: int = 0
    current_player: Literal["B", "W"]
    game_over: bool = False
    ai_move: Optional[AIMoveInfo] = None

class BoardDiff(BaseModel):
    added: dict[str, Literal["B", "W"]] = {}  # Quân mới thêm: {"x,y": "B"}
    removed: list[str] = []  # Quân bị xóa: ["x,y", ...]
```

#### Ví dụ response:
```json
{
  "status": "accepted",
  "move": {"x": 2, "y": 2, "color": "B"},
  "captured": [[1, 1], [1, 2]],  // 2 quân trắng bị bắt
  "board_diff": {
    "added": {"2,2": "B"},  // Quân đen mới
    "removed": ["1,1", "1,2"]  // 2 quân trắng bị xóa
  },
  "prisoners_black": 2,
  "prisoners_white": 0,
  "current_player": "W",
  "game_over": false,
  "ai_move": {
    "x": 3,
    "y": 3,
    "color": "W",
    "captured": [],
    "board_diff": {
      "added": {"3,3": "W"},
      "removed": []
    }
  }
}
```

---

### **2. Cải thiện MatchService.record_move()**

#### Thay đổi trong `record_move()`:
```python
async def record_move(self, match: match_model.Match, move: match_schema.MoveRequest, current_user_id: Optional[str] = None) -> dict:
    # ... existing code ...
    
    # Apply move
    board.make_move(go_move)
    
    # Lấy thông tin captured stones từ undo info
    # NOTE: make_move() trả về UndoInfo với captured stones
    # Nhưng hiện tại không lưu lại, cần sửa
    
    # Tạm thời: Lấy board state trước và sau để tính diff
    board_before = await self._get_or_create_board(match)
    # ... apply move ...
    board_after = board
    
    # Tính captured stones
    captured_stones = []
    for x in range(match.board_size):
        for y in range(match.board_size):
            before = board_before.at(x, y)
            after = board_after.at(x, y)
            if before != go.Stone.Empty and after == go.Stone.Empty:
                # Quân bị bắt
                captured_stones.append([x, y])
    
    # Tính board diff
    board_diff = {
        "added": {},
        "removed": []
    }
    
    # Quân mới thêm
    if not go_move.is_pass:
        board_diff["added"][f"{move.x},{move.y}"] = move.color
    
    # Quân bị xóa (captured)
    for x, y in captured_stones:
        board_diff["removed"].append(f"{x},{y}")
    
    # ... save to MongoDB ...
    
    result = {
        "status": "accepted",
        "move": {"x": move.x, "y": move.y, "color": move.color},
        "captured": captured_stones,  # NEW
        "board_diff": board_diff,  # NEW
        "prisoners_black": board.get_prisoners(go.Color.Black),
        "prisoners_white": board.get_prisoners(go.Color.White),
        "current_player": "W" if board.current_player() == go.Color.White else "B",
        "game_over": is_game_over,
    }
    
    # ... AI move với captured info ...
    
    return result
```

---

### **3. Cải thiện Frontend để hiển thị capture**

#### MainWindow.jsx:
```javascript
const handleMove = async (x, y) => {
  // ... existing code ...
  
  const response = await moveApi.post(`/matches/${currentMatch.id}/move`, {
    x, y, move_number, color,
  })
  
  // Xử lý captured stones với animation
  if (response.data.captured && response.data.captured.length > 0) {
    // Hiển thị animation capture
    response.data.captured.forEach(([cx, cy], index) => {
      setTimeout(() => {
        // Xóa quân với animation
        setBoardState(prev => {
          const newStones = { ...prev.stones }
          delete newStones[`${cx},${cy}`]
          return { ...prev, stones: newStones }
        })
      }, index * 100) // Stagger animation
    })
  }
  
  // Xử lý board diff
  if (response.data.board_diff) {
    const { added, removed } = response.data.board_diff
    
    setBoardState(prev => {
      const newStones = { ...prev.stones }
      
      // Thêm quân mới
      Object.entries(added).forEach(([key, color]) => {
        newStones[key] = color
      })
      
      // Xóa quân bị bắt
      removed.forEach(key => {
        delete newStones[key]
      })
      
      return {
        ...prev,
        stones: newStones,
        prisonersBlack: response.data.prisoners_black,
        prisonersWhite: response.data.prisoners_white,
        currentPlayer: response.data.current_player,
      }
    })
  }
  
  // ... AI move handling ...
}
```

#### Board.jsx - Thêm capture animation:
```javascript
const Board = ({ boardSize, stones, onCellClick, lastMove, captured = [], disabled }) => {
  // captured: list of positions that were just captured
  
  return (
    <div className="board">
      {cells.map((cell, idx) => {
        const key = `${cell.x},${cell.y}`
        const isCaptured = captured.includes(key)
        const stoneColor = stones[key]
        
        return (
          <div
            key={key}
            className={`cell ${isCaptured ? 'captured' : ''}`}
            onClick={() => onCellClick(cell.x, cell.y)}
          >
            {stoneColor && !isCaptured && (
              <div className={`stone stone-${stoneColor.toLowerCase()}`} />
            )}
            {isCaptured && (
              <div className="capture-animation">
                {/* Animation khi quân bị bắt */}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}
```

#### Board.css - Thêm animation:
```css
.captured {
  animation: capturePulse 0.5s ease-out;
}

@keyframes capturePulse {
  0% {
    transform: scale(1);
    opacity: 1;
  }
  50% {
    transform: scale(1.2);
    opacity: 0.7;
  }
  100% {
    transform: scale(0);
    opacity: 0;
  }
}

.capture-animation {
  position: absolute;
  width: 100%;
  height: 100%;
  background: radial-gradient(circle, rgba(255,0,0,0.3) 0%, transparent 70%);
  animation: captureFlash 0.3s ease-out;
}

@keyframes captureFlash {
  0% {
    opacity: 0;
  }
  50% {
    opacity: 1;
  }
  100% {
    opacity: 0;
  }
}
```

---

### **4. Tối ưu hóa Board State trong MongoDB**

#### Cấu trúc mới cho game document:
```javascript
{
  "match_id": "...",
  "board_size": 9,
  "moves": [
    {
      "number": 1,
      "color": "B",
      "position": [3, 3],
      "captured": []  // NEW: Quân bị bắt trong move này
    },
    {
      "number": 2,
      "color": "W",
      "position": [2, 2],
      "captured": []  // NEW
    },
    {
      "number": 3,
      "color": "B",
      "position": [1, 1],
      "captured": [[2, 2]]  // NEW: Bắt 1 quân trắng
    }
  ],
  "current_player": "W",
  "prisoners_black": 1,
  "prisoners_white": 0,
  "board_position": {  // Current board state (để tối ưu query)
    "3,3": "B",
    "1,1": "B"
    // Không có "2,2" vì đã bị bắt
  }
}
```

#### Lợi ích:
- Có thể replay từ moves với captured info
- Board position chỉ chứa quân hiện tại (không có quân bị bắt)
- Dễ dàng tính diff khi cần

---

### **5. Cải thiện get_match_state()**

#### Tối ưu hóa:
```python
async def get_match_state(self, match: match_model.Match) -> dict | None:
    collection = self.mongo_db.get_collection("games")
    game_doc = await collection.find_one({"match_id": match.id})
    if not game_doc:
        return None
    
    moves = game_doc.get("moves", [])
    
    # Nếu có board_position trong DB → dùng luôn (đã được cập nhật)
    if "board_position" in game_doc:
        board_position = game_doc["board_position"]
    else:
        # Fallback: Rebuild từ moves
        if go:
            board = go.Board(match.board_size)
            for move_doc in moves:
                # ... replay moves ...
            # Build board_position
            board_position = {}
            for x in range(match.board_size):
                for y in range(match.board_size):
                    stone = board.at(x, y)
                    if stone != go.Stone.Empty:
                        board_position[f"{x},{y}"] = "B" if stone == go.Stone.Black else "W"
        else:
            board_position = None
    
    return {
        "moves": moves,
        "current_player": game_doc.get("current_player", "B"),
        "prisoners_black": game_doc.get("prisoners_black", 0),
        "prisoners_white": game_doc.get("prisoners_white", 0),
        "board_position": board_position,
    }
```

---

## 📊 SO SÁNH HIỆU NĂNG

### **Trước (Current):**
- Response size: ~5-10KB (toàn bộ board_position)
- Frontend phải reload toàn bộ state
- Không có capture info
- Không có animation

### **Sau (Optimized):**
- Response size: ~0.5-1KB (chỉ diff)
- Frontend chỉ update thay đổi
- Có capture info đầy đủ
- Có animation mượt mà

**Tiết kiệm:** ~90% bandwidth cho mỗi move!

---

## 🎯 IMPLEMENTATION PLAN

### Phase 1: Backend Changes
1. ✅ Sửa `record_move()` để trả về captured stones
2. ✅ Thêm `board_diff` vào response
3. ✅ Cập nhật MongoDB schema để lưu captured trong moves
4. ✅ Cập nhật `get_match_state()` để tối ưu

### Phase 2: Frontend Changes
1. ✅ Cập nhật MainWindow để xử lý captured và board_diff
2. ✅ Thêm capture animation trong Board component
3. ✅ Cập nhật CSS cho animation
4. ✅ Test với các scenarios capture

### Phase 3: Testing
1. ✅ Test capture single stone
2. ✅ Test capture multiple stones
3. ✅ Test capture large group
4. ✅ Test performance với nhiều moves

---

## 💡 LƯU Ý

1. **Backward compatibility**: Giữ fallback cho clients cũ
2. **Error handling**: Xử lý trường hợp board_diff không khớp
3. **Performance**: Cache board_position trong MongoDB để tránh rebuild mỗi lần
4. **Animation**: Đảm bảo animation không block UI thread

