# Tính năng Undo (Hoàn tác) - Implementation Guide

## 📋 Tổng quan

Tính năng undo cho phép người chơi hoàn tác nước đi cuối cùng của mình trong trận đấu.

## ✅ Đã Implement

### Backend

1. **Endpoint**: `POST /matches/{match_id}/undo`
   - File: `backend/app/routers/matches.py`
   - Yêu cầu authentication
   - Chỉ cho phép undo move của chính user

2. **Service Method**: `MatchService.undo_move()`
   - File: `backend/app/services/match_service.py`
   - Hỗ trợ cả `gogame_py` mode và fallback mode
   - Rebuild board state từ moves còn lại

### Frontend

1. **UI Button**: Nút "Hoàn tác" trong `GameControls`
   - File: `frontend-web/src/components/GameControls.jsx`
   - Disabled khi không có moves hoặc game over

2. **Handler**: `handleUndo()` trong `MainWindow`
   - File: `frontend-web/src/components/MainWindow.jsx`
   - Xử lý API call và cập nhật UI

## 🛡️ Edge Cases & Error Handling

### 1. Match đã kết thúc
- **Kiểm tra**: `if match.finished_at`
- **Error**: "Không thể undo: Trận đấu đã kết thúc"
- **Status**: ✅ Đã xử lý

### 2. Không có moves
- **Kiểm tra**: `if not moves`
- **Error**: "Không có nước đi nào để undo"
- **Status**: ✅ Đã xử lý

### 3. Move cuối cùng không phải của user
- **AI Match**: 
  - **Logic đặc biệt**: 
    - Nếu move cuối cùng là của AI (White) → Undo cả AI move và user move trước đó
    - Nếu move cuối cùng là của user (Black) → Chỉ undo user move
    - Lý do: Trong AI match, sau khi user đánh, AI đánh ngay lập tức, nên khi undo cần undo cả 2 nước
- **PvP Match**:
  - Kiểm tra user có phải là player của màu đó không
  - Nếu không → Error
  - Chỉ undo 1 move (của chính user)
- **Status**: ✅ Đã xử lý

### 4. Rebuild Board State

#### Fallback Mode:
- Rebuild `board_position` từ moves còn lại
- Tính lại `prisoners_black` và `prisoners_white`
- Tính lại `current_player`
- Tính lại `ko_position` (nếu có)

#### gogame_py Mode:
- Tạo Board mới và apply tất cả moves còn lại
- Extract `board_position` từ board
- Extract `prisoners` từ board
- Extract `current_player` từ board
- `ko_position` tạm thời set None (sẽ được tính lại khi có move tiếp theo)

### 5. Prisoners Calculation
- **Lưu ý**: Prisoners là số quân đối phương bị bắt
- **Logic**: 
  - Nếu Black đánh → `prisoners_white += 1` (Black bắt White)
  - Nếu White đánh → `prisoners_black += 1` (White bắt Black)
- **Status**: ✅ Đã sửa logic

### 6. Ko Position Calculation
- **Fallback Mode**: 
  - Kiểm tra move trước đó có capture 1 quân không
  - Kiểm tra nhóm quân mình có chỉ 1 quân không
  - Nếu đúng → set `ko_position` = vị trí quân bị bắt
- **gogame_py Mode**: 
  - Tạm thời set None (có thể cải thiện sau)
- **Status**: ✅ Đã xử lý (có thể cải thiện)

### 7. Race Conditions
- **Vấn đề**: Nếu user undo trong khi đang có request khác
- **Giải pháp**: 
  - Sử dụng `isProcessing` state để disable button
  - Disable button khi `isProcessing || gameOver`
- **Status**: ✅ Đã xử lý

### 8. UI Synchronization
- **Vấn đề**: Sau khi undo, UI cần cập nhật board state
- **Giải pháp**:
  - Cập nhật `boardState` từ response
  - Gọi `loadMatchState()` để đảm bảo đồng bộ
- **Status**: ✅ Đã xử lý

## ⚠️ Lưu ý & Hạn chế

### 1. AI Match
- **Logic đặc biệt**: 
  - Nếu move cuối cùng là của AI → Undo cả AI move và user move trước đó
  - Nếu move cuối cùng là của user → Chỉ undo user move
- **Lý do**: Trong AI match, sau khi user đánh, AI đánh ngay lập tức, nên user không có thời gian để undo nước của mình trước khi AI đánh. Do đó, khi undo, hệ thống sẽ undo cả 2 nước (AI + User) để quay về trạng thái trước khi user đánh.
- **Status**: ✅ Đã implement

### 2. Ko Position trong gogame_py Mode
- **Hạn chế**: `ko_position` tạm thời set None sau khi undo
- **Lý do**: Board không expose `ko_index` trực tiếp
- **Giải pháp tương lai**: Có thể thêm method để lấy `ko_index` từ board

### 3. Multiple Undos
- **Hạn chế**: Chỉ có thể undo 1 move tại một thời điểm
- **Lý do**: Đơn giản hóa logic
- **Giải pháp tương lai**: Có thể implement "undo multiple moves" nếu cần

### 4. Concurrent Undos (PvP)
- **Vấn đề**: Nếu cả 2 players cùng undo cùng lúc
- **Giải pháp**: Backend chỉ cho phép undo move của chính user
- **Status**: ✅ Đã xử lý

## 🧪 Testing Checklist

- [ ] Undo move của user trong AI match (chỉ undo user move)
- [ ] Undo khi move cuối cùng là của AI trong AI match (undo cả AI + User)
- [ ] Undo move của user trong PvP match
- [ ] Không thể undo move của đối thủ (PvP)
- [ ] Không thể undo khi match đã kết thúc
- [ ] Không thể undo khi không có moves
- [ ] Không thể undo khi chỉ có 1 move và đó là của AI (cần ít nhất 2 moves)
- [ ] Board state được cập nhật đúng sau undo
- [ ] Prisoners được tính lại đúng sau undo
- [ ] Current player được cập nhật đúng sau undo
- [ ] Ko position được tính lại đúng (nếu có)
- [ ] UI được cập nhật đúng sau undo
- [ ] Race conditions được xử lý đúng
- [ ] Confirm message hiển thị đúng cho AI match và PvP match

## 📝 API Response Format

```json
{
  "status": "undone",
  "undone_moves": [
    {
      "number": 5,
      "color": "W",
      "position": [4, 5],
      "captured": []
    },
    {
      "number": 4,
      "color": "B",
      "position": [3, 4],
      "captured": [[2, 4]]
    }
  ],
  "undone_move": {
    "number": 5,
    "color": "W",
    "position": [4, 5],
    "captured": []
  },
  "board_position": {
    "0,0": "B",
    "1,1": "W",
    ...
  },
  "current_player": "B",
  "prisoners_black": 2,
  "prisoners_white": 1,
  "remaining_moves": 3
}
```

**Lưu ý**: 
- `undone_moves`: Danh sách tất cả moves đã undo (có thể có 1 hoặc 2 moves)
- `undone_move`: Move đầu tiên trong `undone_moves` (giữ backward compatibility)
- Trong AI match, nếu move cuối cùng là của AI, `undone_moves` sẽ có 2 moves (AI + User)

## 🔄 Future Improvements

1. **Undo AI Move**: Cho phép undo move của AI (cần tính lại AI move)
2. **Multiple Undos**: Cho phép undo nhiều moves cùng lúc
3. **Undo History**: Lưu lịch sử undo để có thể redo
4. **Ko Position**: Cải thiện logic tính ko_position trong gogame_py mode
5. **Visual Feedback**: Hiển thị animation khi undo

## 📚 Files Modified

### Backend:
- `backend/app/services/match_service.py` - Thêm `undo_move()` method
- `backend/app/routers/matches.py` - Thêm `/undo` endpoint

### Frontend:
- `frontend-web/src/components/MainWindow.jsx` - Thêm `handleUndo()` function
- `frontend-web/src/components/GameControls.jsx` - Thêm nút Undo

## ✅ Status

**Hoàn thành**: Tính năng undo đã được implement đầy đủ với error handling và edge cases.

**Cần test**: Cần test kỹ các scenarios để đảm bảo không có lỗi.

