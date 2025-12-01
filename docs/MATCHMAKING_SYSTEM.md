# Hệ Thống Ghép Trận (Matchmaking System)

## Tổng Quan

Hệ thống ghép trận tự động tìm và ghép 2 người chơi có cùng kích thước bàn cờ và ELO rating tương thích để bắt đầu một trận đấu PvP.

## Kiến Trúc

### 1. **Frontend (React)**
- `MatchmakingDialog.jsx`: UI để người chơi tham gia queue
- Polling mechanism: Kiểm tra queue status và match mỗi 2 giây

### 2. **Backend (FastAPI)**
- `MatchmakingService`: Quản lý queue và matching algorithm
- Background thread: Chạy matching loop độc lập
- Singleton pattern: Queue được chia sẻ giữa tất cả requests

## Luồng Hoạt Động Chi Tiết

### Bước 1: Người Chơi Tham Gia Queue

#### Frontend:
```javascript
// User click "Tìm đối thủ"
handleJoinQueue() {
  // Gọi API POST /matchmaking/queue/join
  api.post('/matchmaking/queue/join', { board_size: 9 })
}
```

#### Backend:
1. **Auto-resign active matches** (nếu có):
   - Kiểm tra số lượng matches đang active
   - Tự động resign tối đa 5 matches để giải phóng người chơi
   - Không block request nếu có lỗi

2. **Thêm vào queue**:
   ```python
   join_queue(user_id, elo_rating, board_size)
   ```
   - Kiểm tra user đã có trong queue chưa
   - Tạo `QueueEntry` với:
     - `user_id`: ID người chơi
     - `elo_rating`: ELO hiện tại
     - `board_size`: Kích thước bàn cờ (9, 13, 19)
     - `joined_at`: Thời gian tham gia
     - `elo_range`: Khoảng ELO ban đầu (±200)

3. **Khởi động matching thread** (nếu chưa chạy):
   - Tạo background thread với event loop riêng
   - Thread này chạy matching loop độc lập

4. **Trả về queue status**:
   ```json
   {
     "in_queue": true,
     "board_size": 9,
     "elo_rating": 1500,
     "wait_time": 0,
     "queue_size": 1,
     "elo_range": 200
   }
   ```

### Bước 2: Polling Queue Status và Match

#### Frontend:
```javascript
// Polling mỗi 2 giây
setInterval(async () => {
  // 1. Check queue status
  const statusRes = await api.get('/matchmaking/queue/status')
  setQueueStatus(statusRes.data)
  
  // 2. Check if match found
  const matchRes = await api.get('/matchmaking/queue/match')
  if (matchRes.data.matched) {
    onMatchFound(matchRes.data.match) // Navigate to game
  }
}, 2000)
```

#### Backend Endpoints:

**GET `/matchmaking/queue/status`**:
- Đọc từ memory (nhanh)
- Trả về thông tin queue hiện tại:
  - `wait_time`: Thời gian đã chờ (giây)
  - `queue_size`: Số người trong queue
  - `elo_range`: Khoảng ELO hiện tại

**GET `/matchmaking/queue/match`**:
- Query database để tìm match mới được tạo
- Filter:
  - User là player (black hoặc white)
  - Match có cả 2 players
  - Match chưa kết thúc
  - Match không phải AI match
- Nếu tìm thấy:
  - Tự động leave queue
  - Trả về match info

### Bước 3: Matching Algorithm (Background Thread)

#### Matching Loop:
```python
# Chạy mỗi 2 giây (MATCHING_INTERVAL)
while running:
    _try_match_players()  # Tìm và tạo matches
    await asyncio.sleep(2)
```

#### Matching Process:

1. **Lặp qua từng board_size queue**:
   ```python
   for board_size, queue in _shared_queue.items():
       if len(queue) < 2:
           continue  # Cần ít nhất 2 người
   ```

2. **Mở rộng ELO range theo thời gian**:
   ```python
   # Mỗi 5 giây, mở rộng thêm 50 ELO
   elapsed = (now - joined_at).total_seconds()
   expansions = int(elapsed / 5)
   elo_range = min(200 + (expansions * 50), 500)
   ```
   - Ban đầu: ±200 ELO
   - Sau 5 giây: ±250 ELO
   - Sau 10 giây: ±300 ELO
   - Tối đa: ±500 ELO

3. **Tìm compatible pairs**:
   ```python
   for entry1 in queue:
       for entry2 in queue:
           if entry1.is_compatible(entry2):
               # Match found!
   ```
   
   **Điều kiện compatibility**:
   - Cùng `board_size`
   - ELO difference ≤ `entry1.elo_range` VÀ ≤ `entry2.elo_range`
   - (Bidirectional check - cả 2 phải chấp nhận nhau)

4. **Tạo match TRƯỚC KHI remove khỏi queue**:
   ```python
   # Lấy entries nhưng chưa pop
   entry1 = queue[i]
   entry2 = queue[j]
   
   # Tạo match synchronously
   _create_match_sync(entry1, entry2, board_size)
   
   # CHỈ remove nếu tạo thành công
   if success:
       queue.pop(i)
       queue.pop(j)
   ```

5. **Tạo match trong database**:
   ```python
   _create_match_sync(entry1, entry2, board_size):
       # 1. Validate users tồn tại
       # 2. Kiểm tra duplicate match
       # 3. Xác định Black/White (ELO cao hơn = Black)
       # 4. Tạo room_code 6 ký tự duy nhất
       # 5. Tạo Match object
       # 6. Commit to database
   ```

### Bước 4: Người Chơi Nhận Match

#### Frontend:
```javascript
// Polling phát hiện match
if (matchRes.data.matched) {
  // Navigate to game board
  onMatchFound(matchRes.data.match)
  // Match object chứa:
  // - id, board_size
  // - black_player_id, white_player_id
  // - black_player_username, white_player_username
  // - room_code
}
```

#### Backend:
- `check_match` endpoint tự động leave queue khi tìm thấy match
- Match đã được tạo trong database với cả 2 players

## Cấu Trúc Dữ Liệu

### QueueEntry
```python
class QueueEntry:
    user_id: str           # User ID
    elo_rating: int       # ELO hiện tại
    board_size: int       # 9, 13, hoặc 19
    joined_at: datetime   # Thời gian tham gia
    elo_range: int       # Khoảng ELO hiện tại (mở rộng theo thời gian)
```

### Queue Structure
```python
_shared_queue: Dict[int, List[QueueEntry]]
# {
#   9: [QueueEntry(user1), QueueEntry(user2), ...],
#   13: [QueueEntry(user3), ...],
#   19: [QueueEntry(user4), ...]
# }
```

## Thuật Toán Matching

### ELO Range Expansion
- **Ban đầu**: ±200 ELO
- **Mỗi 5 giây**: +50 ELO
- **Tối đa**: ±500 ELO
- **Ví dụ**:
  - User ELO 1500, chờ 0 giây: Match với 1300-1700
  - User ELO 1500, chờ 10 giây: Match với 1200-1800
  - User ELO 1500, chờ 30 giây: Match với 1000-2000 (max)

### Compatibility Check
```python
def is_compatible(self, other):
    # 1. Cùng board_size
    if self.board_size != other.board_size:
        return False
    
    # 2. ELO difference trong range của CẢ 2
    elo_diff = abs(self.elo_rating - other.elo_rating)
    return (
        elo_diff <= self.elo_range AND
        elo_diff <= other.elo_range
    )
```

**Ví dụ**:
- User A: ELO 1500, range ±200 (1300-1700)
- User B: ELO 1600, range ±150 (1450-1750)
- ELO diff = 100
- ✅ Compatible: 100 ≤ 200 (A) VÀ 100 ≤ 150 (B)

## Thread Safety

### Singleton Pattern
- `_shared_queue`: Class-level variable, chia sẻ giữa tất cả instances
- `_shared_lock`: Threading lock để đảm bảo thread-safe
- Tất cả operations trên queue đều được bảo vệ bởi lock

### Background Thread
- Matching thread chạy độc lập với FastAPI event loop
- Có event loop riêng để chạy async operations
- Không block main thread

## Error Handling

### Frontend:
- Timeout handling: Không hiển thị error cho polling timeout
- Network errors: Log và tiếp tục polling
- Match found: Dừng polling và navigate

### Backend:
- Matching loop: Tiếp tục chạy ngay cả khi có lỗi
- Match creation: Nếu fail, entries vẫn còn trong queue
- Auto-resign: Không fail request nếu có lỗi

## Timeout và Cleanup

### Queue Timeout
- **QUEUE_TIMEOUT = 60 giây**
- Sau 60 giây, player tự động bị remove khỏi queue
- Matching loop tự động cleanup timed-out entries

### Frontend Cleanup
- Khi component unmount, tự động leave queue
- Clear tất cả intervals

## Ví Dụ Hoạt Động

### Scenario 1: Match Ngay Lập Tức
1. User A (ELO 1500, 9x9) join queue → Queue size: 1
2. User B (ELO 1520, 9x9) join queue → Queue size: 2
3. Matching loop (chạy mỗi 2 giây) phát hiện:
   - ELO diff = 20
   - 20 ≤ 200 (A) VÀ 20 ≤ 200 (B) → ✅ Compatible
4. Tạo match trong database
5. Remove cả 2 khỏi queue
6. Frontend polling phát hiện match → Navigate to game

### Scenario 2: Chờ Mở Rộng ELO Range
1. User A (ELO 1500, 9x9) join queue
2. User B (ELO 1800, 9x9) join queue
3. ELO diff = 300 > 200 → ❌ Không compatible
4. Sau 5 giây: A's range = ±250, B's range = ±250
   - 300 > 250 → ❌ Vẫn không compatible
5. Sau 10 giây: A's range = ±300, B's range = ±300
   - 300 ≤ 300 → ✅ Compatible!
6. Tạo match và remove khỏi queue

### Scenario 3: Timeout
1. User A join queue, chờ 60 giây không có match
2. Matching loop tự động remove A khỏi queue
3. Frontend polling phát hiện `in_queue = false`
4. Hiển thị message hoặc tự động đóng dialog

## Performance

### Optimizations:
1. **Memory-based queue**: Không query database mỗi lần
2. **Background thread**: Không block API requests
3. **Batch operations**: Tạo match trước khi remove khỏi queue
4. **Efficient matching**: Chỉ check compatibility, không query DB

### Scalability:
- Queue được tổ chức theo `board_size` → O(n) matching
- Thread-safe operations → Có thể handle nhiều requests đồng thời
- Singleton pattern → Chỉ 1 matching thread cho toàn bộ server

## Monitoring và Debugging

### Logs:
- `join_queue`: Log user ID, ELO, board size, queue size
- `_try_match_players`: Log compatible pairs found
- `_create_match_sync`: Log match creation success/failure
- `check_match`: Log khi user tìm thấy match

### Metrics:
- Queue size per board size
- Average wait time
- Match success rate
- Timeout rate

