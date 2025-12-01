# Phân Tích Logic UI - Kiểm Tra Toàn Diện

## 1. FLOW TỔNG QUAN

### 1.1. Flow Matchmaking (Ghép Trận Online)

```
HomePage
  └─> Click "Bắt đầu chơi"
      └─> MatchDialog
          └─> Chọn "Ghép người chơi online"
              └─> MatchmakingDialog
                  ├─> Chọn board size
                  ├─> Click "Tìm đối thủ"
                  │   └─> POST /matchmaking/queue/join
                  │       └─> Polling mỗi 1 giây:
                  │           ├─> GET /matchmaking/queue/status
                  │           └─> GET /matchmaking/queue/match
                  │               └─> Nếu matched:
                  │                   └─> MatchFoundDialog
                  │                       ├─> Initialize ready status từ server
                  │                       ├─> Polling mỗi 1 giây để check opponent ready
                  │                       ├─> User click "Sẵn sàng"
                  │                       │   └─> POST /matches/{id}/ready
                  │                       └─> Nếu both ready:
                  │                           └─> onStart(match)
                  │                               └─> HomePage.handleMatchStart
                  │                                   └─> onStartMatch(match)
                  │                                       └─> MainWindow (với initialMatch)
                  └─> Click "Hủy tìm kiếm"
                      └─> POST /matchmaking/queue/leave
```

### 1.2. Flow PvP Join by Code

```
HomePage
  └─> Click "Bắt đầu chơi"
      └─> MatchDialog
          └─> Chọn "Đấu với người (Mã tham gia)"
              └─> PvPDialog
                  ├─> Mode "Tạo bàn"
                  │   └─> POST /matches/pvp
                  │       └─> onMatchCreated(match)
                  │           └─> MainWindow (với initialMatch)
                  └─> Mode "Tham gia bàn"
                      └─> Nhập mã 6 ký tự
                          └─> POST /matches/pvp/join_by_code
                              └─> onMatchJoined(match)
                                  └─> MainWindow (với initialMatch)
```

### 1.3. Flow AI Match

```
HomePage
  └─> Click "Bắt đầu chơi"
      └─> MatchDialog
          └─> Chọn level AI + board size
              └─> POST /matches/ai
                  └─> MainWindow (với initialMatch)
```

### 1.4. Flow Game Play (MainWindow)

```
MainWindow (với initialMatch)
  ├─> useEffect: Load initial data
  ├─> useEffect: Load match state từ initialMatch
  ├─> useEffect: Xác định player color (PvP only)
  ├─> useEffect: Polling cho PvP matches (mỗi 2 giây)
  ├─> useEffect: Đếm ngược thời gian (PvP với time control)
  │
  ├─> handleBoardClick
  │   ├─> Ràng buộc 1: Check !currentMatch || isProcessing || gameOver
  │   ├─> Ràng buộc 2: Check position không có stone
  │   ├─> POST /matches/{id}/move
  │   ├─> Xử lý board_diff (remove captured, add new)
  │   ├─> Check game_over từ response
  │   └─> Nếu AI match: Xử lý AI move
  │
  ├─> handlePass
  │   ├─> Ràng buộc 1: Check !currentMatch || isProcessing || gameOver
  │   ├─> POST /matches/{id}/pass
  │   ├─> Check game_over từ response
  │   └─> Nếu AI match: Xử lý AI pass
  │
  ├─> handleResign
  │   ├─> Ràng buộc 1: Check !currentMatch
  │   ├─> Confirm dialog
  │   ├─> POST /matches/{id}/resign
  │   └─> Set gameOver = true
  │
  └─> handleUndo
      ├─> Ràng buộc 1: Check !currentMatch || isProcessing || gameOver
      ├─> Confirm dialog
      ├─> POST /matches/{id}/undo
      └─> Reload match state
```

## 2. RÀNG BUỘC VÀ VALIDATION

### 2.1. MatchmakingDialog

**Ràng buộc:**
- ✅ `inQueue` state được quản lý đúng
- ✅ Polling cleanup khi component unmount
- ✅ Leave queue khi đóng dialog
- ✅ Error handling cho timeout, 404, 401
- ⚠️ **VẤN ĐỀ**: `onMatchFound` được gọi nhưng không clear `inQueue` state → có thể gây race condition

**Cần sửa:**
```javascript
// Trong MatchmakingDialog, khi match found:
if (matchRes.data && matchRes.data.matched && matchRes.data.match) {
  clearInterval(interval)
  setCheckingInterval(null)
  setInQueue(false) // ← THIẾU: Cần set inQueue = false
  if (waitTimeIntervalRef.current) {
    clearInterval(waitTimeIntervalRef.current)
    waitTimeIntervalRef.current = null
  }
  onMatchFound(matchData)
}
```

### 2.2. MatchFoundDialog

**Ràng buộc:**
- ✅ Validate match object trước khi render
- ✅ Initialize ready status từ server khi mount
- ✅ Polling để sync ready status
- ✅ Auto-start khi both ready
- ⚠️ **VẤN ĐỀ**: Polling dependency có `isReady` → có thể gây re-render loop

**Cần kiểm tra:**
```javascript
// useEffect polling dependency
}, [match?.id, bothReady, userIsBlack, onStart, isReady])
// ↑ isReady trong dependency có thể gây re-render loop
// Nên chỉ dùng match?.id, bothReady, userIsBlack, onStart
```

### 2.3. MainWindow - Game State Management

**Ràng buộc:**
- ✅ `gameOver` check trong tất cả actions (click, pass, undo)
- ✅ Backend validation: `if match.finished_at or match.result: raise ValueError`
- ✅ Polling check `result` và `finished_at` từ backend
- ✅ `loadMatchState` check `result` và `finished_at`
- ⚠️ **VẤN ĐỀ**: Race condition giữa polling và user actions

**Cần cải thiện:**
1. **Polling race condition**: Polling có thể update `gameOver` trong khi user đang thực hiện action
   - **Giải pháp**: Check `isProcessing` trong polling, hoặc dùng flag để prevent polling update khi đang process

2. **State synchronization**: `gameOver` có thể bị reset nếu `loadMatchState` được gọi sau khi game over
   - **Giải pháp**: Không reset `gameOver` nếu match đã có `result` hoặc `finished_at`

### 2.4. MainWindow - Board Click Validation

**Ràng buộc hiện tại:**
```javascript
if (!currentMatch || isProcessing || gameOver) {
  return // ✅ Đúng
}

if (boardState.stones[key]) {
  return // ✅ Đúng
}
```

**Thiếu:**
- ❌ Không check `currentPlayer` với user color (PvP)
- ❌ Không check match đã có đủ 2 người chơi (PvP)
- ❌ Không check thời gian còn lại (PvP với time control)

**Cần thêm:**
```javascript
// Check đúng lượt (PvP only)
if (!currentMatch.ai_level) {
  const userColor = determineUserColor() // B hoặc W
  if (boardState.currentPlayer !== userColor) {
    alert('Không phải lượt của bạn')
    return
  }
}

// Check đủ người chơi (PvP only)
if (!currentMatch.ai_level) {
  if (!currentMatch.black_player_id || !currentMatch.white_player_id) {
    alert('Chưa đủ người chơi')
    return
  }
}
```

### 2.5. MainWindow - Polling Logic

**Ràng buộc:**
- ✅ Chỉ poll cho PvP matches
- ✅ Dừng poll khi `gameOver`
- ✅ Check `isProcessing` trước khi poll
- ⚠️ **VẤN ĐỀ**: Polling có thể gây race condition với user actions

**Cần cải thiện:**
```javascript
// Thêm flag để prevent polling update khi đang process
const [isUpdatingFromPoll, setIsUpdatingFromPoll] = useState(false)

// Trong polling:
if (isProcessing || isUpdatingFromPoll) {
  return // Skip polling
}

// Khi update từ polling:
setIsUpdatingFromPoll(true)
await loadMatchState(currentMatch.id)
setIsUpdatingFromPoll(false)
```

## 3. EDGE CASES VÀ ERROR HANDLING

### 3.1. Matchmaking Edge Cases

**Case 1: User đóng dialog khi đang trong queue**
- ✅ **Đã xử lý**: Cleanup effect leave queue khi unmount
- ⚠️ **Vấn đề**: Nếu user đóng dialog trước khi match found, match sẽ bị mất
- **Giải pháp**: Check match khi mở lại dialog, hoặc lưu match vào localStorage

**Case 2: Network timeout khi polling**
- ✅ **Đã xử lý**: Catch timeout error, không exit queue
- ✅ **Đã xử lý**: Retry logic trong backend

**Case 3: Match found nhưng user đã rời queue**
- ⚠️ **Vấn đề**: `onMatchFound` được gọi nhưng user có thể đã rời queue
- **Giải pháp**: Check `inQueue` trước khi gọi `onMatchFound`

### 3.2. MatchFoundDialog Edge Cases

**Case 1: Opponent cancel ready**
- ⚠️ **Vấn đề**: Không có logic để detect opponent cancel
- **Giải pháp**: Polling đã sync ready status từ server, nhưng cần handle case opponent set ready = false

**Case 2: Match bị xóa khi đang trong ready dialog**
- ✅ **Đã xử lý**: 404 error handling trong polling
- ✅ **Đã xử lý**: Error message hiển thị cho user

**Case 3: Both ready nhưng onStart không được gọi**
- ⚠️ **Vấn đề**: Race condition giữa polling và handleReady
- **Giải pháp**: Đảm bảo `onStart` chỉ được gọi một lần

### 3.3. MainWindow Edge Cases

**Case 1: Game over nhưng user vẫn có thể click**
- ✅ **Đã xử lý**: Check `gameOver` trong `handleBoardClick`
- ✅ **Đã xử lý**: Backend reject moves khi match finished
- ⚠️ **Vấn đề**: Nếu `gameOver` chưa được set (race condition), user có thể click
- **Giải pháp**: Double-check trong `handleBoardClick`:
  ```javascript
  // Check lại từ backend trước khi process
  const matchResponse = await api.get(`/matches/${currentMatch.id}`)
  if (matchResponse.data?.result || matchResponse.data?.finished_at) {
    setGameOver(true)
    return
  }
  ```

**Case 2: Opponent disconnect trong PvP match**
- ✅ **Đã xử lý**: Backend auto-resign khi detect disconnect
- ✅ **Đã xử lý**: Frontend polling sẽ detect `result` và set `gameOver`
- ⚠️ **Vấn đề**: Có thể có delay giữa disconnect và auto-resign
- **Giải pháp**: Thêm check trong polling để detect opponent disconnect sớm hơn

**Case 3: Time out trong PvP match**
- ✅ **Đã xử lý**: Backend auto-resign khi time <= 0
- ✅ **Đã xử lý**: Frontend đếm ngược thời gian
- ⚠️ **Vấn ĐỀ**: Frontend có thể không sync với backend time
- **Giải pháp**: Sync time từ backend trong polling

**Case 4: Undo khi game đã over**
- ✅ **Đã xử lý**: Check `gameOver` trong `handleUndo`
- ✅ **Đã xử lý**: Backend reject undo khi match finished
- ✅ **OK**

## 4. STATE SYNCHRONIZATION

### 4.1. Game Over State

**Nguồn truth:**
- Backend: `match.result` và `match.finished_at`
- Frontend: `gameOver` state

**Sync points:**
1. ✅ `loadMatchState`: Check `result` và `finished_at`
2. ✅ Polling: Check `result` và `finished_at`
3. ✅ `handleBoardClick`: Check `game_over` từ response
4. ✅ `handlePass`: Check `game_over` từ response
5. ✅ `handleResign`: Set `gameOver = true` sau khi resign

**Vấn đề:**
- ⚠️ Race condition: Nhiều nơi có thể set `gameOver` cùng lúc
- **Giải pháp**: Dùng một function để set game over:
  ```javascript
  const setGameOverState = async (matchId) => {
    const response = await api.get(`/matches/${matchId}`)
    if (response.data?.result || response.data?.finished_at) {
      setGameOver(true)
      setGameResult(response.data.result)
      // ... load ELO, show modal
    }
  }
  ```

### 4.2. Board State Synchronization

**Nguồn truth:**
- Backend: `state.board_position` (MongoDB)
- Frontend: `boardState.stones`

**Sync points:**
1. ✅ `loadMatchState`: Load từ `board_position`
2. ✅ `handleBoardClick`: Update từ `board_diff`
3. ✅ Polling: Reload nếu có thay đổi

**Vấn đề:**
- ⚠️ Race condition: User action và polling có thể conflict
- **Giải pháp**: Dùng flag `isProcessing` để prevent polling update

### 4.3. Ready Status Synchronization

**Nguồn truth:**
- Backend: `match.black_ready` và `match.white_ready`
- Frontend: `isReady` và `opponentReady` states

**Sync points:**
1. ✅ `MatchFoundDialog` mount: Fetch từ server
2. ✅ Polling: Sync mỗi 1 giây
3. ✅ `handleReady`: Update từ response

**Vấn đề:**
- ✅ **Đã xử lý tốt**: Polling sync ready status liên tục

## 5. CÁC VẤN ĐỀ ĐÃ SỬA

### 5.1. Critical Issues ✅

1. ✅ **MatchmakingDialog**: Set `inQueue = false` khi match found
2. ✅ **MatchFoundDialog**: Remove `isReady` khỏi polling dependency
3. ✅ **MainWindow**: Double-check game over trong `handleBoardClick` và `handlePass`
4. ✅ **MainWindow**: Check đúng lượt và đủ người chơi trong `handleBoardClick` và `handlePass`

### 5.2. Improvements ✅

1. ✅ **State management**: Tạo helper function `setGameOverState` để set game over state (tránh duplicate code)
2. ⚠️ **Race condition**: Polling đã check `isProcessing`, nhưng có thể cải thiện thêm với flag `isUpdatingFromPoll`
3. ✅ **Error handling**: Đã có error handling tốt, có thể thêm retry logic nếu cần
4. ✅ **User feedback**: Loading state đã được hiển thị rõ ràng

### 5.3. Optional Improvements (Có thể làm sau)

1. **Race condition**: Thêm flag `isUpdatingFromPoll` để prevent polling update khi đang process
2. **Error handling**: Thêm retry logic cho network errors (hiện tại đã có timeout handling)
3. **Performance**: Optimize polling frequency dựa trên game state

## 6. TESTING CHECKLIST

### 6.1. Matchmaking Flow
- [ ] User join queue → match found → ready → start game
- [ ] User join queue → cancel → leave queue
- [ ] User join queue → close dialog → cleanup queue
- [ ] Network timeout → retry → success
- [ ] Match found nhưng user đã rời queue

### 6.2. Game Play Flow
- [ ] User click → move success → board update
- [ ] User click → game over → board locked
- [ ] User pass → game over → board locked
- [ ] Opponent move → polling detect → board update
- [ ] Game over → user cannot click/pass/undo
- [ ] Time out → auto-resign → game over

### 6.3. Edge Cases
- [ ] Opponent disconnect → auto-resign → game over
- [ ] Network error → retry → success
- [ ] Race condition: User action + polling
- [ ] Multiple rapid clicks → only one move processed
- [ ] Undo khi game over → rejected

