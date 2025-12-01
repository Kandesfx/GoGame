# 📊 CHI TIẾT CHẾ ĐỘ KHÓ (LEVEL 3) VÀ SIÊU KHÓ (LEVEL 4)

## 🎯 TỔNG QUAN

Level 3 và 4 sử dụng **Hybrid Minimax/MCTS Strategy** - kết hợp ưu điểm của cả hai thuật toán:
- **Giai đoạn đầu (1/3 trận)**: Minimax với đầy đủ tính năng bổ trợ
- **Giai đoạn sau (2/3 trận)**: MCTS với heuristic rollout

---

## 🔴 LEVEL 3 (KHÓ)

### Cấu hình cơ bản:
- **Algorithm chính**: MCTS
- **Playouts**: 500 playouts
- **Heuristics**: ✅ Bật (heuristic rollout)
- **Randomness**: ❌ Không có (luôn chọn best move)
- **Threads**: 1 thread

### Hybrid Strategy:

#### **Giai đoạn đầu (1/3 trận) - Minimax:**
- **Algorithm**: Minimax với đầy đủ tính năng bổ trợ
- **Depth** (tự động theo board size):
  - 9×9: **Depth 4**
  - 13×13: **Depth 3**
  - 19×19: **Depth 2**
- **Alpha-Beta Pruning**: ✅ Bật (depth >= 2)
- **Move Ordering**: ✅ Bật (depth >= 2)
  - Capture moves: +1000 điểm
  - Saves atari: +500 điểm
  - Star points: +30 điểm
  - Center position: bonus
- **Transposition Table**: ✅ Bật (depth >= 3)
  - Cache size: 1,000,000 entries
- **Time Limit**: Không giới hạn

#### **Giai đoạn sau (2/3 trận) - MCTS:**
- **Algorithm**: MCTS với heuristic rollout
- **Playouts**: 500 playouts
- **Heuristic Rollout**: ✅ Bật
  - Quick evaluation (không test board)
  - Prioritize captures, atari, connections
  - Top 30% moves selection
- **UCB Constant**: 1.414 (√2)
- **Time Limit**: Không giới hạn (dùng số playouts)

### Ngưỡng chuyển đổi:
- **9×9**: ~18 moves đầu → Minimax, sau đó → MCTS
- **13×13**: ~34 moves đầu → Minimax, sau đó → MCTS
- **19×19**: ~72 moves đầu → Minimax, sau đó → MCTS

### Timeout (Backend):
- **9×9**: 45 giây
- **19×19**: 60 giây

---

## 🔴 LEVEL 4 (SIÊU KHÓ)

### Cấu hình cơ bản:
- **Algorithm chính**: MCTS
- **Playouts**: 1200 playouts
- **Heuristics**: ✅ Bật (heuristic rollout)
- **Randomness**: ❌ Không có (luôn chọn best move)
- **Threads**: 1 thread

### Hybrid Strategy:

#### **Giai đoạn đầu (1/3 trận) - Minimax:**
- **Algorithm**: Minimax với đầy đủ tính năng bổ trợ
- **Depth** (tự động theo board size):
  - 9×9: **Depth 4**
  - 13×13: **Depth 3**
  - 19×19: **Depth 2**
- **Alpha-Beta Pruning**: ✅ Bật (depth >= 2)
- **Move Ordering**: ✅ Bật (depth >= 2)
  - Capture moves: +1000 điểm
  - Saves atari: +500 điểm
  - Star points: +30 điểm
  - Center position: bonus
- **Transposition Table**: ✅ Bật (depth >= 3)
  - Cache size: 1,000,000 entries
- **Time Limit**: Không giới hạn

#### **Giai đoạn sau (2/3 trận) - MCTS:**
- **Algorithm**: MCTS với heuristic rollout
- **Playouts**: 1200 playouts (nhiều hơn level 3)
- **Heuristic Rollout**: ✅ Bật
  - Quick evaluation (không test board)
  - Prioritize captures, atari, connections
  - Top 30% moves selection
- **UCB Constant**: 1.414 (√2)
- **Time Limit**: Không giới hạn (dùng số playouts)

### Ngưỡng chuyển đổi:
- **9×9**: ~18 moves đầu → Minimax, sau đó → MCTS
- **13×13**: ~34 moves đầu → Minimax, sau đó → MCTS
- **19×19**: ~72 moves đầu → Minimax, sau đó → MCTS

### Timeout (Backend):
- **9×9**: 90 giây
- **19×19**: 120 giây

---

## 🔧 CHI TIẾT KỸ THUẬT

### Minimax Engine (Giai đoạn đầu):

#### Evaluator:
- **Territory**: Đánh giá vùng đất
- **Prisoners**: Đánh giá quân bắt được
- **Group Strength**: Đánh giá sức mạnh nhóm quân
- **Influence**: Đánh giá ảnh hưởng
- **Patterns**: Nhận diện pattern

#### Move Ordering:
- Sắp xếp moves theo priority trước khi search
- Giúp Alpha-Beta pruning hiệu quả hơn
- Ưu tiên: Captures > Atari > Star points > Center

#### Transposition Table:
- Cache kết quả đã tính
- Tránh tính lại các position giống nhau
- Sử dụng Zobrist hashing

### MCTS Engine (Giai đoạn sau):

#### Heuristic Rollout:
- **Quick Evaluation**: Chỉ check neighbors, không test board
- **Priority System**:
  - Potential capture/atari: +100 điểm
  - Connection: +20 điểm
  - Center bonus: +20 điểm
- **Selection**: Top 30% moves (không phải random hoàn toàn)

#### UCB Selection:
- **Formula**: `exploitation + exploration`
- **Exploitation**: Win rate
- **Exploration**: UCB constant × √(log(parent_visits) / visits)
- **UCB Constant**: 1.414 (√2)

#### Best Child Selection:
- **Robust Child**: Chọn move có nhiều visits nhất
- **Không random**: Luôn chọn best move

---

## 📈 SO SÁNH LEVEL 3 VÀ 4

| Tính năng | Level 3 (Khó) | Level 4 (Siêu Khó) |
|-----------|---------------|---------------------|
| **MCTS Playouts** | 500 | 1200 |
| **Minimax Depth (9×9)** | 4 | 4 |
| **Minimax Depth (13×13)** | 3 | 3 |
| **Minimax Depth (19×19)** | 2 | 2 |
| **Heuristics** | ✅ | ✅ |
| **Randomness** | ❌ | ❌ |
| **Timeout (9×9)** | 45s | 90s |
| **Timeout (19×19)** | 60s | 120s |
| **Độ mạnh** | Mạnh | Rất mạnh |

---

## 🎮 CÁCH HOẠT ĐỘNG

### Flow Chart:

```
Level 3/4 Start
    ↓
Check move_count
    ↓
move_count < 1/3 trận?
    ├─ YES → Minimax (với đầy đủ tính năng)
    │         ↓
    │      Alpha-Beta Pruning
    │         ↓
    │      Move Ordering
    │         ↓
    │      Transposition Table
    │         ↓
    │      Return best move
    │
    └─ NO → MCTS (với heuristic)
              ↓
           Heuristic Rollout
              ↓
           500/1200 playouts
              ↓
           UCB Selection
              ↓
           Return best move
```

---

## ⚙️ TỐI ƯU HÓA

### Tại sao Hybrid Strategy?

1. **Giai đoạn đầu (Minimax)**:
   - Ít quân → Minimax nhanh và chính xác
   - Depth 4 có thể search toàn bộ không gian
   - Alpha-Beta pruning rất hiệu quả

2. **Giai đoạn sau (MCTS)**:
   - Nhiều quân → Minimax quá chậm
   - MCTS với heuristic rollout nhanh và tốt
   - 500-1200 playouts đủ để đánh giá tốt

### Tại sao không random?

- Level 3-4 là "khó" và "siêu khó"
- Người chơi mong đợi AI mạnh, không có nước đi sai
- Random làm giảm chất lượng và không tự nhiên

---

## 🔍 DEBUGGING

### Logs có thể thấy:
- `AI level 3, board size 9x9, timeout: 45s`
- `AI subprocess timeout after 45s (level 3, board 9x9)`

### Kiểm tra:
1. Move count để xem đang ở giai đoạn nào
2. Algorithm đang dùng (Minimax hay MCTS)
3. Timeout có đủ không

---

## 📝 LƯU Ý

1. **Cần rebuild C++** sau khi thay đổi code
2. **Timeout** là safety net, MCTS có thể dừng sớm hơn
3. **Ngưỡng 1/3** có thể điều chỉnh nếu cần
4. **Playouts** có thể tăng/giảm tùy performance

---

**Cập nhật lần cuối**: Sau khi implement hybrid strategy với Minimax đầy đủ tính năng bổ trợ.

