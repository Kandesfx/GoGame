# 🎲 PLAYOUTS LÀ GÌ? - Giải Thích Chi Tiết

## 📖 ĐỊNH NGHĨA

**Playouts** (hay **simulations**) là số lần MCTS (Monte Carlo Tree Search) thực hiện **rollout/simulation** để đánh giá một nước đi.

Mỗi **playout** = 1 lần chơi thử từ vị trí hiện tại đến hết ván (hoặc đến một độ sâu nhất định) để xem kết quả.

---

## 🔄 MCTS HOẠT ĐỘNG NHƯ THẾ NÀO?

MCTS có 4 bước chính:

### 1. **Selection** (Chọn)
- Chọn node tốt nhất để explore dựa trên UCB formula
- Đi từ root xuống leaf node

### 2. **Expansion** (Mở rộng)
- Thêm node con mới vào tree
- Chọn một move chưa được thử

### 3. **Simulation/Playout** (Mô phỏng) ⭐
- **Đây chính là PLAYOUTS!**
- Chơi thử từ node này đến hết ván (random hoặc heuristic)
- Tính kết quả (thắng/thua/hòa)

### 4. **Backpropagation** (Lan truyền ngược)
- Cập nhật thông tin (visits, wins) lên tất cả nodes từ leaf đến root

---

## 🎯 VÍ DỤ CỤ THỂ

### Ví dụ: 500 playouts

```
AI đang ở vị trí này:
  ● ○ ●
○ ● ○ ●
  ● ○

AI muốn đánh ở đâu?

MCTS sẽ:
1. Chọn một move để thử (ví dụ: đánh ở (2,2))
2. Thực hiện 500 playouts:
   - Playout 1: Chơi thử từ (2,2) → Kết quả: Thắng
   - Playout 2: Chơi thử từ (2,2) → Kết quả: Thua
   - Playout 3: Chơi thử từ (2,2) → Kết quả: Thắng
   - ...
   - Playout 500: Chơi thử từ (2,2) → Kết quả: Thắng

3. Tính win rate: 300/500 = 60% (ví dụ)
4. So sánh với các moves khác
5. Chọn move có win rate cao nhất
```

---

## 📊 PLAYOUTS TRONG DỰ ÁN NÀY

### Level 3 (Khó):
- **500 playouts** = MCTS sẽ thực hiện 500 lần simulation cho mỗi move candidate
- Mỗi playout = chơi thử từ vị trí đó đến hết ván

### Level 4 (Siêu Khó):
- **1200 playouts** = MCTS sẽ thực hiện 1200 lần simulation
- Nhiều playouts hơn = đánh giá chính xác hơn nhưng chậm hơn

---

## ⚡ HEURISTIC ROLLOUT

Trong dự án này, chúng ta dùng **heuristic rollout** thay vì **random rollout**:

### Random Rollout (Cũ):
```cpp
// Chọn move hoàn toàn ngẫu nhiên
for (mỗi playout) {
    while (chưa hết ván) {
        move = random_move();  // Ngẫu nhiên
        board.make_move(move);
    }
    result = evaluate(board);
}
```

### Heuristic Rollout (Mới - Nhanh hơn):
```cpp
// Chọn move có priority (captures, atari, connections)
for (mỗi playout) {
    while (chưa hết ván) {
        moves = get_legal_moves();
        moves = prioritize(moves);  // Ưu tiên moves tốt
        move = select_from_top_30%(moves);  // Chọn từ top 30%
        board.make_move(move);
    }
    result = evaluate(board);
}
```

**Kết quả**: Heuristic rollout **nhanh hơn 10-20×** và **chất lượng cao hơn** so với random rollout.

---

## 📈 MỐI QUAN HỆ: PLAYOUTS vs CHẤT LƯỢNG

| Playouts | Chất lượng | Thời gian | Phù hợp |
|----------|------------|-----------|---------|
| 100-300 | Thấp | Rất nhanh | Level dễ |
| 500-1000 | Trung bình | Nhanh | Level khó |
| 2000-5000 | Cao | Chậm | Level siêu khó |
| 10000+ | Rất cao | Rất chậm | Tournament |

**Lưu ý**: Với heuristic rollout, 500 playouts có thể tốt bằng 2000-3000 random playouts!

---

## 🎮 TRONG CODE

### Cấu hình:
```cpp
// Level 3
default_mcts_config(500, true, 1)
// 500 = số playouts
// true = dùng heuristic
// 1 = số threads

// Level 4
default_mcts_config(1200, true, 1)
// 1200 = số playouts (nhiều hơn)
```

### Thực thi:
```cpp
// Trong MCTSEngine::search()
for (int i = 0; i < max_playouts; ++i) {
    // 1. Selection
    MCTSNode *selected = selection(root, board);
    
    // 2. Expansion
    selected = expansion(selected, board);
    
    // 3. Simulation (PLAYOUT!)
    double result = simulation(board, player);
    
    // 4. Backpropagation
    backpropagation(selected, result);
}
```

---

## 🔍 TẠI SAO CẦN NHIỀU PLAYOUTS?

### Nhiều playouts = Nhiều thông tin:
- **100 playouts**: Chỉ thử 100 lần → Kết quả không chính xác
- **500 playouts**: Thử 500 lần → Kết quả tốt hơn
- **1200 playouts**: Thử 1200 lần → Kết quả rất tốt

### Nhưng:
- **Nhiều playouts = Chậm hơn**
- **Heuristic rollout** giúp giảm số playouts cần thiết

---

## 💡 TÓM TẮT

**Playouts** = Số lần MCTS "chơi thử" một nước đi để đánh giá nó tốt hay không.

- **500 playouts** = Chơi thử 500 lần
- **1200 playouts** = Chơi thử 1200 lần (chính xác hơn nhưng chậm hơn)
- **Heuristic rollout** = Chơi thử thông minh (nhanh và tốt hơn random)

**Công thức đơn giản**:
```
Nhiều playouts = Chất lượng cao hơn nhưng chậm hơn
Heuristic rollout = Giảm số playouts cần thiết mà vẫn giữ chất lượng
```

---

**Ví dụ thực tế**: 
- AlphaGo Zero dùng hàng triệu playouts (nhưng có GPU mạnh)
- Dự án này dùng 500-1200 playouts (phù hợp với CPU thông thường)
- Với heuristic rollout, 500 playouts đã đủ tốt!

