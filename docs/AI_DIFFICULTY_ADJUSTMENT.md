# Hướng Dẫn Điều Chỉnh Độ Khó AI

## Tổng Quan

AI hiện tại có 4 level (1-4), với level 1 là dễ nhất. Nếu level 1 vẫn quá thông minh, có thể điều chỉnh theo các hướng sau:

## Các Cách Giảm Độ Thông Minh AI

### 1. **Giảm Depth của Minimax** ✅ (Đã áp dụng)
- **Vị trí**: `src/ai/ai_player.cpp` - hàm `AIPlayer::AIPlayer()`
- **Hiện tại**: Level 1 dùng depth 1 (đã giảm từ 2)
- **Cách điều chỉnh**: Giảm `default_minimax_config(1)` xuống depth thấp hơn (nhưng depth 1 là tối thiểu)
- **Ảnh hưởng**: AI chỉ nhìn trước 1 nước, rất yếu

### 2. **Thêm Randomness (Ngẫu Nhiên)** ✅ (Đã áp dụng)
- **Vị trí**: `src/ai/ai_player.cpp` - hàm `AIPlayer::select_move()`
- **Hiện tại**: Level 1 có 30% chance chọn move ngẫu nhiên
- **Cách điều chỉnh**: Tăng tỷ lệ randomness (ví dụ: 0.30 → 0.50 = 50%)
- **Code**: `if (dis(gen) < 0.30)` → đổi thành `0.50` hoặc cao hơn
- **Ảnh hưởng**: AI đôi khi chọn move ngẫu nhiên thay vì best move

### 3. **Thêm Mistake Rate (Tỷ Lệ Sai Lầm)** ✅ (Đã áp dụng)
- **Vị trí**: `src/ai/ai_player.cpp` - hàm `AIPlayer::select_move()`
- **Hiện tại**: Level 1 có 20% chance chọn move không tối ưu (từ top 3-5 moves)
- **Cách điều chỉnh**: Tăng tỷ lệ mistake (ví dụ: 0.20 → 0.40 = 40%)
- **Code**: `if (dis(gen) < 0.20)` → đổi thành `0.40` hoặc cao hơn
- **Ảnh hưởng**: AI đôi khi chọn move tốt nhưng không phải best move

### 4. **Giảm Số Playouts của MCTS** (Nếu level 1 dùng MCTS)
- **Vị trí**: `src/ai/ai_player.cpp` - hàm `default_mcts_config()`
- **Hiện tại**: Level 1 dùng Minimax, không dùng MCTS
- **Cách điều chỉnh**: Nếu muốn level 1 dùng MCTS, giảm playouts xuống (ví dụ: 500-1000)
- **Code**: `default_mcts_config(500, true, 1)` thay vì 3000
- **Ảnh hưởng**: MCTS suy nghĩ ít hơn, yếu hơn

### 5. **Tắt Heuristics** (Nếu dùng MCTS)
- **Vị trí**: `src/ai/ai_player.cpp` - hàm `default_mcts_config()`
- **Cách điều chỉnh**: Đổi `use_heuristics` từ `true` → `false`
- **Code**: `default_mcts_config(playouts, false, 1)`
- **Ảnh hưởng**: MCTS không dùng heuristics, yếu hơn

### 6. **Giảm Time Limit**
- **Vị trí**: `src/ai/ai_player.cpp` - hàm `default_minimax_config()`
- **Hiện tại**: `time_limit_seconds = 0.0` (không giới hạn)
- **Cách điều chỉnh**: Đặt time limit ngắn (ví dụ: 0.5 giây)
- **Code**: `config.time_limit_seconds = 0.5;`
- **Ảnh hưởng**: AI có ít thời gian suy nghĩ, yếu hơn

### 7. **Tắt Alpha-Beta Pruning** (Cho level 1)
- **Vị trí**: `src/ai/ai_player.cpp` - hàm `default_minimax_config()`
- **Hiện tại**: Level 1 đã tắt (vì depth < 3)
- **Cách điều chỉnh**: Đảm bảo `use_alpha_beta = false` cho level 1
- **Ảnh hưởng**: Minimax không tối ưu, yếu hơn

### 8. **Tắt Move Ordering**
- **Vị trí**: `src/ai/ai_player.cpp` - hàm `default_minimax_config()`
- **Hiện tại**: Level 1 đã tắt (vì depth < 3)
- **Cách điều chỉnh**: Đảm bảo `use_move_ordering = false` cho level 1
- **Ảnh hưởng**: Minimax không sắp xếp moves, yếu hơn

## Cấu Hình Hiện Tại

### Level 1 (Dễ - Cho Bàn Cờ 9x9)
```cpp
// Depth 1 (rất yếu)
MinimaxEngine::Config config;
config.max_depth = 1;
config.use_alpha_beta = false;
config.use_move_ordering = false;
config.use_transposition = false;
config.time_limit_seconds = 0.0;  // Giữ nguyên

// 30% randomness + 20% mistake rate
// = 50% chance không chọn best move
```

### Level 2 (Trung Bình - Cho Bàn Cờ 9x9)
```cpp
// Depth 2 (vừa phải)
MinimaxEngine::Config config;
config.max_depth = 2;
config.use_alpha_beta = false;  // depth < 3 nên tắt
config.use_move_ordering = false;
config.use_transposition = false;
```

### Level 3 (Khó - Đã Giảm)
```cpp
// MCTS 1500 playouts (giảm từ 3000)
MCTSEngine::Config config;
config.playouts = 1500;
config.use_heuristics = true;
```

### Level 4 (Siêu Khó - Đã Giảm)
```cpp
// MCTS 4000 playouts (giảm từ 8000)
MCTSEngine::Config config;
config.playouts = 4000;
config.use_heuristics = true;
```

## Khuyến Nghị Điều Chỉnh

Nếu level 1 vẫn quá thông minh, có thể:

1. **Tăng Randomness lên 50-70%**: AI sẽ chọn move ngẫu nhiên nhiều hơn
2. **Tăng Mistake Rate lên 30-50%**: AI sẽ chọn move không tối ưu nhiều hơn
3. **Kết hợp cả hai**: Randomness 50% + Mistake 30% = AI chỉ chọn best move 20% thời gian

## Ví Dụ Code Điều Chỉnh

```cpp
// Trong AIPlayer::select_move(), level 1:
if (level == 1) {
    // Tăng randomness lên 50%
    if (dis(gen) < 0.50) {  // Thay vì 0.30
        // Chọn move ngẫu nhiên
    }
    
    // Tăng mistake rate lên 30%
    if (dis(gen) < 0.30) {  // Thay vì 0.20
        // Chọn move không tối ưu
    }
}
```

## Lưu Ý

- Sau khi sửa code C++, cần rebuild lại project
- Test kỹ để đảm bảo AI không quá yếu hoặc quá mạnh
- Có thể điều chỉnh từng tham số một để tìm mức độ phù hợp

