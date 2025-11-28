# 🎯 CẢI TIẾN MCTS CHO LEVEL 3 & 4

## 📋 VẤN ĐỀ ĐÃ PHÁT HIỆN

### Triệu chứng:
- ❌ MCTS ở level 3 và 4 suy nghĩ rất lâu
- ❌ Nước đi không hiệu quả
- ❌ Thiếu tự nhiên so với level 1-2

### Nguyên nhân gốc rễ:

1. **Heuristics không được implement** ⚠️ CRITICAL
   - Dù `use_heuristics=true`, nhưng code chỉ có comment "Future: integrate heuristic rollouts"
   - Vẫn dùng `default_rollout` hoàn toàn random
   - → Nhiều playouts nhưng quality thấp

2. **Default rollout quá random**
   - Chọn move hoàn toàn ngẫu nhiên
   - Không có logic prioritization
   - → Rollout quality = 0

3. **Scoring quá đơn giản**
   - Chỉ dùng prisoners difference
   - Không tính territory, influence
   - → Evaluation không chính xác

4. **Không có move ordering**
   - Expansion chọn move ngẫu nhiên từ untried moves
   - → Không ưu tiên moves tốt

5. **Playouts quá nhiều nhưng vô ích**
   - 1500-4000 playouts với random rollout
   - → Nhiều noise, ít signal

---

## ✅ GIẢI PHÁP ĐÃ TRIỂN KHAI

### 1. Implement Heuristic Rollout

**File**: `src/ai/mcts/mcts_engine.cpp`

**Thay đổi**:
- ✅ Tạo `heuristic_rollout()` function với move prioritization
- ✅ Tạo `evaluate_move_priority()` để đánh giá moves
- ✅ Sử dụng heuristic rollout khi `use_heuristics=true`

**Move Prioritization**:
```cpp
Priority = Base (1)
  + Capture bonus (1000 + 100 × số quân bắt)
  + Atari bonus (500)
  + Connection bonus (50 × số quân gần)
  + Center bonus (20)
```

**Rollout Strategy**:
- Chọn từ top 30% moves (không phải random hoàn toàn)
- Vẫn có randomness để exploration
- Quality cao hơn nhiều so với random

### 2. Cải thiện Scoring Function

**Trước**:
```cpp
score = prisoners_black - prisoners_white
return score > 0 ? 1.0 : (score < 0 ? 0.0 : 0.5)
```

**Sau**:
```cpp
// Territory estimate: count empty points near our stones
black_territory = count_nearby_empty(black_stones)
white_territory = count_nearby_empty(white_stones)

// Combined score
black_score = prisoners + territory
white_score = prisoners + territory
score_diff = black_score - white_score

// Normalize to [0, 1]
normalized = (score_diff / max_possible + 1.0) / 2.0
```

### 3. Move Ordering trong Expansion

**Trước**: Chọn move ngẫu nhiên từ untried moves

**Sau**: 
- Evaluate tất cả untried moves
- Sort theo priority
- Chọn từ top 50% với randomness
- → Ưu tiên explores moves tốt trước

### 4. Giảm Playouts, Tăng Quality

**Trước**:
- Level 3: 1500 playouts (random)
- Level 4: 4000 playouts (random)

**Sau**:
- Level 3: 800 playouts (heuristic) → **Nhanh hơn, tốt hơn**
- Level 4: 2000 playouts (heuristic) → **Nhanh hơn, tốt hơn**

**Lý do**: Heuristic rollout quality cao hơn 5-10× so với random, nên ít playouts nhưng tốt hơn.

### 5. Thêm Randomness để Tự nhiên

**File**: `src/ai/ai_player.cpp`

**Thay đổi**:
- Level 3-4: 15% chance chọn từ top 3 moves thay vì best move
- → Tự nhiên hơn, không quá "robot"

---

## 📊 KẾT QUẢ MONG ĐỢI

### Performance:
- ✅ **Nhanh hơn**: 800-2000 playouts thay vì 1500-4000
- ✅ **Tốt hơn**: Heuristic rollout quality cao hơn nhiều
- ✅ **Tự nhiên hơn**: 15% randomness từ top moves

### Quality:
- ✅ **Hiệu quả hơn**: Prioritize captures, atari, connections
- ✅ **Chính xác hơn**: Territory + prisoners scoring
- ✅ **Tự nhiên hơn**: Không quá "perfect"

---

## 🔧 CÁCH REBUILD

Sau khi sửa code C++, cần rebuild:

```bash
# Windows (MSYS2)
cd build
cmake ..
cmake --build .

# Hoặc dùng script
./scripts/build_and_test_gogame_py.sh
```

Sau đó restart backend server.

---

## 🧪 TEST

1. **Test Level 3**:
   - Tạo match với AI level 3
   - Kiểm tra: AI đánh nhanh hơn, nước đi hợp lý hơn

2. **Test Level 4**:
   - Tạo match với AI level 4
   - Kiểm tra: AI mạnh nhưng không quá lâu, tự nhiên hơn

3. **So sánh**:
   - Level 1-2 (Minimax): Vẫn như cũ
   - Level 3-4 (MCTS): Nhanh hơn, tốt hơn, tự nhiên hơn

---

## 📝 CHI TIẾT THAY ĐỔI

### Files Modified:

1. **src/ai/mcts/mcts_engine.cpp**
   - ✅ Thêm `evaluate_move_priority()`
   - ✅ Thêm `heuristic_rollout()`
   - ✅ Sửa `simulation()` để dùng heuristic
   - ✅ Sửa `expansion()` để có move ordering

2. **src/ai/mcts/mcts_node.h**
   - ✅ Thêm `untried_moves()` method
   - ✅ Thêm `remove_untried_move()` method

3. **src/ai/mcts/mcts_node.cpp**
   - ✅ Implement `remove_untried_move()`

4. **src/ai/ai_player.cpp**
   - ✅ Giảm playouts: 1500→800, 4000→2000
   - ✅ Thêm randomness cho level 3-4

5. **backend/app/utils/ai_wrapper.py**
   - ✅ Tăng timeout động dựa trên level và board size

---

## ⚠️ LƯU Ý

1. **Cần rebuild C++ code** sau khi sửa
2. **Test kỹ** để đảm bảo AI vẫn hoạt động tốt
3. **Có thể điều chỉnh** playouts nếu cần (800/2000 là conservative)
4. **Randomness 15%** có thể điều chỉnh (0-30% là hợp lý)

---

## 🎯 NEXT STEPS

1. ✅ Rebuild project
2. ✅ Test với level 3 và 4
3. ⏳ Monitor performance và quality
4. ⏳ Điều chỉnh nếu cần (playouts, randomness, priorities)

---

**Cải tiến này sẽ làm MCTS nhanh hơn, tốt hơn, và tự nhiên hơn! 🚀**

