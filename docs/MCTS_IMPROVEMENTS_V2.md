# 🔧 CẢI TIẾN MCTS - Sửa Lỗi Đánh Ở Góc và Thiếu Tấn Công/Phòng Thủ

## 🐛 VẤN ĐỀ ĐÃ PHÁT HIỆN

### Triệu chứng:
- ❌ AI có xu hướng đánh ở góc
- ❌ Thiếu sự tấn công
- ❌ Thiếu sự phòng thủ
- ❌ Không thông minh lắm

### Nguyên nhân:

1. **Không có penalty cho góc** ⚠️
   - Code chỉ có bonus cho center (+20)
   - Không có penalty cho góc → AI có thể chọn góc vì không có lý do tránh

2. **Scoring function quá đơn giản** ⚠️
   - Chỉ dùng prisoners + territory estimate
   - Không tính đến:
     - Influence (ảnh hưởng)
     - Group safety (an toàn của nhóm)
     - Attack opportunities
     - Defense needs

3. **Heuristic rollout không đủ tốt** ⚠️
   - Chỉ check neighbors, không đánh giá strategic value
   - Không có position evaluation trong rollout

4. **Thiếu strategic evaluation** ⚠️
   - AI không biết khi nào nên tấn công
   - AI không biết khi nào nên phòng thủ

---

## ✅ GIẢI PHÁP ĐÃ TRIỂN KHAI

### 1. Thêm Corner Penalty và Center Bonus

**File**: `src/ai/mcts/mcts_engine.cpp` - `evaluate_move_priority()`

**Thay đổi**:
```cpp
// Trước: Chỉ có center bonus
if (dist_from_center < size / 3) {
    priority += 20;  // Center bonus
}

// Sau: Corner penalty + Center bonus mạnh hơn
const bool is_corner = (x == 0 || x == size - 1) && (y == 0 || y == size - 1);
const bool is_edge = (x == 0 || x == size - 1 || y == 0 || y == size - 1);

if (is_corner) {
    priority -= 100;  // Penalty cho góc (trừ điểm)
} else if (is_edge && dist_from_center > size * 2 / 3) {
    priority -= 30;  // Penalty cho edge xa center
}

if (dist_from_center < size / 3) {
    priority += 50;  // Center bonus (tăng từ 20)
} else if (dist_from_center < size / 2) {
    priority += 20;  // Near center
}
```

**Kết quả**: AI sẽ tránh góc và ưu tiên center.

### 2. Cải thiện Scoring Function

**File**: `src/ai/mcts/mcts_engine.cpp` - `heuristic_rollout()`

**Trước**:
```cpp
// Chỉ prisoners + territory
black_score = prisoners + territory
white_score = prisoners + territory
```

**Sau**:
```cpp
// Prisoners + Territory + Influence + Group Safety
black_score = prisoners * 2 +           // Prisoners worth more
              territory +
              influence / 2 +           // Influence bonus
              safe_groups / 2;          // Safety bonus
```

**Cải tiến**:
- **Influence**: Đếm 8 directions (bao gồm diagonals)
- **Group Safety**: Đếm groups có >= 3 liberties
- **Prisoners**: Weight x2 (quan trọng hơn)

**Kết quả**: Scoring chính xác hơn, phản ánh tốt hơn tình thế.

### 3. Thêm Position Evaluation trong Rollout

**File**: `src/ai/mcts/mcts_engine.cpp` - `heuristic_rollout()`

**Thay đổi**:
- Thêm corner penalty trong quick evaluation
- Thêm center bonus trong quick evaluation
- Tránh góc ngay cả trong rollout

**Kết quả**: Rollout quality cao hơn, không chọn góc.

### 4. Thêm Star Points Bonus

**File**: `src/ai/mcts/mcts_engine.cpp` - `evaluate_move_priority()`

**Thay đổi**:
```cpp
// Star points bonus (opening)
if (size == 9) {
    star_points = {{2, 2}, {6, 2}, {2, 6}, {6, 6}, {4, 4}};
    priority += 40;  // Star point bonus
} else if (size == 19) {
    star_points = {{3, 3}, {3, 9}, {3, 15}, ...};
    priority += 40;  // Star point bonus
}
```

**Kết quả**: AI ưu tiên star points (vị trí tốt trong opening).

---

## 📊 SO SÁNH TRƯỚC/SAU

| Tính năng | Trước | Sau |
|-----------|-------|-----|
| **Corner handling** | Không có penalty | -100 penalty |
| **Center bonus** | +20 | +50 (mạnh hơn) |
| **Scoring** | Prisoners + Territory | + Influence + Safety |
| **Influence** | 4 directions | 8 directions |
| **Star points** | Không có | +40 bonus |
| **Edge penalty** | Không có | -30 (xa center) |

---

## 🎯 KẾT QUẢ MONG ĐỢI

### 1. Tránh góc:
- ✅ Corner penalty -100 → AI sẽ tránh góc
- ✅ Center bonus +50 → AI ưu tiên center
- ✅ Edge penalty -30 → AI tránh edge xa center

### 2. Tấn công tốt hơn:
- ✅ Influence evaluation → AI biết vị trí có ảnh hưởng
- ✅ Capture priority +1000 → AI ưu tiên captures
- ✅ Atari priority +500 → AI ưu tiên atari

### 3. Phòng thủ tốt hơn:
- ✅ Group safety evaluation → AI biết groups nào an toàn
- ✅ Connection bonus +50 → AI kết nối groups
- ✅ Saves atari +500 → AI cứu groups bị đe dọa

### 4. Strategic hơn:
- ✅ Star points bonus → AI chơi opening tốt hơn
- ✅ Position evaluation → AI đánh giá vị trí tốt hơn
- ✅ Improved scoring → AI đánh giá tình thế chính xác hơn

---

## 🔧 CÁCH REBUILD

Sau khi sửa code C++, cần rebuild:

```bash
cd build
cmake ..
cmake --build .
```

Sau đó restart backend server.

---

## 🧪 TEST

1. **Test tránh góc**:
   - Tạo match với AI level 3-4
   - Kiểm tra: AI không đánh ở góc (0,0), (0,8), (8,0), (8,8)

2. **Test tấn công**:
   - Tạo match với AI level 3-4
   - Kiểm tra: AI có tấn công khi có cơ hội capture/atari

3. **Test phòng thủ**:
   - Tạo match với AI level 3-4
   - Kiểm tra: AI có phòng thủ khi groups bị đe dọa

4. **Test strategic**:
   - Tạo match với AI level 3-4
   - Kiểm tra: AI chơi star points trong opening

---

## 📝 CHI TIẾT THAY ĐỔI

### Files Modified:

1. **src/ai/mcts/mcts_engine.cpp**
   - ✅ Thêm corner penalty trong `evaluate_move_priority()`
   - ✅ Tăng center bonus từ 20 → 50
   - ✅ Thêm star points bonus
   - ✅ Cải thiện scoring function (influence + safety)
   - ✅ Thêm position evaluation trong rollout

---

## ⚠️ LƯU Ý

1. **Cần rebuild C++ code** sau khi sửa
2. **Test kỹ** để đảm bảo AI không còn đánh ở góc
3. **Có thể điều chỉnh** penalty/bonus nếu cần:
   - Corner penalty: -100 (có thể -50 đến -150)
   - Center bonus: +50 (có thể +30 đến +70)
   - Star points: +40 (có thể +20 đến +60)

---

## 🎯 NEXT STEPS

1. ✅ Rebuild project
2. ✅ Test với level 3 và 4
3. ⏳ Monitor performance và quality
4. ⏳ Điều chỉnh penalty/bonus nếu cần

---

**Cải tiến này sẽ làm AI thông minh hơn, tránh góc, và có tấn công/phòng thủ tốt hơn! 🚀**

