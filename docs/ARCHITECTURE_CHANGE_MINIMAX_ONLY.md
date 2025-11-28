# 🔄 THAY ĐỔI KIẾN TRÚC - Minimax Only cho Level 3 & 4

## 📋 THAY ĐỔI

### Trước:
- **Level 1**: Minimax depth 1 + randomness
- **Level 2**: Minimax depth 2
- **Level 3**: MCTS 500 playouts + heuristic
- **Level 4**: MCTS 1200 playouts + heuristic

### Sau:
- **Level 1**: Minimax depth 1 + randomness
- **Level 2**: Minimax depth 2
- **Level 3**: **Minimax depth 4** + đầy đủ tính năng bổ trợ
- **Level 4**: **Minimax depth 5** + đầy đủ tính năng bổ trợ

---

## ✅ CẤU HÌNH MỚI

### Level 3 (Khó):
- **Algorithm**: Minimax
- **Base Depth**: 4 (cho 9×9)
- **Tự động điều chỉnh**:
  - 9×9: Depth 4
  - 13×13: Depth 3 (giảm 1)
  - 19×19: Depth 2 (giảm 2)
- **Alpha-Beta Pruning**: ✅ Bật (depth >= 2)
- **Move Ordering**: ✅ Bật (depth >= 2)
- **Transposition Table**: ✅ Bật (depth >= 3)
- **Time Limit**: Không giới hạn

### Level 4 (Siêu Khó):
- **Algorithm**: Minimax
- **Base Depth**: 5 (cho 9×9)
- **Tự động điều chỉnh**:
  - 9×9: Depth 5
  - 13×13: Depth 4 (giảm 1)
  - 19×19: Depth 3 (giảm 2)
- **Alpha-Beta Pruning**: ✅ Bật (depth >= 2)
- **Move Ordering**: ✅ Bật (depth >= 2)
- **Transposition Table**: ✅ Bật (depth >= 3)
- **Time Limit**: Không giới hạn

---

## 📊 SO SÁNH

| Level | Algorithm | Depth (9×9) | Depth (13×13) | Depth (19×19) | Tính năng |
|-------|-----------|-------------|--------------|--------------|-----------|
| **1** | Minimax | 1 | 1 | 1 | Random + Mistake |
| **2** | Minimax | 2 | 2 | 2 | Basic |
| **3** | **Minimax** | **4** | **3** | **2** | **Full features** |
| **4** | **Minimax** | **5** | **4** | **3** | **Full features** |

---

## ⚙️ TÍNH NĂNG BỔ TRỢ

Tất cả level 3-4 đều có:

### 1. Alpha-Beta Pruning
- Giảm số nodes cần search
- Tăng tốc độ đáng kể
- Bật khi depth >= 2

### 2. Move Ordering
- Sắp xếp moves theo priority:
  - Capture moves: +1000 điểm
  - Saves atari: +500 điểm
  - Star points: +30 điểm
  - Center position: bonus
- Bật khi depth >= 2

### 3. Transposition Table
- Cache kết quả đã tính
- Tránh tính lại các position giống nhau
- Cache size: 1,000,000 entries
- Bật khi depth >= 3

### 4. Evaluator
- Territory evaluation
- Prisoners evaluation
- Group strength evaluation
- Influence evaluation
- Pattern recognition

---

## ⏱️ TIMEOUT (Backend)

| Level | 9×9 | 13×13 | 19×19 |
|-------|-----|-------|-------|
| **1** | 15s | 15s | 15s |
| **2** | 20s | 20s | 20s |
| **3** | 20s | 40s | 60s |
| **4** | 40s | 80s | 120s |

---

## 🎯 ƯU ĐIỂM

### 1. Đơn giản hơn
- Chỉ dùng Minimax, không cần MCTS
- Dễ debug và maintain
- Consistent với level 1-2

### 2. Tốc độ
- Minimax với Alpha-Beta nhanh hơn MCTS ở giai đoạn đầu
- Transposition table giúp cache kết quả
- Move ordering giúp pruning hiệu quả

### 3. Chất lượng
- Depth 4-5 đủ để đánh giá tốt
- Đầy đủ tính năng bổ trợ
- Đánh giá chính xác hơn MCTS với ít playouts

### 4. Tự động điều chỉnh
- Depth tự động giảm theo board size
- Đảm bảo không quá chậm trên bàn cờ lớn
- Vẫn mạnh trên bàn cờ nhỏ

---

## 🔧 THAY ĐỔI CODE

### Files Modified:

1. **src/ai/ai_player.cpp**
   - ✅ Level 3: MCTS → Minimax depth 4
   - ✅ Level 4: MCTS → Minimax depth 5
   - ✅ Xóa hybrid Minimax/MCTS strategy
   - ✅ Thêm logic tự động điều chỉnh depth theo board size
   - ✅ Bật đầy đủ tính năng bổ trợ

2. **backend/app/utils/ai_wrapper.py**
   - ✅ Cập nhật timeout cho level 3-4
   - ✅ Timeout dựa trên board size

---

## 🧪 TEST

1. **Test Level 3**:
   - Tạo match với AI level 3
   - Kiểm tra: AI đánh nhanh, thông minh, không random

2. **Test Level 4**:
   - Tạo match với AI level 4
   - Kiểm tra: AI đánh mạnh, thông minh, không random

3. **Test Board Sizes**:
   - 9×9: Depth 4/5 (mạnh)
   - 13×13: Depth 3/4 (vừa phải)
   - 19×19: Depth 2/3 (nhanh)

---

## 📝 LƯU Ý

1. **Cần rebuild C++ code** sau khi sửa
2. **Depth cao hơn = Chậm hơn** nhưng thông minh hơn
3. **Có thể điều chỉnh depth** nếu cần:
   - Level 3: 4 → 3 (nhanh hơn, yếu hơn)
   - Level 4: 5 → 4 (nhanh hơn, yếu hơn)
4. **Transposition table** giúp cache, nên lần sau sẽ nhanh hơn

---

## 🎯 NEXT STEPS

1. ✅ Rebuild project
2. ✅ Test với level 3 và 4
3. ⏳ Monitor performance và quality
4. ⏳ Điều chỉnh depth nếu cần

---

**Kiến trúc mới: Tất cả levels đều dùng Minimax, chỉ khác depth và tính năng bổ trợ! 🚀**

