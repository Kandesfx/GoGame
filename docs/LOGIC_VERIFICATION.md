# 🔍 KIỂM TRA LOGIC CAPTURE TRONG C++

## 📋 PHÂN TÍCH LOGIC HIỆN TẠI

### 1. **collect_group()** - Tính số khí (liberties)

```cpp
Board::GroupInfo Board::collect_group(int index) const {
    // BFS để thu thập tất cả quân cùng màu liên kết
    // Đếm liberties = các giao điểm trống (Stone::Empty) liền kề
    for (const int neighbor : neighbors(current)) {
        if (neighbor_stone == Stone::Empty && !liberty_seen[neighbor]) {
            liberty_seen[neighbor] = true;
            info.liberties.push_back(neighbor);
        }
    }
}
```

**✅ Logic đúng:**
- Chỉ đếm các vị trí trống (Stone::Empty)
- Không đếm vị trí có quân (dù là quân mình hay đối phương)
- Dùng BFS để thu thập toàn bộ nhóm

### 2. **apply_move()** - Logic Capture

```cpp
// Bước 1: Đặt quân
grid_[index] = stone_from_color(move.color());

// Bước 2: Kiểm tra các neighbor
for (const int neighbor : neighbors(index)) {
    if (grid_[neighbor] == opponent_stone) {
        // Bước 3: Thu thập nhóm đối phương
        const GroupInfo opponent_group = collect_group(neighbor);
        
        // Bước 4: Kiểm tra nếu nhóm không còn khí
        if (opponent_group.liberties.empty()) {
            // Bước 5: Capture toàn bộ nhóm
            captured_indices_set.insert(...);
        }
    }
}
```

**✅ Logic đúng:**
- Sau khi đặt quân tại `index`, vị trí đó không còn là liberty của nhóm đối phương
- `collect_group()` được gọi SAU KHI đã đặt quân, nên nó sẽ không đếm `index` là liberty
- Nếu nhóm không còn liberties nào khác → bị bắt

---

## 🧪 KIỂM TRA VỚI VÍ DỤ

### Ví dụ: Capture nhóm trắng (1,1) và (1,2)

**Trạng thái ban đầu:**
```
  y=0  y=1  y=2  y=3
x=0  .    .    .    .
x=1  .    W    W    .  ← Nhóm trắng: (1,1), (1,2)
x=2  .    B    .    .  ← Quân đen: (2,1)
x=3  .    .    .    .
```

**Khí của nhóm trắng:**
- `collect_group()` sẽ tìm tất cả neighbors trống:
  - (1,1) có neighbors: (0,1), (2,1), (1,0), (1,2)
  - (1,2) có neighbors: (0,2), (2,2), (1,1), (1,3)
  - Tổng hợp: (0,1), (0,2), (2,1), (2,2), (1,0), (1,3) = **6 khí** ✅

**Bước 1: Đen đặt quân tại (2,2)**
```cpp
index = to_index(2, 2)  // Đặt quân đen tại (2,2)
grid_[index] = Stone::Black

// Kiểm tra neighbors của (2,2): (1,2), (3,2), (2,1), (2,3)
// neighbor (1,2) là quân trắng → collect_group(1,2)
opponent_group = collect_group(1,2)
// Nhóm trắng bây giờ có liberties: (0,1), (0,2), (1,0), (1,3)
// (2,2) đã bị quân đen chiếm → không còn là liberty
// liberties.empty() = false → CHƯA BỊ BẮT ✅
```

**Bước 2-5: Tương tự, mỗi nước đi chặn 1 khí**

**Bước 5: Đen đặt quân tại (1,0)**
```cpp
index = to_index(1, 0)  // Đặt quân đen tại (1,0)
grid_[index] = Stone::Black

// Kiểm tra neighbors của (1,0): (0,0), (2,0), (1,1), (1,-1) [out of bounds]
// neighbor (1,1) là quân trắng → collect_group(1,1)
opponent_group = collect_group(1,1)
// Nhóm trắng bây giờ:
// - (1,1) có neighbors: (0,1)[B], (2,1)[B], (1,0)[B], (1,2)[W]
// - (1,2) có neighbors: (0,2)[B], (2,2)[B], (1,1)[W], (1,3)[B]
// Tất cả neighbors đều có quân → KHÔNG CÒN KHÍ
// liberties.empty() = true → BỊ BẮT ✅
```

---

## ⚠️ VẤN ĐỀ TIỀM ẨN

### Vấn đề 1: Capture nhiều nhóm cùng lúc

**Trường hợp:**
```
  .  .  .  .
  .  W  W  .
  .  B  B  .
  .  .  .  .
```

Nếu đen đặt quân tại (1,1), nó có thể bắt cả nhóm trắng (1,2) và (1,3) nếu chúng không còn khí.

**Logic hiện tại:**
- Kiểm tra từng neighbor một
- Mỗi nhóm được xử lý riêng
- ✅ Đúng: Nếu nhiều nhóm đều không còn khí → tất cả đều bị bắt

### Vấn đề 2: Capture sau khi đặt quân

**Logic hiện tại:**
1. Đặt quân trước
2. Sau đó mới kiểm tra capture

**✅ Đúng:** Đây là cách đúng vì:
- Quân mới đặt chiếm một liberty của nhóm đối phương
- Nếu nhóm không còn liberties nào khác → bị bắt
- Nếu nhóm còn ít nhất 1 liberty → không bị bắt

---

## ✅ KẾT LUẬN

**Logic C++ là ĐÚNG:**

1. ✅ `collect_group()` tính đúng số khí (chỉ đếm vị trí trống)
2. ✅ Capture logic đúng: Kiểm tra sau khi đặt quân
3. ✅ Xử lý đúng trường hợp capture nhiều nhóm
4. ✅ Xử lý đúng trường hợp capture nhóm lớn

**Không cần sửa gì trong logic C++!**

---

## 🧪 TEST CASE ĐỀ XUẤT

Để chắc chắn, nên test với:

1. **Capture single stone**
2. **Capture multiple stones (small group)**
3. **Capture large group (10+ stones)**
4. **Capture multiple groups in one move**
5. **Edge case: Capture at board edge**
6. **Edge case: Capture at board corner**

