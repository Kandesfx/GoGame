# 🔍 CƠ CHẾ BẮT QUÂN CHI TIẾT

## 📋 TỔNG QUAN

Khi một quân được đặt xuống bàn cờ, hệ thống sẽ:
1. Đặt quân vào vị trí
2. Kiểm tra các nhóm đối phương xung quanh
3. Tính lại số khí của các nhóm đó
4. Nếu nhóm không còn khí → BẮT
5. Xóa quân bị bắt khỏi bàn cờ
6. Cập nhật prisoners (tù binh)

---

## 🔄 FLOW CHI TIẾT

### **BƯỚC 1: Đặt Quân** (`apply_move()` - Line 220-223)

```cpp
const int index = to_index(move.x(), move.y());
grid_[index] = stone_from_color(move.color());  // Đặt quân vào grid
hash_ ^= zobrist_table_[index][stone_index(grid_[index])];  // Cập nhật hash
ko_index_ = -1;  // Reset ko index
```

**Ví dụ:**
- Đen đặt quân tại (1,0)
- `grid_[to_index(1,0)] = Stone::Black`
- Bàn cờ bây giờ có quân đen tại (1,0)

---

### **BƯỚC 2: Kiểm Tra Neighbors** (`apply_move()` - Line 232-255)

```cpp
for (const int neighbor : neighbors(index)) {
    if (grid_[neighbor] == opponent_stone && !processed_group[neighbor]) {
        // Tìm thấy quân đối phương → Kiểm tra nhóm
    }
}
```

**`neighbors(index)` trả về:**
- 4 vị trí liền kề: trên, dưới, trái, phải
- Chỉ các vị trí trong bounds

**Ví dụ với (1,0):**
```cpp
neighbors(to_index(1,0)) = [
    to_index(0,0),  // Trên
    to_index(2,0),  // Dưới
    to_index(1,-1), // Trái (out of bounds → bỏ qua)
    to_index(1,1)   // Phải ← QUAN TRỌNG: Có quân trắng tại (1,1)
]
```

**Kiểm tra:**
- `grid_[to_index(1,1)] == Stone::White` → ✅ Là quân đối phương
- `!processed_group[to_index(1,1)]` → ✅ Chưa xử lý nhóm này

---

### **BƯỚC 3: Thu Thập Nhóm Đối Phương** (`collect_group()` - Line 166-198)

```cpp
const GroupInfo opponent_group = collect_group(neighbor);
```

**`collect_group(neighbor)` làm gì:**

#### 3.1. Khởi tạo
```cpp
GroupInfo info{};
const Stone color = grid_[neighbor];  // Màu của quân tại neighbor
std::vector<bool> visited(grid_.size(), false);
std::vector<bool> liberty_seen(grid_.size(), false);
std::queue<int> frontier;
```

#### 3.2. BFS (Breadth-First Search) để thu thập nhóm
```cpp
frontier.push(neighbor);  // Bắt đầu từ neighbor
visited[neighbor] = true;

while (!frontier.empty()) {
    const int current = frontier.front();
    frontier.pop();
    info.stones.push_back(current);  // Thêm quân vào nhóm
    
    // Kiểm tra 4 neighbors của current
    for (const int n : neighbors(current)) {
        const Stone n_stone = grid_[n];
        
        // Nếu là quân cùng màu → Thêm vào nhóm
        if (n_stone == color && !visited[n]) {
            visited[n] = true;
            frontier.push(n);
        }
        // Nếu là vị trí trống → Đếm là khí (liberty)
        else if (n_stone == Stone::Empty && !liberty_seen[n]) {
            liberty_seen[n] = true;
            info.liberties.push_back(n);
        }
    }
}
```

**Ví dụ với nhóm trắng (1,1) và (1,2):**

```
Bước 1: Bắt đầu từ (1,1)
  - current = (1,1)
  - info.stones = [(1,1)]
  - Kiểm tra neighbors của (1,1):
    * (0,1): Stone::Black → Bỏ qua
    * (2,1): Stone::Black → Bỏ qua
    * (1,0): Stone::Black → Bỏ qua (quân vừa đặt)
    * (1,2): Stone::White → Thêm vào frontier

Bước 2: Xử lý (1,2)
  - current = (1,2)
  - info.stones = [(1,1), (1,2)]
  - Kiểm tra neighbors của (1,2):
    * (0,2): Stone::Black → Bỏ qua
    * (2,2): Stone::Black → Bỏ qua
    * (1,1): Stone::White → Đã visited → Bỏ qua
    * (1,3): Stone::Black → Bỏ qua

Kết quả:
  - info.stones = [(1,1), (1,2)]  ✅
  - info.liberties = []  ✅ (KHÔNG CÒN KHÍ!)
```

**Tại sao không còn khí?**
- Tất cả neighbors của (1,1) và (1,2) đều có quân (đen hoặc trắng)
- Không có vị trí trống nào → `liberties.empty() == true`

---

### **BƯỚC 4: Kiểm Tra Capture** (`apply_move()` - Line 249)

```cpp
if (opponent_group.liberties.empty()) {
    // Nhóm không còn khí → BẮT
    for (const int stone_index_value : opponent_group.stones) {
        captured_indices_set.insert(stone_index_value);
    }
}
```

**Ví dụ:**
```cpp
opponent_group.liberties.empty() == true  // ✅ Không còn khí
opponent_group.stones = [to_index(1,1), to_index(1,2)]

captured_indices_set = {to_index(1,1), to_index(1,2)}
```

---

### **BƯỚC 5: Xóa Quân Bị Bắt** (`remove_stone()` - Line 275-288)

```cpp
for (const int captured_index : captured_indices_set) {
    remove_stone(captured_index, undo);
}
```

**`remove_stone()` làm gì:**

```cpp
void Board::remove_stone(int index, UndoInfo &undo) {
    const Stone stone = grid_[index];  // Lưu lại để undo
    if (stone == Stone::Empty) {
        return;  // Không có gì để xóa
    }
    
    // 1. Lưu vào undo info (để có thể undo sau này)
    undo.captured.push_back({index, stone});
    
    // 2. Cập nhật prisoners (tù binh)
    const Color color = color_from_stone(stone);
    prisoners_[color_index(opposite_color(color))] += 1;
    // Ví dụ: Nếu stone là White → prisoners_[Black] += 1
    
    // 3. Cập nhật hash
    hash_ ^= zobrist_table_[index][stone_index(stone)];
    
    // 4. XÓA QUÂN KHỎI BÀN CỜ
    grid_[index] = Stone::Empty;
}
```

**Ví dụ với (1,1):**
```cpp
remove_stone(to_index(1,1), undo):
  1. undo.captured.push_back({to_index(1,1), Stone::White})
  2. prisoners_[Black] += 1  // Đen bắt được 1 quân trắng
  3. hash_ ^= zobrist_table_[to_index(1,1)][Stone::White]
  4. grid_[to_index(1,1)] = Stone::Empty  // ✅ XÓA QUÂN
```

**Ví dụ với (1,2):**
```cpp
remove_stone(to_index(1,2), undo):
  1. undo.captured.push_back({to_index(1,2), Stone::White})
  2. prisoners_[Black] += 1  // Đen bắt được thêm 1 quân trắng
  3. hash_ ^= zobrist_table_[to_index(1,2)][Stone::White]
  4. grid_[to_index(1,2)] = Stone::Empty  // ✅ XÓA QUÂN
```

**Kết quả:**
- `prisoners_[Black] = 2` (đen bắt được 2 quân trắng)
- `grid_[to_index(1,1)] = Stone::Empty`
- `grid_[to_index(1,2)] = Stone::Empty`
- Bàn cờ không còn quân trắng tại (1,1) và (1,2)

---

### **BƯỚC 6: Kiểm Tra Suicide** (`apply_move()` - Line 262-268)

```cpp
const GroupInfo own_group = collect_group(index);
if (own_group.liberties.empty()) {
    // Suicide → Revert và throw error
    undo_move(undo);
    throw std::runtime_error("Suicide move applied unexpectedly");
}
```

**Tại sao cần kiểm tra?**
- Sau khi bắt quân đối phương, nhóm của mình có thể có thêm khí
- Nhưng nếu vẫn không còn khí → Đây là bug (đã được check ở `is_legal_move()`)

**Ví dụ:**
```cpp
// Sau khi bắt quân trắng, nhóm đen tại (1,0) có khí:
own_group = collect_group(to_index(1,0))
// Neighbors: (0,0), (2,0), (1,1)[Empty], (1,-1)[out of bounds]
// → Có ít nhất 1 khí tại (1,1) → Không phải suicide ✅
```

---

### **BƯỚC 7: Cập Nhật Ko Index** (`apply_move()` - Line 270-272)

```cpp
if (captured_indices_set.size() == 1 && own_group.stones.size() == 1) {
    ko_index_ = *captured_indices_set.begin();
}
```

**Ko Rule:**
- Nếu chỉ bắt 1 quân và nhóm mình chỉ có 1 quân → Ko
- Ghi nhớ vị trí quân bị bắt để tránh lặp lại

**Ví dụ:**
```cpp
// Nếu bắt 1 quân tại (1,1) và nhóm đen chỉ có 1 quân tại (1,0)
ko_index_ = to_index(1,1)  // Trắng không được đặt lại tại (1,1) ngay
```

---

## 🎯 VÍ DỤ HOÀN CHỈNH

### Scenario: Đen bắt 2 quân trắng

**Trạng thái trước:**
```
  y=0  y=1  y=2  y=3
x=0  .    B    B    .
x=1  .    W    W    .  ← Nhóm trắng: (1,1), (1,2)
x=2  .    B    B    .
x=3  .    .    .    .
```

**Đen đặt quân tại (1,0):**

#### Step 1: Đặt quân
```cpp
grid_[to_index(1,0)] = Stone::Black
```

#### Step 2: Kiểm tra neighbors
```cpp
neighbors(to_index(1,0)) = [(0,0), (2,0), (1,1)]
// (1,1) là quân trắng → Kiểm tra nhóm
```

#### Step 3: Thu thập nhóm trắng
```cpp
opponent_group = collect_group(to_index(1,1))
// BFS:
//   - (1,1) → neighbors: (0,1)[B], (2,1)[B], (1,0)[B], (1,2)[W]
//   - (1,2) → neighbors: (0,2)[B], (2,2)[B], (1,1)[W], (1,3)[B]
// Kết quả:
//   - stones = [(1,1), (1,2)]
//   - liberties = []  ← KHÔNG CÒN KHÍ!
```

#### Step 4: Capture
```cpp
captured_indices_set = {to_index(1,1), to_index(1,2)}
```

#### Step 5: Xóa quân
```cpp
remove_stone(to_index(1,1), undo):
  - prisoners_[Black] += 1
  - grid_[to_index(1,1)] = Stone::Empty

remove_stone(to_index(1,2), undo):
  - prisoners_[Black] += 1
  - grid_[to_index(1,2)] = Stone::Empty
```

**Trạng thái sau:**
```
  y=0  y=1  y=2  y=3
x=0  .    B    B    .
x=1  B    .    .    B  ← 2 quân trắng đã bị xóa
x=2  .    B    B    .
x=3  .    .    .    .

Prisoners: Black = 2
```

---

## 📊 TÓM TẮT FLOW

```
Đặt quân
  ↓
Kiểm tra neighbors
  ↓
Tìm quân đối phương?
  ↓ YES
collect_group() → Thu thập nhóm
  ↓
Tính liberties
  ↓
liberties.empty()?
  ↓ YES
Thêm vào captured_indices_set
  ↓
remove_stone() cho mỗi quân bị bắt
  ↓
  - Lưu vào undo.captured
  - Cập nhật prisoners
  - Cập nhật hash
  - Xóa khỏi grid (grid_[index] = Empty)
  ↓
Kiểm tra suicide (own_group)
  ↓
Cập nhật ko_index (nếu cần)
  ↓
HOÀN TẤT
```

---

## 🔑 ĐIỂM QUAN TRỌNG

1. **Quân được đặt TRƯỚC, sau đó mới kiểm tra capture**
   - Quân mới đặt chiếm một liberty của nhóm đối phương
   - Nếu nhóm không còn liberties nào khác → Bị bắt

2. **`collect_group()` được gọi SAU KHI đặt quân**
   - Nó sẽ không đếm vị trí quân mới đặt là liberty
   - Chỉ đếm các vị trí trống (Stone::Empty)

3. **Quân bị bắt được XÓA NGAY LẬP TỨC**
   - `grid_[index] = Stone::Empty`
   - Không còn trên bàn cờ nữa

4. **Prisoners được cập nhật tự động**
   - Mỗi quân bị bắt → prisoners tăng 1
   - Dùng để tính điểm cuối ván

5. **Undo info được lưu lại**
   - Có thể undo move và khôi phục quân bị bắt
   - Quan trọng cho AI search và replay

---

## ✅ KẾT LUẬN

Logic bắt quân hoạt động chính xác:
- ✅ Đặt quân trước
- ✅ Kiểm tra capture sau
- ✅ Xóa quân bị bắt ngay lập tức
- ✅ Cập nhật prisoners
- ✅ Lưu undo info

**Không có vấn đề gì với logic này!**

