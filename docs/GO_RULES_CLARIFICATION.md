# CLARIFICATION: GO GAME RULES

## 1. Board Representation - Giao Điểm (Intersections)

### ✅ ĐÚNG: Quân cờ được đặt tại các GIAO ĐIỂM, không phải trong các ô vuông

**Trong cờ vây:**
- Bàn cờ có các đường kẻ ngang và dọc tạo thành lưới
- Quân cờ được đặt tại **giao điểm** của các đường kẻ này
- Không đặt quân trong các ô vuông

**Trong code:**
- Backend (C++): `Point {x, y}` đại diện cho giao điểm tại tọa độ (x, y)
- Frontend: Mỗi "cell" trong grid thực chất đại diện cho một giao điểm
  - Quân cờ được render ở center của cell (tức là giao điểm)
  - Grid lines được vẽ để hiển thị đúng các đường kẻ

**Ví dụ:**
- Bàn cờ 9x9 có 9×9 = 81 giao điểm
- Giao điểm (0,0) là góc trên bên trái
- Giao điểm (4,4) là trung tâm bàn cờ

## 2. Capture Rule - Luật Ăn Quân

### ✅ ĐÚNG: Quân bị ăn khi nhóm quân không còn "khí" (liberties)

**Luật ăn quân:**
1. Các quân cùng màu liên kết với nhau (theo chiều ngang/dọc) tạo thành một **nhóm** (group)
2. Mỗi nhóm có các **khí** (liberties) - các giao điểm trống liền kề
3. Khi một nhóm **không còn khí nào** (liberties = 0), toàn bộ nhóm bị bắt và loại khỏi bàn cờ
4. Số quân bị bắt được đếm vào **prisoners** của đối phương

**Trong code C++ (`board.cpp`):**
```cpp
// Kiểm tra các nhóm đối phương xung quanh nước đi mới
for (const int neighbor : neighbors(index)) {
    if (grid_[neighbor] == opponent_stone) {
        const GroupInfo opponent_group = collect_group(neighbor);
        if (opponent_group.liberties.empty()) {  // ← Không còn khí
            // Capture toàn bộ nhóm
            for (const int stone_index_value : opponent_group.stones) {
                captured_indices.push_back(stone_index_value);
            }
        }
    }
}
```

**Ví dụ minh họa:**
```
Trước khi ăn (bàn cờ 4x4):
  y=0  y=1  y=2  y=3
x=0  .    .    .    .
x=1  .    W    W    .
x=2  .    B    .    .
x=3  .    .    .    .

Nhóm trắng (W) tại (1,1) và (1,2) có 3 khí: (0,1), (2,1), (2,2)

Sau khi đen đặt quân tại (2,1):
  y=0  y=1  y=2  y=3
x=0  .    .    .    .
x=1  .    W    W    .  ← Nhóm trắng chỉ còn 1 khí: (2,2)
x=2  .    B    B    .  ← Đen đặt quân tại (2,1)
x=3  .    .    .    .

Sau khi đen đặt quân tại (2,2):
  y=0  y=1  y=2  y=3
x=0  .    .    .    .
x=1  .    .    .    .  ← Nhóm trắng không còn khí → BỊ BẮT
x=2  .    B    B    B  ← Đen đặt quân tại (2,2)
x=3  .    .    .    .

Kết quả cuối cùng:
  y=0  y=1  y=2  y=3
x=0  .    .    .    .
x=1  .    .    .    .
x=2  .    B    B    B
x=3  .    .    .    .
  Prisoners: Black = 2 (đã bắt 2 quân trắng)
```

## 3. Suicide Rule - Luật Tự Sát

### ✅ ĐÚNG: Không được đặt quân vào vị trí khiến nhóm của mình không còn khí

**Luật tự sát:**
- Nếu đặt quân vào vị trí khiến nhóm của mình không còn khí, nước đi đó **không hợp lệ**
- Trừ khi nước đi đó ăn được quân đối phương (khi đó nhóm đối phương bị bắt trước)

**Trong code:**
```cpp
// Kiểm tra sau khi đặt quân
const GroupInfo own_group = collect_group(index);
if (own_group.liberties.empty()) {
    // Suicide - revert và throw error
    undo_move(undo);
    throw std::runtime_error("Suicide move applied unexpectedly");
}
```

## 4. Ko Rule - Luật Ko

### ✅ ĐÚNG: Không được lặp lại trạng thái bàn cờ ngay lập tức

**Luật Ko:**
- Sau khi ăn 1 quân, không được đặt quân lại đúng vị trí vừa bị ăn ngay lập tức
- Điều này ngăn vòng lặp vô tận

**Trong code:**
```cpp
if (captured_indices.size() == 1 && own_group.stones.size() == 1) {
    ko_index_ = captured_indices.front();  // Ghi nhớ vị trí ko
}
```

## 5. Scoring - Tính Điểm

**Luật Trung Quốc (Area Scoring):**
- Điểm = Territory (đất) + Prisoners (quân bắt được)
- Trắng được cộng thêm Komi (6.5 cho 9x9, 7.5 cho 13x13/19x19)

**Trong code:**
```python
black_score = territory_black + prisoners_black
white_score = territory_white + prisoners_white + komi
```

## TÓM TẮT

✅ **Code đã được kiểm tra và đúng về:**
- ✅ Board representation: Quân cờ được đặt tại giao điểm (intersections), không phải ô vuông
- ✅ Capture logic: Ăn quân khi nhóm không còn khí (liberties = 0)
- ✅ Suicide prevention: Ngăn đặt quân vào vị trí tự sát
- ✅ Ko rule: Ngăn lặp lại trạng thái bàn cờ ngay lập tức
- ✅ Scoring: Tính điểm theo luật Trung Quốc (territory + prisoners + komi)

✅ **Đã cải thiện:**
- ✅ Thêm comments trong code C++ về giao điểm và capture logic
- ✅ Thêm comments trong frontend về intersections
- ✅ Tạo documentation chi tiết về các luật chơi
- ✅ Cải thiện ví dụ minh họa về capture rule

## LƯU Ý QUAN TRỌNG

**Tọa độ trong code:**
- Sử dụng hệ tọa độ 0-indexed: (0,0) đến (size-1, size-1)
- (0,0) là góc trên bên trái
- x tăng sang phải, y tăng xuống dưới
- Mỗi (x, y) đại diện cho một giao điểm trên bàn cờ

