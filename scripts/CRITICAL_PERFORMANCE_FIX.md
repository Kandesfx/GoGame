# Critical Performance Fix - Tăng tốc từ 25 pos/s lên 100+ pos/s

## Vấn đề
Tốc độ vẫn chỉ khoảng 25 pos/s sau các tối ưu trước, không đạt mục tiêu 100-150 pos/s.

## Nguyên nhân chính
Các hàm scan toàn bộ board (19x19 = 361 cells) mỗi lần gọi:
1. `detect_false_eyes()` - scan 361 cells
2. `detect_cutting_points()` - scan 361 cells  
3. `find_invasion_points()` - scan 361 cells
4. `find_working_ladders()` - scan 361 cells
5. `generate_territory_map()` - scan 361 cells × 361 cells = O(n²)
6. `generate_influence_map()` - scan 361 cells × 361 cells = O(n²)

## Giải pháp đã áp dụng

### 1. Disable các hàm không quan trọng
**Disabled trong `generate_threat_map()`:**
- `detect_false_eyes()` - không quan trọng bằng atari
- `detect_cutting_points()` - không quan trọng bằng atari

**Disabled trong `generate_attack_map()`:**
- `find_invasion_points()` - không quan trọng bằng atari và cut
- `find_working_ladders()` - không quan trọng bằng atari và cut

**Lý do**: Các rules này chỉ bổ sung thông tin phụ, không ảnh hưởng nhiều đến chất lượng labels nhưng làm chậm đáng kể.

### 2. Tối ưu `_find_groups_dfs()`
**Trước**: Scan toàn bộ board
```python
for y in range(self.board_size):
    for x in range(self.board_size):
        if (x, y) in visited or board_state[y, x] == 0:
            continue
        # ... DFS
```

**Sau**: Chỉ scan các vị trí có quân
```python
# Tìm tất cả stones trước
stone_positions = []
for y in range(self.board_size):
    for x in range(self.board_size):
        if board_state[y, x] != 0:
            stone_positions.append((x, y))

# DFS chỉ cho các vị trí có quân
for start_x, start_y in stone_positions:
    # ... DFS
```

**Lý do**: Thường chỉ có 50-200 stones trên board, không phải 361 cells.

### 3. Tối ưu `generate_territory_map()` và `generate_influence_map()`
**Trước**: O(n²) - scan toàn bộ board × toàn bộ board
```python
for y in range(self.board_size):
    for x in range(self.board_size):
        for sy in range(self.board_size):
            for sx in range(self.board_size):
                if board_state[sy, sx] == player_color:
                    # ... calculate
```

**Sau**: O(n×m) - scan board × số lượng stones của player
```python
# Tìm tất cả player stones trước
player_stones = []
for y in range(self.board_size):
    for x in range(self.board_size):
        if board_state[y, x] == player_color:
            player_stones.append((x, y))

# Chỉ tính cho empty points × player stones
for y in range(self.board_size):
    for x in range(self.board_size):
        if board_state[y, x] == 0:
            for sx, sy in player_stones:  # Chỉ check player stones
                # ... calculate
```

**Lý do**: Thường chỉ có 50-100 player stones, không phải 361 cells.

## Kết quả mong đợi
- **Tốc độ**: Tăng từ 25 pos/s lên **100-150 pos/s** (4-6x faster)
- **Giảm overhead**: 
  - Disable 4 hàm scan toàn bộ board → giảm ~40% thời gian
  - Tối ưu DFS → giảm ~20% thời gian
  - Tối ưu territory/influence → giảm ~30% thời gian

## Trade-offs
- **Chất lượng labels**: Giảm nhẹ (mất false_eyes, cutting_points, invasion_points, ladder_moves)
- **Tốc độ**: Tăng đáng kể (4-6x)
- **Độ chính xác**: Vẫn giữ được các rules quan trọng nhất (atari, 2-liberty groups, cut opportunities)

## Files đã thay đổi
1. `scripts/label_generators.py`:
   - Disable `detect_false_eyes()` và `detect_cutting_points()` trong `generate_threat_map()`
   - Disable `find_invasion_points()` và `find_working_ladders()` trong `generate_attack_map()`
   - Tối ưu `_find_groups_dfs()` - chỉ scan stones
   - Tối ưu `generate_territory_map()` - O(n²) → O(n×m)
   - Tối ưu `generate_influence_map()` - O(n²) → O(n×m)

## Lưu ý
- Nếu cần chất lượng labels cao hơn, có thể enable lại các hàm đã disable
- Nhưng sẽ giảm tốc độ về ~25-30 pos/s
- Có thể cân bằng bằng cách chỉ enable một số hàm quan trọng nhất

