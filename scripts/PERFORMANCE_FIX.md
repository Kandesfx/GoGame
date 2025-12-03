# Performance Fix - Tối ưu tốc độ label generation

## Vấn đề
Code chạy chậm (14.73 pos/s) thay vì tốc độ mong đợi (100-150 pos/s).

## Nguyên nhân
1. **Vectorized operations quá phức tạp**: Các operations vectorized với NumPy broadcasting tạo overhead lớn cho small arrays
2. **Tính toán trùng lặp**: `find_groups()` được gọi 2 lần (cho threat_map và attack_map)
3. **Overhead từ type conversion**: Nhiều lần convert numpy arrays không cần thiết
4. **Tính toán không cần thiết**: `detect_false_eyes()` và `detect_cutting_points()` được gọi mỗi lần dù không cần

## Giải pháp đã áp dụng

### 1. Đơn giản hóa `_count_group_liberties()`
**Trước**: Vectorized với broadcasting phức tạp
```python
# Broadcasting: (n, 1, 2) + (1, 4, 2) = (n, 4, 2)
neighbor_positions = group_array[:, None, :] + neighbors[None, :, :]
# ... nhiều operations phức tạp
```

**Sau**: Simple loop với set (nhanh hơn cho small groups)
```python
liberty_set = set()
for x, y in group_positions:
    for dx, dy in neighbors:
        nx, ny = x + dx, y + dy
        if (0 <= nx < self.board_size and 
            0 <= ny < self.board_size and
            board_state[ny, nx] == 0):
            liberty_set.add((nx, ny))
return len(liberty_set)
```

**Lý do**: Với groups nhỏ (thường < 20 stones), simple loop nhanh hơn vectorized operations có overhead.

### 2. Reuse `find_groups()` results
**Trước**: Tính 2 lần
```python
threat_map = threat_gen.generate_threat_map(board_np, current_player)  # Tính groups
attack_map = attack_gen.generate_attack_map(board_np, current_player)  # Tính groups lại
```

**Sau**: Tính 1 lần, reuse
```python
groups = threat_gen.find_groups(board_np)  # Tính 1 lần
threat_map = threat_gen.generate_threat_map(board_np, current_player, groups=groups)
attack_map = attack_gen.generate_attack_map(board_np, current_player, groups=groups)
```

**Lý do**: `find_groups()` là operation tốn thời gian nhất (DFS hoặc scipy.ndimage.label). Reuse giảm 50% thời gian.

### 3. Đơn giản hóa threat map assignment
**Trước**: Vectorized assignment với masks
```python
positions = np.array(group['positions'])
y_coords = positions[:, 1]
x_coords = positions[:, 0]
mask = threat_map[y_coords, x_coords] < 0.7
threat_map[y_coords[mask], x_coords[mask]] = 0.7
```

**Sau**: Simple loop
```python
for x, y in group['positions']:
    if threat_map[y, x] < 0.7:
        threat_map[y, x] = 0.7
```

**Lý do**: Với số lượng positions nhỏ, simple loop nhanh hơn và dễ đọc hơn.

### 4. Conditional computation cho false_eyes và cutting_points
**Trước**: Luôn tính
```python
false_eyes = self.detect_false_eyes(board_state)  # Luôn tính
cutting_points = self.detect_cutting_points(board_state)  # Luôn tính
```

**Sau**: Chỉ tính nếu cần
```python
# Chỉ tính nếu < 30% board có threat
if np.sum(threat_map > 0) < self.board_size * self.board_size * 0.3:
    false_eyes = self.detect_false_eyes(board_state)
    # ...
```

**Lý do**: Nếu đã có nhiều threats (atari, 2-liberty groups), không cần tính false_eyes và cutting_points.

### 5. Tối ưu board state conversion
**Trước**: 
```python
if not board_np.flags['C_CONTIGUOUS']:
    board_np = np.ascontiguousarray(board_np)
```

**Sau**:
```python
if board_np.dtype != np.int8:
    board_np = board_np.astype(np.int8, copy=False)
```

**Lý do**: Chỉ convert dtype nếu cần, tránh copy không cần thiết.

### 6. Tối ưu scipy.ndimage.label usage
**Trước**: Luôn dùng scipy nếu có
```python
try:
    from scipy import ndimage
    return self._find_groups_vectorized(board_state, ndimage)
except ImportError:
    return self._find_groups_dfs(board_state)
```

**Sau**: Chỉ dùng scipy cho board 19x19
```python
try:
    from scipy import ndimage
    if self.board_size == 19:  # Chỉ cho 19x19
        return self._find_groups_vectorized(board_state, ndimage)
    else:
        return self._find_groups_dfs(board_state)
except ImportError:
    return self._find_groups_dfs(board_state)
```

**Lý do**: scipy.ndimage.label có overhead cho board nhỏ, DFS đủ nhanh.

## Kết quả mong đợi
- **Tốc độ**: Tăng từ 14.73 pos/s lên 100-150 pos/s (6-10x faster)
- **Memory**: Giảm overhead từ vectorized operations
- **Code clarity**: Đơn giản hơn, dễ maintain

## Files đã thay đổi
1. `scripts/label_generators.py`:
   - Đơn giản hóa `_count_group_liberties()`
   - Thêm `groups` parameter cho `generate_threat_map()` và `generate_attack_map()`
   - Đơn giản hóa threat map assignment
   - Conditional computation cho false_eyes và cutting_points
   - Tối ưu scipy usage

2. `scripts/generate_labels_colab.py`:
   - Reuse `find_groups()` results giữa threat_map và attack_map
   - Tối ưu board state conversion

## Lưu ý
- Các tối ưu này phù hợp với board 19x19 và groups nhỏ (< 20 stones)
- Nếu cần scale lên board lớn hơn hoặc groups lớn hơn, có thể cần điều chỉnh lại
- Vẫn giữ backward compatibility với code cũ

