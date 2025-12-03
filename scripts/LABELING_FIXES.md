# 🔧 CÁC SỬA ĐỔI GÁN NHÃN - LABELING FIXES

## 📋 Tổng Quan

Các sửa đổi này đảm bảo quá trình gán nhãn (labeling) **chặt chẽ, đúng quỹ đạo tài liệu** và xử lý đầy đủ các trường hợp đặc biệt.

## ✅ Các Vấn Đề Đã Sửa

### 1. **Pass Moves Không Được Xử Lý** ❌ → ✅

**Vấn đề cũ:**
- Pass moves (`None` hoặc `(-1, -1)`) bị crash khi unpack
- Pass moves bị bỏ qua hoàn toàn trong parsing
- Policy label không có index cho pass move

**Đã sửa:**
- ✅ `generate_policy_label()` bây giờ xử lý `None`, `(-1, -1)` cho pass moves
- ✅ Policy vector có thêm 1 index cuối cùng cho pass: `[board_size * board_size + 1]`
- ✅ Parsing lưu pass moves với `move = None` thay vì bỏ qua

**Code:**
```python
# Pass move → index cuối cùng
if move is None or move == (-1, -1):
    policy[-1] = 1.0  # Pass move index
```

### 2. **Value Label Thiếu Validation** ❌ → ✅

**Vấn đề cũ:**
- Không validate `current_player` format
- Không xử lý trường hợp `winner` không hợp lệ
- Có thể trả về giá trị sai nếu `current_player` không khớp

**Đã sửa:**
- ✅ Validate `current_player` phải là 'B', 'W', 'b', hoặc 'w'
- ✅ Normalize `current_player` và `winner` về uppercase
- ✅ Parse `winner` từ `game_result` nếu `winner` không hợp lệ
- ✅ Trả về 0.5 cho các trường hợp không xác định

**Code:**
```python
# Validate current_player
if current_player not in ('B', 'W', 'b', 'w'):
    raise ValueError(f"Invalid current_player: '{current_player}'")

# Normalize
current_player = current_player.upper()
```

### 3. **Thiếu Validation Trong Processing** ❌ → ✅

**Vấn đề cũ:**
- Không validate move coordinates trước khi tạo policy label
- Không kiểm tra value label có trong range [0.0, 1.0]
- Lỗi không rõ ràng khi có vấn đề

**Đã sửa:**
- ✅ Validate move coordinates trước khi xử lý
- ✅ Validate value label trong range [0.0, 1.0]
- ✅ Error messages chi tiết với error type
- ✅ Return error info thay vì crash

**Code:**
```python
# Validate move
if isinstance(move, (tuple, list)) and len(move) == 2:
    mx, my = move
    if not (0 <= mx < board_size and 0 <= my < board_size):
        return None, {'error': '...', 'type': 'invalid_move'}

# Validate value
if not (0.0 <= value <= 1.0):
    return None, {'error': '...', 'type': 'invalid_value'}
```

### 4. **Parsing Bỏ Qua Pass Moves** ❌ → ✅

**Vấn đề cũ:**
- `parse_sgf_local.py` và `parse_sgf_colab.py` bỏ qua pass moves
- Chỉ lưu positions có move hợp lệ

**Đã sửa:**
- ✅ Lưu pass moves với `move = None`
- ✅ Tăng `move_count` cho pass moves
- ✅ Không apply move cho pass (đúng logic)

**Code:**
```python
if x is not None and y is not None:
    # Normal move
    positions.append({..., 'move': (x, y)})
else:
    # Pass move
    positions.append({..., 'move': None})
    move_count += 1
```

## 📊 Format Dữ Liệu Sau Khi Sửa

### Policy Label Format

**Trước:**
```python
policy: Tensor[board_size * board_size]  # Không có pass move
```

**Sau:**
```python
policy: Tensor[board_size * board_size + 1]  # +1 cho pass move
# Index 0 đến (board_size * board_size - 1): board positions
# Index (board_size * board_size): pass move
```

### Position Format

**Trước:**
```python
{
    'move': (x, y)  # Chỉ có normal moves
}
```

**Sau:**
```python
{
    'move': (x, y) | None  # Normal move hoặc None cho pass
}
```

### Value Label Format

**Không thay đổi nhưng có validation chặt chẽ:**
```python
value: float  # 0.0 (lose), 0.5 (draw/unknown), 1.0 (win)
# Đảm bảo: 0.0 <= value <= 1.0
```

## 🔍 Validation Rules

### 1. Move Validation
- ✅ `move` phải là `None`, `(-1, -1)`, hoặc `(x, y)` tuple
- ✅ Nếu tuple, `x` và `y` phải là integers
- ✅ Nếu tuple, `0 <= x < board_size` và `0 <= y < board_size`
- ✅ Nếu không hợp lệ → treat as pass move (với warning)

### 2. Current Player Validation
- ✅ Phải là 'B', 'W', 'b', hoặc 'w'
- ✅ Tự động normalize về uppercase
- ✅ Raise `ValueError` nếu không hợp lệ

### 3. Value Label Validation
- ✅ Phải trong range [0.0, 1.0]
- ✅ Validate `winner` format
- ✅ Parse từ `game_result` nếu `winner` không hợp lệ
- ✅ Return 0.5 cho unknown/draw

### 4. Board State Validation
- ✅ Board shape phải khớp với `board_size`
- ✅ Board values phải là 0, 1, hoặc 2

## 🧪 Testing

### Test Cases

1. **Pass Move Policy Label**
   ```python
   policy = generate_policy_label(None, 19)
   assert policy.shape == (19 * 19 + 1,)
   assert policy[-1] == 1.0  # Pass move index
   ```

2. **Normal Move Policy Label**
   ```python
   policy = generate_policy_label((5, 5), 19)
   assert policy[5 * 19 + 5] == 1.0
   assert policy[-1] == 0.0  # Not pass
   ```

3. **Value Label Validation**
   ```python
   # Should raise ValueError
   try:
       value = generate_value_label('B', 'X')
   except ValueError:
       pass
   ```

4. **Invalid Move Handling**
   ```python
   # Move outside board → treated as pass
   policy = generate_policy_label((20, 20), 19)
   assert policy[-1] == 1.0  # Treated as pass
   ```

## 📝 Files Đã Sửa

1. ✅ `scripts/generate_features_colab.py`
   - Sửa `generate_policy_label()` để xử lý pass moves
   - Sửa `generate_value_label()` để validate chặt chẽ

2. ✅ `scripts/generate_labels_colab.py`
   - Thêm validation trong `process_single_position()`
   - Validate move, current_player, và value label

3. ✅ `scripts/generate_labels_local.py`
   - Áp dụng các validation tương tự

4. ✅ `scripts/parse_sgf_local.py`
   - Lưu pass moves với `move = None`

5. ✅ `scripts/parse_sgf_colab.py`
   - Lưu pass moves với `move = None`

## 🎯 Kết Quả

- ✅ **Pass moves được xử lý đúng**: Có index riêng trong policy vector
- ✅ **Validation chặt chẽ**: Tất cả inputs được validate trước khi xử lý
- ✅ **Error handling tốt hơn**: Error messages rõ ràng với error types
- ✅ **Đúng quỹ đạo tài liệu**: Format phù hợp với ML_COMPREHENSIVE_GUIDE.md
- ✅ **Backward compatible**: Vẫn hỗ trợ format cũ (normal moves)

## ⚠️ Breaking Changes

**LƯU Ý**: Policy vector bây giờ có thêm 1 dimension:
- **Trước**: `[board_size * board_size]`
- **Sau**: `[board_size * board_size + 1]`

Nếu bạn đã train models với format cũ, cần:
1. Retrain models với format mới, HOẶC
2. Map policy vector cũ sang format mới (thêm 0 cho pass index)

## 📚 Tài Liệu Liên Quan

- `docs/ML_COMPREHENSIVE_GUIDE.md` - Tài liệu chính về ML training
- `docs/COLAB_LABELING_GUIDE.md` - Hướng dẫn labeling trên Colab
- `scripts/generate_features_colab.py` - Code generate labels

---

**Last updated**: 2025-01-27
**Status**: ✅ Completed

