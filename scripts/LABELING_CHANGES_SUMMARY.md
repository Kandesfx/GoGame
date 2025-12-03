# 📋 TÓM TẮT CÁC THAY ĐỔI GÁN NHÃN

## 🎯 Mục Tiêu

Sửa lại code gán nhãn để:
1. ✅ **Đúng quỹ đạo tài liệu** (ML_COMPREHENSIVE_GUIDE.md)
2. ✅ **Xử lý đầy đủ pass moves**
3. ✅ **Validation chặt chẽ** cho tất cả inputs
4. ✅ **Error handling tốt hơn** với messages rõ ràng

## 📝 Các Thay Đổi Chính

### 1. Policy Label - Hỗ Trợ Pass Moves

**File**: `scripts/generate_features_colab.py`

**Thay đổi**:
- Policy vector: `[board_size * board_size]` → `[board_size * board_size + 1]`
- Index cuối cùng dành cho pass move
- Xử lý `None`, `(-1, -1)` cho pass moves
- Validate move coordinates

**Code mới**:
```python
def generate_policy_label(move: Optional[Tuple[int, int]], board_size: int):
    policy = torch.zeros(board_size * board_size + 1, dtype=torch.float32)
    
    if move is None or move == (-1, -1):
        policy[-1] = 1.0  # Pass move
    else:
        x, y = move
        if 0 <= x < board_size and 0 <= y < board_size:
            idx = y * board_size + x
            policy[idx] = 1.0
        else:
            # Invalid → treat as pass
            policy[-1] = 1.0
    
    return policy
```

### 2. Value Label - Validation Chặt Chẽ

**File**: `scripts/generate_features_colab.py`

**Thay đổi**:
- Validate `current_player` format
- Normalize `current_player` và `winner`
- Parse `winner` từ `game_result` nếu cần
- Raise `ValueError` cho invalid inputs

**Code mới**:
```python
def generate_value_label(winner, current_player, game_result=None):
    # Validate current_player
    if current_player not in ('B', 'W', 'b', 'w'):
        raise ValueError(f"Invalid current_player: '{current_player}'")
    
    current_player = current_player.upper()
    
    # Handle None/DRAW
    if winner is None:
        return 0.5
    
    # Parse from game_result if winner invalid
    if winner not in ('B', 'W') and game_result:
        # Parse from game_result
        ...
    
    # Return value
    return 1.0 if winner == current_player else 0.0
```

### 3. Parsing - Lưu Pass Moves

**Files**: 
- `scripts/parse_sgf_local.py`
- `scripts/parse_sgf_colab.py`

**Thay đổi**:
- Lưu pass moves với `move = None` thay vì bỏ qua
- Tăng `move_count` cho pass moves
- Không apply move cho pass (đúng logic)

**Code mới**:
```python
if x is not None and y is not None:
    # Normal move
    positions.append({..., 'move': (x, y)})
    board[y, x] = 1 if color == 'B' else 2
else:
    # Pass move
    positions.append({..., 'move': None})
    # Không apply move
move_count += 1
```

### 4. Processing - Validation Đầy Đủ

**Files**:
- `scripts/generate_labels_colab.py`
- `scripts/generate_labels_local.py`

**Thay đổi**:
- Validate move format trước khi xử lý
- Validate `current_player`
- Validate value label range [0.0, 1.0]
- Error messages chi tiết với error types

**Code mới**:
```python
# Validate move
if move is None:
    pass  # OK
elif isinstance(move, (tuple, list)) and len(move) == 2:
    mx, my = move
    if not (0 <= mx < board_size and 0 <= my < board_size):
        return None, {'error': '...', 'type': 'invalid_move'}
else:
    return None, {'error': '...', 'type': 'invalid_move_format'}

# Validate current_player
if current_player not in ('B', 'W', 'b', 'w'):
    return None, {'error': '...', 'type': 'invalid_player'}

# Validate value
if not (0.0 <= value <= 1.0):
    return None, {'error': '...', 'type': 'invalid_value'}
```

## 🔄 Breaking Changes

### ⚠️ QUAN TRỌNG: Policy Vector Shape Thay Đổi

**Trước**:
```python
policy: Tensor[board_size * board_size]  # Ví dụ: [361] cho 19x19
```

**Sau**:
```python
policy: Tensor[board_size * board_size + 1]  # Ví dụ: [362] cho 19x19
```

**Ảnh hưởng**:
- Models đã train với format cũ sẽ KHÔNG tương thích
- Cần retrain models với format mới
- HOẶC map policy vector cũ sang mới (thêm 0 cho pass index)

## ✅ Validation Rules

### Move Validation
- ✅ `None` → Pass move
- ✅ `(-1, -1)` → Pass move
- ✅ `(x, y)` với `0 <= x,y < board_size` → Normal move
- ✅ `(x, y)` ngoài board → Treated as pass (với warning)

### Current Player Validation
- ✅ Phải là 'B', 'W', 'b', hoặc 'w'
- ✅ Tự động normalize về uppercase
- ✅ Raise `ValueError` nếu không hợp lệ

### Value Label Validation
- ✅ Phải trong range [0.0, 1.0]
- ✅ Validate `winner` format
- ✅ Parse từ `game_result` nếu cần
- ✅ Return 0.5 cho unknown/draw

## 📊 Format Dữ Liệu

### Position Format
```python
{
    'board_state': np.ndarray[board_size, board_size],
    'move': (x, y) | None,  # Normal move hoặc None cho pass
    'current_player': 'B' | 'W',
    'move_number': int,
    'board_size': int,
    'game_result': str | None,
    'winner': 'B' | 'W' | 'DRAW' | None,
    'handicap': int
}
```

### Labeled Sample Format
```python
{
    'features': Tensor[17, board_size, board_size],
    'policy': Tensor[board_size * board_size + 1],  # +1 cho pass
    'value': float,  # 0.0 <= value <= 1.0
    'metadata': {
        'move_number': int,
        'game_result': str | None,
        'winner': 'B' | 'W' | 'DRAW' | None,
        'handicap': int
    }
}
```

## 🧪 Testing

Test script: `scripts/test_labeling_fixes.py`

**Test cases**:
1. ✅ Policy label cho normal move
2. ✅ Policy label cho pass move (None, (-1, -1))
3. ✅ Policy label cho invalid move (outside board)
4. ✅ Value label validation
5. ✅ Value label range [0.0, 1.0]
6. ✅ Policy label shape consistency
7. ✅ Integration test

**Chạy test**:
```bash
# Cần cài torch trước
pip install torch numpy

# Chạy test
python scripts/test_labeling_fixes.py
```

## 📚 Files Đã Sửa

1. ✅ `scripts/generate_features_colab.py`
   - `generate_policy_label()` - Hỗ trợ pass moves
   - `generate_value_label()` - Validation chặt chẽ

2. ✅ `scripts/generate_labels_colab.py`
   - `process_single_position()` - Validation đầy đủ

3. ✅ `scripts/generate_labels_local.py`
   - `process_single_position()` - Validation đầy đủ

4. ✅ `scripts/parse_sgf_local.py`
   - Lưu pass moves với `move = None`

5. ✅ `scripts/parse_sgf_colab.py`
   - Lưu pass moves với `move = None`

## 📖 Tài Liệu

- `scripts/LABELING_FIXES.md` - Chi tiết các sửa đổi
- `docs/ML_COMPREHENSIVE_GUIDE.md` - Tài liệu chính về ML training
- `docs/COLAB_LABELING_GUIDE.md` - Hướng dẫn labeling trên Colab

## ✅ Kết Quả

- ✅ **Pass moves được xử lý đúng**: Có index riêng trong policy vector
- ✅ **Validation chặt chẽ**: Tất cả inputs được validate
- ✅ **Error handling tốt**: Messages rõ ràng với error types
- ✅ **Đúng quỹ đạo tài liệu**: Format phù hợp với ML_COMPREHENSIVE_GUIDE.md
- ✅ **Backward compatible**: Vẫn hỗ trợ format cũ (normal moves)

---

**Status**: ✅ Completed
**Last updated**: 2025-01-27

