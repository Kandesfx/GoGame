# Parse SGF Format 9x9 - Hướng Dẫn

## Format SGF 9x9

Các file SGF 9x9 có thể có các format khác nhau:

### Format 1: Đơn giản (Ví dụ 1, 2)
```
(;GM[1]FF[4]CA[UTF-8]RU[Chinese]SZ[9]KM[7.0]TM[300]
;B[cd]BL[300];W[dd]WL[296];B[fg]BL[300]
;B[]BL[300];W[ii]WL[151]  ← Pass moves
)
```

### Format 2: Có CC property (Ví dụ 3, 4)
```
(;GM[1]FF[4]CA[UTF-8]RU[Chinese]SZ[9]KM[7.0]TM[300]
;B[df]BL[295]CC[{"winrate":0.5546875,...}]
;W[fd]WL[300]
;C[Resignation]  ← Comment node
)
```

## Các Tính Năng Đã Hỗ Trợ

### 1. Pass Moves ✅
- `;B[]` - Pass move của Black
- `;W[]` - Pass move của White
- Được lưu với `move = None`
- `move_number` vẫn tăng (đúng logic)

### 2. Comment Nodes ✅
- `;C[Resignation]` - Comment về resignation
- Nếu có comment "Resignation", parsing sẽ dừng lại
- Các comment nodes khác sẽ bị skip (không ảnh hưởng)

### 3. CC Property ✅
- Property `CC` chứa JSON data (winrate, score, visits, etc.)
- Được bỏ qua khi parse (không ảnh hưởng đến parsing)
- Có thể extract sau nếu cần

### 4. Board Size 9x9 ✅
- Tự động detect từ `SZ[9]`
- Validate board size (chỉ chấp nhận 9, 13, 19)

## Cách Parse

### Sử dụng script hiện có:
```bash
python scripts/parse_by_year.py \
  --input data/raw_sgf \
  --output data/processed \
  --year 2024 \
  --board-sizes 9
```

Hoặc parse tất cả:
```bash
python scripts/parse_by_year.py \
  --input data/raw_sgf \
  --output data/processed \
  --board-sizes 9
```

## Format Output

Mỗi position sẽ có format:
```python
{
    'board_state': np.ndarray,  # [9, 9] cho 9x9
    'move': (x, y) | None,      # Normal move hoặc None cho pass
    'current_player': 'B' | 'W',
    'move_number': int,         # Bắt đầu từ 0
    'board_size': 9,
    'game_result': str,         # "B+12.5", "W+Resign", etc.
    'winner': 'B' | 'W' | 'DRAW' | None,
    'handicap': int             # 0 nếu không có handicap
}
```

## Lưu Ý

1. **Pass Moves**: Được lưu với `move = None`, không apply vào board
2. **Comment Nodes**: Bị skip, không tạo position
3. **Resignation**: Nếu có `C[Resignation]`, parsing dừng tại đó
4. **CC Property**: Bị bỏ qua, không extract (có thể thêm sau nếu cần)

## Test

Để test với file mẫu:
```python
from scripts.parse_sgf_local import parse_single_sgf_file

positions, error = parse_single_sgf_file('test_9x9.sgf')
if error:
    print(f"Error: {error}")
else:
    print(f"Parsed {len(positions)} positions")
    print(f"Pass moves: {sum(1 for p in positions if p['move'] is None)}")
```

