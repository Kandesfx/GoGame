# 🔍 KIỂM TRA FORMAT POSITION FILES

## Cách Kiểm Tra

### Option 1: Chạy Script Kiểm Tra (Cần torch)

```bash
# Cài torch nếu chưa có
pip install torch numpy

# Kiểm tra một file
python scripts/check_positions_format.py data/processed/positions_19x19_2012.pt

# Kiểm tra tất cả files trong thư mục
python scripts/check_positions_format.py data/processed/
```

### Option 2: Kiểm Tra Thủ Công (Không cần torch)

Tạo file Python tạm thời:

```python
# check_manual.py
import sys
sys.path.insert(0, 'scripts')

# Cần cài torch trước
import torch

file_path = 'data/processed/positions_19x19_2012.pt'
data = torch.load(file_path, map_location='cpu', weights_only=False)

positions = data['positions']
print(f"Total: {len(positions):,}")

# Check sample
sample = positions[0]
print("\nFields:", list(sample.keys()))

# Check pass moves
pass_count = sum(1 for p in positions[:1000] if p.get('move') is None)
print(f"Pass moves: {pass_count}/1000")

# Check required fields
required = ['board_state', 'move', 'current_player', 'winner', 'game_result']
for field in required:
    if field in sample:
        print(f"✅ {field}: Present")
    else:
        print(f"❌ {field}: MISSING")
```

## Kết Quả Mong Đợi

### ✅ KHÔNG CẦN PARSE LẠI nếu:

1. **Có đủ fields**:
   - ✅ `board_state`
   - ✅ `move` (có thể là `None` cho pass)
   - ✅ `current_player`
   - ✅ `winner` (hoặc `game_result`)
   - ✅ `game_result`

2. **Hỗ trợ pass moves**:
   - ✅ `move = None` cho pass moves
   - ✅ Có ít nhất một vài pass moves trong file

### ❌ CẦN PARSE LẠI nếu:

1. **Thiếu fields**:
   - ❌ Không có `board_state`
   - ❌ Không có `move`
   - ❌ Không có `current_player`

2. **Không hỗ trợ pass moves**:
   - ❌ Tất cả moves đều là tuple `(x, y)`
   - ❌ Không có `move = None` nào
   - ⚠️ Nếu games có pass moves nhưng bị bỏ qua khi parse

## Format Mong Đợi

```python
{
    'positions': [
        {
            'board_state': np.ndarray,  # [19, 19]
            'move': (x, y) | None,      # Normal hoặc pass
            'current_player': 'B' | 'W',
            'move_number': int,
            'board_size': int,
            'game_result': str | None,
            'winner': 'B' | 'W' | 'DRAW' | None,
            'handicap': int
        },
        ...
    ],
    'board_size': int,
    'total': int,
    'year': int (optional)
}
```

## Quyết Định

Sau khi kiểm tra:

- **Nếu file có đủ fields và hỗ trợ pass moves**:
  - ✅ **KHÔNG CẦN parse lại**
  - Chạy labeling script ngay: `python scripts/generate_labels_local.py`

- **Nếu file thiếu fields hoặc không có pass moves**:
  - ❌ **CẦN parse lại**
  - Chạy: `python scripts/parse_sgf_local.py`

