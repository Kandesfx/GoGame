# 📝 CẬP NHẬT SGF PARSING - HANDICAP SUPPORT

## ✅ Đã cập nhật

### 1. `scripts/parse_sgf_colab.py`

**Thêm hỗ trợ:**
- ✅ Handicap stones (`;AB[...]`) - Black stones đặt sẵn
- ✅ White handicap stones (`;AW[...]`) - White stones đặt sẵn (hiếm)
- ✅ Handicap number (`;HA[n]`)
- ✅ Starting player tự động: White đi trước nếu có handicap
- ✅ Lưu thông tin handicap trong training data

**Thay đổi chính:**

```python
# Trước: Không xử lý handicap
board = np.zeros((board_size, board_size))
current_player = 'B'  # Luôn Black đi trước

# Sau: Xử lý handicap
handicap = int(root.properties.get('HA', ['0'])[0])
handicap_stones_black = root.properties.get('AB', [])

# Đặt handicap stones
for stone_coord in handicap_stones_black:
    x, y = parse_sgf_coord(stone_coord, board_size)
    board[y, x] = 1  # Black stones

# Starting player thay đổi
current_player = 'W' if handicap > 0 else 'B'
```

### 2. `backend/app/utils/sgf.py`

**Cập nhật `parse_sgf()`:**
- ✅ Extract handicap number (`;HA[n]`)
- ✅ Extract handicap stones (`;AB[...]`, `;AW[...]`)
- ✅ Xử lý SGF coordinates đúng (skip 'i')
- ✅ Return handicap info trong result dict

**Cập nhật `export_sgf()`:**
- ✅ Hỗ trợ export handicap stones
- ✅ Tự động set komi (7.5 cho game bình thường, 0.5 cho handicap)
- ✅ Export handicap number

---

## 📋 Cấu trúc SGF với Handicap

### Ví dụ file SGF có handicap:

```sgf
;GM[1]
;FF[4]
;SZ[19]
;PW[ghost49]
;WR[7d]
;PB[HiraBot44]
;BR[5d]
;DT[2019-04-29]
;HA[2]              ← Handicap: 2 stones
;AB[pd][dp]         ← Black handicap stones tại (p,d) và (d,p)
;KM[0.50]           ← Komi thấp (vì có handicap)
;RE[W+Resign]
;RU[Chinese]
;W[cd]              ← White đi trước (vì có handicap)
;B[pq]
...
```

### Các properties quan trọng:

| Property | Mô tả | Ví dụ |
|----------|-------|-------|
| `;HA[n]` | Số handicap stones | `;HA[2]` = 2 stones |
| `;AB[xy]` | Black handicap stones | `;AB[pd][dp]` = 2 stones |
| `;AW[xy]` | White handicap stones (hiếm) | `;AW[dd]` = 1 stone |
| `;KM[n]` | Komi | `;KM[0.50]` hoặc `;KM[7.50]` |

---

## 🔧 Cách sử dụng

### Parse SGF file với handicap:

```python
from scripts.parse_sgf_colab import parse_sgf_file

# Parse file
positions = parse_sgf_file('game_with_handicap.sgf')

# Mỗi position có thông tin handicap
for pos in positions:
    print(f"Handicap: {pos['handicap']}")
    print(f"Board state: {pos['board_state']}")
    print(f"Current player: {pos['current_player']}")  # 'W' nếu có handicap
```

### Parse SGF string (backend):

```python
from backend.app.utils.sgf import parse_sgf

sgf_content = "(;FF[4];SZ[19];HA[2];AB[pd][dp];KM[0.50];W[cd];B[pq];...)"

game_data = parse_sgf(sgf_content)

print(f"Handicap: {game_data['handicap']}")  # 2
print(f"Handicap stones: {game_data['handicap_stones_black']}")  # ['pd', 'dp']
print(f"Moves: {len(game_data['moves'])}")
```

### Export SGF với handicap:

```python
from backend.app.utils.sgf import export_sgf

sgf = export_sgf(
    moves=[...],
    board_size=19,
    handicap=2,
    handicap_stones_black=[(15, 3), (3, 15)],  # (x, y) coordinates
    komi=0.5
)
```

---

## ⚠️ Lưu ý quan trọng

### 1. Starting Player

**Quy tắc:**
- **Không có handicap**: Black đi trước
- **Có handicap**: White đi trước (vì Black đã có lợi thế từ handicap stones)

**Code:**
```python
current_player = 'W' if handicap > 0 else 'B'
```

### 2. Komi

**Quy tắc:**
- **Không có handicap**: Komi = 7.5 (bù cho White vì Black đi trước)
- **Có handicap**: Komi = 0.5 (thấp hơn vì Black đã có lợi thế)

**Code:**
```python
if handicap > 0:
    komi = 0.5
else:
    komi = 7.5
```

### 3. SGF Coordinates

**Quan trọng:** SGF không có chữ 'i' trong bảng chữ cái Go coordinates!

- `a-h` = columns/rows 0-7
- `j-z` = columns/rows 8-25 (bỏ qua 'i')

**Conversion:**
```python
# SGF → 0-indexed
x = ord(sgf_coord[0]) - ord('a')
if x >= 8:  # Skip 'i'
    x -= 1

# 0-indexed → SGF
sgf_x = chr(ord('a') + x + (1 if x >= 8 else 0))
```

### 4. Training Data

**Handicap games vẫn có thể dùng để train:**
- ✅ Vẫn là dữ liệu hợp lệ
- ✅ Model học được cách chơi với handicap
- ✅ Có thể filter sau nếu cần (dựa vào `handicap` field)

**Filter handicap games:**
```python
# Chỉ lấy games không có handicap
normal_positions = [p for p in positions if p['handicap'] == 0]

# Hoặc chỉ lấy handicap games
handicap_positions = [p for p in positions if p['handicap'] > 0]
```

---

## 🧪 Test Cases

### Test 1: Parse file không có handicap

```python
sgf = "(;FF[4];SZ[19];B[dd];W[ee];B[ed];RE[B+2.5])"
data = parse_sgf(sgf)
assert data['handicap'] == 0
assert data['handicap_stones_black'] == []
assert len(data['moves']) == 3
```

### Test 2: Parse file có handicap

```python
sgf = "(;FF[4];SZ[19];HA[2];AB[pd][dp];KM[0.50];W[cd];B[pq];RE[W+R])"
data = parse_sgf(sgf)
assert data['handicap'] == 2
assert data['handicap_stones_black'] == ['pd', 'dp']
assert len(data['moves']) == 2
```

### Test 3: Export với handicap

```python
sgf = export_sgf(
    moves=[{"color": "W", "position": [2, 3]}, {"color": "B", "position": [15, 16]}],
    board_size=19,
    handicap=2,
    handicap_stones_black=[(15, 3), (3, 15)],
    komi=0.5
)
assert ";HA[2]" in sgf
assert ";AB[" in sgf
assert ";KM[0.50]" in sgf
```

---

## 📚 Tài liệu tham khảo

- **SGF Format Specification**: http://www.red-bean.com/sgf/
- **Handicap Rules**: https://senseis.xmp.net/?Handicap
- **KGS Archive**: https://u-go.net/gamerecords/

---

## ✅ Checklist

Trước khi parse SGF files:

- [x] Script hỗ trợ handicap stones (`;AB[...]`)
- [x] Script xử lý starting player đúng (White nếu có handicap)
- [x] Script lưu thông tin handicap trong training data
- [x] Backend parser hỗ trợ handicap
- [x] Backend exporter hỗ trợ handicap
- [x] SGF coordinates được xử lý đúng (skip 'i')

---

**Cập nhật:** 2025-01-27
**Version:** 2.0

