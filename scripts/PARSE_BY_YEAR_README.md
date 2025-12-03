# 📅 Parse SGF Files Theo Năm

Script đơn giản để parse các file SGF từ thư mục `data/raw_sgf` theo từng năm.

## 🚀 Cài Đặt

Trước khi sử dụng, cần cài đặt dependencies:

```bash
pip install -r scripts/requirements_local.txt
```

Hoặc cài đặt thủ công:

```bash
pip install sgf numpy torch tqdm
```

### ⚠️ Lưu Ý về Python Environment

Nếu bạn đang dùng **MSYS2/Git Bash**, Python mặc định có thể không có các packages đã cài. Có 2 cách:

**Cách 1: Dùng Windows Python trực tiếp**
```bash
# Trong Git Bash/MSYS2
/c/Users/Hai/AppData/Local/Programs/Python/Python312/python.exe scripts/parse_by_year.py --year 2000
```

**Cách 2: Dùng wrapper script (Windows)**
```cmd
scripts\parse_by_year_wrapper.bat --year 2000
```

**Cách 3: Dùng wrapper script (Git Bash/MSYS2)**
```bash
bash scripts/parse_by_year_wrapper.sh --year 2000
```

## 📋 Cách Sử Dụng

### 1. Xem các năm có sẵn

```bash
python scripts/parse_by_year.py --list-years
```

Output:
```
Các năm có sẵn trong thư mục:
  2000: 15 files
  2001: 20 files
  ...
```

### 2. Parse một năm cụ thể

```bash
python scripts/parse_by_year.py --year 2000
```

### 3. Parse nhiều năm

```bash
python scripts/parse_by_year.py --year 2000 --year 2001 --year 2002
```

### 4. Parse tất cả các năm

```bash
python scripts/parse_by_year.py --year all
```

### 5. Parse và generate labels luôn

```bash
python scripts/parse_by_year.py --year 2000 --generate-labels
```

Điều này sẽ:
1. Parse SGF files → tạo `positions_*.pt` trong `data/processed/`
2. Generate labels → tạo `labeled_*.pt` trong `data/datasets/`

## 📁 Format File Input

Script hỗ trợ cả hai format tên file:
- `YYYY-M-D-X.sgf` (ví dụ: `2000-7-19-1.sgf`)
- `YYYY-MM-DD-XX.sgf` (ví dụ: `2000-07-19-01.sgf`)

## 📂 Output

### Parse SGF → Positions

Output được lưu trong `data/processed/`:
- `positions_19x19_2000.pt` - Positions cho board 19x19 năm 2000
- `positions_13x13_2000.pt` - Positions cho board 13x13 năm 2000
- `positions_9x9_2000.pt` - Positions cho board 9x9 năm 2000

### Generate Labels

Output được lưu trong `data/datasets/`:
- `labeled_19x19_2000.pt` - Labeled dataset cho board 19x19 năm 2000
- `labeled_13x13_2000.pt` - Labeled dataset cho board 13x13 năm 2000
- `labeled_9x9_2000.pt` - Labeled dataset cho board 9x9 năm 2000

## ⚙️ Tùy Chọn

### Thay đổi thư mục input/output

```bash
python scripts/parse_by_year.py \
    --year 2000 \
    --input data/raw_sgf \
    --output data/processed \
    --labels-output data/datasets
```

### Chỉ parse một số board sizes

```bash
python scripts/parse_by_year.py --year 2000 --board-sizes 19
```

### Điều chỉnh số workers (cho máy có RAM thấp)

```bash
python scripts/parse_by_year.py --year 2000 --workers 4
```

### Giữ lại handicap positions khi generate labels

```bash
python scripts/parse_by_year.py --year 2000 --generate-labels --no-filter-handicap
```

## 📊 Ví Dụ Workflow Hoàn Chỉnh

```bash
# 1. Xem các năm có sẵn
python scripts/parse_by_year.py --list-years

# 2. Parse năm 2000
python scripts/parse_by_year.py --year 2000

# 3. Parse và generate labels cho năm 2000
python scripts/parse_by_year.py --year 2000 --generate-labels

# 4. Parse tất cả các năm và generate labels
python scripts/parse_by_year.py --year all --generate-labels
```

## 🔍 Format Dữ Liệu

### Positions File (`positions_*.pt`)

```python
{
    'positions': [
        {
            'board_state': np.ndarray,  # Board state trước khi đặt quân
            'move': (x, y) | None,      # Move hoặc None (pass)
            'current_player': 'B' | 'W',
            'move_number': int,
            'board_size': int,
            'game_result': str,
            'winner': 'B' | 'W' | 'DRAW' | None,
            'handicap': int
        },
        ...
    ],
    'board_size': int,
    'total': int,
    'year': int,
    'metadata': {...}
}
```

### Labeled Dataset (`labeled_*.pt`)

Theo format trong `MULTI_TASK_LABELS_IMPLEMENTATION.md`:

```python
{
    'labeled_data': [
        {
            'features': Tensor[17, board_size, board_size],
            'labels': {
                'threat_map': Tensor[board_size, board_size],
                'attack_map': Tensor[board_size, board_size],
                'intent': {
                    'type': str,  # 'territory', 'attack', 'defense', 'connection', 'cut'
                    'confidence': float,
                    'region': List[Tuple[int, int]]
                },
                'evaluation': {
                    'win_probability': float,
                    'territory_map': Tensor[board_size, board_size],
                    'influence_map': Tensor[board_size, board_size]
                }
            },
            'policy': Tensor[board_size * board_size + 1],
            'value': float,
            'metadata': {...}
        },
        ...
    ],
    'board_size': int,
    'total': int,
    'year': int,
    'metadata': {...}
}
```

## ⚠️ Lưu Ý

1. **Memory Usage**: Với dataset lớn (>1M positions), script sẽ tự động dùng incremental save để tránh RAM overflow.

2. **Error Handling**: Script sẽ bỏ qua các file lỗi và tiếp tục xử lý. Xem log file để biết chi tiết lỗi.

3. **Performance**: 
   - Parse: ~100-500 files/phút (tùy độ phức tạp)
   - Label generation: ~1000-5000 positions/phút (tùy CPU/RAM)

4. **Log Files**:
   - `parse_by_year.log` - Log chính
   - `parse_sgf_local.log` - Log từ parse_sgf_local
   - `generate_labels_local.log` - Log từ generate_labels_local
   - `data/processed/parse_errors_*.log` - Chi tiết lỗi parse
   - `data/datasets/label_errors_*.log` - Chi tiết lỗi label generation

## 🔗 Liên Quan

- `scripts/parse_sgf_local.py` - Script parse SGF chính
- `scripts/generate_labels_local.py` - Script generate labels
- `scripts/MULTI_TASK_LABELS_IMPLEMENTATION.md` - Tài liệu về multi-task labels

