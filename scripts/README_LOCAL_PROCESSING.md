# 🚀 HƯỚNG DẪN XỬ LÝ DATASET TRÊN LOCAL

## 📋 Tổng Quan

Scripts tối ưu để xử lý 220,000 trận đấu SGF trên local máy tính với:
- ✅ Xử lý theo năm (từ tên file `YYYY-MM-DD-XX.sgf`)
- ✅ Multiprocessing để tăng tốc
- ✅ Error handling và logging chi tiết
- ✅ Bỏ qua file lỗi và tiếp tục
- ✅ Output theo năm để dễ quản lý

## 📁 Scripts

| Script | Mô tả |
|--------|-------|
| `parse_sgf_local.py` | Parse SGF files → positions (theo năm) |
| `generate_labels_local.py` | Generate labels từ positions (multiprocessing) |
| `process_by_year.sh` | Script tự động xử lý tất cả các năm |
| `merge_datasets.py` | Merge datasets từ nhiều năm |

## 🎯 Workflow

```
1. Parse SGF (theo năm) → processed/positions_*_YYYY.pt
2. Generate Labels (theo năm) → datasets/labeled_*_YYYY.pt
3. Merge tất cả năm → datasets/labeled_*_merged.pt
4. Upload lên Drive → Training trên Colab
```

## 📝 Cách Sử Dụng

### Option 1: Xử Lý Từng Năm (Khuyến Nghị)

#### Bước 1: Parse SGF cho một năm

```bash
# Nếu dùng virtual environment, activate trước:
source venv/bin/activate  # Linux/Mac/MSYS2
# hoặc
venv\Scripts\activate.bat  # Windows

# Sau đó chạy:
python scripts/parse_sgf_local.py \
    --input data/raw_sgf \
    --output data/processed \
    --year 2019 \
    --board-sizes 9 13 19 \
    --workers 8 \
    --min-positions 10

# Hoặc dùng helper script:
bash scripts/activate_venv.sh --input data/raw_sgf --output data/processed --year 2019
```

**Output:**
- `data/processed/positions_9x9_2019.pt`
- `data/processed/positions_13x13_2019.pt`
- `data/processed/positions_19x19_2019.pt`
- `data/processed/parse_errors_2019.log` (nếu có lỗi)

#### Bước 2: Generate Labels cho một năm

```bash
python scripts/generate_labels_local.py \
    --input data/processed/positions_9x9_2019.pt \
    --output data/datasets/labeled_9x9_2019.pt \
    --filter-handicap \
    --workers 8 \
    --batch-size 5000
```

**Output:**
- `data/datasets/labeled_9x9_2019.pt`
- `data/datasets/label_errors_2019.log` (nếu có lỗi)

### Option 2: Xử Lý Tất Cả Năm Tự Động

```bash
# Chỉnh sửa YEARS trong script nếu cần
bash scripts/process_by_year.sh
```

Script sẽ tự động:
1. Parse SGF cho từng năm
2. Generate labels cho từng năm
3. Hiển thị progress và errors

### Option 3: Merge Tất Cả Năm

Sau khi xử lý xong tất cả năm, merge lại:

```bash
# Merge cho board 9x9
python scripts/merge_datasets.py \
    --input data/datasets \
    --output data/datasets/labeled_9x9_merged.pt \
    --board-size 9

# Merge cho board 13x13
python scripts/merge_datasets.py \
    --input data/datasets \
    --output data/datasets/labeled_13x13_merged.pt \
    --board-size 13

# Merge cho board 19x19
python scripts/merge_datasets.py \
    --input data/datasets \
    --output data/datasets/labeled_19x19_merged.pt \
    --board-size 19
```

## ⚙️ Configuration

### Parse SGF Options

```bash
--input DIR          # Thư mục chứa SGF files
--output DIR         # Thư mục output
--year YEAR          # Năm cần xử lý (2015-2024)
--board-sizes        # 9 13 19 (default: tất cả)
--workers N          # Số worker processes (default: auto, max 8)
--min-positions N    # Số positions tối thiểu mỗi game (default: 10)
```

### Generate Labels Options

```bash
--input FILE         # File positions (.pt)
--output FILE        # File labeled dataset (.pt)
--filter-handicap    # Bỏ qua handicap positions (default: True)
--workers N          # Số worker processes (default: auto, max 8)
                     # ⚠️ Giảm nếu RAM bị chiếm nhiều (khuyến nghị: 6-8)
--batch-size N       # Batch size (default: 5000, tối ưu cho performance)
                     # ⚠️ Giảm nếu RAM bị chiếm nhiều (khuyến nghị: 2000-5000)
```

## 📊 Performance

### Ước Tính Thời Gian

Với 220,000 games (~10-20M positions):

| Step | Time (8 workers) | Notes |
|------|-------------------|-------|
| Parse SGF (1 year) | 30-60 phút | ~20K games/năm |
| Generate Labels (1 year) | 20-40 phút | ~1M positions/năm |
| **Total (all years)** | **10-20 giờ** | Có thể chạy qua đêm |

### Tối Ưu Hóa

1. **Cân bằng Workers và Memory:**
   - **16GB RAM:** `--workers 6-8`, `--batch-size 2000-5000`
   - **32GB RAM:** `--workers 8-12`, `--batch-size 5000-10000`
   - ⚠️ **Không nên dùng > 12 workers** vì mỗi worker process cần memory

2. **Nếu RAM bị chiếm nhiều:**
   ```bash
   --workers 6        # Giảm workers
   --batch-size 2000  # Giảm batch size
   ```

3. **Theo dõi memory trong log:**
   - Script tự động log memory usage mỗi 15 giây
   - Nếu thấy > 3GB, giảm workers hoặc batch-size

4. **Xử lý từng năm** để dễ kiểm soát và resume

## 📝 Logging

### Parse Logs

- `parse_sgf_local.log`: Log chi tiết
- `processed/parse_errors_YYYY.log`: Errors cho từng năm

**Format:**
```
File: 2019-04-30-62.sgf
Type: parse_error
Error: Invalid board size: 21
```

### Label Generation Logs

- `generate_labels_local.log`: Log chi tiết
- `datasets/label_errors_YYYY.log`: Errors cho từng năm

**Format:**
```
Type: size_mismatch
Error: Board size mismatch: (9, 9) vs 13
Position: {'move_number': 42, 'current_player': 'B'}
```

## 🔍 Error Handling

### Các Loại Lỗi

1. **Parse Errors:**
   - `empty`: File rỗng
   - `parse_error`: Không parse được SGF
   - `invalid_board_size`: Board size không hợp lệ
   - `no_moves`: Không có moves hợp lệ
   - `exception`: Lỗi khác

2. **Label Errors:**
   - `size_mismatch`: Board size không khớp
   - `exception`: Lỗi khác

### Xử Lý Lỗi

- ✅ **Tự động bỏ qua** file/position lỗi
- ✅ **Log chi tiết** vào file
- ✅ **Tiếp tục** xử lý các file khác
- ✅ **Statistics** về success rate

## 📦 Output Format

### Positions File

```python
{
    'positions': [
        {
            'board_state': np.ndarray,  # [board_size, board_size]
            'move': (x, y),
            'current_player': 'B' or 'W',
            'move_number': int,
            'board_size': int,
            'game_result': str,
            'winner': 'B' or 'W' or None,
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

### Labeled Dataset File

```python
{
    'labeled_data': [
        {
            'features': torch.Tensor,  # [17, board_size, board_size]
            'policy': torch.Tensor,     # [board_size * board_size]
            'value': float,            # 0.0 - 1.0
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

## ⚠️ LƯU Ý QUAN TRỌNG

**Nếu bạn dùng MSYS2/Git Bash:** MSYS2 Python không có pre-built wheels cho numpy/torch. 

**Giải pháp nhanh:** Dùng Python Windows với `py` command (khuyến nghị) hoặc xem `scripts/QUICK_FIX.md`.

**Nếu gặp lỗi:** Xem `scripts/QUICK_FIX.md` để biết cách fix nhanh.

## 🚀 Quick Start

### 1. Setup

#### Option A: Dùng Virtual Environment (Khuyến nghị cho MSYS2)

```bash
# Tạo virtual environment
python -m venv venv

# Activate (Linux/Mac/MSYS2)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate.bat

# Cài đặt dependencies
pip install -r scripts/requirements_local.txt
# Hoặc:
pip install sgf numpy torch tqdm
```

#### Option B: Dùng Python từ Windows (nếu có)

```bash
# Trên Windows, dùng Python từ Windows (không phải MSYS2):
py -m pip install -r scripts/requirements_local.txt

# Hoặc cài thủ công:
py -m pip install sgf numpy torch tqdm
```

#### Option C: Dùng MSYS2 Packages

```bash
# Cài đặt qua pacman (nếu có sẵn trong MSYS2)
pacman -S mingw-w64-x86_64-python-sgf
pacman -S mingw-w64-x86_64-python-numpy
pacman -S mingw-w64-x86_64-python-torch
pacman -S mingw-w64-x86_64-python-tqdm
```

# Tạo thư mục
mkdir -p data/{raw_sgf,processed,datasets}

# Copy SGF files vào data/raw_sgf/
# Format: YYYY-MM-DD-XX.sgf
```

### 2. Test với một năm

```bash
# Parse năm 2019
python scripts/parse_sgf_local.py \
    --input data/raw_sgf \
    --output data/processed \
    --year 2019

bash scripts/run_with_python_windows.sh scripts/parse_sgf_local.py \
    --input data/raw_sgf \
    --output data/processed \
    --year 2019

# Generate labels
python scripts/generate_labels_local.py \
    --input data/processed/positions_9x9_2019.pt \
    --output data/datasets/labeled_9x9_2019.pt
```
/c/Users/HAI/AppData/Local/Programs/Python/Python312/python.exe scripts/generate_labels_local.py \
    --input data/processed/positions_19x19_2019.pt \
    --output data/datasets/labeled_19x19_2019.pt \
    --filter-handicap \
    --workers 8 \
    --batch-size 2000

    
### 3. Xử lý tất cả năm

```bash
# Chỉnh sửa YEARS trong process_by_year.sh
bash scripts/process_by_year.sh
```

### 4. Merge và upload

```bash
# Merge
python scripts/merge_datasets.py \
    --input data/datasets \
    --output data/datasets/labeled_9x9_merged.pt \
    --board-size 9

# Upload lên Google Drive
# Sau đó train trên Colab
```

## 🐛 Troubleshooting

### Vấn đề: RAM bị chiếm nhiều nhưng CPU thấp

**Nguyên nhân:**
- Quá nhiều workers (mỗi worker process cần memory)
- Batch size quá lớn (giữ nhiều data trong memory)
- Tất cả labeled data được giữ trong RAM cho đến khi xong

**Giải pháp:**
1. **Giảm số workers:**
   ```bash
   --workers 8  # Thay vì 20
   ```

2. **Giảm batch size:**
   ```bash
   --batch-size 2000  # Thay vì 5000
   ```

3. **Theo dõi memory trong log:**
   - Script sẽ log memory usage mỗi 15 giây
   - Nếu thấy > 3GB, giảm workers hoặc batch-size

4. **Khuyến nghị cho 16GB RAM:**
   - `--workers 6-8`
   - `--batch-size 1000-2000`
   - Xử lý từng năm thay vì tất cả cùng lúc

### Lỗi: "Out of memory"

**Giải pháp:**
- Giảm `--batch-size` (5000 → 2000 → 1000)
- Giảm `--workers` (8 → 6 → 4)
- Xử lý từng năm thay vì tất cả
- Kiểm tra Task Manager để xem memory usage

### Lỗi: "Too many open files"

**Giải pháp:**
```bash
ulimit -n 4096  # Tăng limit
```

### Lỗi: "Parse error: ..."

**Giải pháp:**
- Kiểm tra `parse_errors_*.log`
- File lỗi sẽ tự động bỏ qua
- Có thể xử lý lại file cụ thể sau

## 📚 Next Steps

Sau khi xử lý xong:

1. **Verify datasets:**
   ```python
   import torch
   data = torch.load('data/datasets/labeled_9x9_merged.pt')
   print(f"Total: {data['total']:,} samples")
   ```

2. **Upload lên Google Drive:**
   - Upload `labeled_*_merged.pt` files
   - Size: ~50-200MB mỗi file

3. **Train trên Colab:**
   - Sử dụng `train_colab.py`
   - Xem `docs/ML_TRAINING_COLAB_GUIDE.md`

---

**Status**: ✅ Ready for 220K games processing!

