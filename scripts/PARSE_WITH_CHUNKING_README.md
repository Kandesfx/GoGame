# Parse SGF với Tự Động Chia File (Chunking)

## Mô Tả

Script `parse_sgf_with_chunking.py` được thiết kế để parse các file SGF **không có năm trong tên file** (ví dụ: `1547679.sgf`, `1547692.sgf`) và tự động chia thành các file output với kích thước hợp lý.

## Tính Năng

✅ **Tự động chia file**: Chia positions thành các chunks với kích thước cố định  
✅ **Đánh dấu rõ ràng**: Mỗi chunk có số thứ tự và metadata đầy đủ  
✅ **Index file**: Tạo file JSON index để quản lý các chunks  
✅ **Không cần năm**: Không cần filter theo năm, parse tất cả file trong thư mục  

## Cách Sử Dụng

### Cơ Bản

```bash
python scripts/parse_sgf_with_chunking.py \
  --input data/raw_sgf \
  --output data/processed \
  --board-sizes 9
```

### Tùy Chỉnh Chunk Size

```bash
# 100K positions mỗi chunk
python scripts/parse_sgf_with_chunking.py \
  --input data/raw_sgf \
  --output data/processed \
  --board-sizes 9 \
  --positions-per-chunk 100000
```

### Nhiều Board Sizes

```bash
python scripts/parse_sgf_with_chunking.py \
  --input data/raw_sgf \
  --output data/processed \
  --board-sizes 9 13 19
```

### Tùy Chỉnh Prefix

```bash
python scripts/parse_sgf_with_chunking.py \
  --input data/raw_sgf \
  --output data/processed \
  --board-sizes 9 \
  --chunk-prefix "9x9_data"
```

## Tham Số

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--input` | **Bắt buộc** | Thư mục chứa file SGF |
| `--output` | **Bắt buộc** | Thư mục lưu kết quả |
| `--board-sizes` | `9 13 19` | Các kích thước bàn cờ cần parse |
| `--positions-per-chunk` | `50000` | Số positions mỗi chunk file |
| `--chunk-prefix` | `chunk` | Prefix cho tên file chunk |
| `--num-workers` | `auto` | Số worker processes (None = auto) |
| `--min-positions-per-game` | `10` | Số positions tối thiểu mỗi game |

## Format Output

### Chunk Files

Mỗi chunk file có format: `{prefix}_{board_size}x{board_size}_{chunk_num:04d}.pt`

Ví dụ:
- `chunk_9x9_0001.pt`
- `chunk_9x9_0002.pt`
- `chunk_9x9_0003.pt`

### Chunk File Structure

```python
{
    'positions': [...],              # List các positions
    'board_size': 9,
    'chunk_num': 1,                   # Số thứ tự chunk
    'total_chunks': 5,                # Tổng số chunks
    'positions_in_chunk': 50000,      # Số positions trong chunk này
    'start_index': 0,                 # Index bắt đầu
    'end_index': 49999,               # Index kết thúc
    'metadata': {
        'source_files': 14,           # Số file SGF đã parse
        'source_file_range': "1547679.sgf to 1547692.sgf",
        'errors': 0,
        'date_processed': "2025-01-20T...",
        'positions_per_chunk': 50000
    }
}
```

### Index File

File JSON index: `{prefix}_{board_size}x{board_size}_index.json`

```json
{
  "board_size": 9,
  "total_positions": 250000,
  "total_chunks": 5,
  "positions_per_chunk": 50000,
  "chunks": [
    {
      "chunk_num": 1,
      "filename": "chunk_9x9_0001.pt",
      "positions": 50000
    },
    {
      "chunk_num": 2,
      "filename": "chunk_9x9_0002.pt",
      "positions": 50000
    }
  ],
  "source_files": 14,
  "date_created": "2025-01-20T..."
}
```

## Ví Dụ Output

```
📊 Found 14 SGF files
📁 Files range: 1547679.sgf to 1547692.sgf
✅ 9x9: 250,000 positions (250,000 before filtering)
💾 Saving 250,000 positions for 9x9 into 5 chunk(s) (50,000 positions/chunk)
  ✅ Chunk 1/5: 50,000 positions → chunk_9x9_0001.pt
  ✅ Chunk 2/5: 50,000 positions → chunk_9x9_0002.pt
  ✅ Chunk 3/5: 50,000 positions → chunk_9x9_0003.pt
  ✅ Chunk 4/5: 50,000 positions → chunk_9x9_0004.pt
  ✅ Chunk 5/5: 50,000 positions → chunk_9x9_0005.pt
📋 Index file saved: chunk_9x9_index.json
```

## Lưu Ý

1. **Chunk Size**: Mặc định 50K positions/chunk (~2.5GB mỗi file). Có thể tăng nếu có nhiều RAM.
2. **Index File**: File JSON index giúp quản lý và load các chunks dễ dàng hơn.
3. **Metadata**: Mỗi chunk chứa đầy đủ metadata về source files và processing info.
4. **Error Log**: Lỗi được ghi vào `parse_errors.log` trong output directory.

## So Sánh với `parse_by_year.py`

| Tính năng | `parse_by_year.py` | `parse_sgf_with_chunking.py` |
|-----------|-------------------|------------------------------|
| Filter theo năm | ✅ Có | ❌ Không (parse tất cả) |
| Chia file output | ❌ 1 file/năm | ✅ Nhiều chunks |
| Phù hợp cho | File có năm trong tên | File không có năm |
| Index file | ❌ | ✅ |

## Troubleshooting

### Lỗi: "Missing dependencies"
```bash
pip install sgf numpy torch
```

### Lỗi: "Cannot import parse_sgf_local"
Đảm bảo bạn đang chạy từ thư mục gốc của project:
```bash
cd /path/to/GoGame
python scripts/parse_sgf_with_chunking.py --input ... --output ...
```

### Memory Issues
Giảm `--positions-per-chunk` nếu gặp vấn đề về memory:
```bash
python scripts/parse_sgf_with_chunking.py \
  --input data/raw_sgf \
  --output data/processed \
  --board-sizes 9 \
  --positions-per-chunk 25000
```

