# Quick Start: Parse 9x9 Data

## Thông tin

- **Input**: `data/raw_sgf_9x9` (61,541 file SGF)
- **Output**: `data/processed_9x9`
- **Format**: Tự động chia thành chunks (50K positions/chunk)

## Cách chạy

### Option 1: Dùng script helper (Khuyến nghị)

**Windows Command Prompt:**
```cmd
scripts\parse_9x9_data.bat
```

**Git Bash:**
```bash
bash scripts/parse_9x9_data.sh
```

### Option 2: Chạy trực tiếp (1 dòng, không dùng backslash)

**Git Bash:**
```bash
python scripts/parse_sgf_with_chunking.py --input data/raw_sgf_9x9 --output data/processed_9x9 --board-sizes 9 --positions-per-chunk 50000 --chunk-prefix "9x9_chunk"
```

**Windows Command Prompt:**
```cmd
python scripts/parse_sgf_with_chunking.py --input data/raw_sgf_9x9 --output data/processed_9x9 --board-sizes 9 --positions-per-chunk 50000 --chunk-prefix "9x9_chunk"
```

### Option 3: Dùng Python path đầy đủ (nếu cần)

**Git Bash:**
```bash
/c/Users/Hai/AppData/Local/Programs/Python/Python312/python.exe scripts/parse_sgf_with_chunking.py --input data/raw_sgf_9x9 --output data/processed_9x9 --board-sizes 9 --positions-per-chunk 50000 --chunk-prefix "9x9_chunk"
```

## Lưu ý quan trọng

⚠️ **Trong Git Bash, KHÔNG dùng backslash (`\`) để tiếp tục dòng!**

❌ **SAI:**
```bash
python scripts/parse_sgf_with_chunking.py \
  --input data/raw_sgf_9x9 \
  --output data/processed_9x9
```

✅ **ĐÚNG:**
```bash
python scripts/parse_sgf_with_chunking.py --input data/raw_sgf_9x9 --output data/processed_9x9 --board-sizes 9 --positions-per-chunk 50000 --chunk-prefix "9x9_chunk"
```

## Kết quả

Sau khi chạy, bạn sẽ có:

```
data/processed_9x9/
├── 9x9_chunk_9x9_0001.pt
├── 9x9_chunk_9x9_0002.pt
├── 9x9_chunk_9x9_0003.pt
├── ...
└── 9x9_chunk_9x9_index.json
```

## Tùy chỉnh

Nếu muốn thay đổi số positions mỗi chunk:

```bash
python scripts/parse_sgf_with_chunking.py --input data/raw_sgf_9x9 --output data/processed_9x9 --board-sizes 9 --positions-per-chunk 100000 --chunk-prefix "9x9_chunk"
```

## Troubleshooting

### Lỗi: "unrecognized arguments"
- **Nguyên nhân**: Dùng backslash (`\`) trong Git Bash
- **Giải pháp**: Viết lệnh trên 1 dòng hoặc dùng script helper

### Lỗi: "Cannot import parse_sgf_local"
- **Nguyên nhân**: Đang dùng Python từ MSYS2 thay vì Windows Python
- **Giải pháp**: Dùng đường dẫn đầy đủ đến Windows Python:
  ```bash
  /c/Users/Hai/AppData/Local/Programs/Python/Python312/python.exe scripts/parse_sgf_with_chunking.py ...
  ```

### Lỗi: "Missing dependencies"
- **Giải pháp**: Cài đặt dependencies:
  ```bash
  pip install sgf numpy torch
  ```

