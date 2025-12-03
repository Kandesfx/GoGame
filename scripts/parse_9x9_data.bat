@echo off
REM Script để parse data 9x9 từ raw_sgf_9x9
REM Tự động chia thành các chunks với kích thước hợp lý

echo ========================================
echo Parse SGF 9x9 Data with Chunking
echo ========================================
echo.

REM Sử dụng Python từ Windows (không phải MSYS2)
python scripts/parse_sgf_with_chunking.py ^
  --input data/raw_sgf_9x9 ^
  --output data/processed_9x9 ^
  --board-sizes 9 ^
  --positions-per-chunk 50000 ^
  --chunk-prefix "9x9_chunk"

echo.
echo ========================================
echo Done!
echo ========================================
pause

