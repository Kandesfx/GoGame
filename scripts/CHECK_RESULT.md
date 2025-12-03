# 📋 HƯỚNG DẪN KIỂM TRA POSITION FILES

## 🚀 Cách Kiểm Tra Nhanh

### Bước 1: Cài Dependencies (Nếu Chưa Có)

```bash
pip install torch numpy
```

### Bước 2: Chạy Script Kiểm Tra

```bash
# Kiểm tra một file cụ thể
python scripts/check_positions_simple.py data/processed/positions_19x19_2012.pt

# Hoặc kiểm tra tất cả files trong thư mục
python scripts/check_positions_simple.py
```

## 📊 Kết Quả

Script sẽ hiển thị:

1. **Total positions**: Số lượng positions trong file
2. **Sample fields**: Các fields có trong position
3. **Required fields check**: Kiểm tra fields bắt buộc
4. **Pass moves check**: Kiểm tra hỗ trợ pass moves
5. **Summary**: Kết luận có cần parse lại không

## ✅ Kết Luận

### KHÔNG CẦN PARSE LẠI nếu:

- ✅ Tất cả required fields đều có
- ✅ Pass moves được hỗ trợ (`move = None`)
- ✅ File format đúng

### CẦN PARSE LẠI nếu:

- ❌ Thiếu required fields
- ❌ Không hỗ trợ pass moves (mà games có pass moves)
- ❌ Format không đúng

## 📝 Lưu Ý

- Files đã được parse **SAU KHI SỬA** (có pass moves support) → **KHÔNG CẦN** parse lại
- Files được parse **TRƯỚC KHI SỬA** (không có pass moves) → **CẦN** parse lại

## 🎯 Next Steps

Sau khi kiểm tra:

1. **Nếu không cần parse lại**:
   ```bash
   # Chạy labeling script
   python scripts/generate_labels_local.py \
     --input data/processed/positions_19x19_2012.pt \
     --output data/datasets/labeled_19x19_2012.pt
   ```

2. **Nếu cần parse lại**:
   ```bash
   # Parse lại SGF files
   python scripts/parse_sgf_local.py \
     --input data/raw_sgf/ \
     --output data/processed/ \
     --year 2012
   ```

