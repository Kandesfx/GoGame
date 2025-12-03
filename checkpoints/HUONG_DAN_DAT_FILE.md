# 📍 HƯỚNG DẪN ĐẶT FILE MODEL

## ✅ Bước 1: Tìm thư mục checkpoints

Thư mục `checkpoints/` nằm ở **root của project GoGame-master**.

```
GoGame-master/
└── checkpoints/    ← Bạn đang ở đây
```

## ✅ Bước 2: Copy file model vào đây

Copy file `final_model.pt` (hoặc file model khác) vào thư mục này.

**Cách 1: Kéo thả**
- Mở thư mục `checkpoints/` trong File Explorer
- Kéo file `final_model.pt` vào đây

**Cách 2: Copy/Paste**
- Copy file `final_model.pt`
- Paste vào thư mục `checkpoints/`

## ✅ Bước 3: Kiểm tra

Sau khi copy, bạn sẽ thấy:

```
checkpoints/
├── final_model.pt    ← File của bạn
└── README.md
```

## ✅ Bước 4: Sử dụng trong code

```python
from pathlib import Path

# Đường dẫn đến model
checkpoint_path = 'checkpoints/final_model.pt'

# Hoặc dùng Path
checkpoint_path = Path('checkpoints/final_model.pt')
```

## ❓ Câu hỏi thường gặp

**Q: File có tên khác (ví dụ: `dataset_2019_final_model.pt`) thì sao?**  
A: Vẫn đặt vào đây, và dùng đúng tên file khi load:
```python
checkpoint_path = 'checkpoints/dataset_2019_final_model.pt'
```

**Q: Có thể đặt ở thư mục khác không?**  
A: Có, nhưng cần chỉnh đường dẫn trong code cho đúng.

**Q: Làm sao biết file đã đặt đúng?**  
A: Chạy code kiểm tra:
```python
from pathlib import Path
checkpoint_path = Path('checkpoints/final_model.pt')
if checkpoint_path.exists():
    print(f"✅ File found: {checkpoint_path}")
else:
    print(f"❌ File not found: {checkpoint_path}")
```

## 📚 Xem thêm

Xem chi tiết cách sử dụng model: `docs/HUONG_DAN_SU_DUNG_MODEL.md`

