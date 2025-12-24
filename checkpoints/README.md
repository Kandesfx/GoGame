# 📁 Thư Mục Checkpoints

> **⚠️ Lưu ý**: Thư mục này đã được thay thế bằng `backend/models/` cho deploy.
> Code vẫn tìm trong thư mục này để backward compatibility, nhưng **khuyến nghị đặt model ở `backend/models/`**.

## 📍 Vị Trí Đặt File Model (Mới)

**Khuyến nghị**: Đặt file model vào `backend/models/`:

```
GoGame/
└── backend/
    └── models/
        └── final_model.pt      ← Đặt file của bạn ở đây (khuyến nghị)
```

## 📍 Vị Trí Cũ (Backward Compatibility)

Thư mục này vẫn được hỗ trợ nhưng không khuyến nghị:

```
GoGame-master/
└── checkpoints/
    ├── final_model.pt          ← Vẫn hoạt động nhưng không khuyến nghị
    ├── best_model.pt           
    └── checkpoint_epoch_X.pt   
```

## 📝 Hướng Dẫn

### 1. Đặt file `final_model.pt` vào đây

Copy file `final_model.pt` vào thư mục `checkpoints/` (thư mục này).

### 2. Sử dụng trong code

Sau khi đặt file vào đây, bạn có thể load model như sau:

```python
from pathlib import Path

# Đường dẫn đến model
checkpoint_path = Path('checkpoints/final_model.pt')

# Hoặc đường dẫn tuyệt đối
checkpoint_path = Path(__file__).parent / 'checkpoints' / 'final_model.pt'
```

### 3. Kiểm tra file đã đặt đúng chưa

```python
from pathlib import Path

checkpoint_path = Path('checkpoints/final_model.pt')
if checkpoint_path.exists():
    print(f"✅ Model found: {checkpoint_path}")
    print(f"   Size: {checkpoint_path.stat().st_size / 1024 / 1024:.2f} MB")
else:
    print(f"❌ Model not found: {checkpoint_path}")
```

## 📂 Cấu Trúc File Model

File checkpoint thường có cấu trúc:

```python
{
    'policy_net_state_dict': {...},      # Weights của Policy Network
    'value_net_state_dict': {...},       # Weights của Value Network
    'policy_config': {...},               # Config của Policy Network
    'value_config': {...},                # Config của Value Network
    'board_size': 9,                      # Kích thước bàn cờ (9, 13, hoặc 19)
    'val_policy_loss': 0.5234,            # Validation loss (nếu có)
    'val_value_loss': 0.1234              # Validation loss (nếu có)
}
```

## ⚠️ Lưu Ý

- Đảm bảo file có đuôi `.pt` (PyTorch format)
- File thường có kích thước từ vài MB đến vài trăm MB tùy model size
- Nếu file có tên khác (ví dụ: `dataset_2019_final_model.pt`), vẫn đặt vào đây và dùng đúng tên file khi load

## 🔗 Xem Thêm

Xem chi tiết cách sử dụng model trong: `docs/HUONG_DAN_SU_DUNG_MODEL.md`

