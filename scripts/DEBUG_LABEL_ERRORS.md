# Debug Label Generation Errors

## Vấn đề
Sau khi tối ưu, có 624,736 errors trên 624,736 positions (100% lỗi).

## Các Fix Đã Áp Dụng

### 1. Error Handling trong Vectorized Functions
- ✅ Thêm try-except trong `detect_false_eyes()`
- ✅ Thêm try-except trong `detect_cutting_points()`
- ✅ Thêm try-except trong `find_invasion_points()`
- ✅ Fallback: return empty list nếu có lỗi (thay vì crash)

### 2. Type Safety
- ✅ Đảm bảo `board_state` là numpy array
- ✅ Convert sang `int` khi cần (tránh numpy scalar issues)
- ✅ Validate `groups` trước khi dùng

### 3. Better Error Messages
- ✅ Thêm traceback trong error messages để debug dễ hơn

## Cách Debug

### Bước 1: Kiểm tra Log File
```bash
# Trên Colab
cat /content/drive/MyDrive/GoGame_ML/datasets/label_errors_2019.log
```

### Bước 2: Xem Error Types
Log file sẽ group errors theo type:
- `threat_map_error`: Lỗi khi generate threat map
- `attack_map_error`: Lỗi khi generate attack map
- `intent_label_error`: Lỗi khi generate intent label
- `evaluation_label_error`: Lỗi khi generate evaluation label
- `exception`: Lỗi chung (có traceback)

### Bước 3: Test với Sample
```python
# Test với một position
from label_generators import ThreatLabelGenerator
import numpy as np

threat_gen = ThreatLabelGenerator(19)
board = np.zeros((19, 19), dtype=np.int8)
board[9, 9] = 1  # Đặt một quân đen

try:
    groups = threat_gen.find_groups(board)
    threat_map = threat_gen.generate_threat_map(board, 'W', groups=groups)
    print("Success!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
```

## Các Lỗi Có Thể Xảy Ra

### 1. NumPy Array Issues
**Triệu chứng**: `TypeError: 'numpy.ndarray' object is not callable`
**Nguyên nhân**: Board state không phải numpy array hoặc có vấn đề với dtype
**Fix**: Đã thêm check và convert trong code

### 2. Index Out of Bounds
**Triệu chứng**: `IndexError: index out of bounds`
**Nguyên nhân**: Coordinates từ `np.where` không đúng
**Fix**: Đã thêm `int()` conversion và bounds checking

### 3. Groups Format Issues
**Triệu chứng**: `KeyError: 'color'` hoặc `AttributeError`
**Nguyên nhân**: `find_groups()` trả về format không đúng
**Fix**: Đã thêm validation

## Test Script

Tạo file `test_label_generation.py`:

```python
import numpy as np
import torch
from label_generators import (
    ThreatLabelGenerator,
    AttackLabelGenerator,
    IntentLabelGenerator,
    EvaluationLabelGenerator
)

def test_label_generation():
    board_size = 19
    board = np.zeros((board_size, board_size), dtype=np.int8)
    
    # Tạo một board đơn giản
    board[9, 9] = 1  # Black stone
    board[9, 10] = 2  # White stone
    
    threat_gen = ThreatLabelGenerator(board_size)
    attack_gen = AttackLabelGenerator(board_size)
    intent_gen = IntentLabelGenerator(board_size)
    eval_gen = EvaluationLabelGenerator(board_size)
    
    # Test threat map
    try:
        groups = threat_gen.find_groups(board)
        print(f"Found {len(groups)} groups")
        
        threat_map = threat_gen.generate_threat_map(board, 'W', groups=groups)
        print(f"Threat map shape: {threat_map.shape}")
        print("✅ Threat map OK")
    except Exception as e:
        print(f"❌ Threat map error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test attack map
    try:
        attack_map = attack_gen.generate_attack_map(board, 'W', groups=groups)
        print(f"Attack map shape: {attack_map.shape}")
        print("✅ Attack map OK")
    except Exception as e:
        print(f"❌ Attack map error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test intent label
    try:
        intent = intent_gen.generate_intent_label(board, (9, 8), [], 'W')
        print(f"Intent: {intent}")
        print("✅ Intent label OK")
    except Exception as e:
        print(f"❌ Intent label error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test evaluation
    try:
        eval_result = eval_gen.generate_evaluation(board, 'W', None, None)
        print(f"Evaluation: {eval_result.keys()}")
        print("✅ Evaluation OK")
    except Exception as e:
        print(f"❌ Evaluation error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_label_generation()
```

## Next Steps

1. **Chạy test script** để xác định hàm nào bị lỗi
2. **Kiểm tra log file** để xem error messages chi tiết
3. **Fix lỗi cụ thể** dựa trên error messages
4. **Test lại** với một sample nhỏ trước khi chạy toàn bộ dataset

## Lưu Ý

- Tất cả các hàm vectorized đã có error handling
- Nếu có lỗi, sẽ return empty list thay vì crash
- Error messages có traceback để debug dễ hơn
- Cần kiểm tra log file để biết chính xác lỗi gì

