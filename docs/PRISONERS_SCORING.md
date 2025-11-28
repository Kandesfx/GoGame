# Prisoners và Scoring Logic - Hướng dẫn

## 📋 Khái niệm Prisoners

Trong cờ vây, **prisoners** là số quân đối phương bị bắt.

### Quy tắc:
- `prisoners_black` = Số quân **Black** bị bắt = **Điểm của White**
- `prisoners_white` = Số quân **White** bị bắt = **Điểm của Black**

## ✅ Logic Đúng

### Khi Black đánh và bắt White:
```python
# Black đánh → bắt White → tăng prisoners_white
prisoners_white += len(captured_stones)
# prisoners_white = số quân White bị bắt = điểm của Black
```

### Khi White đánh và bắt Black:
```python
# White đánh → bắt Black → tăng prisoners_black
prisoners_black += len(captured_stones)
# prisoners_black = số quân Black bị bắt = điểm của White
```

## 🎯 Tính Điểm Cuối Game

### Công thức đúng:
```python
# Black điểm = territory + quân White bị bắt
black_score = territory_black + prisoners_white

# White điểm = territory + quân Black bị bắt + komi
white_score = territory_white + prisoners_black + komi
```

### So sánh điểm:
```python
if black_score > white_score:
    result = f"B+{black_score - white_score}"  # Black thắng
elif white_score > black_score:
    result = f"W+{white_score - black_score}"  # White thắng
else:
    result = "DRAW"
```

## ⚠️ Lỗi Thường Gặp

### ❌ SAI:
```python
# SAI: Dùng prisoners_black cho điểm của Black
black_score = territory_black + prisoners_black  # SAI!
white_score = territory_white + prisoners_white  # SAI!

# SAI: So sánh prisoners sai
if prisoners_black > prisoners_white:
    result = "B+"  # SAI! Phải là prisoners_white > prisoners_black
```

### ✅ ĐÚNG:
```python
# ĐÚNG: Dùng prisoners_white cho điểm của Black
black_score = territory_black + prisoners_white  # ĐÚNG!
white_score = territory_white + prisoners_black  # ĐÚNG!

# ĐÚNG: So sánh điểm đúng
black_score = prisoners_white  # Điểm Black = quân White bị bắt
white_score = prisoners_black  # Điểm White = quân Black bị bắt
if black_score > white_score:
    result = "B+"  # ĐÚNG!
```

## 📍 Các Chỗ Đã Sửa

1. ✅ `_calculate_game_result()` - Tính điểm cuối game (gogame_py mode)
2. ✅ Fallback mode scoring - Tính điểm khi không có gogame_py
3. ✅ AI move scoring - Tính điểm sau AI move
4. ✅ Pass move scoring - Tính điểm sau pass move
5. ✅ Undo move - Tính lại prisoners từ moves còn lại

## 🔍 Kiểm Tra Logic

Khi implement tính điểm, luôn nhớ:
- **Prisoners của đối phương = Điểm của mình**
- Black bắt White → `prisoners_white` tăng → Điểm Black tăng
- White bắt Black → `prisoners_black` tăng → Điểm White tăng

## 📝 Ví Dụ

**Scenario**: Black bắt 3 quân White, White bắt 1 quân Black

```python
prisoners_black = 1  # 1 quân Black bị bắt
prisoners_white = 3  # 3 quân White bị bắt

# Tính điểm (chỉ dùng prisoners, không có territory)
black_score = prisoners_white = 3  # Black có 3 điểm
white_score = prisoners_black = 1  # White có 1 điểm

# Kết quả
if black_score > white_score:  # 3 > 1
    result = "B+2"  # Black thắng 2 điểm
```

