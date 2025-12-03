# 🧪 HƯỚNG DẪN TEST MODEL TRONG GAME

## ✅ Kết Quả Test

Model đã được test và hoạt động tốt:
- ✅ Load model thành công
- ✅ Predict move chính xác
- ✅ Performance tốt (~12ms per prediction)
- ✅ Hoạt động với nhiều board states khác nhau

## 🧪 Cách Test

### 1. Test Cơ Bản (Không cần chạy server)

Chạy script test đơn giản:

```bash
python scripts/test_model_in_game.py
```

Script này sẽ test:
- Load model
- Predict với board states khác nhau
- Performance test
- Test với Black và White player

**Kết quả mong đợi:**
```
✅ Model loaded successfully!
✅ Prediction successful!
   Best move: (x, y)
   Policy probability: 0.xxxx
   Win probability: 0.xxxx
```

### 2. Test Trong Game Thực Tế

#### Bước 1: Khởi động Backend Server

```bash
cd backend
python -m app.main
```

Hoặc nếu dùng script:
```bash
cd backend
.\run.bat
```

#### Bước 2: Tạo AI Match

Sử dụng API hoặc frontend để tạo AI match mới:

**API Request:**
```bash
POST /api/matches/ai
{
  "board_size": 19,
  "level": 1,
  "player_color": "black"
}
```

**Hoặc qua Frontend:**
- Mở game
- Chọn "Play with AI"
- Chọn level và màu quân
- Bắt đầu game

#### Bước 3: Kiểm Tra Logs

Khi AI đánh, bạn sẽ thấy logs trong console:

```
🤖 [ML] Trying ML model AI move
✅ ML model AI move successful
🤖 ML model AI move: (x, y), prob=0.xxxx, win_prob=0.xxxx
```

Nếu thấy logs này, nghĩa là ML model đang được sử dụng!

#### Bước 4: Quan Sát Nước Đi

- AI sẽ đánh nước đi dựa trên ML model
- Nước đi sẽ được hiển thị trên bàn cờ
- Kiểm tra xem nước đi có hợp lý không

## 🔍 Kiểm Tra Model Có Được Sử Dụng Không

### Cách 1: Kiểm Tra Logs

Trong backend console, tìm các dòng:
- `🤖 [ML] Trying ML model AI move`
- `✅ ML model AI move successful`
- `🤖 ML model AI move: (x, y), prob=..., win_prob=...`

### Cách 2: Kiểm Tra Code

Model được sử dụng trong `match_service.py`:
- Hàm `_make_ai_move()` sẽ thử ML model trước
- Nếu ML model không available, sẽ fallback về MCTS/minimax

### Cách 3: So Sánh Nước Đi

- **ML Model**: Nước đi dựa trên deep learning, có thể khác với MCTS
- **MCTS**: Nước đi dựa trên tree search

Nếu thấy nước đi khác với trước, có thể ML model đang được sử dụng.

## ⚠️ Troubleshooting

### Model không được sử dụng

**Kiểm tra:**
1. File `checkpoints/final_model.pt` có tồn tại không?
2. Model có load được không? (chạy `test_model_in_game.py`)
3. Logs có hiển thị lỗi không?

**Giải pháp:**
- Đảm bảo file model đã được đặt đúng vị trí
- Kiểm tra logs để xem lỗi cụ thể
- Model sẽ tự động fallback về MCTS nếu có lỗi

### Model chạy chậm

**Nguyên nhân:**
- Đang dùng CPU (chậm hơn GPU)
- Board size lớn (19x19)

**Giải pháp:**
- Model vẫn hoạt động tốt trên CPU (~12ms per move)
- Nếu có GPU, có thể set `device='cuda'` trong code

### Nước đi không hợp lý

**Nguyên nhân:**
- Model mới train, chưa tối ưu
- Cần train thêm với nhiều data hơn

**Giải pháp:**
- Model sẽ tự động validate move trước khi apply
- Nếu move không hợp lệ, sẽ fallback về pass hoặc MCTS

## 📊 Performance Metrics

Từ test script:
- **Load time**: < 1 giây
- **Prediction time**: ~12ms per move (CPU)
- **Memory usage**: ~200-500MB (tùy model size)

## 🎯 Test Cases

Script test đã cover:
1. ✅ Load model
2. ✅ Board state đơn giản
3. ✅ Board state phức tạp
4. ✅ Board trống (đầu game)
5. ✅ White player
6. ✅ Performance test

## 💡 Tips

1. **Test thường xuyên**: Chạy `test_model_in_game.py` sau mỗi lần update model
2. **Kiểm tra logs**: Luôn kiểm tra logs khi test trong game
3. **So sánh**: So sánh nước đi của ML model với MCTS để đánh giá
4. **Performance**: Monitor performance để đảm bảo game không bị lag

## 📚 Xem Thêm

- **Hướng dẫn sử dụng model**: `docs/HUONG_DAN_SU_DUNG_MODEL.md`
- **Tích hợp model**: `docs/ML_MODEL_INTEGRATION.md`
- **Training guide**: `scripts/README_COLAB_TRAINING.md`

---

**Model đã sẵn sàng! Hãy test và tận hưởng! 🎮**

