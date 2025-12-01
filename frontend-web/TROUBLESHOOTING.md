# Troubleshooting Guide

## Đăng ký không thành công

### 1. Check Backend đang chạy

```bash
# Terminal 1: Start backend
cd backend
uvicorn app.main:app --reload
```

Backend phải chạy tại: `http://localhost:8000`

### 2. Check CORS Configuration

Backend cần cho phép frontend origin. Check `backend/.env`:

```env
CORS_ORIGINS=["http://localhost:3000", "http://127.0.0.1:3000"]
```

Nếu không có, thêm vào `backend/.env` và restart backend.

### 3. Check Browser Console

Mở Browser DevTools (F12) → Console tab:
- Xem có lỗi CORS không?
- Xem có network errors không?
- Check error messages từ API

### 4. Common Errors

#### "Network error: Could not reach server"
→ Backend không chạy hoặc URL sai
→ Check `VITE_API_URL` trong `frontend-web/.env`

#### "Username or email exists"
→ Username/email đã được sử dụng
→ Thử username/email khác

#### "Validation error"
→ Check:
- Username: 3-32 characters
- Email: valid email format
- Password: minimum 8 characters

#### CORS Error
→ Backend chưa config CORS
→ Thêm vào `backend/.env`:
```env
CORS_ORIGINS=["http://localhost:3000"]
```

### 5. Test API trực tiếp

Test với curl hoặc Postman:

```bash
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "email": "test@example.com",
    "password": "testpass123"
  }'
```

### 6. Check Database

Đảm bảo PostgreSQL đang chạy và database đã được tạo:

```bash
# Check connection
cd backend
python scripts/test_db_connection.py
```

### 7. Debug Steps

1. **Check Network Tab trong Browser:**
   - F12 → Network tab
   - Thử đăng ký
   - Xem request/response

2. **Check Backend Logs:**
   - Xem terminal chạy backend
   - Có error messages không?

3. **Check Frontend Console:**
   - F12 → Console
   - Xem error messages

4. **Test với test script:**
   ```bash
   cd backend
   python scripts/test_api.py
   ```

### 8. Quick Fix Checklist

- [ ] Backend đang chạy tại port 8000
- [ ] Frontend đang chạy tại port 3000
- [ ] CORS đã được config trong backend
- [ ] Database connection OK
- [ ] `.env` files đã được setup đúng
- [ ] Browser console không có errors
- [ ] Network tab shows request được gửi

