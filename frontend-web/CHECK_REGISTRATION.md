# Debug Registration Issues

## Quick Checklist

1. **Backend đang chạy?**
   ```bash
   cd backend
   uvicorn app.main:app --reload
   ```
   Check: http://localhost:8000/docs (Swagger UI)

2. **Frontend đang chạy?**
   ```bash
   cd frontend-web
   npm run dev
   ```
   Check: http://localhost:3000

3. **CORS configured?**
   - Backend tự động allow `http://localhost:3000` trong development
   - Nếu vẫn lỗi CORS, check browser console

4. **Database connected?**
   ```bash
   cd backend
   python scripts/test_db_connection.py
   ```

## Test Registration với Browser DevTools

1. **Mở Browser DevTools (F12)**
2. **Tab Console:** Xem error messages
3. **Tab Network:** 
   - Thử đăng ký
   - Tìm request `register`
   - Check:
     - Status code
     - Request payload
     - Response data

## Common Issues

### Issue 1: "Network error"
**Nguyên nhân:** Backend không chạy hoặc URL sai

**Fix:**
- Check backend đang chạy: `curl http://localhost:8000/health`
- Check `VITE_API_URL` trong `frontend-web/.env`

### Issue 2: "Username or email exists"
**Nguyên nhân:** Username/email đã được sử dụng

**Fix:**
- Thử username/email khác
- Hoặc xóa user cũ trong database

### Issue 3: "Validation error"
**Nguyên nhân:** Input không đúng format

**Fix:**
- Username: 3-32 characters
- Email: valid email format
- Password: minimum 8 characters

### Issue 4: CORS Error
**Nguyên nhân:** Backend chưa allow frontend origin

**Fix:**
- Backend đã tự động allow `localhost:3000` trong development
- Nếu vẫn lỗi, check backend logs

## Manual Test với curl

```bash
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser123",
    "email": "test123@example.com",
    "password": "testpass123"
  }'
```

Expected response:
```json
{
  "user_id": "...",
  "token": {
    "access_token": "...",
    "refresh_token": "...",
    "token_type": "bearer"
  }
}
```

## Debug trong Frontend

Thêm console.log để debug:

```javascript
// In LoginDialog.jsx handleRegister
console.log('Registering:', { username, email, password: '***' })
console.log('API URL:', import.meta.env.VITE_API_URL)
```

