# 🚀 Quick Start Guide

## Start Backend và Frontend

### Bước 1: Start Backend (Terminal 1)

**Windows:**
```bash
cd backend
start.bat
```

**Linux/Mac/Git Bash:**
```bash
cd backend
bash start.sh
```

**Hoặc manual:**
```bash
cd backend
uvicorn app.main:app --reload
```

✅ **Verify:** Mở http://localhost:8000/docs trong browser

### Bước 2: Start Frontend (Terminal 2)

**Git Bash:**
```bash
cd frontend-web
unset NODE_OPTIONS
export PATH="/c/Program Files/nodejs:$PATH"
npm run dev
```

**Hoặc dùng script:**
```bash
cd frontend-web
bash fix_and_run.sh
```

✅ **Verify:** Mở http://localhost:3000 trong browser

## Test Registration

1. Mở http://localhost:3000
2. Click tab "Register"
3. Nhập thông tin:
   - Username: `testuser123` (3-32 characters)
   - Email: `test123@example.com`
   - Password: `testpass123` (minimum 8 characters)
4. Click "Register"

## Troubleshooting

### "Network error: Could not reach server"

**Nguyên nhân:** Backend chưa chạy

**Fix:**
1. Start backend (Bước 1 ở trên)
2. Verify: http://localhost:8000/health
3. Check browser console (F12) để xem error messages

### "Port 8000 already in use"

**Fix:**
- Kill process đang dùng port 8000
- Hoặc đổi port trong backend và update `VITE_API_URL` trong frontend

### "Database connection failed"

**Fix:**
- Check PostgreSQL đang chạy
- Check MongoDB đang chạy
- Verify database connection: `cd backend && python scripts/test_db_connection.py`

## Files Created

- `backend/start.bat` - Windows script để start backend
- `backend/start.sh` - Linux/Mac script để start backend
- `frontend-web/START_BACKEND.md` - Hướng dẫn start backend
- `frontend-web/QUICK_START_FULL.md` - Full setup guide
- `frontend-web/test_backend_connection.js` - Test script

