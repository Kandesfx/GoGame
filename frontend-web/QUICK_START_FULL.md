# Quick Start - Full Setup

## Bước 1: Start Backend

**Mở Terminal 1:**

```bash
cd backend

# Activate virtual environment (nếu có)
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Start server
uvicorn app.main:app --reload
```

**Verify:** Mở http://localhost:8000/docs trong browser

## Bước 2: Start Frontend

**Mở Terminal 2 (mới):**

```bash
cd frontend-web

# Fix Node.js PATH (nếu dùng Git Bash)
unset NODE_OPTIONS
export PATH="/c/Program Files/nodejs:$PATH"

# Install dependencies (lần đầu)
npm install

# Start dev server
npm run dev
```

**Verify:** Mở http://localhost:3000 trong browser

## Bước 3: Test Registration

1. Mở http://localhost:3000
2. Click tab "Register"
3. Nhập:
   - Username: `testuser123` (3-32 chars)
   - Email: `test123@example.com`
   - Password: `testpass123` (min 8 chars)
4. Click "Register"

## Nếu gặp "Network error"

### Check 1: Backend có đang chạy?

```bash
curl http://localhost:8000/health
```

Nếu không response → Backend chưa start

### Check 2: API URL đúng chưa?

File: `frontend-web/.env`
```env
VITE_API_URL=http://localhost:8000
```

### Check 3: Browser Console

F12 → Console tab → Xem error messages

### Check 4: Network Tab

F12 → Network tab → Tìm request `register` → Check status code

## Common Issues

### "Backend not running"
→ Start backend: `cd backend && uvicorn app.main:app --reload`

### "Port 8000 already in use"
→ Kill process hoặc đổi port

### "Database connection failed"
→ Check PostgreSQL đang chạy và database đã được tạo

### "CORS error"
→ Backend đã tự động allow localhost:3000, nếu vẫn lỗi check backend logs

