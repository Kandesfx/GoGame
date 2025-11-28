# Start Backend Server

## Quick Start

### 1. Start Backend (Terminal 1)

```bash
cd backend
uvicorn app.main:app --reload
```

Backend sẽ chạy tại: **http://localhost:8000**

### 2. Verify Backend Running

Mở browser: http://localhost:8000/docs (Swagger UI)

Hoặc test với curl:
```bash
curl http://localhost:8000/health
```

Expected response: `{"status":"ok"}`

### 3. Start Frontend (Terminal 2)

```bash
cd frontend-web
unset NODE_OPTIONS
export PATH="/c/Program Files/nodejs:$PATH"
npm run dev
```

Frontend sẽ chạy tại: **http://localhost:3000**

## Troubleshooting

### Backend không start

**Check PostgreSQL:**
```bash
cd backend
python scripts/test_db_connection.py
```

**Check MongoDB:**
- Đảm bảo MongoDB đang chạy
- Default: `mongodb://localhost:27017`

**Check dependencies:**
```bash
cd backend
pip install -r requirements.txt
```

### Port 8000 đã được sử dụng

Thay đổi port trong backend:
```bash
uvicorn app.main:app --reload --port 8001
```

Và update `frontend-web/.env`:
```env
VITE_API_URL=http://localhost:8001
```

### Network Error

1. **Check backend đang chạy:**
   ```bash
   curl http://localhost:8000/health
   ```

2. **Check API URL trong frontend:**
   - File: `frontend-web/.env`
   - Should be: `VITE_API_URL=http://localhost:8000`

3. **Check browser console:**
   - F12 → Console
   - Xem có CORS errors không?

4. **Check backend logs:**
   - Xem terminal chạy backend
   - Có error messages không?

## Quick Test

```bash
# Terminal 1: Start backend
cd backend
uvicorn app.main:app --reload

# Terminal 2: Test API
curl http://localhost:8000/health

# Terminal 3: Start frontend
cd frontend-web
npm run dev
```

## Verify Setup

1. Backend: http://localhost:8000/docs ✅
2. Frontend: http://localhost:3000 ✅
3. Test registration trong frontend ✅

