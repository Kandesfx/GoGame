# Quick Start Guide

## Bước 1: Cài Node.js (Nếu chưa có)

### Windows:
1. Download từ: https://nodejs.org/
2. Chọn LTS version
3. Chạy installer
4. **Restart terminal** sau khi cài

### Verify:
```bash
node --version
npm --version
```

## Bước 2: Setup Project

```bash
# Vào thư mục frontend-web
cd frontend-web

# Cài dependencies
npm install

# Tạo file .env
echo "VITE_API_URL=http://localhost:8000" > .env
```

## Bước 3: Chạy Development Server

```bash
npm run dev
```

Mở browser: **http://localhost:3000**

## Bước 4: Start Backend (Terminal khác)

```bash
cd backend
uvicorn app.main:app --reload
```

## Test

1. Mở http://localhost:3000
2. Register/Login
3. Tạo match
4. Chơi game!

## Troubleshooting

**npm: command not found**
→ Cần cài Node.js (xem Bước 1)

**Port 3000 already in use**
→ Thay đổi port trong `vite.config.js`

**CORS errors**
→ Check backend CORS config trong `.env`

