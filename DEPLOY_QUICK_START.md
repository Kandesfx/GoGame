# 🚀 Quick Start - Deploy GoGame Online

## Option 1: Docker Compose (Dễ nhất)

### Bước 1: Chuẩn bị

```bash
# Clone repository
git clone https://github.com/Kandesfx/GoGame.git
cd GoGame
```

### Bước 2: Cấu hình Environment

Tạo file `.env`:
```env
POSTGRES_PASSWORD=your_secure_password
JWT_SECRET_KEY=your_very_long_random_secret_key_min_32_chars
JWT_REFRESH_SECRET_KEY=your_refresh_secret_key
CORS_ORIGINS=http://localhost:3000,http://localhost:5173
VITE_API_URL=http://localhost:8000
```

### Bước 3: Deploy

```bash
# Build và start tất cả services
docker-compose up -d

# Xem logs
docker-compose logs -f

# Kiểm tra status
docker-compose ps
```

### Bước 4: Chạy Migrations

```bash
# Vào container backend
docker-compose exec backend bash

# Chạy migrations
alembic upgrade head

# Exit
exit
```

### Bước 5: Kiểm tra

- Backend: http://localhost:8000/docs
- Frontend: http://localhost

---

## Option 2: Railway (Không cần server)

### Backend

1. Truy cập: https://railway.app
2. Đăng ký với GitHub
3. New Project → Deploy from GitHub
4. Chọn repository GoGame
5. Add service: Backend
   - Root directory: `backend`
   - Build: `pip install -r requirements.txt`
   - Start: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
6. Add PostgreSQL database
7. Environment variables:
   ```
   POSTGRES_DSN=${{Postgres.DATABASE_URL}}
   MONGO_DSN=mongodb://... (hoặc dùng MongoDB Atlas)
   JWT_SECRET_KEY=...
   ```

### Frontend

1. Truy cập: https://vercel.com
2. Import project từ GitHub
3. Root directory: `frontend-web`
4. Build command: `npm run build`
5. Environment: `VITE_API_URL=https://your-backend.railway.app`

---

## Option 3: VPS với Script Tự Động

Xem chi tiết trong `docs/DEPLOYMENT_GUIDE.md`

---

## 🔧 Troubleshooting

### Docker không start

```bash
# Check logs
docker-compose logs backend
docker-compose logs frontend

# Restart
docker-compose restart
```

### Database connection failed

```bash
# Check database đang chạy
docker-compose ps

# Test connection
docker-compose exec backend python -c "from app.database import get_db; next(get_db())"
```

### Frontend không load

- Kiểm tra `VITE_API_URL` trong `.env`
- Rebuild frontend: `docker-compose build frontend`

---

Xem hướng dẫn chi tiết: `docs/DEPLOYMENT_GUIDE.md`

