# 🚀 Quick Start - Deploy trên Fly.io

## Bước 1: Cài Fly CLI

**Windows:**
```powershell
iwr https://fly.io/install.ps1 -useb | iex
```

**Linux/Mac:**
```bash
curl -L https://fly.io/install.sh | sh
```

## Bước 2: Đăng Nhập

```bash
fly auth login
```

## Bước 3: Deploy Backend

```bash
cd backend

# Tạo app (chọn Yes cho PostgreSQL)
fly launch

# Set secrets
fly secrets set JWT_SECRET_KEY="your_very_long_random_secret_key"
fly secrets set JWT_REFRESH_SECRET_KEY="your_refresh_secret_key"
fly secrets set MONGO_DSN="mongodb+srv://user:pass@cluster.mongodb.net/gogame"
fly secrets set CORS_ORIGINS="https://gogame-frontend.fly.dev"

# Deploy
fly deploy

# Lấy URL backend
fly status
# Ghi lại URL: https://gogame-backend.fly.dev
```

## Bước 4: Deploy Frontend

```bash
cd frontend-web

# Tạo app
fly launch

# Cập nhật fly.toml - thay VITE_API_URL bằng backend URL
# Trong file fly.toml, tìm dòng:
# build_args = { VITE_API_URL = "https://gogame-backend.fly.dev" }

# Deploy
fly deploy
```

## Bước 5: Kiểm Tra

- Backend: `https://gogame-backend.fly.dev/docs`
- Frontend: `https://gogame-frontend.fly.dev`

## Troubleshooting

```bash
# Xem logs
fly logs -a gogame-backend
fly logs -a gogame-frontend

# SSH vào container
fly ssh console -a gogame-backend

# Check secrets
fly secrets list -a gogame-backend
```

Xem hướng dẫn chi tiết: `docs/DEPLOY_FLYIO.md`

