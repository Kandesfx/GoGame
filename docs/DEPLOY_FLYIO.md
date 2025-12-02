# 🚀 Deploy GoGame trên Fly.io

Fly.io là platform dễ dùng để deploy ứng dụng với Docker containers, có free tier và scaling tự động.

## 📋 Tổng Quan

Fly.io sẽ deploy:
- **Backend**: FastAPI trên port 8000
- **Frontend**: React build với Nginx
- **PostgreSQL**: Fly Postgres (managed)
- **MongoDB**: Có thể dùng MongoDB Atlas hoặc Fly volume

## 🎯 Bước 1: Cài Đặt Fly CLI

### Windows

```bash
# Dùng PowerShell
iwr https://fly.io/install.ps1 -useb | iex
```

### Linux/Mac

```bash
curl -L https://fly.io/install.sh | sh
```

### Verify

```bash
fly version
```

## 🎯 Bước 2: Đăng Nhập Fly.io

```bash
fly auth login
```

Mở browser và đăng nhập với GitHub/Email.

## 🎯 Bước 3: Deploy Backend

### 3.1. Tạo Fly App cho Backend

```bash
cd backend
fly launch
```

Fly sẽ hỏi:
- App name: `gogame-backend` (hoặc tên bạn muốn)
- Region: Chọn region gần bạn (ví dụ: `sin` cho Singapore, `iad` cho US East)
- PostgreSQL: Chọn "Yes" để tạo database
- MongoDB: Chọn "No" (sẽ dùng MongoDB Atlas)

### 3.2. Cấu Hình Environment Variables

```bash
# Set JWT secret
fly secrets set JWT_SECRET_KEY="your_very_long_random_secret_key_min_32_chars"
fly secrets set JWT_REFRESH_SECRET_KEY="your_refresh_secret_key"

# Set MongoDB (nếu dùng Atlas)
fly secrets set MONGO_DSN="mongodb+srv://user:pass@cluster.mongodb.net/gogame"

# Set CORS origins
fly secrets set CORS_ORIGINS="https://your-frontend.fly.dev,https://yourdomain.com"
```

### 3.3. Cấu Hình Database Connection

Fly tự động tạo PostgreSQL và inject `DATABASE_URL`. Cập nhật `fly.toml`:

```toml
[env]
  POSTGRES_DSN = "${DATABASE_URL}"
```

Hoặc trong code, sử dụng `DATABASE_URL` trực tiếp.

### 3.4. Deploy Backend

```bash
fly deploy
```

### 3.5. Chạy Migrations

```bash
# SSH vào container
fly ssh console

# Trong container
cd /app
alembic upgrade head

# Exit
exit
```

Hoặc tạo release command trong `fly.toml`:

```toml
[deploy]
  release_command = "alembic upgrade head"
```

### 3.6. Kiểm Tra Backend

```bash
# Xem logs
fly logs

# Check status
fly status

# Open app
fly open
```

Backend sẽ có URL: `https://gogame-backend.fly.dev`

## 🎯 Bước 4: Setup MongoDB Atlas (Khuyến nghị)

### 4.1. Tạo MongoDB Atlas Account

1. Truy cập: https://www.mongodb.com/cloud/atlas
2. Đăng ký free tier
3. Tạo cluster (chọn region gần Fly.io region)

### 4.2. Tạo Database User

1. Database Access → Add New Database User
2. Username/Password
3. Network Access → Add IP Address → "Allow Access from Anywhere" (0.0.0.0/0)

### 4.3. Lấy Connection String

1. Clusters → Connect → Connect your application
2. Copy connection string
3. Update password: `mongodb+srv://username:password@cluster.mongodb.net/gogame`

### 4.4. Set MongoDB DSN trong Fly.io

```bash
cd backend
fly secrets set MONGO_DSN="mongodb+srv://username:password@cluster.mongodb.net/gogame"
```

## 🎯 Bước 5: Deploy Frontend

### 5.1. Tạo Fly App cho Frontend

```bash
cd frontend-web
fly launch
```

- App name: `gogame-frontend`
- Region: Cùng region với backend
- PostgreSQL: "No"
- MongoDB: "No"

### 5.2. Cấu Hình Environment Variables

```bash
# Set backend URL
fly secrets set VITE_API_URL="https://gogame-backend.fly.dev"
```

**Lưu ý**: Vite environment variables cần được set lúc build time, không phải runtime. Cần cập nhật Dockerfile.

### 5.3. Cập Nhật Dockerfile cho Frontend

File `frontend-web/Dockerfile` cần build với env vars:

```dockerfile
# Build stage
FROM node:18-alpine AS builder

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .

# Build với environment variable
ARG VITE_API_URL
ENV VITE_API_URL=$VITE_API_URL
RUN npm run build

# Production stage
FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

### 5.4. Cập Nhật fly.toml

```toml
[build]
  build_args = { VITE_API_URL = "https://gogame-backend.fly.dev" }
```

### 5.5. Deploy Frontend

```bash
fly deploy
```

### 5.6. Kiểm Tra Frontend

```bash
fly open
```

Frontend sẽ có URL: `https://gogame-frontend.fly.dev`

## 🎯 Bước 6: Cấu Hình Custom Domain (Tùy chọn)

### 6.1. Add Domain

```bash
# Backend
cd backend
fly certs add api.yourdomain.com

# Frontend
cd frontend-web
fly certs add yourdomain.com
fly certs add www.yourdomain.com
```

### 6.2. Cấu Hình DNS

Thêm CNAME records:
- `api.yourdomain.com` → `gogame-backend.fly.dev`
- `yourdomain.com` → `gogame-frontend.fly.dev`
- `www.yourdomain.com` → `gogame-frontend.fly.dev`

### 6.3. Update Environment Variables

```bash
# Backend
cd backend
fly secrets set CORS_ORIGINS="https://yourdomain.com,https://www.yourdomain.com"

# Frontend
cd frontend-web
fly secrets set VITE_API_URL="https://api.yourdomain.com"
fly deploy
```

## 🔧 Cấu Hình Nâng Cao

### Scale Backend

```bash
cd backend
fly scale count 2  # 2 instances
fly scale vm shared-cpu-1x  # CPU size
fly scale memory 512  # RAM in MB
```

### Persistent Volumes (nếu cần)

```bash
# Tạo volume
fly volumes create data --size 10 --region sin

# Mount trong fly.toml
[mounts]
  source = "data"
  destination = "/data"
```

### Health Checks

Fly.io tự động health check với endpoint `/health`. Đảm bảo backend có endpoint này.

### Monitoring

```bash
# Xem metrics
fly metrics

# Xem logs real-time
fly logs

# SSH vào container
fly ssh console
```

## 📝 File Cấu Hình Mẫu

### backend/fly.toml

```toml
app = "gogame-backend"
primary_region = "sin"

[build]
  dockerfile = "Dockerfile"

[env]
  POSTGRES_DSN = "${DATABASE_URL}"
  DEBUG = "false"

[http_service]
  internal_port = 8000
  force_https = true
  auto_stop_machines = true
  auto_start_machines = true
  min_machines_running = 1
  processes = ["app"]

  [[http_service.checks]]
    grace_period = "10s"
    interval = "30s"
    method = "GET"
    timeout = "5s"
    path = "/health"

[deploy]
  release_command = "alembic upgrade head"
```

### frontend-web/fly.toml

```toml
app = "gogame-frontend"
primary_region = "sin"

[build]
  dockerfile = "Dockerfile"
  build_args = { VITE_API_URL = "https://gogame-backend.fly.dev" }

[http_service]
  internal_port = 80
  force_https = true
  auto_stop_machines = true
  auto_start_machines = true
  min_machines_running = 1

  [[http_service.checks]]
    grace_period = "10s"
    interval = "30s"
    method = "GET"
    timeout = "5s"
    path = "/"
```

## 🚨 Troubleshooting

### Backend không start

```bash
# Xem logs
fly logs -a gogame-backend

# SSH vào container
fly ssh console -a gogame-backend

# Check environment variables
fly secrets list -a gogame-backend
```

### Database connection failed

```bash
# Check DATABASE_URL
fly secrets list -a gogame-backend

# Test connection trong container
fly ssh console -a gogame-backend
python -c "from app.database import get_db; next(get_db())"
```

### Frontend không kết nối backend

- Kiểm tra `VITE_API_URL` trong build args
- Kiểm tra CORS settings trong backend
- Kiểm tra network connectivity

### Build failed

```bash
# Xem build logs
fly logs -a gogame-frontend

# Rebuild
fly deploy --build-only
```

## 💰 Pricing

**Free Tier:**
- 3 shared-cpu-1x VMs
- 3GB persistent volume storage
- 160GB outbound data transfer

**Paid:**
- $1.94/month per shared-cpu-1x VM
- $0.15/GB per month for volumes
- $0.02/GB for outbound data

## ✅ Checklist

- [ ] Fly CLI đã cài
- [ ] Đã đăng nhập Fly.io
- [ ] Backend app đã tạo
- [ ] PostgreSQL đã setup
- [ ] MongoDB Atlas đã setup
- [ ] Environment variables đã set
- [ ] Backend đã deploy
- [ ] Migrations đã chạy
- [ ] Frontend app đã tạo
- [ ] Frontend đã deploy
- [ ] Custom domain đã setup (nếu có)
- [ ] SSL certificates đã có
- [ ] Health checks đang hoạt động

---

**Chúc bạn deploy thành công trên Fly.io! 🚀**

