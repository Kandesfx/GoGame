# 🚀 Hướng Dẫn Deploy Dự Án GoGame Online

## 📋 Tổng Quan

Dự án GoGame bao gồm:
- **Backend**: FastAPI (Python) - Port 8000
- **Frontend**: React (Vite) - Port 3000/5173
- **Database**: PostgreSQL + MongoDB
- **AI Engine**: C++ module (gogame_py) - Optional

## 🎯 Các Phương Án Deploy

### Option 1: VPS/Cloud Server (Khuyến nghị)

**Platforms:**
- **DigitalOcean**: $6-12/tháng (Droplet)
- **AWS EC2**: Pay-as-you-go
- **Google Cloud Compute**: Free tier available
- **Azure VM**: Free tier available
- **Vultr**: $6/tháng
- **Linode**: $5/tháng

**Ưu điểm:**
- Full control
- Có thể cài đặt mọi thứ
- Phù hợp cho production

### Option 2: Platform as a Service (PaaS)

**Backend:**
- **Railway**: Dễ dùng, $5/tháng
- **Render**: Free tier available
- **Fly.io**: Free tier (xem [DEPLOY_FLYIO.md](DEPLOY_FLYIO.md) để biết chi tiết)
- **Heroku**: $7/tháng (không còn free tier)

**Frontend:**
- **Vercel**: Free tier, tự động deploy từ Git
- **Netlify**: Free tier
- **Cloudflare Pages**: Free tier

**Ưu điểm:**
- Dễ deploy
- Tự động CI/CD
- Không cần quản lý server

### Option 3: Docker + Cloud

**Platforms:**
- **AWS ECS/Fargate**
- **Google Cloud Run**
- **Azure Container Instances**
- **DigitalOcean App Platform**

**Ưu điểm:**
- Scalable
- Containerized
- Dễ quản lý

---

## 📦 Option 1: Deploy trên VPS (Chi Tiết)

### Bước 1: Chuẩn Bị Server

#### 1.1. Tạo VPS

**DigitalOcean Example:**
1. Tạo Droplet: Ubuntu 22.04 LTS
2. Chọn plan: $12/tháng (2GB RAM, 1 vCPU) - đủ cho development
3. Chọn datacenter gần bạn nhất
4. Add SSH key hoặc password

#### 1.2. Kết Nối Server

```bash
ssh root@your-server-ip
```

#### 1.3. Cập Nhật System

```bash
apt update && apt upgrade -y
```

### Bước 2: Cài Đặt Dependencies

#### 2.1. Cài Python 3.10+

```bash
apt install -y python3.10 python3.10-venv python3-pip
```

#### 2.2. Cài Node.js 18+

```bash
curl -fsSL https://deb.nodesource.com/setup_18.x | bash -
apt install -y nodejs
```

#### 2.3. Cài PostgreSQL

```bash
apt install -y postgresql postgresql-contrib
systemctl start postgresql
systemctl enable postgresql
```

#### 2.4. Cài MongoDB

```bash
curl -fsSL https://www.mongodb.org/static/pgp/server-6.0.asc | gpg -o /usr/share/keyrings/mongodb-server-6.0.gpg --dearmor
echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-6.0.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/6.0 multiverse" | tee /etc/apt/sources.list.d/mongodb-org-6.0.list
apt update
apt install -y mongodb-org
systemctl start mongod
systemctl enable mongod
```

#### 2.5. Cài Nginx (Reverse Proxy)

```bash
apt install -y nginx
systemctl start nginx
systemctl enable nginx
```

#### 2.6. Cài PM2 (Process Manager cho Node.js)

```bash
npm install -g pm2
```

### Bước 3: Setup Database

#### 3.1. PostgreSQL

```bash
sudo -u postgres psql
```

Trong PostgreSQL shell:
```sql
CREATE DATABASE gogame;
CREATE USER gogame_user WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE gogame TO gogame_user;
\q
```

#### 3.2. MongoDB

MongoDB đã chạy, không cần setup thêm (mặc định không có authentication).

### Bước 4: Deploy Backend

#### 4.1. Clone Repository

```bash
cd /opt
git clone https://github.com/Kandesfx/GoGame.git
cd GoGame/backend
```

#### 4.2. Tạo Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

#### 4.3. Cài Dependencies

```bash
pip install -r requirements.txt
```

#### 4.4. Cấu Hình Environment

```bash
cp env.example .env
nano .env
```

Cập nhật `.env`:
```env
POSTGRES_DSN=postgresql+psycopg://gogame_user:your_secure_password@localhost:5432/gogame
MONGO_DSN=mongodb://localhost:27017
JWT_SECRET_KEY=your_very_long_random_secret_key_here_min_32_chars
DEBUG=false
```

#### 4.5. Chạy Migrations

```bash
alembic upgrade head
```

#### 4.6. Test Backend

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Kiểm tra: `curl http://localhost:8000/health`

#### 4.7. Tạo Systemd Service

```bash
sudo nano /etc/systemd/system/gogame-backend.service
```

Nội dung:
```ini
[Unit]
Description=GoGame Backend API
After=network.target postgresql.service mongod.service

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/GoGame/backend
Environment="PATH=/opt/GoGame/backend/venv/bin"
ExecStart=/opt/GoGame/backend/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable và start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable gogame-backend
sudo systemctl start gogame-backend
sudo systemctl status gogame-backend
```

### Bước 5: Deploy Frontend

#### 5.1. Build Frontend

```bash
cd /opt/GoGame/frontend-web
npm install
npm run build
```

#### 5.2. Cấu Hình Environment

Tạo file `.env.production`:
```env
VITE_API_URL=https://api.yourdomain.com
```

Build lại:
```bash
npm run build
```

#### 5.3. Serve với Nginx

```bash
sudo nano /etc/nginx/sites-available/gogame
```

Nội dung:
```nginx
server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;

    # Frontend
    location / {
        root /opt/GoGame/frontend-web/dist;
        try_files $uri $uri/ /index.html;
    }

    # Backend API
    location /api {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }

    # WebSocket support (nếu có)
    location /ws {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

Enable site:
```bash
sudo ln -s /etc/nginx/sites-available/gogame /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### Bước 6: SSL/HTTPS với Let's Encrypt

#### 6.1. Cài Certbot

```bash
apt install -y certbot python3-certbot-nginx
```

#### 6.2. Cấu Hình Domain

Đảm bảo domain đã trỏ về IP server:
- A record: `yourdomain.com` → `your-server-ip`
- A record: `www.yourdomain.com` → `your-server-ip`

#### 6.3. Lấy SSL Certificate

```bash
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com
```

Certbot sẽ tự động cấu hình Nginx với HTTPS.

#### 6.4. Auto Renewal

```bash
sudo certbot renew --dry-run
```

### Bước 7: Firewall

```bash
ufw allow 22/tcp    # SSH
ufw allow 80/tcp    # HTTP
ufw allow 443/tcp   # HTTPS
ufw enable
```

---

## 📦 Option 2: Deploy với Railway (Dễ Dàng)

### Backend trên Railway

#### 1. Tạo Account

1. Truy cập: https://railway.app
2. Đăng ký với GitHub

#### 2. Tạo Project

1. Click "New Project"
2. Chọn "Deploy from GitHub repo"
3. Chọn repository GoGame

#### 3. Cấu Hình Backend Service

1. Add service: "Backend"
2. Root directory: `backend`
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

#### 4. Environment Variables

Thêm trong Railway dashboard:
```
POSTGRES_DSN=postgresql+psycopg://...
MONGO_DSN=mongodb://...
JWT_SECRET_KEY=...
```

#### 5. Add PostgreSQL Database

1. Click "New" → "Database" → "PostgreSQL"
2. Railway tự động tạo và inject `DATABASE_URL`
3. Update `POSTGRES_DSN` trong env vars

#### 6. Deploy

Railway tự động deploy khi push code lên GitHub.

### Frontend trên Vercel

#### 1. Tạo Account

1. Truy cập: https://vercel.com
2. Đăng ký với GitHub

#### 2. Import Project

1. Click "Add New" → "Project"
2. Import từ GitHub repository
3. Root directory: `frontend-web`

#### 3. Cấu Hình Build

- Framework Preset: Vite
- Build Command: `npm run build`
- Output Directory: `dist`

#### 4. Environment Variables

```
VITE_API_URL=https://your-backend.railway.app
```

#### 5. Deploy

Vercel tự động deploy khi push code.

---

## 📦 Option 3: Docker Deployment

### Tạo Dockerfile cho Backend

```dockerfile
# backend/Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Expose port
EXPOSE 8000

# Run
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Tạo Dockerfile cho Frontend

```dockerfile
# frontend-web/Dockerfile
FROM node:18-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  postgres:
    image: postgres:14
    environment:
      POSTGRES_DB: gogame
      POSTGRES_USER: gogame_user
      POSTGRES_PASSWORD: your_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  mongodb:
    image: mongo:6
    volumes:
      - mongo_data:/data/db
    ports:
      - "27017:27017"

  backend:
    build: ./backend
    environment:
      POSTGRES_DSN: postgresql+psycopg://gogame_user:your_password@postgres:5432/gogame
      MONGO_DSN: mongodb://mongodb:27017
      JWT_SECRET_KEY: your_secret_key
    ports:
      - "8000:8000"
    depends_on:
      - postgres
      - mongodb

  frontend:
    build: ./frontend-web
    ports:
      - "80:80"
    depends_on:
      - backend

volumes:
  postgres_data:
  mongo_data:
```

Deploy:
```bash
docker-compose up -d
```

---

## 🔧 Cấu Hình Quan Trọng

### Environment Variables

**Backend (.env):**
```env
# Database
POSTGRES_DSN=postgresql+psycopg://user:pass@host:5432/dbname
MONGO_DSN=mongodb://host:27017

# Security
JWT_SECRET_KEY=your_very_long_random_secret_key_min_32_chars
DEBUG=false

# CORS (nếu frontend ở domain khác)
CORS_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
```

**Frontend (.env.production):**
```env
VITE_API_URL=https://api.yourdomain.com
```

### Security Checklist

- [ ] Đổi tất cả default passwords
- [ ] Sử dụng HTTPS (SSL/TLS)
- [ ] Cấu hình CORS đúng
- [ ] Enable firewall
- [ ] Disable debug mode trong production
- [ ] Sử dụng strong JWT secret key
- [ ] Backup database định kỳ
- [ ] Monitor logs
- [ ] Update dependencies thường xuyên

---

## 📊 Monitoring & Maintenance

### Logs

**Backend logs:**
```bash
sudo journalctl -u gogame-backend -f
```

**Nginx logs:**
```bash
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

### Backup Database

**PostgreSQL:**
```bash
pg_dump -U gogame_user gogame > backup_$(date +%Y%m%d).sql
```

**MongoDB:**
```bash
mongodump --out /backup/mongodb_$(date +%Y%m%d)
```

### Update Application

```bash
cd /opt/GoGame
git pull origin master
cd backend
source venv/bin/activate
pip install -r requirements.txt
alembic upgrade head
sudo systemctl restart gogame-backend
```

---

## 🚨 Troubleshooting

### Backend không start

```bash
# Check logs
sudo journalctl -u gogame-backend -n 50

# Check database connection
cd backend
python scripts/test_db_connection.py
```

### Frontend không load

```bash
# Check Nginx
sudo nginx -t
sudo systemctl status nginx

# Check build
ls -la /opt/GoGame/frontend-web/dist
```

### Database connection failed

- Kiểm tra PostgreSQL/MongoDB đang chạy
- Kiểm tra firewall rules
- Kiểm tra connection string trong .env

---

## 📚 Tài Liệu Tham Khảo

- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [Vite Production Build](https://vitejs.dev/guide/build.html)
- [Nginx Configuration](https://nginx.org/en/docs/)
- [Let's Encrypt](https://letsencrypt.org/)

---

## ✅ Checklist Deploy

- [ ] Server/VPS đã setup
- [ ] Dependencies đã cài (Python, Node.js, PostgreSQL, MongoDB)
- [ ] Database đã tạo và migrate
- [ ] Backend đã deploy và chạy
- [ ] Frontend đã build và serve
- [ ] Nginx đã cấu hình
- [ ] SSL/HTTPS đã setup
- [ ] Domain đã trỏ về server
- [ ] Firewall đã cấu hình
- [ ] Monitoring đã setup
- [ ] Backup strategy đã có

---

**Chúc bạn deploy thành công! 🚀**

