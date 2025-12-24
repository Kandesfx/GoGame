# 🚀 Quick Start Guide - GoGame Backend

Hướng dẫn nhanh để chạy backend trên local và server.

## 📋 Yêu cầu

- **Python 3.10+** (kiểm tra: `python --version`)
- **PostgreSQL 14+** (hoặc sử dụng Docker)
- **MongoDB 6+** (tùy chọn, cho AI features)

## 🏃 Cách 1: Setup Tự Động (Khuyến nghị)

### Windows

```bash
cd backend
setup.bat
```

Sau khi setup xong:
```bash
run.bat
```

### Linux/Mac

```bash
cd backend
chmod +x setup.sh run.sh
./setup.sh
```

Sau khi setup xong:
```bash
./run.sh
```

## 🔧 Cách 2: Setup Thủ Công

### 1. Tạo Virtual Environment

```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 2. Cài Đặt Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Cấu Hình Environment

```bash
# Copy file env.example thành .env
# Windows
copy env.example .env

# Linux/Mac
cp env.example .env
```

Chỉnh sửa `.env` với các thông tin của bạn:

```env
# PostgreSQL
POSTGRES_DSN=postgresql+psycopg://postgres:password@localhost:5432/gogame

# MongoDB (tùy chọn)
MONGO_DSN=mongodb://localhost:27017
MONGO_DATABASE=gogame

# JWT Secret (tạo random string dài ít nhất 32 ký tự)
JWT_SECRET_KEY=your-secret-key-here-min-32-chars-long
JWT_REFRESH_SECRET_KEY=your-refresh-secret-key-here-min-32-chars-long
```

### 4. Setup Database

#### Option A: Sử dụng Script Tự Động

```bash
python scripts/setup_database.py
```

#### Option B: Manual

1. Tạo database PostgreSQL:
```sql
CREATE DATABASE gogame;
```

2. Chạy migrations:
```bash
alembic upgrade head
```

### 5. Build C++ AI Engine (Tùy chọn - chỉ cần nếu muốn dùng AI)

Nếu muốn sử dụng AI features, cần build module `gogame_py`:

**Xem hướng dẫn chi tiết**: [`README_GOGAME_PY.md`](../README_GOGAME_PY.md)

**Tóm tắt nhanh:**
```bash
# Trong MSYS2 MinGW 64-bit shell
cd build
cmake .. -G "Ninja" -DCMAKE_BUILD_TYPE=Release
cmake --build . --target gogame_py
```

**⚠️ Lưu ý**: Module phải được build và đặt trong thư mục `build/` để backend nhận diện đúng.

### 6. Chạy Server

```bash
# Cách 1: Sử dụng script
run.bat  # Windows
./run.sh # Linux/Mac

# Cách 2: Chạy trực tiếp
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## ✅ Kiểm Tra

Sau khi server chạy, mở browser:

- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **API Base**: http://localhost:8000

## 🐳 Chạy với Docker (Tùy chọn)

Nếu bạn muốn chạy PostgreSQL và MongoDB bằng Docker:

```bash
# Từ thư mục root
docker-compose up -d
```

Sau đó cấu hình `.env` để kết nối với Docker containers.

## 🔍 Troubleshooting

### Lỗi: `ModuleNotFoundError: No module named 'uvicorn'`

**Giải pháp:**
1. Đảm bảo virtual environment đã được activate
2. Chạy: `pip install -r requirements.txt`
3. Hoặc chạy `setup.bat` / `setup.sh` để setup tự động

### Lỗi: `Could not connect to database`

**Giải pháp:**
1. Kiểm tra PostgreSQL đang chạy: `pg_isready` hoặc kiểm tra service
2. Kiểm tra connection string trong `.env`
3. Đảm bảo database `gogame` đã được tạo

### Lỗi: `alembic: command not found`

**Giải pháp:**
```bash
# Đảm bảo virtual environment đã activate
pip install alembic
# Hoặc
pip install -r requirements.txt
```

### Lỗi khi chạy trên Server (Production)

**Giải pháp:**
1. Sử dụng production server (không dùng `--reload`):
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

2. Hoặc sử dụng Gunicorn:
```bash
pip install gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

3. Sử dụng reverse proxy (Nginx) cho production

## 📚 Tài Liệu Thêm

- [README.md](README.md) - Tài liệu chi tiết
- [README_GOGAME_PY.md](../README_GOGAME_PY.md) - Hướng dẫn build C++ AI module
- [scripts/README.md](scripts/README.md) - Hướng dẫn scripts
- [docs/](../docs/) - Tài liệu deployment

## 💡 Tips

- **Development**: Sử dụng `--reload` để auto-reload khi code thay đổi
- **Production**: Không dùng `--reload`, sử dụng multiple workers
- **Environment Variables**: Luôn sử dụng `.env` file, không commit vào Git
- **Database Migrations**: Chạy `alembic upgrade head` sau mỗi lần pull code mới

