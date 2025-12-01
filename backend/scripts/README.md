# Database Setup Scripts

Các script để tự động setup database cho GoGame backend.

## Các script có sẵn

### 1. `setup_database.py` (Khuyến nghị - Cross-platform)

Script Python tự động:
- Đọc cấu hình từ `.env`
- Tạo database nếu chưa tồn tại
- Tạo user nếu cần
- Chạy migrations
- Kiểm tra kết nối

**Sử dụng:**
```bash
# Từ thư mục backend
python scripts/setup_database.py

# Hoặc
python -m scripts.setup_database
```

**Yêu cầu:**
```bash
pip install python-dotenv psycopg[binary]
```

### 2. `setup_database.sh` (Linux/Mac)

Script bash cho Linux và macOS.

**Sử dụng:**
```bash
# Từ thư mục backend
chmod +x scripts/setup_database.sh
./scripts/setup_database.sh
```

**Yêu cầu:**
- PostgreSQL client (`psql`)
- Python và Alembic

### 3. `setup_database.bat` (Windows)

Script batch cho Windows (gọi Python script).

**Sử dụng:**
```bash
# Từ thư mục backend
scripts\setup_database.bat
```

### 4. `create_database.sql` (Manual)

SQL script để tạo database thủ công.

**Sử dụng:**
```bash
# Với quyền superuser
psql -U postgres -f scripts/create_database.sql

# Hoặc chạy từ psql shell
psql -U postgres
\i scripts/create_database.sql
```

## Quy trình setup

### Bước 1: Cấu hình .env

Đảm bảo file `.env` có thông tin đúng:

```env
POSTGRES_DSN=postgresql+psycopg://postgres:postgres@localhost:5432/gogame
```

### Bước 2: Chạy script

```bash
# Cách 1: Python script (khuyến nghị)
python scripts/setup_database.py

# Cách 2: Bash script (Linux/Mac)
./scripts/setup_database.sh

# Cách 3: Batch script (Windows)
scripts\setup_database.bat
```

### Bước 3: Kiểm tra

Script sẽ tự động:
- ✅ Tạo database nếu chưa có
- ✅ Chạy migrations
- ✅ Kiểm tra kết nối

## Troubleshooting

### Lỗi: "psycopg not found"

```bash
pip install psycopg[binary]
```

### Lỗi: "Cannot connect to PostgreSQL"

1. Kiểm tra PostgreSQL đang chạy:
   ```bash
   # Linux
   sudo systemctl status postgresql
   
   # Windows
   # Kiểm tra Services → PostgreSQL
   ```

2. Kiểm tra thông tin trong `.env`:
   ```env
   POSTGRES_DSN=postgresql+psycopg://user:password@host:port/database
   ```

3. Kiểm tra kết nối thủ công:
   ```bash
   psql -U postgres -d postgres -c "SELECT version();"
   ```

### Lỗi: "Permission denied"

Đảm bảo user có quyền tạo database:
```sql
-- Với quyền superuser
ALTER USER your_user CREATEDB;
```

### Lỗi migration

Xem [../migrations/TROUBLESHOOTING.md](../migrations/TROUBLESHOOTING.md) để biết cách xử lý.

## Manual Setup (nếu script không hoạt động)

### 1. Tạo database thủ công

```bash
# Kết nối đến PostgreSQL
psql -U postgres

# Tạo database
CREATE DATABASE gogame;

# Thoát
\q
```

### 2. Chạy migrations

```bash
# Từ thư mục backend
alembic upgrade head
```

### 3. Kiểm tra

```bash
psql -U postgres -d gogame -c "\dt"
```

## Liên hệ

Nếu gặp vấn đề, vui lòng:
1. Kiểm tra logs chi tiết
2. Xem [../migrations/TROUBLESHOOTING.md](../migrations/TROUBLESHOOTING.md)
3. Tạo issue trên GitHub

