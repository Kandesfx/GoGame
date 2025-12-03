# 🗄️ Hướng Dẫn Cấu Hình Database

## Vấn Đề Thường Gặp

Nếu bạn gặp lỗi:
```
password authentication failed for user "postgres"
connection to server at "127.0.0.1", port 5432 failed
```

Đây là lỗi kết nối PostgreSQL. Làm theo các bước sau:

## Bước 1: Kiểm Tra PostgreSQL Đã Chạy Chưa

### Windows
```bash
# Kiểm tra service
sc query postgresql-x64-*

# Hoặc kiểm tra trong Services (services.msc)
# Tìm "PostgreSQL" và đảm bảo nó đang chạy
```

### Linux/Mac
```bash
# Kiểm tra process
ps aux | grep postgres

# Hoặc kiểm tra service
sudo systemctl status postgresql
```

## Bước 2: Tạo File .env

Tạo file `.env` trong thư mục `backend/`:

```bash
cd backend
cp env.example .env
```

## Bước 3: Cấu Hình Database Connection

Mở file `.env` và chỉnh sửa `POSTGRES_DSN` hoặc `DATABASE_URL`:

### Option 1: Sử dụng POSTGRES_DSN (Local Development)
```env
POSTGRES_DSN=postgresql+psycopg://username:password@localhost:5432/gogame
```

**Thay thế:**
- `username`: Tên user PostgreSQL của bạn (thường là `postgres`)
- `password`: Mật khẩu PostgreSQL của bạn
- `localhost:5432`: Địa chỉ và port (mặc định là 5432)
- `gogame`: Tên database (tạo database này nếu chưa có)

### Option 2: Sử dụng DATABASE_URL (Production/Fly.io)
```env
DATABASE_URL=postgresql+psycopg://username:password@host:5432/gogame
```

## Bước 4: Tạo Database

Nếu database `gogame` chưa tồn tại:

### Windows (psql)
```bash
# Kết nối PostgreSQL
psql -U postgres

# Tạo database
CREATE DATABASE gogame;

# Thoát
\q
```

### Linux/Mac
```bash
sudo -u postgres psql
CREATE DATABASE gogame;
\q
```

## Bước 5: Chạy Migrations

Sau khi database đã được tạo và cấu hình đúng:

```bash
cd backend

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows

# Chạy migrations
alembic upgrade head
```

## Bước 6: Kiểm Tra Kết Nối

Test kết nối bằng cách chạy server:

```bash
python -m uvicorn app.main:app --reload
```

Nếu vẫn lỗi, kiểm tra:

1. **PostgreSQL đang chạy:**
   ```bash
   # Windows
   netstat -an | findstr 5432
   
   # Linux/Mac
   netstat -an | grep 5432
   ```

2. **Username và password đúng:**
   - Thử kết nối bằng psql:
   ```bash
   psql -U postgres -d gogame
   ```

3. **Database đã được tạo:**
   ```sql
   \l  -- List databases trong psql
   ```

## Cấu Hình Nhanh Cho Local Development

Nếu bạn dùng PostgreSQL mặc định (user: postgres, password: postgres):

1. Tạo file `.env`:
```env
POSTGRES_DSN=postgresql+psycopg://postgres:postgres@localhost:5432/gogame
```

2. Tạo database:
```bash
createdb -U postgres gogame
```

3. Chạy migrations:
```bash
alembic upgrade head
```

## Production (Fly.io)

Trên Fly.io, `DATABASE_URL` được tự động set. Bạn không cần cấu hình gì thêm.

Nếu deploy lên platform khác, set biến môi trường:
```bash
export DATABASE_URL=postgresql+psycopg://user:pass@host:5432/dbname
```

## Troubleshooting

### Lỗi: "password authentication failed"
- Kiểm tra username và password trong `.env`
- Đảm bảo PostgreSQL cho phép password authentication
- Kiểm tra file `pg_hba.conf` nếu cần

### Lỗi: "connection refused"
- PostgreSQL chưa chạy
- Port không đúng (mặc định là 5432)
- Firewall chặn kết nối

### Lỗi: "database does not exist"
- Tạo database trước: `CREATE DATABASE gogame;`
- Kiểm tra tên database trong connection string

## Liên Kết Hữu Ích

- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [SQLAlchemy Connection Strings](https://docs.sqlalchemy.org/en/20/core/engines.html#database-urls)
- [Psycopg Documentation](https://www.psycopg.org/docs/)

