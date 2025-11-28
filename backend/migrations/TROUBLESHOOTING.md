# Troubleshooting Migrations

## Lỗi: `relation "users" does not exist`

### Nguyên nhân

Lỗi này xảy ra khi:
1. Database mới (chưa có bảng nào)
2. Migration đầu tiên (`06aeee49f6ae`) chỉ có `pass` (giả định schema đã tồn tại)
3. Migration thứ hai (`6f554950ac0e`) cố gắng thêm cột vào bảng `users` nhưng bảng chưa tồn tại

### Giải pháp

Migration `6f554950ac0e` đã được sửa để tự động tạo bảng `users` nếu chưa tồn tại. Nếu vẫn gặp lỗi:

#### Cách 1: Reset và chạy lại migrations (Khuyến nghị cho database mới)

```bash
# Xóa tất cả bảng (CẨN THẬN: Sẽ mất dữ liệu!)
psql -U postgres -d gogame -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"

# Chạy lại migrations
alembic upgrade head
```

#### Cách 2: Xóa bảng alembic_version và chạy lại

```bash
# Xóa bảng tracking migrations
psql -U postgres -d gogame -c "DROP TABLE IF EXISTS alembic_version;"

# Chạy lại migrations từ đầu
alembic upgrade head
```

#### Cách 3: Tạo bảng users thủ công trước

```sql
-- Kết nối vào database
psql -U postgres -d gogame

-- Tạo bảng users
CREATE TABLE users (
    id VARCHAR(36) PRIMARY KEY,
    username VARCHAR(32) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    elo_rating INTEGER DEFAULT 1500,
    coins INTEGER DEFAULT 0,
    display_name VARCHAR(64),
    avatar_url VARCHAR(255),
    preferences JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE
);

-- Tạo indexes
CREATE INDEX ix_users_username ON users(username);
CREATE INDEX ix_users_email ON users(email);

-- Sau đó chạy lại migration
-- alembic upgrade head
```

## Lỗi: `column "xxx" already exists`

### Nguyên nhân

Migration đã được chạy một phần hoặc cột đã tồn tại.

### Giải pháp

Migration đã có logic kiểm tra `IF NOT EXISTS`, nhưng nếu vẫn gặp lỗi:

```bash
# Kiểm tra trạng thái migrations
alembic current

# Xem lịch sử migrations
alembic history

# Nếu cần, đánh dấu migration đã chạy
alembic stamp head
```

## Lỗi: `duplicate key value violates unique constraint`

### Nguyên nhân

Có dữ liệu trùng lặp trong database.

### Giải pháp

```sql
-- Kiểm tra dữ liệu trùng lặp
SELECT username, COUNT(*) FROM users GROUP BY username HAVING COUNT(*) > 1;
SELECT email, COUNT(*) FROM users GROUP BY email HAVING COUNT(*) > 1;

-- Xóa dữ liệu trùng lặp (cẩn thận!)
```

## Lỗi: `could not connect to server`

### Nguyên nhân

PostgreSQL không chạy hoặc cấu hình sai.

### Giải pháp

1. **Kiểm tra PostgreSQL đang chạy:**
   ```bash
   # Windows
   services.msc  # Tìm PostgreSQL service
   
   # Linux
   sudo systemctl status postgresql
   
   # macOS
   brew services list
   ```

2. **Kiểm tra file `.env`:**
   ```bash
   # Đảm bảo POSTGRES_DSN đúng
   POSTGRES_DSN=postgresql+psycopg://user:password@localhost:5432/gogame
   ```

3. **Kiểm tra kết nối:**
   ```bash
   psql -U postgres -d gogame -c "SELECT version();"
   ```

## Lỗi: `alembic.util.exc.CommandError: Target database is not up to date`

### Nguyên nhân

Database đang ở version cũ hơn so với code.

### Giải pháp

```bash
# Xem version hiện tại
alembic current

# Xem version mới nhất
alembic heads

# Upgrade lên version mới nhất
alembic upgrade head
```

## Lỗi: `alembic.util.exc.CommandError: Can't locate revision identified by 'xxx'`

### Nguyên nhân

Migration file bị thiếu hoặc revision ID không khớp.

### Giải pháp

1. **Kiểm tra migration files:**
   ```bash
   ls backend/migrations/versions/
   ```

2. **Kiểm tra revision chain:**
   ```bash
   alembic history
   ```

3. **Nếu migration file bị thiếu, tạo lại:**
   ```bash
   alembic revision -m "description"
   ```

## Best Practices

1. **Luôn backup database trước khi chạy migrations:**
   ```bash
   pg_dump -U postgres gogame > backup.sql
   ```

2. **Test migrations trên database dev trước:**
   ```bash
   # Tạo database test
   createdb gogame_test
   
   # Chạy migrations trên test
   POSTGRES_DSN=postgresql+psycopg://user:password@localhost:5432/gogame_test alembic upgrade head
   ```

3. **Kiểm tra migration trước khi commit:**
   ```bash
   # Xem SQL sẽ được chạy (không thực thi)
   alembic upgrade head --sql
   ```

4. **Sử dụng transactions khi có thể:**
   - PostgreSQL tự động sử dụng transactions cho migrations
   - Nếu có lỗi, tất cả thay đổi sẽ được rollback

## Liên hệ

Nếu gặp vấn đề khác, vui lòng:
1. Kiểm tra logs chi tiết
2. Tạo issue trên GitHub với thông tin:
   - Lỗi đầy đủ (full traceback)
   - Version của PostgreSQL
   - Version của Alembic
   - Trạng thái migrations hiện tại (`alembic current`)

