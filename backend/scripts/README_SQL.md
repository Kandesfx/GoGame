# GoGame Database SQL Scripts

Bộ script SQL để quản lý database GoGame thủ công (không tự động).

## 📋 Danh sách Scripts

### 1. `database_schema.sql` - Tạo toàn bộ schema

Script chính để tạo database và tất cả các bảng.

**Sử dụng:**
```bash
# Từ command line
psql -U postgres -f scripts/database_schema.sql

# Hoặc từ psql shell
psql -U postgres
\i scripts/database_schema.sql
```

**Nội dung:**
- ✅ Tạo database `gogame`
- ✅ Tạo extension `uuid-ossp`
- ✅ Tạo bảng `users`
- ✅ Tạo bảng `matches`
- ✅ Tạo bảng `refresh_tokens`
- ✅ Tạo bảng `coin_transactions`
- ✅ Tạo bảng `premium_requests`
- ✅ Tạo bảng `alembic_version`
- ✅ Tạo tất cả indexes
- ✅ Tạo foreign keys
- ✅ Thêm comments

### 2. `database_drop.sql` - Xóa database

Script để xóa hoàn toàn database.

**⚠️ CẢNH BÁO:** Sẽ xóa TẤT CẢ dữ liệu!

**Sử dụng:**
```bash
psql -U postgres -f scripts/database_drop.sql
```

### 3. `database_reset.sql` - Reset dữ liệu

Script để xóa tất cả dữ liệu nhưng giữ lại cấu trúc bảng.

**⚠️ CẢNH BÁO:** Sẽ xóa TẤT CẢ dữ liệu trong các bảng!

**Sử dụng:**
```bash
psql -U postgres -d gogame -f scripts/database_reset.sql
```

### 4. `database_backup.sql` - Hướng dẫn backup

Script hiển thị hướng dẫn và thông tin về backup.

**Sử dụng:**
```bash
psql -U postgres -d gogame -f scripts/database_backup.sql
```

**Hoặc dùng pg_dump:**
```bash
# Backup toàn bộ
pg_dump -U postgres -d gogame -f backup.sql

# Backup chỉ schema
pg_dump -U postgres -d gogame --schema-only -f schema.sql

# Backup chỉ dữ liệu
pg_dump -U postgres -d gogame --data-only -f data.sql

# Backup dạng custom (nén)
pg_dump -U postgres -d gogame -F c -f backup.dump
```

**Restore:**
```bash
# Từ SQL file
psql -U postgres -d gogame < backup.sql

# Từ custom dump
pg_restore -U postgres -d gogame backup.dump
```

### 5. `database_sample_data.sql` - Dữ liệu mẫu

Script để insert dữ liệu mẫu cho testing.

**Sử dụng:**
```bash
psql -U postgres -d gogame -f scripts/database_sample_data.sql
```

**Nội dung:**
- ✅ 3 sample users
- ✅ 3 sample matches
- ✅ 3 sample coin transactions

### 6. `database_queries.sql` - Các query hữu ích

Script chứa các câu query để kiểm tra và quản lý database.

**Sử dụng:**
```bash
psql -U postgres -d gogame -f scripts/database_queries.sql
```

**Nội dung:**
- 📊 Thông tin database
- 👥 Thống kê users
- 🎮 Thống kê matches
- 💰 Thống kê coins
- ⭐ Thống kê premium requests
- 🔧 Maintenance queries

## 🚀 Quy trình Setup Database

### Bước 1: Tạo database và schema

```bash
psql -U postgres -f scripts/database_schema.sql
```

### Bước 2: (Tùy chọn) Insert dữ liệu mẫu

```bash
psql -U postgres -d gogame -f scripts/database_sample_data.sql
```

### Bước 3: Kiểm tra

```bash
psql -U postgres -d gogame -f scripts/database_queries.sql
```

## 🔧 Migration từ VARCHAR(36) sang UUID

Nếu database đã được tạo với `VARCHAR(36)` và gặp lỗi type mismatch, chạy migration script:

```bash
psql -U postgres -d gogame -f scripts/database_migrate_varchar_to_uuid.sql
```

Script này sẽ:
- ✅ Chuyển đổi tất cả các cột ID từ `VARCHAR(36)` sang `UUID`
- ✅ Giữ nguyên dữ liệu
- ✅ Tạo lại foreign keys

**⚠️ CẢNH BÁO:** Backup database trước khi chạy migration!

## 🔄 Quy trình Reset Database

### Cách 1: Xóa và tạo lại (Hoàn toàn)

```bash
# Xóa database
psql -U postgres -f scripts/database_drop.sql

# Tạo lại
psql -U postgres -f scripts/database_schema.sql
```

### Cách 2: Chỉ xóa dữ liệu (Giữ schema)

```bash
psql -U postgres -d gogame -f scripts/database_reset.sql
```

## 📦 Backup và Restore

### Backup

```bash
# Backup toàn bộ
pg_dump -U postgres -d gogame -f backup_$(date +%Y%m%d_%H%M%S).sql

# Backup với timestamp
pg_dump -U postgres -d gogame -f backup_$(date +%Y%m%d_%H%M%S).sql
```

### Restore

```bash
# Từ SQL file
psql -U postgres -d gogame < backup_YYYYMMDD_HHMMSS.sql

# Hoặc tạo database mới trước
psql -U postgres -c "CREATE DATABASE gogame_restore;"
psql -U postgres -d gogame_restore < backup_YYYYMMDD_HHMMSS.sql
```

## 🔍 Kiểm tra Database

### Kết nối

```bash
psql -U postgres -d gogame
```

### Xem danh sách bảng

```sql
\dt
```

### Xem cấu trúc bảng

```sql
\d+ users
\d+ matches
```

### Xem dữ liệu

```sql
SELECT * FROM users LIMIT 10;
SELECT * FROM matches LIMIT 10;
```

### Thoát

```sql
\q
```

## 🛠️ Maintenance

### Vacuum (dọn dẹp database)

```sql
VACUUM ANALYZE;
```

### Xóa dữ liệu cũ

```sql
-- Xóa refresh tokens đã hết hạn
DELETE FROM refresh_tokens WHERE expires_at < NOW();

-- Xóa matches cũ hơn 30 ngày
DELETE FROM matches 
WHERE finished_at IS NOT NULL 
  AND finished_at < NOW() - INTERVAL '30 days';
```

### Kiểm tra kích thước

```sql
SELECT pg_size_pretty(pg_database_size('gogame'));
```

## ⚠️ Lưu ý

1. **Luôn backup trước khi chạy script xóa/reset**
2. **Kiểm tra quyền user trước khi chạy script**
3. **Đọc kỹ script trước khi chạy (đặc biệt là drop/reset)**
4. **Test trên database dev trước khi chạy trên production**

## 📚 Tài liệu tham khảo

- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [pg_dump Documentation](https://www.postgresql.org/docs/current/app-pgdump.html)
- [psql Documentation](https://www.postgresql.org/docs/current/app-psql.html)

