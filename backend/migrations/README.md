# Database Migrations

## Thêm cột room_code vào bảng matches

Chạy migration script để thêm cột `room_code`:

```bash
# PostgreSQL
psql -U your_username -d your_database -f migrations/add_room_code.sql

# Hoặc nếu dùng connection string
psql "postgresql://user:password@localhost/dbname" -f migrations/add_room_code.sql
```

### Hoặc chạy SQL trực tiếp:

```sql
ALTER TABLE matches 
ADD COLUMN IF NOT EXISTS room_code VARCHAR(6);

CREATE INDEX IF NOT EXISTS idx_matches_room_code ON matches(room_code) 
WHERE room_code IS NOT NULL;
```

### Kiểm tra:

```sql
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'matches' AND column_name = 'room_code';
```

## Troubleshooting

Nếu gặp lỗi khi chạy migrations (ví dụ: `relation "users" does not exist`), xem [TROUBLESHOOTING.md](TROUBLESHOOTING.md) để biết cách xử lý.
