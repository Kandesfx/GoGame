# Fix Machine Health Check Timeout

## Vấn đề
Machine `e78464eae54dd8` không thể restart vì health check fail.

## Giải pháp

### Option 1: Destroy và tạo lại machine (Khuyến nghị)

```powershell
cd backend

# Destroy machine bị lỗi
fly machines destroy e78464eae54dd8 -a gogame-backend --force

# Fly.io sẽ tự động tạo machine mới
# Hoặc scale up để tạo machine mới
fly scale count 2 -a gogame-backend
```

### Option 2: Scale down rồi scale up

```powershell
cd backend

# Scale về 1 machine (sẽ destroy machine bị lỗi)
fly scale count 1 -a gogame-backend

# Đợi 30 giây để machine còn lại stable

# Scale lại 2 machines
fly scale count 2 -a gogame-backend
```

### Option 3: Kiểm tra logs trước khi destroy

```powershell
cd backend

# Xem logs của machine bị lỗi
fly logs -a gogame-backend | grep -i "error\|exception\|failed\|database" | tail -30

# Hoặc xem logs real-time
fly logs -a gogame-backend
```

**Các lỗi thường gặp:**
- `Database connection failed` → Check `DATABASE_URL`
- `Alembic migration failed` → Check migration files
- `Module import error` → Check C++ module build

### Option 4: Tạm thời chạy 1 machine

Nếu không cần 2 machines ngay:

```powershell
cd backend

# Scale về 1 machine
fly scale count 1 -a gogame-backend

# Test backend
curl https://gogame-backend.fly.dev/health
```

Sau đó có thể scale lên 2 machines khi cần.

## Quick Fix (Khuyến nghị)

```powershell
cd backend

# 1. Destroy machine bị lỗi
fly machines destroy e78464eae54dd8 -a gogame-backend --force

# 2. Scale up để tạo machine mới
fly scale count 2 -a gogame-backend

# 3. Đợi 1-2 phút, rồi test
curl https://gogame-backend.fly.dev/health
```

## Kiểm tra sau khi fix

```powershell
# Check status
fly status -a gogame-backend

# Test health
curl https://gogame-backend.fly.dev/health

# Xem logs
fly logs -a gogame-backend | tail -20
```

## Nếu vẫn lỗi

Có thể do:
1. **Database connection issue** → Check `DATABASE_URL` trong secrets
2. **Migration failed** → Check `fly logs` để xem lỗi migration
3. **C++ module build failed** → Backend vẫn chạy được nhưng AI features bị disable

Xem logs chi tiết:
```powershell
fly logs -a gogame-backend | grep -i "error\|exception\|failed" | tail -50
```

