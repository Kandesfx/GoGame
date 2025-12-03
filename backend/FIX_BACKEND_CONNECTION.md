# Fix Backend Connection Issue

## Vấn đề
Frontend không kết nối được đến backend: `https://gogame-backend.fly.dev`

## Giải pháp

### Bước 1: Kiểm tra Backend có đang chạy không

```powershell
# Xem status
cd backend
fly status -a gogame-backend

# Xem logs
fly logs -a gogame-backend

# Test health endpoint
curl https://gogame-backend.fly.dev/health
```

Nếu backend không chạy hoặc có lỗi → Xem logs và fix

### Bước 2: Config CORS để cho phép Frontend

Backend cần cho phép frontend domain trong CORS:

```powershell
cd backend
fly secrets set CORS_ORIGINS="https://gogame-frontend.fly.dev,http://localhost:3000" -a gogame-backend
```

**Lưu ý:** Sau khi set secrets, backend sẽ tự động restart. Đợi 30-60 giây.

### Bước 3: Verify CORS Config

```powershell
# Xem secrets
fly secrets list -a gogame-backend

# Xem logs để verify CORS origins
fly logs -a gogame-backend | grep -i "CORS"
```

Bạn sẽ thấy log: `🌐 CORS allowed origins: ['https://gogame-frontend.fly.dev', 'http://localhost:3000']`

### Bước 4: Test Backend từ Browser

Mở browser console và chạy:

```javascript
// Test health endpoint
fetch('https://gogame-backend.fly.dev/health')
  .then(r => r.json())
  .then(console.log)
  .catch(console.error)

// Test với CORS
fetch('https://gogame-backend.fly.dev/health', {
  headers: {
    'Origin': 'https://gogame-frontend.fly.dev'
  }
})
  .then(r => {
    console.log('Status:', r.status)
    console.log('CORS Headers:', r.headers.get('Access-Control-Allow-Origin'))
    return r.json()
  })
  .then(console.log)
  .catch(console.error)
```

### Bước 5: Nếu Backend không chạy

```powershell
# Restart backend
fly apps restart -a gogame-backend

# Hoặc redeploy
cd backend
fly deploy
```

### Bước 6: Kiểm tra Database Connection

Nếu backend crash do database:

```powershell
# Xem logs
fly logs -a gogame-backend | grep -i "database\|postgres\|error"

# SSH vào container
fly ssh console -a gogame-backend

# Trong container, test database
python -c "from app.database import get_db; next(get_db()); print('✅ DB OK')"
```

## Quick Fix Commands

```powershell
# 1. Set CORS
cd backend
fly secrets set CORS_ORIGINS="https://gogame-frontend.fly.dev,http://localhost:3000" -a gogame-backend

# 2. Restart backend
fly apps restart -a gogame-backend

# 3. Test
curl https://gogame-backend.fly.dev/health

# 4. Check logs
fly logs -a gogame-backend
```

## Troubleshooting

### Backend trả về 502/503
→ Backend đang crash hoặc không start được
→ Xem logs: `fly logs -a gogame-backend`

### CORS error trong browser
→ CORS_ORIGINS chưa được set đúng
→ Verify: `fly secrets list -a gogame-backend`

### Connection refused
→ Backend không chạy
→ Restart: `fly apps restart -a gogame-backend`

### Database connection failed
→ DATABASE_URL không đúng hoặc database không accessible
→ Check: `fly secrets list -a gogame-backend | grep DATABASE`

