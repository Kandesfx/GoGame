# Hướng dẫn Setup Backend Tối Thiểu (Không Cần C++ Engine)

Tài liệu này hướng dẫn setup backend GoGame **mà không cần** build C++ AI engine (gogame_py). Backend sẽ chạy được nhưng một số tính năng sẽ bị hạn chế.

## ✅ Backend Có Thể Chạy Được

Backend FastAPI **có thể chạy được** mà không cần:
- ❌ MSYS2 / MinGW
- ❌ CMake
- ❌ C++ Compiler (GCC/G++)
- ❌ Build C++ AI engine (gogame_py)

## ⚠️ Tính Năng Bị Hạn Chế

Khi không có `gogame_py` module, các tính năng sau sẽ bị ảnh hưởng:

### 1. AI Features (Bị Disable)
- ❌ Chơi với AI (PvAI matches)
- ❌ AI sẽ không thể chơi được
- ✅ PvP matches vẫn hoạt động bình thường

### 2. Premium Features (Fallback Mode)
- ⚠️ Premium hints - Sử dụng fallback logic (có thể không chính xác)
- ⚠️ Premium analysis - Sử dụng fallback logic (có thể không đầy đủ)
- ⚠️ Game review - Sử dụng fallback logic

### 3. Tính Năng Hoạt Động Bình Thường
- ✅ Authentication & User management
- ✅ PvP matches (Player vs Player)
- ✅ Matchmaking
- ✅ Coin system
- ✅ User profiles & statistics
- ✅ Database operations
- ✅ API endpoints (trừ AI-related)

## 📋 Yêu Cầu Tối Thiểu

Chỉ cần các công cụ sau:

### Bắt buộc:
- ✅ **Python 3.10+**
- ✅ **PostgreSQL 14+**
- ✅ **MongoDB 5.0+** (tùy chọn, backend vẫn chạy được nếu không có)

### Không cần:
- ❌ CMake
- ❌ C++ Compiler
- ❌ MSYS2 / MinGW
- ❌ Build tools

## 🚀 Setup Backend Minimal

### Bước 1: Cài đặt Python và Databases

Xem [INSTALLATION.md](INSTALLATION.md) để cài đặt:
- Python 3.10+
- PostgreSQL 14+
- MongoDB 5.0+ (tùy chọn)

**Bỏ qua** các bước về:
- CMake
- C++ Compiler
- MSYS2

### Bước 2: Setup Backend

```bash
# 1. Clone repository
git clone <repository-url>
cd GoGame/backend

# 2. Tạo virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# 3. Cài đặt dependencies
pip install -r requirements.txt

# 4. Cấu hình .env
cp env.example .env
# Chỉnh sửa .env với thông tin database

# 5. Setup database
python scripts/setup_database.py
# hoặc
psql -U postgres -f scripts/database_schema.sql

# 6. Chạy migrations (nếu cần)
alembic upgrade head
```

### Bước 3: Chạy Backend

```bash
# Chạy server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# hoặc
uvicorn app.main:app --reload
```

Backend sẽ chạy và hiển thị warning:
```
WARNING:root:gogame_py module not found. AI features will be disabled.
WARNING:root:gogame_py module not found. Premium features will use fallback.
```

**Đây là bình thường** - backend vẫn hoạt động được!

## 🧪 Kiểm Tra

### 1. Health Check

```bash
curl http://localhost:8000/health
```

Kết quả mong đợi:
```json
{
  "status": "healthy",
  "postgres": true,
  "mongo": true/false
}
```

### 2. API Docs

Truy cập: http://localhost:8000/docs

Bạn sẽ thấy tất cả các endpoints, nhưng:
- Endpoints liên quan đến AI sẽ trả về lỗi hoặc không hoạt động
- Endpoints PvP vẫn hoạt động bình thường

### 3. Test Registration

```bash
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "email": "test@example.com",
    "password": "testpassword123"
  }'
```

## 📝 Lưu Ý

### Khi Nào Cần Build C++ Engine?

Bạn **cần** build C++ engine nếu muốn:
- ✅ Chơi với AI (PvAI matches)
- ✅ Sử dụng premium features đầy đủ
- ✅ Phân tích game chính xác
- ✅ Test toàn bộ tính năng

### Khi Nào Không Cần?

Bạn **không cần** build C++ engine nếu chỉ muốn:
- ✅ Test backend API
- ✅ Test authentication
- ✅ Test PvP matches
- ✅ Test database operations
- ✅ Phát triển frontend
- ✅ Phát triển các tính năng không liên quan đến AI

## 🔄 Nâng Cấp Lên Full Setup

Khi muốn có đầy đủ tính năng, bạn có thể:

1. Cài đặt MSYS2 / MinGW (xem [INSTALLATION.md](INSTALLATION.md))
2. Cài đặt CMake
3. Build C++ engine:
   ```bash
   mkdir -p build
   cd build
   cmake ..
   cmake --build .
   ```
4. Copy `gogame_py.pyd` (Windows) hoặc `gogame_py.so` (Linux) vào thư mục backend
5. Restart backend server

## ❓ Troubleshooting

### Lỗi: "gogame_py module not found"

**Đây không phải lỗi!** Đây chỉ là warning. Backend vẫn chạy được.

Nếu muốn tắt warning, bạn có thể:
- Bỏ qua (backend vẫn hoạt động)
- Build C++ engine để có đầy đủ tính năng

### Lỗi khi tạo AI match

Nếu bạn cố tạo AI match mà không có `gogame_py`, API sẽ trả về lỗi. Đây là hành vi mong đợi.

**Giải pháp:** Chỉ sử dụng PvP matches, hoặc build C++ engine.

### MongoDB không chạy

Backend vẫn chạy được nếu MongoDB không có, nhưng:
- Game state sẽ không được lưu vào MongoDB
- Một số tính năng có thể bị ảnh hưởng

**Giải pháp:** Cài đặt và chạy MongoDB, hoặc bỏ qua nếu chỉ test backend.

## 📚 Tài liệu Liên Quan

- [SETUP.md](SETUP.md) - Setup đầy đủ (có C++ engine)
- [INSTALLATION.md](INSTALLATION.md) - Hướng dẫn cài đặt tất cả công cụ
- [backend/README.md](backend/README.md) - Chi tiết về backend

