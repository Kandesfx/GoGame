# GoGame Backend

FastAPI backend cho ứng dụng chơi Cờ Vây với AI.

## Yêu cầu

### Tối thiểu (Backend cơ bản):
- Python 3.10+
- PostgreSQL 14+
- MongoDB 6+ (tùy chọn)

### Đầy đủ (Có AI features):
- Python 3.10+
- PostgreSQL 14+
- MongoDB 6+
- Module `gogame_py` (C++ AI engine bindings)

> **Lưu ý:** Backend có thể chạy được mà không cần `gogame_py`, nhưng AI features sẽ bị disable. Xem [SETUP_MINIMAL.md](../SETUP_MINIMAL.md) để biết cách setup tối thiểu.

## Cài đặt

### ⚡ Quick Start (Khuyến nghị)

**Windows:**
```bash
cd backend
setup.bat    # Setup môi trường
run.bat      # Chạy server
```

**Linux/Mac:**
```bash
cd backend
chmod +x setup.sh run.sh
./setup.sh   # Setup môi trường
./run.sh     # Chạy server
```

Xem [QUICK_START.md](QUICK_START.md) để biết chi tiết.

### 📝 Setup Thủ Công

### 1. Tạo virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 2. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 3. Cấu hình môi trường

Sao chép `env.example` thành `.env` và chỉnh sửa:

```bash
# Windows
copy env.example .env

# Linux/Mac
cp env.example .env
```

Chỉnh sửa các biến trong `.env`:
- `POSTGRES_DSN`: Connection string PostgreSQL
- `MONGO_DSN`: Connection string MongoDB
- `JWT_SECRET_KEY`: Secret key cho JWT (tạo random string dài ít nhất 32 ký tự)

### 4. Setup Database

#### Cách 1: Sử dụng script tự động (Khuyến nghị)

```bash
# Từ thư mục backend
python scripts/setup_database.py
```

Script sẽ tự động:
- ✅ Tạo database nếu chưa tồn tại
- ✅ Chạy migrations
- ✅ Kiểm tra kết nối

Xem [scripts/README.md](scripts/README.md) để biết thêm chi tiết.

#### Cách 2: Manual Setup

##### PostgreSQL

Tạo database:

```sql
CREATE DATABASE gogame;
```

Chạy migrations:

```bash
cd backend
alembic upgrade head
```

**⚠️ QUAN TRỌNG: Nếu gặp lỗi `column matches.room_code does not exist`**

Chạy migration để thêm cột `room_code`:

```bash
cd backend
alembic upgrade head
```

Hoặc chạy SQL trực tiếp (nếu không dùng Alembic):

```sql
ALTER TABLE matches ADD COLUMN IF NOT EXISTS room_code VARCHAR(6);
CREATE INDEX IF NOT EXISTS idx_matches_room_code ON matches(room_code) WHERE room_code IS NOT NULL;
```

Xem file `migrations/add_room_code.sql` để biết thêm chi tiết.

#### MongoDB

MongoDB không cần migration, chỉ cần đảm bảo service đang chạy.

### 5. Build C++ AI Engine (nếu chưa có)

Xem hướng dẫn trong `README.md` ở root project để build module `gogame_py`.

### 6. Chạy server

**QUAN TRỌNG**: Phải chạy từ thư mục `backend`!

```bash
# Đảm bảo đang ở thư mục backend
cd backend

# Cách 1: Dùng script run.bat/run.sh (khuyến nghị - tự động kiểm tra)
run.bat  # Windows
# hoặc
bash run.sh  # Linux/Mac

# Cách 2: Dùng script start.bat/start.sh
start.bat  # Windows
# hoặc
bash start.sh  # Linux/Mac

# Cách 3: Dùng script trong thư mục scripts
scripts/run_server.bat  # Windows
# hoặc
bash scripts/run_server.sh  # Linux/Mac

# Cách 4: Chạy trực tiếp (phải ở trong thư mục backend)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
# Hoặc
uvicorn app.main:app --reload
```


**Lưu ý**: 
- Nếu gặp lỗi `ModuleNotFoundError: No module named 'app'`, đảm bảo bạn đang ở trong thư mục `backend`.
- Script `run.bat`/`run.sh` sẽ tự động kiểm tra và báo lỗi nếu chạy từ sai thư mục.

Server sẽ chạy tại `http://localhost:8000`

API docs: `http://localhost:8000/docs`

## Cấu trúc thư mục

```
backend/
├── app/
│   ├── main.py              # FastAPI app entry point
│   ├── config.py            # Configuration
│   ├── database.py           # DB connections
│   ├── dependencies.py       # FastAPI dependencies
│   ├── routers/              # API endpoints
│   ├── services/             # Business logic
│   ├── models/               # SQLAlchemy & Pydantic models
│   ├── schemas/              # Pydantic request/response schemas
│   ├── utils/                # Utilities
│   └── tasks/                # Background tasks
├── migrations/               # Alembic migrations
├── tests/                    # Unit tests
├── alembic.ini               # Alembic config
├── requirements.txt          # Python dependencies
└── .env.example              # Environment variables template
```

## Database Migrations

Xem `migrations/README.md` để biết cách sử dụng Alembic.

## Testing

### Unit Tests

```bash
pytest tests/
```

### API Integration Tests

Chạy server trước (trong terminal khác):

```bash
uvicorn app.main:app --reload
```

Sau đó chạy test script:

```bash
python scripts/test_api.py
```

Script sẽ test:
- ✅ Health check endpoint
- ✅ Authentication (register/login)
- ✅ Create AI match
- ✅ Get match state
- ✅ Submit move (với AI response)

### Manual Testing với Swagger UI

1. Mở browser: `http://localhost:8000/docs`
2. Test các endpoints trực tiếp trong Swagger UI
3. Có thể xem request/response examples

## API Endpoints

- `GET /health` - Health check
- `POST /auth/register` - Đăng ký
- `POST /auth/login` - Đăng nhập
- `POST /matches/ai` - Tạo trận đấu với AI
- `POST /matches/{id}/move` - Đi nước cờ
- `POST /premium/hint` - Gợi ý nước đi (premium)
- ... và nhiều endpoints khác

Xem chi tiết tại `/docs` khi server đang chạy.

## Development

### Format code

```bash
black app/
isort app/
```

### Type checking

```bash
mypy app/
```

## Troubleshooting

### Lỗi kết nối database

- Kiểm tra PostgreSQL/MongoDB đang chạy
- Kiểm tra connection string trong `.env`
- Kiểm tra firewall/network

### Lỗi import `gogame_py`

- Đảm bảo đã build C++ module
- Kiểm tra `PYTHONPATH` hoặc copy `.pyd` vào thư mục Python
- Xem `README.md` ở root để biết cách build

### Lỗi migration

- Kiểm tra database đã được tạo chưa
- Kiểm tra user có quyền tạo tables
- Xem log trong `alembic.ini` để debug

