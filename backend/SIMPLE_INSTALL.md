# 🚀 Hướng Dẫn Cài Đặt Đơn Giản

Nếu gặp lỗi với `orjson` hoặc các package cần Rust, làm theo các bước sau:

## Bước 1: Activate Virtual Environment

```bash
cd backend

# Trong Git Bash (Windows)
source venv/bin/activate

# Hoặc trong Windows CMD
venv\Scripts\activate
```

## Bước 2: Cài Đặt Package Cơ Bản

Chạy từng lệnh này:

```bash
# Upgrade pip
pip install --upgrade pip

# Cài đặt uvicorn (quan trọng nhất)
pip install uvicorn==0.30.1

# Cài đặt các package cơ bản
pip install fastapi==0.111.0
pip install sqlalchemy==2.0.30
pip install alembic==1.13.1
pip install "psycopg[binary]==3.1.19"
pip install python-dotenv==1.0.1
pip install PyJWT==2.9.0
pip install httpx==0.27.0
```

## Bước 3: Cài Đặt Pydantic (có thể bỏ qua nếu lỗi)

```bash
pip install "pydantic==2.7.1"
```

Nếu lỗi, thử:
```bash
pip install pydantic --prefer-binary
```

## Bước 4: Cài Đặt Các Package Khác (Optional)

```bash
pip install "motor>=3.5.0"
pip install "pymongo>=4.5.0,<5.0.0"
pip install pydantic-settings==2.2.1
pip install argon2-cffi==23.1.0
pip install "passlib[argon2]==1.7.4"
pip install pytest==8.2.2
```

## Bước 5: Kiểm Tra

```bash
python -c "import uvicorn; print('✅ uvicorn OK')"
python -c "import fastapi; print('✅ fastapi OK')"
python -c "import sqlalchemy; print('✅ sqlalchemy OK')"
```

Nếu tất cả đều OK, bạn có thể chạy server!

## Bước 6: Chạy Server

```bash
# Activate venv (nếu chưa)
source venv/bin/activate  # Git Bash
# hoặc
venv\Scripts\activate     # Windows CMD

# Chạy server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Lưu Ý

- **orjson** là optional - không cần thiết để chạy server
- **watchfiles** là optional - chỉ cần cho `--reload` mode
- Nếu thiếu package nào, server sẽ báo lỗi khi import - cài thêm package đó

## Nếu Vẫn Lỗi

Thử cài đặt từ requirements_minimal.txt:

```bash
pip install -r requirements_minimal.txt
```

Hoặc cài đặt từng package một và bỏ qua các package lỗi.

