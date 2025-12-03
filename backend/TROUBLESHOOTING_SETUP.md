# 🔧 Troubleshooting Setup Issues

## Vấn đề: Lỗi cài đặt `orjson` (cần Rust)

**Lỗi:**
```
ERROR: Failed to build 'orjson' when installing build dependencies
```

**Giải pháp:**

`orjson` là optional dependency của FastAPI, không bắt buộc. Có 2 cách:

### Cách 1: Cài đặt từng package (Khuyến nghị)

Chạy script `install_deps.sh` hoặc `install_deps.bat`:

```bash
# Git Bash / Linux / Mac
./install_deps.sh

# Windows CMD
install_deps.bat
```

### Cách 2: Cài đặt thủ công

```bash
# Activate virtual environment
# Windows
venv\Scripts\activate

# Git Bash / Linux / Mac  
source venv/bin/activate  # hoặc source venv/Scripts/activate trên Windows Git Bash

# Cài đặt từng package
pip install fastapi==0.111.0
pip install "uvicorn[standard]==0.30.1"
pip install sqlalchemy==2.0.30
pip install alembic==1.13.1
pip install "psycopg[binary]==3.1.19"
pip install "motor>=3.5.0"
pip install "pymongo>=4.5.0,<5.0.0"
pip install "pydantic[email]==2.7.1"
pip install pydantic-settings==2.2.1
pip install python-dotenv==1.0.1
pip install PyJWT==2.9.0
pip install argon2-cffi==23.1.0
pip install "passlib[argon2]==1.7.4"
pip install httpx==0.27.0
pip install pytest==8.2.2
```

### Cách 3: Cài đặt Rust (nếu muốn có orjson)

1. Cài đặt Rust: https://rustup.rs/
2. Sau đó chạy lại: `pip install -r requirements.txt`

## Vấn đề: Git Bash không chạy được `.bat` files

**Lỗi:**
```
bash: setup.bat: command not found
```

**Giải pháp:**

Trong Git Bash, sử dụng script `.sh`:

```bash
# Thay vì setup.bat
./setup.sh

# Thay vì run.bat
./run.sh
```

Hoặc chạy `.bat` qua `cmd`:

```bash
cmd //c setup.bat
cmd //c run.bat
```

## Vấn đề: Virtual environment không activate

**Lỗi:**
```
venv/Scripts/activate: No such file or directory
```

**Giải pháp:**

### Windows Git Bash

```bash
# Thử cả 2 đường dẫn
source venv/Scripts/activate
# hoặc
source venv/bin/activate
```

### Windows CMD

```cmd
venv\Scripts\activate.bat
```

### Linux/Mac

```bash
source venv/bin/activate
```

## Vấn đề: Python không tìm thấy

**Lỗi:**
```
python: command not found
```

**Giải pháp:**

1. Kiểm tra Python đã cài đặt: `python --version` hoặc `python3 --version`
2. Nếu không có, cài đặt từ: https://www.python.org/
3. Đảm bảo Python trong PATH
4. Thử `py` thay vì `python` trên Windows

## Vấn đề: pip không tìm thấy

**Lỗi:**
```
pip: command not found
```

**Giải pháp:**

```bash
# Cài đặt pip
python -m ensurepip --upgrade

# Hoặc
python -m pip install --upgrade pip
```

## Kiểm tra cài đặt

Sau khi cài đặt, kiểm tra:

```bash
# Activate venv
source venv/bin/activate  # hoặc venv\Scripts\activate trên Windows

# Kiểm tra uvicorn
python -c "import uvicorn; print('✅ uvicorn OK')"

# Kiểm tra fastapi
python -c "import fastapi; print('✅ fastapi OK')"

# Kiểm tra tất cả
python -c "import uvicorn, fastapi, sqlalchemy, alembic, psycopg, motor, pymongo, pydantic; print('✅ All core packages OK')"
```

## Chạy server sau khi setup

```bash
# Activate venv
source venv/bin/activate  # hoặc venv\Scripts\activate

# Chạy server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Hoặc sử dụng script:

```bash
./run.sh  # Git Bash / Linux / Mac
run.bat   # Windows CMD
```

