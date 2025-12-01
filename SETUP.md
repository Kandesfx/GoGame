# Hướng dẫn Setup Dự án GoGame

> **Lưu ý**: Nếu bạn đang setup trên máy mới và chưa có các công cụ cần thiết, vui lòng xem [INSTALLATION.md](INSTALLATION.md) trước để cài đặt tất cả các dependencies.

## Yêu cầu hệ thống

### Tối thiểu (Backend cơ bản - không có AI):
- **Python**: 3.10+
- **PostgreSQL**: 14+
- **MongoDB**: 5.0+ (tùy chọn)

### Đầy đủ (Có AI features):
- **Python**: 3.10+
- **Node.js**: 18+ (cho frontend-web)
- **PostgreSQL**: 14+ (cho database)
- **MongoDB**: 5.0+ (cho game state storage)
- **CMake**: 3.15+ (cho build C++ AI engine)
- **C++ Compiler**: GCC/G++ hoặc MSVC (cho build AI engine)

> **Chưa có các công cụ trên?** Xem [INSTALLATION.md](INSTALLATION.md) để biết hướng dẫn cài đặt chi tiết.

> **Chỉ muốn test backend mà không cần AI?** Xem [SETUP_MINIMAL.md](SETUP_MINIMAL.md) để biết hướng dẫn setup tối thiểu (không cần C++ engine).

## Bước 1: Clone repository

```bash
git clone <repository-url>
cd GoGame
```

## Bước 2: Setup Backend

### 2.1. Tạo virtual environment

```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 2.2. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2.3. Cấu hình môi trường

```bash
# Copy file env.example thành .env
cp env.example .env

# Chỉnh sửa .env với thông tin database của bạn
# POSTGRES_DSN=postgresql+psycopg://user:password@localhost:5432/gogame
# MONGO_DSN=mongodb://localhost:27017
# JWT_SECRET_KEY=<generate-a-random-secret-key>
```

### 2.4. Setup Database

#### PostgreSQL

```bash
# Tạo database
createdb gogame

# Hoặc dùng psql
psql -U postgres
CREATE DATABASE gogame;
```

#### MongoDB

```bash
# MongoDB thường chạy tự động sau khi cài đặt
# Kiểm tra: mongosh hoặc mongo
```

### 2.5. Setup Database

Có 2 cách để setup database:

#### Cách 1: Sử dụng script tự động (Khuyến nghị)

```bash
# Từ thư mục backend
python scripts/setup_database.py

# Hoặc trên Linux/Mac
./scripts/setup_database.sh

# Hoặc trên Windows
scripts\setup_database.bat
```

Script sẽ tự động:
- ✅ Tạo database nếu chưa tồn tại
- ✅ Chạy migrations
- ✅ Kiểm tra kết nối

Xem [backend/scripts/README.md](backend/scripts/README.md) để biết thêm chi tiết.

#### Cách 2: Manual setup

```bash
# Tạo database thủ công
psql -U postgres -c "CREATE DATABASE gogame;"

# Chạy migrations
cd backend
alembic upgrade head
```

### 2.6. Build C++ AI Engine (nếu cần)

> **Lưu ý**: Module `gogame_py` cần **pybind11** để build. Nếu không có pybind11, module sẽ không được tạo.

#### Bước 1: Kiểm tra và cài đặt pybind11

**Windows (MSYS2):**
```bash
# Mở MSYS2 MinGW 64-bit shell
pacman -S mingw-w64-x86_64-pybind11

# Kiểm tra đã cài chưa
pacman -Q mingw-w64-x86_64-pybind11
```

**Linux:**
```bash
# Cài qua pip (khuyến nghị)
pip install pybind11

# Hoặc qua package manager
sudo apt install python3-pybind11  # Ubuntu/Debian
```

**macOS:**
```bash
# Cài qua pip
pip install pybind11

# Hoặc qua Homebrew
brew install pybind11
```

#### Bước 2: Build module

**Cách 1: Dùng script tự động (Khuyến nghị)**

```bash
# Windows (CMD)
scripts\build_gogame_py_simple.bat

# Linux/Mac/Git Bash
bash scripts/build_gogame_py_simple.sh
```

Script sẽ tự động:
- Kiểm tra CMake và pybind11
- Build module
- Kiểm tra file output
- Hiển thị hướng dẫn sử dụng

**Cách 2: Build thủ công**

```bash
# Từ thư mục root
mkdir -p build
cd build

# Configure CMake
cmake ..

# Kiểm tra output - phải thấy: "pybind11 found. Building Python bindings."
# Nếu thấy "pybind11 not found. Skipping Python bindings." → quay lại Bước 1

# Build target gogame_py (quan trọng!)
cmake --build . --target gogame_py

# Hoặc build tất cả
cmake --build .
```

#### Bước 3: Kiểm tra file đã được tạo

**Windows:**
```bash
# File sẽ có tên dạng: gogame_py.cp3XX-*.pyd
# Ví dụ: gogame_py.cp312-mingw_x86_64_msvcrt_gnu.pyd
ls build/gogame_py*.pyd
```

**Linux/macOS:**
```bash
# File sẽ có tên dạng: gogame_py.cpython-*.so
ls build/gogame_py*.so
```

#### Bước 4: Sử dụng module

**Option A: Thêm build/ vào PYTHONPATH**
```bash
# Windows (PowerShell)
$env:PYTHONPATH = "$PWD\build;$env:PYTHONPATH"

# Linux/Mac
export PYTHONPATH="$PWD/build:$PYTHONPATH"
```

**Option B: Copy vào backend/**
```bash
# Tìm file module (tên có thể khác tùy Python version)
# Windows
copy build\gogame_py*.pyd backend\gogame_py.pyd

# Linux/Mac
cp build/gogame_py*.so backend/gogame_py.so
```

**Option C: Dùng script tự động**
```bash
# Windows (MSYS2 shell)
python scripts/install_gogame_py.py

# Linux/Mac
python3 scripts/install_gogame_py.py
```

#### Troubleshooting

**Vấn đề 1: Không thấy file `.pyd` hoặc `.so` sau khi build**

**Nguyên nhân:**
- pybind11 chưa được cài đặt hoặc CMake không tìm thấy
- Build target không được chỉ định

**Giải pháp:**
```bash
# 1. Kiểm tra pybind11
python -c "import pybind11; print(pybind11.get_cmake_dir())"

# 2. Xem output của cmake .. - phải có "pybind11 found"
cd build
cmake ..  # Xem output

# 3. Build lại với target cụ thể
cmake --build . --target gogame_py --verbose

# 4. Kiểm tra file trong build/
ls -la build/ | grep gogame_py
```

**Vấn đề 2: Module không import được**

**Nguyên nhân:**
- File ở vị trí khác
- Thiếu DLL dependencies (Windows)
- Python version không khớp

**Giải pháp:**
```bash
# 1. Tìm file module
find build -name "gogame_py*" -type f

# 2. Kiểm tra Python version khớp với file
python --version
# File .pyd có cp312 → cần Python 3.12

# 3. Test import
cd build
python -c "import gogame_py; print('OK')"
```

**Vấn đề 3: Lỗi DLL không tìm thấy (Windows)**

**Giải pháp:**
- Copy các DLL cần thiết từ MSYS2 vào cùng thư mục với `.pyd`
- Hoặc dùng script: `scripts/setup_gogame_py_for_backend.sh`

### 2.7. Chạy Backend Server

```bash
# Windows
python run.bat
# hoặc
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Linux/Mac
./run.sh
# hoặc
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend sẽ chạy tại: `http://localhost:8000`

API docs: `http://localhost:8000/docs`

## Bước 3: Setup Frontend Web

### 3.1. Cài đặt dependencies

```bash
cd frontend-web
npm install
```

### 3.2. Cấu hình môi trường (nếu cần)

```bash
# Tạo file .env nếu cần thay đổi backend URL
# BACKEND_URL=http://localhost:8000
```

### 3.3. Chạy Development Server

```bash
npm run dev
```

Frontend sẽ chạy tại: `http://localhost:5173` (hoặc port khác nếu 5173 đã được sử dụng)

### 3.4. Build Production

```bash
npm run build
```

Output sẽ ở trong thư mục `dist/`

## Bước 4: Kiểm tra

1. **Backend Health Check**: Truy cập `http://localhost:8000/health`
2. **Frontend**: Truy cập `http://localhost:5173`
3. **API Docs**: Truy cập `http://localhost:8000/docs`

## Troubleshooting

### Lỗi kết nối database

- Kiểm tra PostgreSQL và MongoDB đang chạy
- Kiểm tra thông tin trong file `.env`
- Kiểm tra firewall/port

### Lỗi build C++

- Đảm bảo đã cài CMake và C++ compiler
- Kiểm tra CMakeLists.txt
- Xem `docs/SystemSpec.md` để biết thêm chi tiết

### Lỗi frontend

- Xóa `node_modules` và `package-lock.json`, sau đó chạy lại `npm install`
- Kiểm tra Node.js version: `node --version` (cần 18+)

## Cấu trúc dự án

```
GoGame/
├── backend/           # FastAPI backend
│   ├── app/          # Application code
│   ├── migrations/   # Database migrations
│   └── requirements.txt
├── frontend-web/     # React frontend
│   ├── src/         # Source code
│   └── package.json
├── src/             # C++ AI engine source
├── docs/            # Documentation
└── README.md        # Tổng quan dự án
```

## Tài liệu tham khảo

- `docs/SystemSpec.md`: Thiết kế tổng quan
- `docs/BackendDesign.md`: Thiết kế backend
- `docs/FRONTEND_GUIDE.md`: Hướng dẫn frontend
- `backend/README.md`: Chi tiết backend
- `frontend-web/README.md`: Chi tiết frontend

