# Hướng dẫn Cài đặt từ Đầu - Máy Mới

Tài liệu này hướng dẫn cài đặt tất cả các công cụ và dependencies cần thiết cho dự án GoGame trên máy mới (chưa có gì được cài đặt).

## 📋 Mục lục

1. [Windows](#windows)
2. [Linux (Ubuntu/Debian)](#linux-ubuntudebian)
3. [macOS](#macos)
4. [Kiểm tra sau khi cài đặt](#kiểm-tra-sau-khi-cài-đặt)

---

## Windows

### 1. Cài đặt Git

1. Tải Git từ: https://git-scm.com/download/win
2. Chạy installer và chọn các tùy chọn mặc định
3. Kiểm tra:
   ```bash
   git --version
   ```

### 2. Cài đặt Python 3.10+

1. Tải Python từ: https://www.python.org/downloads/
2. **Quan trọng**: Khi cài đặt, chọn "Add Python to PATH"
3. Kiểm tra:
   ```bash
   python --version
   # hoặc
   python3 --version
   pip --version
   ```

### 3. Cài đặt Node.js 18+

1. Tải Node.js từ: https://nodejs.org/
2. Chọn phiên bản LTS (Long Term Support)
3. Chạy installer với các tùy chọn mặc định
4. Kiểm tra:
   ```bash
   node --version
   npm --version
   ```

### 4. Cài đặt PostgreSQL 14+

#### Cách 1: Sử dụng Installer

1. Tải PostgreSQL từ: https://www.postgresql.org/download/windows/
2. Chạy installer
3. **Ghi nhớ password** cho user `postgres` (mặc định)
4. Port mặc định: `5432`
5. Kiểm tra:
   ```bash
   psql --version
   ```

#### Cách 2: Sử dụng Docker (Khuyến nghị)

1. Cài đặt Docker Desktop: https://www.docker.com/products/docker-desktop/
2. Chạy PostgreSQL container:
   ```bash
   docker run --name gogame-postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=gogame -p 5432:5432 -d postgres:14
   ```

### 5. Cài đặt MongoDB 5.0+

#### Cách 1: Sử dụng Installer

1. Tải MongoDB Community Server từ: https://www.mongodb.com/try/download/community
2. Chọn Windows x64
3. Chạy installer với các tùy chọn mặc định
4. MongoDB sẽ chạy như một Windows Service
5. Kiểm tra:
   ```bash
   mongosh --version
   # hoặc
   mongo --version
   ```

#### Cách 2: Sử dụng Docker (Khuyến nghị)

1. Chạy MongoDB container:
   ```bash
   docker run --name gogame-mongo -p 27017:27017 -d mongo:5.0
   ```

### 6. Cài đặt CMake 3.15+

1. Tải CMake từ: https://cmake.org/download/
2. Chọn "Windows x64 Installer"
3. **Quan trọng**: Khi cài đặt, chọn "Add CMake to system PATH"
4. Kiểm tra:
   ```bash
   cmake --version
   ```

### 7. Cài đặt C++ Compiler và MSYS2 (Cho AI Engine)

#### Cách 1: MSYS2 MinGW (Khuyến nghị cho Windows - Cần cho AI Engine)

MSYS2 là môi trường phát triển Unix-like trên Windows, cần thiết để build và chạy C++ AI engine.

##### Bước 1: Tải và cài đặt MSYS2

1. Tải MSYS2 từ: https://www.msys2.org/
2. Chạy installer (`msys2-x86_64-*.exe`)
3. Chọn thư mục cài đặt (mặc định: `C:\msys64`)
4. Hoàn tất cài đặt

##### Bước 2: Cập nhật MSYS2

1. Mở **"MSYS2 MSYS"** terminal (không phải MinGW)
2. Cập nhật package database:
   ```bash
   pacman -Syu
   ```
   ⚠️ **Lưu ý**: Sau khi cập nhật xong, terminal sẽ tự đóng. Bạn cần **mở lại terminal** và chạy lại lệnh `pacman -Syu` một lần nữa để hoàn tất cập nhật.

##### Bước 3: Cài đặt C++ Compiler và Build Tools

1. Mở **"MSYS2 MinGW 64-bit"** terminal (không phải MSYS)
2. Cài đặt GCC/G++:
   ```bash
   pacman -S mingw-w64-x86_64-gcc
   pacman -S mingw-w64-x86_64-gdb
   pacman -S mingw-w64-x86_64-make
   ```
3. Cài đặt CMake:
   ```bash
   pacman -S mingw-w64-x86_64-cmake
   ```
4. Cài đặt Python 3 (cần cho AI wrapper):
   ```bash
   pacman -S mingw-w64-x86_64-python3
   pacman -S mingw-w64-x86_64-python-pip
   ```
5. Cài đặt pybind11 (cần cho Python bindings):
   ```bash
   pacman -S mingw-w64-x86_64-pybind11
   ```

##### Bước 4: Thêm MSYS2 vào PATH (Windows)

1. Mở **"Edit the system environment variables"**:
   - Nhấn `Win + R`
   - Gõ `sysdm.cpl` và nhấn Enter
   - Chọn tab "Advanced"
   - Click "Environment Variables"
2. Trong "System variables", tìm và chọn "Path", click "Edit"
3. Thêm các đường dẫn sau (nếu chưa có):
   - `C:\msys64\mingw64\bin`
   - `C:\msys64\usr\bin`
4. Click "OK" để lưu
5. **Quan trọng**: Đóng tất cả terminal/command prompt và mở lại để áp dụng thay đổi PATH

##### Bước 5: Kiểm tra cài đặt

Mở **Command Prompt** hoặc **PowerShell** mới và kiểm tra:

```bash
# Kiểm tra GCC
g++ --version
# Kết quả mong đợi: g++ (RevX, Built by MSYS2 project) x.x.x

# Kiểm tra CMake
cmake --version
# Kết quả mong đợi: cmake version x.x.x

# Kiểm tra Python từ MSYS2
C:\msys64\mingw64\bin\python3.exe --version
# Kết quả mong đợi: Python 3.x.x
```

##### Bước 6: Kiểm tra Python trong MSYS2

Mở **"MSYS2 MinGW 64-bit"** terminal và kiểm tra:

```bash
# Kiểm tra Python
python3 --version

# Kiểm tra pip
pip3 --version

# Kiểm tra pybind11
python3 -c "import pybind11; print(pybind11.__version__)"
```

##### Lưu ý quan trọng về MSYS2

- **MSYS2 MSYS**: Terminal Unix-like, dùng để quản lý packages (`pacman`)
- **MSYS2 MinGW 64-bit**: Terminal với MinGW compiler, dùng để build và chạy code
- **Python trong MSYS2**: Đường dẫn mặc định: `C:\msys64\mingw64\bin\python3.exe`
- **PATH**: Cần thêm `C:\msys64\mingw64\bin` vào PATH để dùng từ Command Prompt

#### Cách 2: Visual Studio Build Tools

1. Tải Visual Studio Build Tools từ: https://visualstudio.microsoft.com/downloads/
2. Chọn "Build Tools for Visual Studio"
3. Trong installer, chọn "Desktop development with C++"
4. Cài đặt
5. Mở "Developer Command Prompt for VS" để sử dụng

#### Cách 3: WSL (Windows Subsystem for Linux)

1. Mở PowerShell với quyền Administrator
2. Cài đặt WSL:
   ```powershell
   wsl --install
   ```
3. Khởi động lại máy
4. Sau khi khởi động lại, làm theo hướng dẫn [Linux](#linux-ubuntudebian)

---

## Linux (Ubuntu/Debian)

### 1. Cập nhật hệ thống

```bash
sudo apt update
sudo apt upgrade -y
```

### 2. Cài đặt Git

```bash
sudo apt install git -y
git --version
```

### 3. Cài đặt Python 3.10+

```bash
sudo apt install python3 python3-pip python3-venv -y
python3 --version
pip3 --version
```

### 4. Cài đặt Node.js 18+

#### Cách 1: Sử dụng NodeSource (Khuyến nghị)

```bash
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs
node --version
npm --version
```

#### Cách 2: Sử dụng nvm (Node Version Manager)

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
source ~/.bashrc
nvm install 18
nvm use 18
node --version
npm --version
```

### 5. Cài đặt PostgreSQL 14+

```bash
sudo apt install postgresql postgresql-contrib -y
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Tạo database và user
sudo -u postgres psql
```

Trong PostgreSQL shell:
```sql
CREATE DATABASE gogame;
CREATE USER gogame_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE gogame TO gogame_user;
\q
```

Hoặc sử dụng Docker:
```bash
docker run --name gogame-postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=gogame -p 5432:5432 -d postgres:14
```

### 6. Cài đặt MongoDB 5.0+

#### Cách 1: Sử dụng MongoDB Repository

```bash
# Import MongoDB public GPG key
curl -fsSL https://www.mongodb.org/static/pgp/server-5.0.asc | sudo gpg -o /usr/share/keyrings/mongodb-server-5.0.gpg --dearmor

# Add MongoDB repository
echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-5.0.gpg ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/5.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-5.0.list

# Install MongoDB
sudo apt update
sudo apt install -y mongodb-org

# Start MongoDB
sudo systemctl start mongod
sudo systemctl enable mongod

# Check status
sudo systemctl status mongod
```

#### Cách 2: Sử dụng Docker

```bash
docker run --name gogame-mongo -p 27017:27017 -d mongo:5.0
```

### 7. Cài đặt CMake 3.15+

```bash
sudo apt install cmake -y
cmake --version
```

### 8. Cài đặt C++ Build Tools

```bash
sudo apt install build-essential -y
g++ --version
make --version
```

### 9. Cài đặt Docker (Tùy chọn - nếu muốn dùng Docker cho databases)

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group (để chạy docker không cần sudo)
sudo usermod -aG docker $USER

# Log out và log in lại để áp dụng thay đổi
```

---

## macOS

### 1. Cài đặt Homebrew (Package Manager)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. Cài đặt Git

```bash
brew install git
git --version
```

### 3. Cài đặt Python 3.10+

```bash
brew install python@3.10
python3 --version
pip3 --version
```

### 4. Cài đặt Node.js 18+

```bash
brew install node@18
node --version
npm --version
```

### 5. Cài đặt PostgreSQL 14+

```bash
brew install postgresql@14
brew services start postgresql@14

# Tạo database
createdb gogame
```

Hoặc sử dụng Docker:
```bash
docker run --name gogame-postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=gogame -p 5432:5432 -d postgres:14
```

### 6. Cài đặt MongoDB 5.0+

```bash
brew tap mongodb/brew
brew install mongodb-community@5.0
brew services start mongodb-community@5.0
```

Hoặc sử dụng Docker:
```bash
docker run --name gogame-mongo -p 27017:27017 -d mongo:5.0
```

### 7. Cài đặt CMake 3.15+

```bash
brew install cmake
cmake --version
```

### 8. Cài đặt C++ Build Tools

```bash
# Xcode Command Line Tools (bao gồm g++, make, etc.)
xcode-select --install
```

### 9. Cài đặt Docker (Tùy chọn)

Tải Docker Desktop từ: https://www.docker.com/products/docker-desktop/

---

## Kiểm tra sau khi cài đặt

Sau khi cài đặt tất cả các công cụ, chạy các lệnh sau để kiểm tra:

```bash
# Git
git --version

# Python
python --version  # hoặc python3 --version
pip --version     # hoặc pip3 --version

# Node.js
node --version
npm --version

# PostgreSQL
psql --version
# Kiểm tra kết nối
psql -U postgres -c "SELECT version();"

# MongoDB
mongosh --version
# Kiểm tra kết nối
mongosh --eval "db.version()"

# CMake
cmake --version

# C++ Compiler
g++ --version
# hoặc
clang++ --version  # trên macOS

# MSYS2 Python (Windows - cho AI engine)
# Từ Command Prompt/PowerShell:
C:\msys64\mingw64\bin\python3.exe --version
# Hoặc trong MSYS2 MinGW 64-bit terminal:
python3 --version
```

## Bước tiếp theo

Sau khi đã cài đặt tất cả các công cụ, tiếp tục với [SETUP.md](SETUP.md) để setup dự án.

## Troubleshooting

### Python không tìm thấy

**Windows:**
- Đảm bảo đã chọn "Add Python to PATH" khi cài đặt
- Thêm thủ công vào PATH: `C:\Users\<username>\AppData\Local\Programs\Python\Python3.x`

**Linux/macOS:**
- Sử dụng `python3` thay vì `python`
- Kiểm tra: `which python3`

### Node.js không tìm thấy

**Windows:**
- Thêm Node.js vào PATH: `C:\Program Files\nodejs`

**Linux:**
- Nếu dùng nvm, đảm bảo đã source `~/.bashrc` hoặc `~/.zshrc`

### PostgreSQL không kết nối được

**Windows:**
- Kiểm tra service đang chạy: Services → PostgreSQL
- Kiểm tra port 5432 không bị firewall chặn

**Linux:**
- Kiểm tra service: `sudo systemctl status postgresql`
- Kiểm tra port: `sudo netstat -tlnp | grep 5432`

### MongoDB không kết nối được

**Windows:**
- Kiểm tra MongoDB service đang chạy
- Kiểm tra port 27017

**Linux:**
- Kiểm tra service: `sudo systemctl status mongod`
- Kiểm tra port: `sudo netstat -tlnp | grep 27017`

### CMake không tìm thấy

**Windows:**
- Đảm bảo đã chọn "Add CMake to system PATH" khi cài đặt
- Thêm thủ công: `C:\Program Files\CMake\bin`

**Linux/macOS:**
- Kiểm tra: `which cmake`
- Nếu không có, cài đặt lại: `sudo apt install cmake` (Linux) hoặc `brew install cmake` (macOS)

### C++ Compiler không tìm thấy

**Windows (MSYS2):**
- Đảm bảo đã thêm `C:\msys64\mingw64\bin` vào PATH
- Mở terminal mới sau khi thêm PATH
- Kiểm tra trong "MSYS2 MinGW 64-bit" terminal: `g++ --version`
- Nếu vẫn không tìm thấy, cài đặt lại: `pacman -S mingw-w64-x86_64-gcc`

### MSYS2 Python không tìm thấy

**Vấn đề**: Backend báo lỗi "MSYS2 Python not found"

**Giải pháp**:
1. Kiểm tra Python có được cài trong MSYS2:
   ```bash
   # Trong MSYS2 MinGW 64-bit terminal
   python3 --version
   ```
2. Nếu không có, cài đặt:
   ```bash
   pacman -S mingw-w64-x86_64-python3
   ```
3. Kiểm tra đường dẫn: `C:\msys64\mingw64\bin\python3.exe` phải tồn tại
4. Nếu đường dẫn khác, cập nhật `backend/app/utils/ai_wrapper.py`:
   ```python
   MSYS2_PYTHON = Path("C:/msys64/mingw64/bin/python3.exe")  # Đổi đường dẫn nếu cần
   ```

**Linux:**
- Cài đặt: `sudo apt install build-essential`

**macOS:**
- Cài đặt Xcode Command Line Tools: `xcode-select --install`

## Cần giúp đỡ?

Nếu gặp vấn đề, vui lòng:
1. Kiểm tra lại các bước cài đặt
2. Xem phần Troubleshooting ở trên
3. Tạo issue trên GitHub với thông tin chi tiết về lỗi

