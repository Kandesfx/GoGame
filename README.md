# GoGame – Dự án AI chơi Cờ Vây

Ứng dụng web chơi Cờ Vây với AI engine mạnh mẽ, hỗ trợ chơi với AI, chơi online (PvP), matchmaking tự động, và hệ thống xếp hạng.

## 🚀 Quick Start

### Backend (Tối thiểu - không cần AI)

Xem [SETUP_MINIMAL.md](SETUP_MINIMAL.md) để setup backend nhanh nhất.

### Backend (Đầy đủ - có AI)

```bash
# 1. Setup backend
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Build C++ AI module (xem README_GOGAME_PY.md để biết chi tiết)
cd ..
# Trong MSYS2 MinGW 64-bit shell:
cd build
cmake .. -G "Ninja" -DCMAKE_BUILD_TYPE=Release
cmake --build . --target gogame_py

# 3. Chạy backend
cd ../backend
uvicorn app.main:app --reload
```

### Frontend

```bash
cd frontend-web
npm install
npm run dev
```

Xem [backend/QUICK_START.md](backend/QUICK_START.md) và [README_GOGAME_PY.md](README_GOGAME_PY.md) để biết chi tiết.

## 📁 Cấu trúc dự án

```
GoGame/
├── backend/              # FastAPI backend
│   ├── app/             # Application code
│   ├── migrations/    # Database migrations
│   ├── scripts/       # Utility scripts
│   └── requirements.txt
├── frontend-web/      # React frontend (Vite)
│   ├── src/          # Source code
│   └── package.json
├── frontend/          # Frontend cũ (nếu có)
├── src/              # C++ AI engine source
│   ├── ai/          # AI algorithms (Minimax, MCTS)
│   │   ├── minimax/ # Minimax engine
│   │   └── mcts/    # MCTS engine
│   ├── game/        # Game logic (Board, Move, etc.)
│   └── bindings/    # Python bindings (pybind11)
├── build/            # Build output (CMake)
│   └── gogame_py*.pyd  # Python module (sau khi build)
├── scripts/          # Utility scripts
├── docs/             # Documentation
└── README.md         # This file
```

## 📚 Tài liệu

### Setup & Installation

- **[README_GOGAME_PY.md](README_GOGAME_PY.md)** - ⭐ **Hướng dẫn build C++ AI module** (QUAN TRỌNG)
- [SETUP_MINIMAL.md](SETUP_MINIMAL.md) - Setup backend tối thiểu (không cần C++ engine)
- [INSTALLATION.md](INSTALLATION.md) - Hướng dẫn cài đặt từ đầu trên máy mới
- [SETUP.md](SETUP.md) - Hướng dẫn setup đầy đủ (có C++ engine)
- [backend/README.md](backend/README.md) - Tài liệu backend chi tiết
- [backend/QUICK_START.md](backend/QUICK_START.md) - Quick start backend

### Design & Architecture

- [docs/SystemSpec.md](docs/SystemSpec.md) - Thiết kế tổng quan, kiến trúc và roadmap
- [docs/BackendDesign.md](docs/BackendDesign.md) - Thiết kế backend
- [docs/FRONTEND_GUIDE.md](docs/FRONTEND_GUIDE.md) - Hướng dẫn frontend
- [docs/AI_OPTIMIZATION.md](docs/AI_OPTIMIZATION.md) - Tối ưu hóa AI

### Deployment

- [DEPLOY_BACKEND.md](DEPLOY_BACKEND.md) - Hướng dẫn deploy backend
- [DEPLOY_QUICK_START.md](DEPLOY_QUICK_START.md) - Quick start deployment
- [FLYIO_QUICK_START.md](FLYIO_QUICK_START.md) - Deploy lên Fly.io

### Development

- [CONTRIBUTING.md](CONTRIBUTING.md) - Hướng dẫn đóng góp
- [scripts/README.md](scripts/README.md) - Hướng dẫn scripts

## 🛠️ Yêu cầu môi trường

### Backend

- **Python 3.10+** (khuyến nghị dùng venv)
- **PostgreSQL 14+**
- **MongoDB 6+** (tùy chọn, cho game state)
- **FastAPI, SQLAlchemy, Alembic** (tự động cài qua `requirements.txt`)

### C++ AI Engine (Tùy chọn - chỉ cần nếu muốn dùng AI)

- **MSYS2 MinGW 64-bit** (Windows) hoặc **GCC/Clang** (Linux/Mac)
- **CMake 3.20+**
- **pybind11** (cài qua MSYS2: `pacman -S mingw-w64-x86_64-pybind11`)
- **Ninja** (tùy chọn, nhanh hơn: `pacman -S mingw-w64-x86_64-ninja`)

### Frontend

- **Node.js 18+**
- **npm** hoặc **yarn**

## 🔨 Build C++ AI Module

**⚠️ QUAN TRỌNG**: Module `gogame_py` phải được build và đặt trong thư mục `build/` để backend nhận diện đúng.

### Windows (MSYS2)

```bash
# 1. Mở "MSYS2 MinGW 64-bit" shell
# 2. Cài dependencies (nếu chưa có)
pacman -S mingw-w64-x86_64-cmake
pacman -S mingw-w64-x86_64-pybind11
pacman -S mingw-w64-x86_64-ninja  # Tùy chọn

# 3. Build module
cd /d/Hai/study/TTNT/GoGame  # Hoặc đường dẫn project của bạn
mkdir -p build
cd build
cmake .. -G "Ninja" -DCMAKE_BUILD_TYPE=Release
cmake --build . --target gogame_py
```

### Linux/Mac

```bash
# 1. Cài dependencies
sudo apt install cmake build-essential  # Linux
# hoặc
brew install cmake  # Mac

pip install pybind11  # Hoặc cài qua package manager

# 2. Build module
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --target gogame_py
```

### Kiểm tra Build Thành Công

```bash
# Kiểm tra file đã được tạo
ls -lh build/gogame_py*.pyd  # Windows/Linux
# hoặc
dir build\gogame_py*.pyd     # Windows CMD

# Test với MSYS2 Python
/c/msys64/mingw64/bin/python3 -c "import sys; sys.path.insert(0, 'build'); import gogame_py; print('OK')"
```

**Xem [README_GOGAME_PY.md](README_GOGAME_PY.md) để biết chi tiết đầy đủ về build, troubleshooting, và sử dụng module.**

## 🎮 Tính năng

- 🎯 **Chơi với AI** - 4 mức độ khó (Dễ, Trung bình, Khó, Siêu khó)
- 👥 **Chơi online (PvP)** - Chơi với người chơi khác
- 🔍 **Matchmaking tự động** - Ghép trận dựa trên ELO và board size
- 📊 **Hệ thống xếp hạng** - ELO rating và leaderboard
- 📈 **Thống kê chi tiết** - Lịch sử trận đấu, win rate, v.v.
- ⏱️ **Time control** - Giới hạn thời gian cho PvP matches
- 🔄 **Undo moves** - Hoàn tác nước đi
- 🎨 **UI/UX hiện đại** - Giao diện đẹp, responsive
- 🏆 **Room code** - Tạo phòng và tham gia bằng mã phòng

## 🛠️ Công nghệ sử dụng

### Backend
- **FastAPI** - Web framework
- **PostgreSQL** - SQL database (users, matches, ratings)
- **MongoDB** - NoSQL database (game states)
- **SQLAlchemy** - ORM
- **Alembic** - Database migrations
- **JWT** - Authentication
- **Motor** - Async MongoDB driver

### Frontend
- **React** - UI framework
- **Vite** - Build tool
- **Axios** - HTTP client
- **React Router** - Routing

### AI Engine
- **C++** - Core AI algorithms
- **Minimax** - Search algorithm với alpha-beta pruning
- **MCTS** - Monte Carlo Tree Search
- **pybind11** - Python bindings
- **ML Models** - PolicyNet, ValueNet (tùy chọn)

## 📦 Cài đặt

### 1. Clone repository

```bash
git clone <repository-url>
cd GoGame
```

### 2. Setup Backend

Xem [backend/README.md](backend/README.md) hoặc [backend/QUICK_START.md](backend/QUICK_START.md) để biết chi tiết.

**Tóm tắt:**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Setup database
python scripts/setup_database.py

# Chạy server
uvicorn app.main:app --reload
```

### 3. Build C++ AI Module (Tùy chọn)

Xem [README_GOGAME_PY.md](README_GOGAME_PY.md) để biết chi tiết.

**Tóm tắt:**
```bash
# Trong MSYS2 MinGW 64-bit shell
cd build
cmake .. -G "Ninja" -DCMAKE_BUILD_TYPE=Release
cmake --build . --target gogame_py
```

### 4. Setup Frontend

```bash
cd frontend-web
npm install
npm run dev
```

## 🧪 Testing

### Test Backend

```bash
cd backend
pytest tests/
```

### Test AI Module

```bash
# Test với MSYS2 Python
/c/msys64/mingw64/bin/python3 scripts/test_gogame_py_msys2.py

# Hoặc test với script
python scripts/check_ai_availability.py
```

### Test API

```bash
# Start server trước
cd backend
uvicorn app.main:app --reload

# Trong terminal khác
python scripts/test_backend_with_ai.py
```

## 🐛 Troubleshooting

### Backend không nhận diện AI module

1. **Kiểm tra file có trong `build/` không:**
   ```bash
   ls -lh build/gogame_py*.pyd
   ```

2. **Kiểm tra với script:**
   ```bash
   python scripts/check_ai_availability.py
   ```

3. **Kiểm tra logs khi start backend:**
   - Nếu thấy: `✅ gogame_py module loaded successfully from build directory` → OK
   - Nếu thấy: `⚠️ gogame_py module found but NOT in build directory` → File ở sai vị trí

4. **Xem [README_GOGAME_PY.md](README_GOGAME_PY.md)** để biết thêm troubleshooting.

### Build errors

- **CMake không tìm thấy pybind11**: Cài qua MSYS2: `pacman -S mingw-w64-x86_64-pybind11`
- **DLL load failed**: Module build với MinGW nhưng Python dùng MSVC → Dùng MSYS2 Python hoặc wrapper
- **"ninja: no work to do"**: Module đã build, không cần rebuild (bình thường)

Xem [README_GOGAME_PY.md](README_GOGAME_PY.md) để biết thêm troubleshooting.

## 📝 License

MIT License - Xem [LICENSE](LICENSE) để biết thêm chi tiết.

## 🤝 Đóng góp

Chúng tôi hoan nghênh mọi đóng góp! Xem [CONTRIBUTING.md](CONTRIBUTING.md) để biết hướng dẫn.

## 📞 Liên hệ & Support

Nếu có câu hỏi hoặc vấn đề:
- Tạo issue trên GitHub
- Xem tài liệu trong thư mục `docs/`
- Kiểm tra [README_GOGAME_PY.md](README_GOGAME_PY.md) cho vấn đề về AI module

## 🎯 Roadmap

- [ ] Cải thiện AI engine (thêm heuristics, opening book)
- [ ] ML model training và integration
- [ ] Tournament mode
- [ ] Replay và analysis tools
- [ ] Mobile app (React Native)
- [ ] Spectator mode
- [ ] Chat system

---

**Lưu ý**: Để sử dụng AI features, bạn cần build C++ module. Xem [README_GOGAME_PY.md](README_GOGAME_PY.md) để biết chi tiết.
