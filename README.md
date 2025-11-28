# GoGame – Dự án AI chơi Cờ Vây

Tài liệu này dùng để ghi chú nhanh các bước setup môi trường và thông tin quan trọng trong quá trình phát triển. Vui lòng cập nhật khi quy trình thay đổi.

## 📁 Cấu trúc dự án

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
│   ├── ai/          # AI algorithms (Minimax, MCTS)
│   └── game/        # Game logic
├── docs/            # Documentation
└── README.md        # This file
```

## 📚 Tài liệu

- [SETUP_MINIMAL.md](SETUP_MINIMAL.md) - **Setup backend tối thiểu (không cần C++ engine)** ⭐ Mới!
- [INSTALLATION.md](INSTALLATION.md) - Hướng dẫn cài đặt từ đầu trên máy mới
- [SETUP.md](SETUP.md) - Hướng dẫn setup đầy đủ (có C++ engine)
- [docs/SystemSpec.md](docs/SystemSpec.md) - Thiết kế tổng quan, kiến trúc và roadmap
- [docs/BackendDesign.md](docs/BackendDesign.md) - Thiết kế backend
- [docs/FRONTEND_GUIDE.md](docs/FRONTEND_GUIDE.md) - Hướng dẫn frontend
- [CONTRIBUTING.md](CONTRIBUTING.md) - Hướng dẫn đóng góp

## 2. Yêu cầu môi trường & toolchain

### 2.1. Windows

Khuyến nghị dùng **MSYS2 MinGW 64-bit**:
```bash
# 1. Cài MSYS2 từ https://www.msys2.org/
# 2. Mở "MSYS2 MSYS" và chạy:
pacman -Syu
# Sau khi update xong, mở lại và chạy:
pacman -Su

# 3. Cài GCC/G++ 64-bit:
pacman -S mingw-w64-x86_64-gcc

# 4. Dùng shell "MSYS2 MinGW 64-bit" để chạy:
g++ --version
```

Nếu cần lựa chọn khác:
- **MinGW-w64 standalone**: tải từ https://www.mingw-w64.org/ và thêm `mingw64/bin` vào `PATH`.
- **MSVC Build Tools**: cài “Desktop development with C++” rồi dùng `cl.exe`.
- **WSL (Ubuntu)**: `sudo apt install build-essential`.

> **Lưu ý:** Nếu shell vẫn báo `g++: command not found`, hãy mở lại đúng terminal (ví dụ “MSYS2 MinGW 64-bit”) hoặc kiểm tra biến `PATH`.

### 2.2. Linux / Server

- Cài đặt trực tiếp: `sudo apt update && sudo apt install build-essential cmake`.
- Khuyến nghị tạo **Docker image** để đảm bảo môi trường đồng nhất:
  ```Dockerfile
  FROM ubuntu:22.04
  RUN apt-get update && apt-get install -y \
      build-essential cmake python3 python3-pip
  ```
- Có thể dùng base image khác (ví dụ `nvidia/cuda`) nếu cần GPU cho ML.

### 2.3. Python & ML

- Python 3.10+ (khuyến nghị dùng venv).
- Cài đặt PyTorch, NumPy:
  ```bash
  python -m venv venv
  source venv/bin/activate     # Windows: venv\Scripts\activate
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # hoặc cpu build
  pip install numpy
  ```
- Các module ML nằm tại `src/ml/` (policy/value networks, self-play skeleton).
- Để chạy self-play training cần build module binding:
  ```bash
  pacman -S mingw-w64-x86_64-pybind11   # MSYS2
  cmake -S . -B build -G "Ninja"
  cmake --build build
  ```
  Sau đó trong venv:
  ```bash
  python -c "import gogame_py"
  ```
  để chắc chắn module đã load được.

## 3. Quy trình build với CMake

### 3.1. Cài đặt CMake (nếu chưa có)

- **MSYS2 MinGW 64-bit**: `pacman -S mingw-w64-x86_64-cmake`
- **Linux**: `sudo apt install cmake`
- **Windows khác**: tải từ https://cmake.org/download/

### 3.2. Build Minimax library

Thực hiện trong shell có compiler (ví dụ “MSYS2 MinGW 64-bit” hoặc WSL):
```bash
cd /d/Hai/study/TTNT/GoGame   # hoặc đường dẫn tương ứng
mkdir -p build
cd build
cmake ..
cmake --build .
```

CMakeLists hiện tạo target `gogame_minimax` (static library). Khi các module khác hoàn thiện, ta sẽ thêm target tương ứng (engine game, binding Python, unit test…).

### 3.3. Kiểm thử nhanh

- Dùng `ctest` khi có test (sẽ bổ sung sau).
- Nếu cần kiểm tra từng file, vẫn có thể dùng `g++ -Isrc -c ...`, nhưng ưu tiên CMake để đảm bảo đồng nhất môi trường.

## ✨ Tính năng

- 🎮 Chơi với AI (4 mức độ khó)
- 👥 Chơi online với người khác (PvP)
- 🎯 Matchmaking tự động dựa trên ELO
- 📊 Hệ thống xếp hạng và leaderboard
- 📈 Thống kê chi tiết
- ⏱️ Time control cho PvP matches
- 🔄 Undo moves
- 🎨 UI/UX hiện đại

## 🛠️ Công nghệ sử dụng

- **Backend**: FastAPI, PostgreSQL, MongoDB, SQLAlchemy, Alembic
- **Frontend**: React, Vite, Axios
- **AI Engine**: C++ (Minimax, MCTS)
- **Authentication**: JWT

## 📝 License

MIT License - Xem [LICENSE](LICENSE) để biết thêm chi tiết.

## 🤝 Đóng góp

Chúng tôi hoan nghênh mọi đóng góp! Xem [CONTRIBUTING.md](CONTRIBUTING.md) để biết hướng dẫn.

## 📞 Liên hệ

Nếu có câu hỏi hoặc vấn đề, vui lòng tạo issue trên GitHub.

