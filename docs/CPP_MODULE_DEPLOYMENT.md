# 🔧 Deploy C++ Module (gogame_py) trên Production

## 📋 Tổng Quan

Module `gogame_py` là Python binding cho C++ AI engine. Khi deploy trên production (Linux), cần build lại từ source vì:
- Module build trên Windows (`.pyd`) không chạy trên Linux
- Cần build cho Linux (`.so`) với cùng Python version

## 🎯 Các Phương Án

### Option 1: Build trong Docker (Khuyến nghị)

**Ưu điểm:**
- Tự động build khi deploy
- Đảm bảo compatibility với Python version trong container
- Không cần setup build environment riêng

**Cách làm:**
- Dùng multi-stage Dockerfile (đã có trong `backend/Dockerfile`)
- Build context là root directory để access `CMakeLists.txt` và `src/`

**Fly.io:**
```toml
[build]
  dockerfile = "backend/Dockerfile"
  dockerfile_context = "../"
```

### Option 2: Pre-build và Copy

**Ưu điểm:**
- Build nhanh hơn khi deploy
- Có thể build trên máy mạnh hơn

**Cách làm:**

1. **Build trên máy local (Linux) hoặc CI/CD:**

```bash
# Trên máy Linux hoặc GitHub Actions
cd /path/to/GoGame
mkdir -p build
cd build
cmake ..
cmake --build . --target gogame_py

# Copy module
cp gogame_py*.so ../backend/gogame_py.so
```

2. **Commit module vào repo (không khuyến nghị cho production)**

3. **Hoặc upload lên artifact storage và download trong Dockerfile:**

```dockerfile
# Download pre-built module
RUN curl -L https://your-artifact-storage.com/gogame_py.so -o /app/gogame_py.so
```

### Option 3: Build trên CI/CD và Inject vào Docker

**Ưu điểm:**
- Build một lần, dùng nhiều lần
- Có thể cache build artifacts

**Cách làm với GitHub Actions:**

```yaml
# .github/workflows/build-cpp.yml
name: Build C++ Module

on:
  push:
    branches: [master]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build gogame_py
        run: |
          mkdir build && cd build
          cmake ..
          cmake --build . --target gogame_py
      - name: Upload artifact
        uses: actions/upload-artifact@v3
        with:
          name: gogame_py
          path: build/gogame_py*.so
```

### Option 4: Không Build (Fallback Mode)

**Khi nào dùng:**
- Chỉ cần PvP matches (không cần AI)
- Chưa sẵn sàng build C++ module
- Testing backend features khác

**Hành vi:**
- Backend vẫn chạy được
- AI features bị disable
- Log: `WARNING: gogame_py module not found. AI features will be disabled.`

**Dùng Dockerfile đơn giản:**
```bash
# Dùng Dockerfile.simple (không build C++)
fly deploy --dockerfile backend/Dockerfile.simple
```

## 🔧 Build trong Docker (Chi Tiết)

### Dockerfile Multi-Stage

```dockerfile
# Stage 1: Build C++ module
FROM python:3.10-slim AS builder

WORKDIR /build

# Install build tools
RUN apt-get update && apt-get install -y \
    build-essential cmake git libpython3.10-dev

# Install pybind11
RUN pip install pybind11

# Copy source
COPY CMakeLists.txt ./
COPY src ./src

# Build
RUN mkdir build && cd build && \
    cmake .. && \
    cmake --build . --target gogame_py

# Stage 2: Python backend
FROM python:3.10-slim

# Copy module from builder
COPY --from=builder /build/build/gogame_py*.so /app/gogame_py.so

# ... rest of Dockerfile
```

### Build Context

**Quan trọng**: Dockerfile cần access `../CMakeLists.txt` và `../src`, nên build context phải là root directory:

```bash
# Build từ root
docker build -f backend/Dockerfile -t gogame-backend .

# Hoặc trong fly.toml
[build]
  dockerfile = "backend/Dockerfile"
  dockerfile_context = "../"
```

## 🚀 Deploy trên Fly.io

### Cách 1: Build trong Docker (Tự động)

```bash
cd backend

# fly.toml đã có dockerfile_context = "../"
fly deploy
```

Fly.io sẽ tự động:
1. Build C++ module trong builder stage
2. Copy module vào final image
3. Deploy container

### Cách 2: Pre-build và Copy

```bash
# 1. Build module trên máy local hoặc CI
cd /path/to/GoGame
mkdir -p build && cd build
cmake .. && cmake --build . --target gogame_py

# 2. Copy vào backend
cp gogame_py*.so ../backend/gogame_py.so

# 3. Dùng Dockerfile.simple
cd backend
fly deploy --dockerfile Dockerfile.simple
```

### Kiểm Tra Module

```bash
# SSH vào container
fly ssh console -a gogame-backend

# Test import
python -c "import gogame_py; print('✅ OK:', gogame_py.__file__)"

# Check AI features
python -c "import gogame_py; ai = gogame_py.AIPlayer(); print('✅ AI OK')"
```

## 🐛 Troubleshooting

### Build Failed: "CMakeLists.txt not found"

**Nguyên nhân**: Build context sai

**Fix:**
```toml
# fly.toml
[build]
  dockerfile = "backend/Dockerfile"
  dockerfile_context = "../"  # Quan trọng!
```

### Build Failed: "pybind11 not found"

**Fix:**
```dockerfile
# Thêm vào builder stage
RUN pip install pybind11
```

### Module không import được

**Nguyên nhân:**
- Python version không khớp
- Module không có trong PYTHONPATH

**Fix:**
```bash
# Check Python version
python --version  # Phải là 3.10

# Check module location
ls -la /app/gogame_py.so

# Test import
python -c "import sys; sys.path.insert(0, '/app'); import gogame_py"
```

### Module build nhưng AI không hoạt động

**Check logs:**
```bash
fly logs -a gogame-backend | grep -i "gogame_py\|ai"
```

**Có thể:**
- Module load được nhưng có lỗi runtime
- Check dependencies (C++ libraries)
- Check AI player initialization

## 📊 Performance

**Build time:**
- C++ module: ~2-5 phút (tùy CPU)
- Total Docker build: ~5-10 phút

**Runtime:**
- Module size: ~1-2 MB
- Memory: +10-20 MB khi load module
- AI move time: 0.1-2s (tùy level)

## ✅ Checklist

- [ ] Build context đúng (`dockerfile_context = "../"`)
- [ ] CMakeLists.txt và src/ có trong build context
- [ ] pybind11 đã install trong builder stage
- [ ] Module được copy từ builder stage
- [ ] Python version khớp (3.10)
- [ ] Test import trong container
- [ ] Check logs để verify module load

---

**Lưu ý**: Nếu không build được C++ module, backend vẫn chạy được nhưng AI features sẽ bị disable. PvP matches và các features khác vẫn hoạt động bình thường.

