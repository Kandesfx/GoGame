# Multi-stage build để build C++ module và Python backend

# Stage 1: Build C++ module (gogame_py)
FROM python:3.10-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install pybind11 (cả pip và cmake package)
RUN pip install --no-cache-dir pybind11 && \
    python3 -c "import pybind11; print('pybind11 installed at:', pybind11.get_cmake_dir())"

# Copy source code (build context là root directory)
COPY CMakeLists.txt ./
COPY src ./src

# Build C++ module
# Set pybind11_DIR để CMake tìm được
RUN mkdir -p build && cd build && \
    PYBIND11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())") && \
    cmake .. -Dpybind11_DIR="$PYBIND11_DIR" && \
    cmake --build . --target gogame_py

# Stage 2: Python backend
FROM python:3.10-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY backend/requirements.txt .
# Upgrade pip, setuptools, wheel for better compatibility with Pydantic v2
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy C++ module from builder stage
COPY --from=builder /build/build/gogame_py*.so /app/
# Rename to standard name for easier import
RUN if ls /app/gogame_py*.so 1> /dev/null 2>&1; then \
        mv /app/gogame_py*.so /app/gogame_py.so; \
    fi

# Copy application code (từ backend directory)
COPY backend/__init__.py ./
COPY backend/app ./app
COPY backend/migrations ./migrations
COPY backend/alembic.ini ./
COPY backend/scripts ./scripts

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
