# Setup Guide - ReactJS Frontend

## Prerequisites

Cần cài đặt **Node.js** và **npm** trước khi chạy frontend.

### Windows

1. **Download Node.js:**
   - Truy cập: https://nodejs.org/
   - Download LTS version (recommended)
   - Chạy installer và follow instructions

2. **Verify Installation:**
   ```bash
   node --version
   npm --version
   ```

3. **Alternative: Sử dụng Chocolatey (nếu đã có):**
   ```bash
   choco install nodejs
   ```

### Linux (Ubuntu/Debian)

```bash
# Sử dụng NodeSource repository
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# Verify
node --version
npm --version
```

### Mac

```bash
# Sử dụng Homebrew
brew install node

# Verify
node --version
npm --version
```

## Setup Project

Sau khi đã cài Node.js:

```bash
# 1. Navigate to frontend-web directory
cd frontend-web

# 2. Install dependencies
npm install

# 3. Create .env file
echo "VITE_API_URL=http://localhost:8000" > .env

# 4. Start development server
npm run dev
```

## Troubleshooting

### npm: command not found

**Windows:**
- Đảm bảo Node.js đã được cài đặt
- Restart terminal/command prompt sau khi cài
- Check PATH environment variable có chứa Node.js

**Linux/Mac:**
- Đảm bảo đã cài Node.js
- Có thể cần thêm vào PATH:
  ```bash
  export PATH=$PATH:/usr/local/bin
  ```

### Port 3000 already in use

Nếu port 3000 đã được sử dụng:
- Thay đổi port trong `vite.config.js`:
  ```js
  server: {
    port: 3001,  // Change to different port
  }
  ```

### CORS Errors

Nếu gặp CORS errors khi gọi API:
- Đảm bảo backend đã config CORS trong `backend/app/main.py`
- Check `CORS_ORIGINS` trong `.env` của backend có chứa `http://localhost:3000`

## Quick Start (After Node.js installed)

```bash
cd frontend-web
npm install
npm run dev
```

Mở browser: `http://localhost:3000`

