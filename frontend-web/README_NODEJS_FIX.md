# Fix Node.js Path trong Git Bash

## Vấn đề

Node.js đã được cài nhưng Git Bash không nhận được.

## Giải pháp

### Option 1: Quick Fix - Sử dụng script tự động

**Cách nhanh nhất:**

```bash
cd frontend-web
bash fix_nodejs_path.sh
```

Script sẽ tự động thêm Node.js vào PATH cho session hiện tại.

**Để làm permanent, thêm vào ~/.bashrc:**
```bash
echo 'export PATH="/c/Program Files/nodejs:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### Option 2: Sử dụng run.sh script

Script `run.sh` đã tự động fix PATH:

```bash
cd frontend-web
bash run.sh
```

### Option 3: Sử dụng Windows Command Prompt hoặc PowerShell

Thay vì dùng Git Bash, dùng **Command Prompt** hoặc **PowerShell**:

```cmd
cd frontend-web
npm install
npm run dev
```

### Option 4: Fix PATH trong Git Bash manually

1. **Tìm đường dẫn Node.js:**
   - Mở **Command Prompt** (cmd.exe)
   - Chạy: `where node`
   - Copy đường dẫn (ví dụ: `C:\Program Files\nodejs\node.exe`)

2. **Thêm vào PATH trong Git Bash:**
   
   Mở file `~/.bashrc` (hoặc `~/.bash_profile`):
   ```bash
   notepad ~/.bashrc
   # hoặc
   code ~/.bashrc
   ```
   
   Thêm dòng này (thay đường dẫn bằng đường dẫn thực tế):
   ```bash
   export PATH="/c/Program Files/nodejs:$PATH"
   ```
   
   Lưu file và restart Git Bash.

3. **Verify:**
   ```bash
   node --version
   npm --version
   ```

### Option 5: Sử dụng full path

Nếu biết đường dẫn Node.js, có thể dùng trực tiếp:

```bash
# Ví dụ:
"/c/Program Files/nodejs/npm" install
"/c/Program Files/nodejs/npm" run dev
```

## Verify Installation

Sau khi fix, verify:
```bash
node --version
npm --version
```

Nếu cả hai đều hiển thị version numbers → ✅ Success!

## Common Node.js Locations

- `C:\Program Files\nodejs\`
- `C:\Program Files (x86)\nodejs\`
- `C:\Users\<YourUsername>\AppData\Roaming\npm\`

## Alternative: Sử dụng nvm-windows

Nếu muốn quản lý nhiều Node.js versions:
1. Download: https://github.com/coreybutler/nvm-windows
2. Install và sử dụng:
   ```cmd
   nvm install lts
   nvm use lts
   ```

