# Quick Fix - Node.js trong Git Bash

## ✅ Node.js đã được tìm thấy tại:
`C:\Program Files\nodejs\`

## ⚠️ Vấn đề: NODE_OPTIONS conflict

Nếu gặp lỗi `--openssl-legacy-providerexport is not allowed in NODE_OPTIONS`, cần unset NODE_OPTIONS trước.

## Giải pháp nhanh nhất:

### Option 1: Thêm vào ~/.bashrc (Permanent)

```bash
# Mở file .bashrc
notepad ~/.bashrc
# hoặc
code ~/.bashrc

# Thêm 2 dòng này vào cuối file:
unset NODE_OPTIONS
export PATH="/c/Program Files/nodejs:$PATH"

# Lưu và restart Git Bash
# Hoặc chạy:
source ~/.bashrc
```

### Option 2: Export trong mỗi session (Temporary)

Mỗi lần mở Git Bash, chạy:
```bash
unset NODE_OPTIONS
export PATH="/c/Program Files/nodejs:$PATH"
```

### Option 3: Sử dụng run.sh script

Script đã tự động fix PATH:
```bash
cd frontend-web
bash run.sh
```

### Option 4: Dùng Command Prompt/PowerShell (Khuyến nghị)

Thay vì Git Bash, dùng **Command Prompt** hoặc **PowerShell**:

```cmd
cd frontend-web
npm install
npm run dev
```

## Verify

Sau khi fix, verify:
```bash
node --version
npm --version
```

Nếu hiển thị version numbers → ✅ Success!

## Troubleshooting

**Nếu vẫn không work:**
1. Check Node.js có thực sự ở `C:\Program Files\nodejs\` không
2. Thử đường dẫn khác: `C:\Program Files (x86)\nodejs\`
3. Restart Git Bash sau khi thay đổi PATH

