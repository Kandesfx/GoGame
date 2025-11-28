# 🚀 HƯỚNG DẪN PUSH CODE LÊN GITHUB

## ✅ Kiểm tra đã hoàn tất

### 1. Git Repository
- ✅ Đã khởi tạo Git repository
- ✅ File `.gitignore` đã được cấu hình đúng

### 2. File nhạy cảm
- ✅ `backend/.env` - Đã được ignore
- ✅ `frontend/.env` - Đã được ignore  
- ✅ `frontend-web/.env` - Đã được ignore

### 3. Thư mục lớn (đã được ignore)
- ✅ `build/` - 3.3MB
- ✅ `venv/` - 916MB
- ✅ `frontend-web/node_modules/` - 172MB

---

## 📝 Các bước để push lên GitHub

### Bước 1: Kiểm tra lại các file sẽ được commit

```bash
# Xem tất cả file sẽ được commit
git status

# Xem chi tiết hơn
git status --short
```

### Bước 2: Thêm tất cả các file

```bash
# Thêm tất cả file (trừ những file trong .gitignore)
git add .

# Kiểm tra lại
git status
```

**Lưu ý**: Các file sau sẽ KHÔNG được thêm:
- `backend/.env`, `frontend/.env`, `frontend-web/.env`
- `build/`, `venv/`, `node_modules/`
- Các file cache và build artifacts

### Bước 3: Tạo commit

```bash
# Commit với message mô tả
git commit -m "Initial commit: GoGame - AI Go Game Platform

- Complete Go game engine with Minimax and MCTS AI
- FastAPI backend with PostgreSQL and MongoDB
- React frontend with modern UI
- ML training infrastructure and documentation
- Full documentation in docs/ directory"
```

**Hoặc commit ngắn gọn hơn:**
```bash
git commit -m "Initial commit: GoGame AI Platform"
```

### Bước 4: Thêm remote repository (nếu chưa có)

**Nếu bạn chưa tạo repository trên GitHub:**

1. Vào https://github.com/new
2. Tạo repository mới (không tích "Initialize with README")
3. Copy URL repository (ví dụ: `https://github.com/username/GoGame.git`)

**Sau đó thêm remote:**

```bash
# Thay <repository-url> bằng URL của bạn
git remote add origin <repository-url>

# Kiểm tra remote
git remote -v
```

**Ví dụ:**
```bash
git remote add origin https://github.com/yourusername/GoGame.git
```

### Bước 5: Push lên GitHub

```bash
# Push lên branch master (hoặc main)
git push -u origin master

# Nếu GitHub dùng branch "main" thay vì "master":
git push -u origin main
```

**Lưu ý**: 
- Lần đầu push có thể cần authenticate (username/password hoặc token)
- Nếu gặp lỗi authentication, xem phần "Troubleshooting" bên dưới

---

## 🔍 Kiểm tra sau khi push

1. **Kiểm tra trên GitHub:**
   - Vào repository trên GitHub
   - Xem tất cả files đã được push chưa
   - Kiểm tra `.gitignore` có hoạt động đúng không

2. **Kiểm tra file nhạy cảm:**
   - Đảm bảo KHÔNG có file `.env` nào trên GitHub
   - Đảm bảo KHÔNG có `venv/`, `node_modules/`, `build/` trên GitHub

3. **Kiểm tra documentation:**
   - README.md hiển thị đúng
   - Các file trong `docs/` đã được push

---

## 🛠️ Troubleshooting

### Lỗi: "remote origin already exists"

```bash
# Xem remote hiện tại
git remote -v

# Xóa remote cũ (nếu cần)
git remote remove origin

# Thêm lại
git remote add origin <repository-url>
```

### Lỗi: Authentication failed

**Option 1: Dùng Personal Access Token (khuyến nghị)**

1. Vào GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token với quyền `repo`
3. Copy token
4. Khi push, dùng token thay vì password:
   ```
   Username: your-username
   Password: <paste-token-here>
   ```

**Option 2: Dùng SSH**

```bash
# Tạo SSH key (nếu chưa có)
ssh-keygen -t ed25519 -C "your_email@example.com"

# Copy public key
cat ~/.ssh/id_ed25519.pub

# Thêm vào GitHub → Settings → SSH and GPG keys

# Đổi remote sang SSH
git remote set-url origin git@github.com:username/GoGame.git

# Push lại
git push -u origin master
```

### Lỗi: "refusing to merge unrelated histories"

```bash
# Nếu repository trên GitHub đã có commits
git pull origin master --allow-unrelated-histories

# Sau đó push lại
git push -u origin master
```

### File lớn (>100MB)

Nếu có file lớn hơn 100MB, GitHub sẽ từ chối. Giải pháp:

1. **Sử dụng Git LFS:**
```bash
# Cài Git LFS
git lfs install

# Track file lớn
git lfs track "*.pyd"
git lfs track "*.so"
git lfs track "*.dll"

# Commit .gitattributes
git add .gitattributes
git commit -m "Add Git LFS tracking"
```

2. **Hoặc loại bỏ file lớn:**
```bash
# Thêm vào .gitignore
echo "*.pyd" >> .gitignore
echo "*.so" >> .gitignore

# Xóa file khỏi git (nếu đã add)
git rm --cached file.pyd
```

---

## 📋 Checklist cuối cùng

Trước khi push, đảm bảo:

- [ ] Đã chạy `git status` và kiểm tra các file
- [ ] Không có file `.env` nào được commit
- [ ] Không có `venv/`, `node_modules/`, `build/` được commit
- [ ] Đã có commit message rõ ràng
- [ ] Đã thêm remote repository
- [ ] Đã test push (hoặc sẵn sàng push)

---

## 🎯 Quick Commands

```bash
# Tất cả trong một (sau khi đã setup remote)
git add .
git commit -m "Initial commit: GoGame AI Platform"
git push -u origin master
```

---

**Chúc bạn thành công! 🚀**

