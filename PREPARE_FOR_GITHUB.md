# Checklist trước khi đẩy lên GitHub

> **Lưu ý**: Nếu bạn đang setup trên máy mới, vui lòng xem [INSTALLATION.md](INSTALLATION.md) trước để cài đặt các công cụ cần thiết.

## ✅ Kiểm tra các file nhạy cảm

Trước khi commit, đảm bảo các file sau **KHÔNG** được commit:

- [ ] `backend/.env` - File này chứa thông tin nhạy cảm (database credentials, JWT secrets)
- [ ] `frontend/.env` - File này chứa cấu hình frontend
- [ ] `frontend-web/.env` - File này chứa cấu hình frontend-web
- [ ] Bất kỳ file `.env` nào khác trong project

**Giải pháp**: File `.gitignore` đã được tạo để tự động ignore các file `.env`. Chỉ commit các file `.env.example`.

## ✅ Kiểm tra các thư mục build

Các thư mục sau không nên được commit:

- [ ] `build/` - Thư mục build C++
- [ ] `venv/` - Python virtual environment
- [ ] `node_modules/` - Node.js dependencies
- [ ] `dist/` - Frontend build output
- [ ] `__pycache__/` - Python cache
- [ ] `.pytest_cache/` - Test cache

**Giải pháp**: File `.gitignore` đã được tạo để tự động ignore các thư mục này.

## ✅ Kiểm tra các file cần thiết

Đảm bảo các file sau đã được tạo:

- [x] `.gitignore` - Ignore các file không cần thiết
- [x] `README.md` - Tài liệu tổng quan
- [x] `SETUP.md` - Hướng dẫn setup chi tiết
- [x] `LICENSE` - Giấy phép (MIT)
- [x] `CONTRIBUTING.md` - Hướng dẫn đóng góp
- [x] `backend/env.example` - Template cho backend .env
- [x] `frontend/env.example` - Template cho frontend .env

## ✅ Kiểm tra thông tin trong README

Đảm bảo README.md có:
- [ ] Mô tả dự án rõ ràng
- [ ] Hướng dẫn setup cơ bản
- [ ] Link đến tài liệu chi tiết
- [ ] Thông tin về license

## ✅ Khởi tạo Git repository (nếu chưa có)

```bash
# Từ thư mục root của project
git init

# Thêm remote (thay <repository-url> bằng URL GitHub của bạn)
git remote add origin <repository-url>

# Kiểm tra remote
git remote -v
```

## ✅ Commit và Push

```bash
# Kiểm tra các file sẽ được commit
git status

# Thêm tất cả các file (trừ những file trong .gitignore)
git add .

# Commit
git commit -m "Initial commit: GoGame - AI Go Game Platform"

# Push lên GitHub (lần đầu tiên)
git push -u origin main
# hoặc
git push -u origin master
```

## ⚠️ Lưu ý quan trọng

1. **KHÔNG commit file `.env`**: File này chứa thông tin nhạy cảm như database passwords và JWT secrets.

2. **KHÔNG commit build artifacts**: Các file build có thể được tạo lại, không cần commit.

3. **KHÔNG commit dependencies**: `node_modules/` và `venv/` rất lớn và có thể được tạo lại từ `package.json` và `requirements.txt`.

4. **Kiểm tra file size**: Nếu có file lớn (>100MB), cân nhắc sử dụng Git LFS hoặc loại bỏ khỏi repository.

5. **Kiểm tra secrets**: Trước khi push, tìm kiếm các từ khóa như "password", "secret", "key" trong code để đảm bảo không có thông tin nhạy cảm bị hardcode.

## 🔍 Kiểm tra cuối cùng

Trước khi push, chạy lệnh sau để xem các file sẽ được commit:

```bash
git status
```

Nếu thấy bất kỳ file `.env` hoặc thư mục `venv/`, `node_modules/`, `build/` nào, hãy kiểm tra lại `.gitignore`.

## 📝 Sau khi push

1. Kiểm tra repository trên GitHub
2. Đảm bảo tất cả các file cần thiết đã được push
3. Kiểm tra `.gitignore` có hoạt động đúng không
4. Tạo README badges (nếu muốn)
5. Tạo tags/releases (nếu cần)

