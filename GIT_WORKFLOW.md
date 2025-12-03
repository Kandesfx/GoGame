# Git Workflow Guide - GoGame Project

## 📋 Cấu trúc nhánh

```
main (master) - Production code, chỉ merge khi code đã test kỹ
  └── develop - Development branch, nhánh chính để phát triển
      ├── feature/your-feature - Nhánh feature của bạn
      └── feature/friend-feature - Nhánh feature của bạn
```

## 🔄 Quy trình làm việc

### **Bạn (Người code chính - Main Developer)**

#### 1. Bắt đầu làm việc mới
```bash
# Cập nhật develop mới nhất
git checkout develop
git pull origin develop

# Tạo nhánh feature mới
git checkout -b feature/my-feature-name

# Hoặc nếu làm bugfix
git checkout -b fix/bug-description
```

#### 2. Code và commit
```bash
# Làm việc, chỉnh sửa code
# ...

# Commit thường xuyên với message rõ ràng
git add .
git commit -m "feat: thêm tính năng X"
# hoặc
git commit -m "fix: sửa lỗi Y"
# hoặc
git commit -m "refactor: tối ưu code Z"
```

#### 3. Push lên remote
```bash
# Push nhánh feature lên remote
git push origin feature/my-feature-name
```

#### 4. Merge vào develop (khi feature hoàn thành)
```bash
# Chuyển về develop
git checkout develop
git pull origin develop  # Cập nhật mới nhất

# Merge feature vào develop
git merge feature/my-feature-name

# Giải quyết conflict nếu có (xem phần Conflict Resolution)

# Push develop
git push origin develop

# Xóa nhánh feature local (tùy chọn)
git branch -d feature/my-feature-name
```

#### 5. Merge develop vào main (khi code ổn định)
```bash
# Chuyển về main
git checkout master
git pull origin master

# Merge develop vào main
git merge develop

# Push main
git push origin master
```

---

### **Bạn của bạn (Collaborator)**

#### 1. Bắt đầu làm việc
```bash
# Fetch và cập nhật develop
git fetch origin
git checkout develop
git pull origin develop

# Tạo nhánh feature riêng
git checkout -b feature/friend-feature-name
```

#### 2. Code và push
```bash
# Code và commit
git add .
git commit -m "feat: mô tả tính năng"

# Push lên remote
git push origin feature/friend-feature-name
```

---

### **Bạn merge code của bạn vào develop**

#### 1. Xem code của bạn
```bash
# Fetch tất cả nhánh mới
git fetch origin

# Xem các nhánh có sẵn
git branch -a

# Xem commit trên nhánh của bạn
git log origin/feature/friend-feature-name --oneline

# Xem diff (thay đổi) so với develop
git diff develop origin/feature/friend-feature-name
```

#### 2. Merge vào develop
```bash
# Đảm bảo develop đã cập nhật
git checkout develop
git pull origin develop

# Merge nhánh của bạn
git merge origin/feature/friend-feature-name

# Nếu có conflict, giải quyết (xem phần Conflict Resolution)

# Test code sau khi merge
# ...

# Push develop
git push origin develop
```

#### 3. Merge develop vào main (khi ổn định)
```bash
git checkout master
git pull origin master
git merge develop
git push origin master
```

---

## 🔧 Giải quyết Conflict (Xung đột)

### Khi merge có conflict:

1. **Git sẽ báo file nào bị conflict:**
```
Auto-merging path/to/file.js
CONFLICT (content): Merge conflict in path/to/file.js
```

2. **Mở file bị conflict, tìm các marker:**
```javascript
<<<<<<< HEAD
// Code từ nhánh hiện tại (develop)
const x = 1;
=======
// Code từ nhánh đang merge (feature/friend-feature)
const x = 2;
>>>>>>> feature/friend-feature
```

3. **Giải quyết conflict:**
   - Giữ code nào phù hợp
   - Hoặc kết hợp cả hai
   - Xóa các marker: `<<<<<<<`, `=======`, `>>>>>>>`

4. **Sau khi sửa xong:**
```bash
# Đánh dấu file đã giải quyết
git add path/to/file.js

# Hoặc add tất cả
git add .

# Hoàn tất merge
git commit -m "Merge feature/friend-feature into develop"
```

### Hủy merge nếu cần:
```bash
git merge --abort
```

---

## 📝 Quy tắc Commit Message

Sử dụng format chuẩn:
```
<type>: <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: Tính năng mới
- `fix`: Sửa lỗi
- `docs`: Cập nhật tài liệu
- `style`: Format code (không ảnh hưởng logic)
- `refactor`: Tối ưu code
- `test`: Thêm/sửa test
- `chore`: Công việc bảo trì

**Ví dụ:**
```bash
git commit -m "feat: thêm tính năng daily bonus"
git commit -m "fix: sửa lỗi token refresh"
git commit -m "refactor: tối ưu CoinDisplay component"
```

---

## 🚨 Lưu ý quan trọng

1. **KHÔNG push trực tiếp lên main/master**
   - Luôn dùng develop hoặc feature branch
   - Chỉ merge vào main khi code đã test kỹ

2. **Luôn pull trước khi merge**
   ```bash
   git pull origin develop  # Trước khi merge
   ```

3. **Test trước khi merge vào main**
   - Merge vào develop trước
   - Test kỹ
   - Mới merge vào main

4. **Commit message rõ ràng**
   - Dễ theo dõi lịch sử
   - Dễ rollback nếu cần

5. **Không commit file lớn hoặc không cần thiết**
   - File trong `data/` đã được ignore
   - Virtual environment (`venv311/`) đã được ignore

---

## 🛠️ Các lệnh hữu ích

### Xem thông tin
```bash
# Xem tất cả nhánh
git branch -a

# Xem commit trên nhánh khác
git log origin/feature/friend-feature --oneline

# Xem diff giữa 2 nhánh
git diff develop origin/feature/friend-feature

# Xem file nào sẽ bị conflict (không merge thật)
git merge --no-commit --no-ff origin/feature/friend-feature
git merge --abort  # Hủy sau khi xem
```

### Quản lý nhánh
```bash
# Xóa nhánh local đã merge xong
git branch -d feature/old-feature

# Xóa nhánh remote
git push origin --delete feature/old-feature

# Đổi tên nhánh
git branch -m old-name new-name
```

### Stash (tạm lưu thay đổi)
```bash
# Lưu thay đổi tạm thời
git stash push -m "Mô tả"

# Xem danh sách stash
git stash list

# Lấy lại thay đổi
git stash pop

# Xóa stash
git stash drop
```

---

## 📊 Workflow Diagram

```
[Feature Branch] → [Develop] → [Main/Master]
     ↑                ↑            ↑
   Code            Test         Production
```

---

## 🎯 Quick Reference

### Bắt đầu feature mới
```bash
git checkout develop
git pull origin develop
git checkout -b feature/my-feature
```

### Hoàn thành feature
```bash
git checkout develop
git pull origin develop
git merge feature/my-feature
git push origin develop
```

### Merge code của bạn
```bash
git fetch origin
git checkout develop
git pull origin develop
git merge origin/feature/friend-feature
# Giải quyết conflict nếu có
git push origin develop
```

### Release lên production
```bash
git checkout master
git pull origin master
git merge develop
git push origin master
```

---

**Lưu ý:** Luôn test kỹ trước khi merge vào main/master!

