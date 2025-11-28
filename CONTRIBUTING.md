# Hướng dẫn Đóng góp

Cảm ơn bạn quan tâm đến việc đóng góp cho dự án GoGame!

## Quy trình đóng góp

1. **Fork** repository
2. **Tạo branch** mới cho feature/fix của bạn:
   ```bash
   git checkout -b feature/ten-feature
   # hoặc
   git checkout -b fix/ten-bug
   ```
3. **Commit** các thay đổi của bạn:
   ```bash
   git commit -m "Add: mô tả ngắn gọn về thay đổi"
   ```
4. **Push** lên branch của bạn:
   ```bash
   git push origin feature/ten-feature
   ```
5. **Tạo Pull Request** trên GitHub

## Quy tắc Code

### Python (Backend)

- Tuân thủ PEP 8
- Sử dụng type hints
- Viết docstrings cho functions/classes
- Tối đa 100 ký tự mỗi dòng

### JavaScript/React (Frontend)

- Sử dụng ESLint và Prettier
- Tuân thủ React best practices
- Sử dụng functional components và hooks
- Tối đa 100 ký tự mỗi dòng

### C++ (AI Engine)

- Tuân thủ Google C++ Style Guide
- Sử dụng const khi có thể
- Tránh raw pointers khi không cần thiết
- Comment cho các thuật toán phức tạp

## Commit Messages

Sử dụng format:
```
<type>: <mô tả ngắn gọn>

<mô tả chi tiết (nếu cần)>
```

Types:
- `Add`: Thêm feature mới
- `Fix`: Sửa bug
- `Update`: Cập nhật code hiện có
- `Refactor`: Refactor code
- `Docs`: Cập nhật documentation
- `Style`: Format code, không thay đổi logic
- `Test`: Thêm/sửa tests
- `Chore`: Các task khác (build, config, etc.)

Ví dụ:
```
Fix: Sửa lỗi sync board state khi reload page

- Đảm bảo board_position được sync đúng từ backend
- Thêm validation cho board state
```

## Testing

- Viết tests cho các features mới
- Đảm bảo tất cả tests pass trước khi commit
- Backend: Sử dụng pytest
- Frontend: Sử dụng Jest/Vitest (nếu có)

## Documentation

- Cập nhật README nếu thay đổi setup
- Thêm comments cho code phức tạp
- Cập nhật docs/ nếu thay đổi architecture

## Questions?

Nếu có câu hỏi, vui lòng tạo issue trên GitHub.

