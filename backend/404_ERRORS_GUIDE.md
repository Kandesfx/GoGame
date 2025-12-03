# 🔍 Hướng Dẫn Lỗi 404 Not Found

## Các Endpoint Có Thể Trả Về 404

### 1. `/matches/{match_id}` - Match Not Found
**Khi nào:** Khi match_id không tồn tại trong database

**Response:**
```json
{
  "detail": "Match not found"
}
```

**Cách xử lý:**
- Kiểm tra match_id có đúng không
- Đảm bảo match đã được tạo trước đó
- Kiểm tra user có quyền truy cập match không

### 2. `/matches/{match_id}/analysis` - Report Not Found
**Khi nào:** Khi premium analysis report chưa được tạo hoặc không tồn tại

**Response:**
```json
{
  "detail": "Report not found"
}
```

**Cách xử lý:**
- Đảm bảo đã request analysis trước đó
- Kiểm tra request_id có đúng không
- Đợi analysis hoàn thành (có thể là async request)

### 3. `/premium/requests/{request_id}` - Request Not Found
**Khi nào:** Khi premium request (analysis/review) không tồn tại

**Response:**
```json
{
  "detail": "Request not found"
}
```

**Cách xử lý:**
- Kiểm tra request_id có đúng không
- Đảm bảo đã tạo request trước đó
- Request có thể đã bị xóa hoặc expired

### 4. `/users/{user_id}` - User Not Found
**Khi nào:** Khi user_id không tồn tại trong database

**Response:**
```json
{
  "detail": "User not found"
}
```

**Cách xử lý:**
- Kiểm tra user_id có đúng không
- Đảm bảo user đã được tạo
- Kiểm tra user có bị xóa không

### 5. `/statistics/{user_id}` - Statistics Not Found
**Khi nào:** Khi statistics của user không tồn tại

**Response:**
```json
{
  "detail": "User not found"
}
```

**Cách xử lý:**
- Đảm bảo user đã chơi ít nhất 1 ván cờ
- Statistics được tạo tự động khi user chơi game
- Có thể cần chờ background task tạo statistics

## Lỗi 404 vs 401

### 401 Unauthorized (Như trong log của bạn)
- **Nguyên nhân:** Token không hợp lệ, hết hạn, hoặc bị revoke
- **Giải pháp:** 
  - Login lại để lấy token mới
  - Refresh token nếu còn valid
  - Kiểm tra token trong localStorage

### 404 Not Found
- **Nguyên nhân:** Resource không tồn tại
- **Giải pháp:**
  - Kiểm tra ID có đúng không
  - Đảm bảo resource đã được tạo
  - Kiểm tra quyền truy cập

## Debug 404 Errors

### 1. Kiểm tra Request
```bash
# Xem request URL có đúng không
curl -X GET http://localhost:8000/matches/{match_id} \
  -H "Authorization: Bearer {token}"
```

### 2. Kiểm tra Database
```sql
-- Kiểm tra match có tồn tại không
SELECT * FROM matches WHERE id = '{match_id}';

-- Kiểm tra user có tồn tại không
SELECT * FROM users WHERE id = '{user_id}';
```

### 3. Kiểm tra Logs
- Xem backend logs để biết endpoint nào trả về 404
- Kiểm tra error message chi tiết
- Xem có exception nào khác không

## Common Issues

### Issue 1: Match Not Found sau khi tạo
**Nguyên nhân:** Race condition hoặc transaction chưa commit

**Giải pháp:**
- Đợi một chút sau khi tạo match
- Refresh match list
- Kiểm tra database xem match đã được tạo chưa

### Issue 2: Premium Request Not Found
**Nguyên nhân:** Request đã bị xóa hoặc expired

**Giải pháp:**
- Tạo request mới
- Kiểm tra request_id có đúng không
- Đảm bảo request chưa quá cũ (có thể có TTL)

### Issue 3: User Not Found
**Nguyên nhân:** User chưa được tạo hoặc đã bị xóa

**Giải pháp:**
- Đảm bảo user đã register
- Kiểm tra user_id có đúng không
- Kiểm tra database

## Best Practices

1. **Luôn kiểm tra response status code** trước khi xử lý data
2. **Hiển thị error message rõ ràng** cho user
3. **Retry logic** cho các request có thể fail tạm thời
4. **Validate IDs** trước khi gửi request
5. **Handle 404 gracefully** - không crash app

## Example Error Handling (Frontend)

```javascript
try {
  const response = await api.get(`/matches/${matchId}`)
  // Handle success
} catch (err) {
  if (err.response?.status === 404) {
    alert('Match không tồn tại. Vui lòng chọn match khác.')
  } else if (err.response?.status === 401) {
    // Token expired, redirect to login
    window.location.href = '/login'
  } else {
    alert('Có lỗi xảy ra. Vui lòng thử lại.')
  }
}
```

