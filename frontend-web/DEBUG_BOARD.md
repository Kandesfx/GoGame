# Hướng dẫn Debug Bàn Cờ

## Vấn đề
Bàn cờ không sáng như mong muốn, có thể do:
1. Browser cache
2. CSS bị override
3. Inline styles
4. Element cha có backdrop-filter

## Các bước Debug

### 1. Hard Refresh Browser
- **Chrome/Edge**: `Ctrl + Shift + R` hoặc `Ctrl + F5`
- **Firefox**: `Ctrl + Shift + R`
- **Safari**: `Cmd + Shift + R`

### 2. Kiểm tra trong DevTools
1. Mở DevTools (F12)
2. Chọn element `.board` bằng Inspector
3. Kiểm tra trong tab **Computed**:
   - `background` - phải là `linear-gradient(135deg, #E8D5B7 0%, #D4C4A0 50%, #C4B08A 100%)`
   - `opacity` - phải là `1`
   - `filter` - phải là `none`
   - `backdrop-filter` - phải là `none`
   - `mix-blend-mode` - phải là `normal`

### 3. Kiểm tra CSS đang được áp dụng
Trong tab **Styles** của DevTools:
- Xem CSS nào đang override `.board`
- Tìm các rule có `!important` đang conflict
- Kiểm tra xem có inline style nào không

### 4. Kiểm tra Element Cha
Kiểm tra các element cha:
- `.board-wrapper`
- `.center-panel`
- `.main-content`
- Xem có `backdrop-filter`, `filter`, `opacity` nào không

### 5. Test CSS mới
Thử thêm CSS này vào DevTools Console:
```css
.board {
  background: linear-gradient(135deg, #E8D5B7 0%, #D4C4A0 50%, #C4B08A 100%) !important;
  opacity: 1 !important;
  filter: none !important;
  backdrop-filter: none !important;
  mix-blend-mode: normal !important;
  isolation: isolate !important;
  transform: translateZ(0) !important;
}
```

### 6. Kiểm tra Video Overlay
Kiểm tra xem `.video-overlay` có đang che phủ bàn cờ không:
- Trong DevTools, tìm `.video-overlay`
- Thử set `display: none` tạm thời
- Xem bàn cờ có sáng hơn không

### 7. Clear Browser Cache
1. Mở DevTools (F12)
2. Right-click vào nút Refresh
3. Chọn "Empty Cache and Hard Reload"

### 8. Kiểm tra Console Errors
Xem có lỗi JavaScript nào đang ảnh hưởng không

## Giải pháp thay thế

Nếu vẫn không được, có thể thử:
1. Tăng z-index của board lên cao hơn
2. Tạo một wrapper div mới với background sáng
3. Sử dụng `::before` pseudo-element để tạo background layer riêng

