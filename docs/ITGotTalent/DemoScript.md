# DEMO SCRIPT - GOGAME

**Thời gian:** 5-7 phút  
**Mục tiêu:** Highlight AI features, show system hoạt động tốt

---

## 🎬 DEMO FLOW

### **BƯỚC 1: GIỚI THIỆU (30 giây)**

**Nói:**
> "Bây giờ em xin demo hệ thống GoGame. Đây là platform học Cờ Vây với AI coach thông minh."

**Action:**
- Mở trình duyệt
- Navigate đến `http://localhost:5173` (hoặc URL deploy)
- Show login page

---

### **BƯỚC 2: LOGIN & DASHBOARD (30 giây)**

**Nói:**
> "Đầu tiên, em đăng nhập vào hệ thống. Sau khi login, chúng ta thấy dashboard với thống kê và leaderboard."

**Action:**
- Login với test account
- Show dashboard
- Point out: ELO rating, win rate, statistics

**Highlight:**
- "Hệ thống có ELO rating system để track skill level"

---

### **BƯỚC 3: TẠO GAME VỚI AI (1 phút)**

**Nói:**
> "Bây giờ em tạo một game với AI Level 3. AI này sử dụng Minimax depth 2-3 với các optimizations."

**Action:**
- Click "Play vs AI"
- Select Level 3 (Hard)
- Select Board Size 9x9
- Start game

**Highlight:**
- "AI sẽ tính toán nước đi trong vài giây"
- "Chúng ta thấy AI đánh nước đầu tiên"

**Wait for AI move:**
- Show AI thinking (nếu có indicator)
- Show AI move

---

### **BƯỚC 4: CHƠI VÀI NƯỚC (1 phút)**

**Nói:**
> "Em đánh vài nước để show gameplay. AI sẽ phản ứng thông minh."

**Action:**
- Đánh 3-4 nước
- Wait for AI responses
- Show board state

**Highlight:**
- "AI đánh nước hợp lý, không random"
- "Có thể thấy AI có strategy"

---

### **BƯỚC 5: HINT SYSTEM (1 phút) - QUAN TRỌNG**

**Nói:**
> "Bây giờ em dùng Hint System - tính năng premium sử dụng Policy Network để gợi ý nước đi tốt."

**Action:**
- Click "Hint" button (hoặc premium menu)
- Show hint results:
  - Top 3 moves
  - Xác suất cho mỗi move
  - Lý do (nếu có)

**Highlight:**
- "Policy Network đã được train trên professional games"
- "Top 3 moves được sắp xếp theo xác suất"
- "Đây là tính năng giúp người học biết nước đi tốt"

**Visual:**
- Show moves trên board (nếu có highlight)
- Show numbers (probabilities)

---

### **BƯỚC 6: POSITION ANALYSIS (1.5 phút) - QUAN TRỌNG**

**Nói:**
> "Tiếp theo, em dùng Position Analysis - tính năng phân tích chi tiết vị trí hiện tại sử dụng Value Network và Multi-Task Model."

**Action:**
- Click "Analysis" button
- Show analysis results:
  - Win probability (Value Network)
  - Territory estimation
  - Threat heatmap (Multi-Task Model)
  - Attack heatmap (Multi-Task Model)
  - Intent prediction (nếu có)

**Highlight:**
- "Value Network cho biết ai đang thắng với xác suất bao nhiêu"
- "Threat heatmap phát hiện các mối đe dọa"
- "Attack heatmap chỉ ra cơ hội tấn công"
- "Đây là insights rất hữu ích cho người học"

**Visual:**
- Show heatmaps trên board
- Show numbers (win probability, territory)
- Toggle between different views

---

### **BƯỚC 7: GAME REVIEW (1 phút) - NẾU CÓ THỜI GIAN**

**Nói:**
> "Cuối cùng, em show Game Review - phân tích toàn bộ ván đấu để highlight mistakes và suggest improvements."

**Action:**
- End game (hoặc dùng game đã chơi trước)
- Click "Review" button
- Show review:
  - Mistakes highlighted
  - Better moves suggested
  - Overall analysis

**Highlight:**
- "Review giúp người học học từ sai lầm"
- "AI phân tích từng nước và đưa ra feedback"

---

### **BƯỚC 8: KẾT THÚC (30 giây)**

**Nói:**
> "Đó là demo của hệ thống GoGame. Như các thầy cô thấy, đây không chỉ là game, mà là platform AI mạnh mẽ với Minimax, MCTS, và Deep Learning, giúp người học Cờ Vây cải thiện kỹ năng."

**Action:**
- Show dashboard một lần nữa
- Point out: "Hệ thống đã lưu game history, statistics, và có thể replay"

---

## 🎯 KEY POINTS CẦN NHẤN MẠNH

### **1. AI là Core**
- ✅ "AI sử dụng Minimax và MCTS"
- ✅ "AI có 4 cấp độ, từ Beginner đến Expert"
- ✅ "AI tính toán thông minh, không random"

### **2. ML Models**
- ✅ "Policy Network cho hint system"
- ✅ "Value Network cho position evaluation"
- ✅ "Multi-Task Model cho threat/attack detection"

### **3. Educational Value**
- ✅ "Hint giúp biết nước đi tốt"
- ✅ "Analysis cho insights sâu về vị trí"
- ✅ "Review giúp học từ sai lầm"

### **4. Production-Ready**
- ✅ "Hệ thống hoạt động ổn định"
- ✅ "Response time nhanh"
- ✅ "UI/UX tốt"

---

## 🚨 BACKUP PLAN

### **Nếu Demo Live Fail:**

**Option 1: Video Demo**
- Có video 5 phút đã quay sẵn
- Play video và comment
- "Đây là video demo hệ thống, em sẽ giải thích các tính năng"

**Option 2: Screenshots**
- Có screenshots các tính năng chính
- Show từng screenshot và giải thích
- "Vì lý do kỹ thuật, em sẽ show screenshots thay vì demo live"

**Option 3: Offline Mode**
- Có local backend chạy sẵn
- Switch sang localhost
- "Em sẽ demo trên local để đảm bảo ổn định"

---

## 📋 CHECKLIST TRƯỚC DEMO

- [ ] Test account đã sẵn sàng
- [ ] Game đã tạo sẵn (nếu cần)
- [ ] Hint system hoạt động
- [ ] Analysis hoạt động
- [ ] Review hoạt động (nếu có)
- [ ] Internet connection ổn định
- [ ] Video backup đã sẵn sàng
- [ ] Screenshots đã chuẩn bị
- [ ] Local backend đã test

---

## 💡 TIPS

1. **Practice nhiều lần** - demo phải smooth
2. **Time management** - không quá 7 phút
3. **Highlight AI features** - đây là điểm mạnh
4. **Show numbers** - probabilities, scores
5. **Be confident** - project tốt, không cần lo

---

**Good luck! 🚀**

