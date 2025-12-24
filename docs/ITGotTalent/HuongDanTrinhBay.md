# HƯỚNG DẪN TRÌNH BÀY - GOGAME

**Bảng thi:** D - Trí tuệ nhân tạo & Công nghệ Chuỗi khối  
**Thời gian:** 10-15 phút trình bày + 5-10 phút Q&A  
**Ngày thi:** 05/01/2026, 07h30

---

## 🎯 MỤC TIÊU TRÌNH BÀY

### **Primary Goal:**
Thuyết phục BGK rằng đây là **dự án AI mạnh mẽ**, không chỉ là game, mà là **hệ thống học tập với AI coach thông minh**.

### **Key Messages:**
1. ✅ **AI là core** - Minimax, MCTS, Deep Learning
2. ✅ **Hybrid architecture** - C++ performance + Python ML
3. ✅ **Educational value** - công cụ học Cờ Vây
4. ✅ **Production-ready** - scalable, deployable

---

## 📋 CẤU TRÚC TRÌNH BÀY (10-15 phút)

### **PHẦN 1: GIỚI THIỆU (1-2 phút)**

#### Slide 1: Title Slide
```
GoGame - Hệ Thống Học Tập Cờ Vây với AI Coach Thông Minh

Team: [Tên đội]
Trường: [Tên trường]
Bảng thi: D - Trí tuệ nhân tạo
```

**Nói gì:**
- "Chào BGK, em xin trình bày dự án GoGame - một hệ thống AI mạnh mẽ để học và phân tích Cờ Vây"
- "Đây không chỉ là game, mà là platform giáo dục với AI coach thông minh"

#### Slide 2: Problem Statement
```
VẤN ĐỀ:
- Học Cờ Vây khó, cần người hướng dẫn
- Thiếu công cụ phân tích và đánh giá
- AI Go hiện tại chủ yếu là game, ít tính giáo dục
```

**Nói gì:**
- "Cờ Vây là game cờ phức tạp nhất, với không gian trạng thái 10^170"
- "Người học cần công cụ để phân tích, học hỏi, và cải thiện kỹ năng"
- "Chúng em xây dựng hệ thống AI để giải quyết vấn đề này"

---

### **PHẦN 2: GIẢI PHÁP TỔNG QUAN (1 phút)**

#### Slide 3: Solution Overview
```
GIẢI PHÁP:
┌─────────────────────────────────────┐
│  AI Engine (C++)                   │
│  - Minimax + MCTS                  │
│  - 4 cấp độ AI                     │
├─────────────────────────────────────┤
│  ML Models (PyTorch)               │
│  - Policy Network                  │
│  - Value Network                   │
│  - Multi-Task Model                │
├─────────────────────────────────────┤
│  Premium Features                  │
│  - Hint System                     │
│  - Position Analysis               │
│  - Game Review                     │
└─────────────────────────────────────┘
```

**Nói gì:**
- "Giải pháp của chúng em gồm 3 lớp: AI Engine, ML Models, và Premium Features"
- "Tất cả đều tập trung vào việc giúp người học cải thiện kỹ năng"

---

### **PHẦN 3: AI ENGINE (3-4 phút) - QUAN TRỌNG NHẤT**

#### Slide 4: AI Architecture
```
KIẾN TRÚC AI:
┌──────────────┐
│   Python     │  ← ML Models, API
│   (PyTorch)  │
└──────┬───────┘
       │ pybind11
┌──────▼───────┐
│     C++      │  ← Core AI Engine
│  (Minimax/   │     (Performance)
│    MCTS)     │
└──────────────┘
```

**Nói gì:**
- "Chúng em sử dụng **hybrid architecture**: C++ cho performance-critical AI engine, Python cho ML models"
- "Binding qua pybind11 để tích hợp seamless"
- "Lý do: Cờ Vây có branching factor ~250, cần xử lý hàng nghìn simulations mỗi nước"

#### Slide 5: Minimax Algorithm
```
MINIMAX VỚI ALPHA-BETA PRUNING:

- Depth: 1-4 tùy level
- Optimizations:
  ✓ Alpha-Beta Pruning
  ✓ Move Ordering
  ✓ Transposition Table
  ✓ Opening Book

Performance:
- Level 3: ~5 giây/nước
- Level 4: ~10-15 giây/nước
```

**Nói gì:**
- "Minimax là thuật toán cổ điển, nhưng chúng em optimize với nhiều kỹ thuật"
- "Alpha-Beta pruning giảm số nodes cần xét từ O(b^d) xuống O(b^(d/2))"
- "Transposition table cache các vị trí đã tính để tránh tính lại"

#### Slide 6: MCTS Algorithm
```
MONTE CARLO TREE SEARCH:

1. Selection (UCB1)
2. Expansion
3. Simulation (Random playout)
4. Backpropagation

Features:
- Không cần evaluation function
- Tự động balance exploration/exploitation
- Có thể kết hợp với ML guidance
```

**Nói gì:**
- "MCTS là thuật toán hiện đại, được dùng trong AlphaGo"
- "Ưu điểm: không cần hàm đánh giá phức tạp, tự học từ simulations"
- "Có thể enhance với Policy Network để guide search"

#### Slide 7: AI Levels
```
4 CẤP ĐỘ AI:

Level 1 (Beginner):  Depth 1, 5s timeout
Level 2 (Intermediate): Depth 2, 8s timeout
Level 3 (Hard): Depth 2-3, 6-15s timeout
Level 4 (Expert): Depth 3-4, 10-20s timeout

+ Opening Book (Level 2-4)
+ Transposition Table (Level 3-4)
```

**Nói gì:**
- "AI có 4 cấp độ để phù hợp với trình độ người chơi"
- "Level cao hơn sử dụng nhiều optimizations hơn"
- "Opening book giúp AI chơi khai cuộc chuyên nghiệp"

---

### **PHẦN 4: MACHINE LEARNING (2-3 phút)**

#### Slide 8: ML Models
```
MACHINE LEARNING MODELS:

1. Policy Network
   - Input: Board state (17 planes)
   - Output: Move probability distribution
   - Usage: Hint system, MCTS guidance

2. Value Network
   - Input: Board state
   - Output: Win probability (0-1)
   - Usage: Position evaluation

3. Multi-Task Model
   - Threat Detection (heatmap)
   - Attack Opportunities (heatmap)
   - Intent Recognition (classification)
```

**Nói gì:**
- "Chúng em train 3 models: Policy, Value, và Multi-Task"
- "Policy Network dự đoán nước đi tốt, dùng cho hint system"
- "Value Network đánh giá vị trí, cho biết ai đang thắng"
- "Multi-Task Model phát hiện threat, attack, và intent - rất hữu ích cho người học"

#### Slide 9: Training Pipeline
```
TRAINING PIPELINE:

1. Data Collection
   - Parse SGF files từ professional games
   - Extract positions và labels

2. Model Training
   - Supervised learning
   - Multi-task loss function
   - PyTorch framework

3. Evaluation
   - Test trên held-out games
   - Accuracy metrics
```

**Nói gì:**
- "Training data từ professional games, đảm bảo chất lượng"
- "Multi-task learning giúp models học tốt hơn vì share features"
- "Evaluation trên test set để đảm bảo generalization"

---

### **PHẦN 5: PREMIUM FEATURES (1-2 phút)**

#### Slide 10: Premium Features
```
TÍNH NĂNG PREMIUM (AI-POWERED):

1. Hint System (10 coins)
   - Policy Network → Top 3 moves
   - Hiển thị xác suất và lý do

2. Position Analysis (20 coins)
   - Value Network → Win probability
   - Territory estimation
   - Threat/Attack heatmaps

3. Game Review (30 coins)
   - Phân tích toàn ván
   - Highlight mistakes
   - Suggest improvements
```

**Nói gì:**
- "Premium features sử dụng ML models để phân tích"
- "Hint system giúp người học biết nước đi tốt"
- "Position analysis cho insights sâu về vị trí"
- "Game review giúp học từ sai lầm"

---

### **PHẦN 6: DEMO (2-3 phút)**

#### Slide 11: Demo Screenshots
```
DEMO:
[Chụp màn hình game board]
[Chụp màn hình hint system]
[Chụp màn hình analysis]
[Chụp màn hình review]
```

**Nói gì:**
- "Bây giờ em xin demo hệ thống"
- **LIVE DEMO hoặc VIDEO:**
  1. Login
  2. Chơi với AI Level 3
  3. Dùng Hint → show top moves
  4. Dùng Analysis → show heatmaps
  5. Xem Review → show mistakes

**Lưu ý:**
- Nếu demo live fail → chuyển sang video ngay
- Highlight AI features rõ ràng
- Show numbers (win probability, move scores)

---

### **PHẦN 7: BENCHMARK & RESULTS (1-2 phút)**

#### Slide 12: Benchmark Results
```
KẾT QUẢ BENCHMARK:

AI Performance:
- Win rate vs random: 85% (Level 3)
- Avg response time: 4.2s (Level 3)
- Memory usage: 320MB

ML Inference:
- Policy Network: 85ms
- Value Network: 92ms
- Multi-Task Model: 180ms

System:
- API response: 150ms
- Game sync: 45ms
```

**Nói gì:**
- "AI đạt win rate 85% so với random player"
- "Response time dưới 5 giây, đủ nhanh cho real-time play"
- "ML inference dưới 200ms, có thể dùng cho premium features"

---

### **PHẦN 8: TECHNOLOGY STACK (30 giây)**

#### Slide 13: Tech Stack
```
CÔNG NGHỆ:

Backend: FastAPI, PostgreSQL, MongoDB
Frontend: React, Vite
AI: C++20, pybind11
ML: PyTorch 2.0+
Deployment: Docker, Fly.io
```

**Nói gì:**
- "Sử dụng các công nghệ hiện đại, production-ready"
- "Architecture scalable, có thể mở rộng"

---

### **PHẦN 9: FUTURE WORK & Q&A (1 phút)**

#### Slide 14: Future Work
```
HƯỚNG PHÁT TRIỂN:

1. Reinforcement Learning (self-play)
2. Larger neural networks
3. Mobile app
4. Tournament system
5. Community features
```

**Nói gì:**
- "Trong tương lai, chúng em sẽ thêm RL để AI tự học"
- "Mở rộng sang mobile để tiếp cận nhiều người dùng hơn"
- "Cảm ơn BGK, em sẵn sàng trả lời câu hỏi"

---

## 💡 TIPS TRÌNH BÀY

### **DO:**
- ✅ **Nhấn mạnh AI/ML** - đây là core của project
- ✅ **Show numbers** - benchmarks, metrics, performance
- ✅ **Be confident** - project tốt, có đủ tính năng
- ✅ **Eye contact** - nhìn BGK khi trình bày
- ✅ **Practice** - rehearsal nhiều lần
- ✅ **Time management** - không vượt quá 15 phút

### **DON'T:**
- ❌ **Đừng nói "chỉ là game"** - đây là AI platform
- ❌ **Đừng quá kỹ thuật** - giải thích dễ hiểu
- ❌ **Đừng đọc slide** - nói tự nhiên
- ❌ **Đừng vượt thời gian** - respect time limit
- ❌ **Đừng panic nếu demo fail** - có backup

---

## 🎤 SCRIPT MẪU (Tham khảo)

### **Opening:**
> "Chào BGK, em xin trình bày dự án GoGame - một hệ thống AI mạnh mẽ để học và phân tích Cờ Vây. Đây không chỉ là game, mà là platform giáo dục với AI coach thông minh, giúp người học cải thiện kỹ năng."

### **AI Engine:**
> "Core của hệ thống là AI Engine viết bằng C++, sử dụng 2 thuật toán chính: Minimax với Alpha-Beta Pruning và Monte Carlo Tree Search. Chúng em optimize với nhiều kỹ thuật như move ordering, transposition table, và opening book để đạt performance cao."

### **ML Models:**
> "Bên cạnh AI cổ điển, chúng em còn tích hợp Machine Learning với 3 models: Policy Network để dự đoán nước đi, Value Network để đánh giá vị trí, và Multi-Task Model để phát hiện threat, attack, và intent. Các models này được train trên professional games và dùng cho premium features."

### **Demo:**
> "Bây giờ em xin demo hệ thống. [Chạy demo] Như các thầy cô thấy, khi dùng Hint, hệ thống hiển thị top 3 nước đi với xác suất. Khi dùng Analysis, chúng ta thấy heatmap về threat và attack opportunities. Đây là những tính năng giúp người học hiểu sâu hơn về vị trí."

### **Closing:**
> "Tóm lại, GoGame là hệ thống AI hoàn chỉnh với Minimax, MCTS, và Deep Learning, có thể ứng dụng trong giáo dục và sẵn sàng cho commercial use. Cảm ơn BGK, em sẵn sàng trả lời câu hỏi."

---

## ❓ CÂU HỎI THƯỜNG GẶP & CÁCH TRẢ LỜI

### **Q1: Tại sao chọn Cờ Vây?**
> "Cờ Vây là game cờ phức tạp nhất với không gian trạng thái 10^170, là thách thức lớn cho AI. Ngoài ra, Cờ Vây có tính giáo dục cao, giúp phát triển tư duy chiến lược."

### **Q2: AI của các em mạnh đến đâu?**
> "AI của chúng em có 4 cấp độ, từ Beginner đến Expert. Level 3-4 đạt win rate 85% so với random player. Chúng em chưa so sánh với AlphaGo vì đó là hệ thống rất lớn, nhưng AI của chúng em đủ mạnh để dạy người học."

### **Q3: ML models được train như thế nào?**
> "Chúng em parse SGF files từ professional games, extract positions và labels, sau đó train với supervised learning. Multi-task learning giúp models học tốt hơn vì share features."

### **Q4: Tại sao dùng hybrid C++/Python?**
> "C++ cho performance-critical AI engine vì cần xử lý hàng nghìn simulations mỗi nước. Python cho ML models vì ecosystem mạnh và dễ phát triển. Binding qua pybind11 để tích hợp seamless."

### **Q5: Sản phẩm có thể thương mại hóa không?**
> "Có, sản phẩm có thể ứng dụng trong giáo dục, như công cụ học Cờ Vây cho học sinh, sinh viên. Có thể monetize qua premium features hoặc subscription model."

### **Q6: Điểm khác biệt với các AI Go khác?**
> "Điểm khác biệt chính là focus vào giáo dục, không chỉ là game. Chúng em có premium features như hint, analysis, review để giúp người học cải thiện kỹ năng. Ngoài ra, architecture scalable và production-ready."

### **Q7: Có thể mở rộng như thế nào?**
> "Có thể thêm Reinforcement Learning để AI tự học, mở rộng sang mobile app, thêm tournament system, và community features. Architecture hiện tại đã support scaling."

---

## 📝 CHECKLIST TRƯỚC KHI TRÌNH BÀY

- [ ] Slides đã hoàn chỉnh
- [ ] Demo đã test và chạy ổn
- [ ] Video backup đã sẵn sàng
- [ ] Benchmarks đã có số liệu
- [ ] Equipment đã check
- [ ] Team đã practice
- [ ] Q&A đã chuẩn bị
- [ ] Tinh thần sẵn sàng

---

**Chúc các em thành công! 🚀**

