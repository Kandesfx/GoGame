# TÀI LIỆU THIẾT KẾ HỆ THỐNG - TRÒ CHƠI CỜ VÂY
## Đồ Án Môn Học: Trí Tuệ Nhân Tạo

**DOCUMENT VERSION:** 2.0  
**DATE:** December 2024  
**STATUS:** ✅ Complete - Based on Current Implementation

---

## PHẦN 1 — SYSTEM OVERVIEW

### 1.1. Giới thiệu dự án

**Tên dự án:** GoGame - Hệ thống trò chơi Cờ Vây thông minh  
**Đề tài:** Đề tài số 18 - Xây dựng AI chơi Cờ Vây  
**Môn học:** Trí Tuệ Nhân Tạo (Artificial Intelligence)  
**Năm học:** 2024-2025

GoGame là một hệ thống trò chơi Cờ Vây hoàn chỉnh với:
- **AI Engine mạnh mẽ** sử dụng Minimax và MCTS (C++)
- **Machine Learning models** cho phân tích vị trí và gợi ý nước đi
- **Backend API** (FastAPI) với PostgreSQL và MongoDB
- **Frontend web** (React + Vite) hiện đại và responsive
- **Hệ thống xếp hạng ELO** và matchmaking tự động
- **Tính năng premium** với coin system

### 1.2. Kiến trúc tổng quan

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (React + Vite)                    │
│  - Game Board UI                                             │
│  - Matchmaking Dialog                                        │
│  - Statistics & Leaderboard                                  │
│  - Premium Features UI                                       │
└───────────────────────────┬─────────────────────────────────┘
                             │ HTTP/WebSocket
┌───────────────────────────▼─────────────────────────────────┐
│              Backend API (FastAPI)                          │
│  - Authentication & Authorization                           │
│  - Match Management                                         │
│  - Matchmaking Service                                      │
│  - Premium & Coin Services                                 │
│  - ML Analysis Service                                      │
└───────┬───────────────────────────────┬─────────────────────┘
        │                               │
┌───────▼────────┐            ┌─────────▼──────────┐
│  PostgreSQL    │            │     MongoDB        │
│  - Users       │            │  - Game States     │
│  - Matches     │            │  - SGF Records     │
│  - Coins       │            │  - ML Analysis     │
│  - Premium     │            │  - AI Logs         │
└───────┬────────┘            └────────────────────┘
        │
┌───────▼──────────────────────────────────────────┐
│         C++ AI Engine (gogame_py)                │
│  - Board Logic (board.cpp)                       │
│  - Minimax Engine (minimax_engine.cpp)           │
│  - MCTS Engine (mcts_engine.cpp)                  │
│  - Opening Book (opening_book.cpp)               │
│  - Python Bindings (pybind11)                    │
└───────┬──────────────────────────────────────────┘
        │
┌───────▼──────────────────────────────────────────┐
│         ML Models (PyTorch)                      │
│  - Policy Network                                │
│  - Value Network                                 │
│  - Multi-Task Model (Threat/Attack/Intent)      │
└──────────────────────────────────────────────────┘
```

### 1.3. Công nghệ sử dụng

#### 1.3.1. Backend
- **Framework:** FastAPI 0.111.0
- **Database:**
  - PostgreSQL 14+ (SQLAlchemy 2.0) - Structured data
  - MongoDB 6+ (Motor) - Game states, SGF records
- **Authentication:** JWT (Access + Refresh tokens), Argon2 password hashing
- **Migration:** Alembic
- **ML Framework:** PyTorch 2.0+ (optional)

#### 1.3.2. Frontend
- **Framework:** React 18.2.0
- **Build Tool:** Vite 7.2.4
- **HTTP Client:** Axios 1.6.0
- **Routing:** React Router DOM 6.20.0
- **UI Libraries:** Framer Motion, React Icons

#### 1.3.3. AI Engine (C++)
- **Language:** C++20
- **Build System:** CMake 3.20+, Ninja
- **Python Bindings:** pybind11
- **Algorithms:**
  - Minimax với Alpha-Beta Pruning
  - Monte Carlo Tree Search (MCTS)
  - Opening Book
  - Transposition Table
  - Move Ordering

#### 1.3.4. Machine Learning
- **Framework:** PyTorch 2.0+
- **Models:**
  - Policy Network (move prediction)
  - Value Network (position evaluation)
  - Multi-Task Model (threat detection, attack opportunities, intent recognition)

### 1.4. Cấu trúc dự án

```
GoGame/
├── backend/                    # FastAPI backend
│   ├── app/
│   │   ├── main.py            # FastAPI app entry
│   │   ├── config.py           # Settings
│   │   ├── database.py        # DB connections
│   │   ├── routers/           # API endpoints
│   │   │   ├── auth.py
│   │   │   ├── users.py
│   │   │   ├── matches.py
│   │   │   ├── matchmaking.py
│   │   │   ├── coins.py
│   │   │   ├── premium.py
│   │   │   ├── ml.py
│   │   │   └── statistics.py
│   │   ├── services/          # Business logic
│   │   │   ├── auth_service.py
│   │   │   ├── match_service.py
│   │   │   ├── matchmaking_service.py
│   │   │   ├── coin_service.py
│   │   │   ├── premium_service.py
│   │   │   ├── ml_service.py
│   │   │   └── statistics_service.py
│   │   ├── models/            # Database models
│   │   │   ├── sql/          # SQLAlchemy models
│   │   │   └── mongo/        # MongoDB models
│   │   ├── schemas/          # Pydantic schemas
│   │   └── utils/            # Utilities
│   ├── migrations/           # Alembic migrations
│   └── requirements.txt
│
├── frontend-web/              # React frontend
│   ├── src/
│   │   ├── components/       # React components
│   │   │   ├── Board.jsx
│   │   │   ├── MatchmakingDialog.jsx
│   │   │   ├── Leaderboard.jsx
│   │   │   ├── PremiumFeatures.jsx
│   │   │   └── ...
│   │   ├── services/        # API client
│   │   └── contexts/        # React contexts
│   └── package.json
│
├── src/                       # C++ AI engine source
│   ├── game/
│   │   ├── board.h/.cpp      # Board logic
│   ├── ai/
│   │   ├── minimax/          # Minimax engine
│   │   │   ├── minimax_engine.h/.cpp
│   │   │   ├── evaluator.h/.cpp
│   │   │   ├── move_ordering.h/.cpp
│   │   │   ├── transposition_table.h/.cpp
│   │   │   └── game_tree.h/.cpp
│   │   ├── mcts/             # MCTS engine
│   │   │   ├── mcts_engine.h/.cpp
│   │   │   └── mcts_node.h/.cpp
│   │   ├── ai_player.h/.cpp   # AI player interface
│   │   └── opening_book.h/.cpp
│   └── bindings/
│       └── python_bindings.cpp  # pybind11 bindings
│
├── src/ml/                    # ML models (Python)
│   ├── policy_network.py
│   ├── value_network.py
│   ├── models/
│   │   ├── multi_task_model.py
│   │   ├── shared_backbone.py
│   │   ├── threat_head.py
│   │   ├── attack_head.py
│   │   └── intent_head.py
│   └── training/
│       └── data_collector.py
│
├── build/                     # Build output
│   └── gogame_py*.pyd        # Python module
│
├── scripts/                   # Utility scripts
│   ├── parse_sgf_*.py        # SGF parsing
│   ├── train_*.py            # Training scripts
│   └── test_*.py             # Test scripts
│
└── docs/                      # Documentation
    ├── SystemSpec.md         # This file
    ├── BackendDesign.md
    └── ...
```

---

## PHẦN 2 — DATABASE DESIGN

### 2.1. PostgreSQL Schema

#### 2.1.1. Users Table
```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(32) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    elo_rating INTEGER DEFAULT 1500 NOT NULL,
    coins INTEGER DEFAULT 0 NOT NULL,
    display_name VARCHAR(64),
    avatar_url VARCHAR(255),
    preferences JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    last_login TIMESTAMP WITH TIME ZONE
);
```

#### 2.1.2. Matches Table
```sql
CREATE TABLE matches (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    black_player_id UUID REFERENCES users(id) ON DELETE SET NULL,
    white_player_id UUID REFERENCES users(id) ON DELETE SET NULL,
    ai_level INTEGER,  -- NULL nếu là PvP
    board_size INTEGER DEFAULT 9 NOT NULL,
    result VARCHAR(32),  -- 'black_wins', 'white_wins', 'draw', 'resign'
    room_code VARCHAR(6) UNIQUE,  -- Mã phòng 6 ký tự
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    finished_at TIMESTAMP WITH TIME ZONE,
    sgf_id VARCHAR(64),  -- Reference to MongoDB
    premium_analysis_id VARCHAR(64),  -- Reference to MongoDB
    
    -- Time control (PvP only)
    time_control_minutes INTEGER,
    black_time_remaining_seconds INTEGER,
    white_time_remaining_seconds INTEGER,
    last_move_at TIMESTAMP WITH TIME ZONE,
    
    -- ELO changes (PvP only)
    black_elo_change INTEGER,
    white_elo_change INTEGER,
    
    -- Ready status (matchmaking)
    black_ready BOOLEAN DEFAULT FALSE NOT NULL,
    white_ready BOOLEAN DEFAULT FALSE NOT NULL
);
```

#### 2.1.3. Coin Transactions Table
```sql
CREATE TABLE coin_transactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    amount INTEGER NOT NULL,  -- Positive = earn, Negative = spend
    type VARCHAR(32) NOT NULL,  -- 'earn', 'spend'
    source VARCHAR(64),  -- 'daily_login', 'win_game', 'hint', etc.
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    CONSTRAINT chk_coin_transactions_amount CHECK (amount != 0)
);
```

#### 2.1.4. Premium Subscriptions Table
```sql
CREATE TABLE premium_subscriptions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
    status VARCHAR(20) NOT NULL,  -- 'active', 'expired', 'cancelled'
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);
```

#### 2.1.5. Premium Requests Table
```sql
CREATE TABLE premium_requests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    match_id UUID NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    feature VARCHAR(32) NOT NULL,  -- 'hint', 'analysis', 'review'
    cost INTEGER NOT NULL,
    status VARCHAR(32) DEFAULT 'pending' NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    completed_at TIMESTAMP WITH TIME ZONE
);
```

#### 2.1.6. Refresh Tokens Table
```sql
CREATE TABLE refresh_tokens (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token TEXT UNIQUE NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    revoked BOOLEAN DEFAULT FALSE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);
```

### 2.2. MongoDB Collections

#### 2.2.1. Game States Collection
```javascript
{
  _id: ObjectId,
  match_id: String,  // UUID reference to PostgreSQL
  board_size: Number,
  moves: [
    {
      move_number: Number,
      color: String,  // 'black' or 'white'
      position: { row: Number, col: Number } | null,  // null = pass
      timestamp: ISODate,
      time_remaining: Number  // seconds
    }
  ],
  current_position: {
    board: [[String]],  // 'black', 'white', 'empty'
    current_player: String,
    move_number: Number,
    prisoners: { black: Number, white: Number }
  },
  sgf_data: String,  // Full SGF string
  created_at: ISODate,
  updated_at: ISODate
}
```

#### 2.2.2. ML Analysis Collection
```javascript
{
  _id: ObjectId,
  match_id: String,
  position_id: String,  // Unique position identifier
  analysis_type: String,  // 'hint', 'analysis', 'review'
  results: {
    best_moves: [
      { position: { row: Number, col: Number }, score: Number }
    ],
    win_probability: Number,
    territory_map: [[Number]],  // Territory estimates
    threat_heatmap: [[Number]],  // Threat detection
    attack_heatmap: [[Number]],  // Attack opportunities
    intent_prediction: {
      primary_intent: String,
      confidence: Number,
      predicted_moves: [Object]
    }
  },
  model_version: String,
  created_at: ISODate
}
```

---

## PHẦN 3 — AI ENGINE DESIGN

### 3.1. Tổng quan

AI Engine được implement bằng C++ với Python bindings (pybind11), hỗ trợ 2 thuật toán chính:
1. **Minimax** với Alpha-Beta Pruning (Level 1-4)
2. **Monte Carlo Tree Search (MCTS)** (Level 5-6, optional)

### 3.2. AI Levels

#### 3.2.1. Level 1: Beginner (Dễ)
- **Algorithm:** Minimax depth 1
- **Features:** 
  - Không dùng alpha-beta, move ordering, transposition table
  - 15% chance random move, 10% chance suboptimal move
- **Timeout:** 5 giây
- **Opening Book:** Không

#### 3.2.2. Level 2: Intermediate (Trung bình)
- **Algorithm:** Minimax depth 2
- **Features:**
  - Alpha-beta pruning
  - Move ordering
  - Opening book
- **Timeout:** 8 giây

#### 3.2.3. Level 3: Hard (Khó)
- **Algorithm:** Minimax depth 2-3 (tùy board size)
- **Features:**
  - Alpha-beta pruning
  - Move ordering
  - Transposition table (nếu depth >= 3)
  - Opening book
- **Timeout:** 6-15 giây (tùy board size)

#### 3.2.4. Level 4: Expert (Siêu khó)
- **Algorithm:** Minimax depth 3-4
- **Features:**
  - Tất cả optimizations
  - Opening book
  - Advanced move ordering
- **Timeout:** 10-20 giây

#### 3.2.5. Level 5-6: MCTS (Optional)
- **Algorithm:** MCTS với ML guidance
- **Features:**
  - Policy network guidance
  - Value network evaluation
  - High playout count (5000-10000)

### 3.3. Minimax Engine

#### 3.3.1. Core Components
- **MinimaxEngine:** Main search algorithm
- **Evaluator:** Position evaluation function
- **MoveOrdering:** Move ordering heuristics
- **TranspositionTable:** Cache for position evaluations
- **GameTree:** Tree structure for visualization

#### 3.3.2. Evaluation Function
```cpp
class Evaluator {
    float evaluate(const Board& board, Color color) {
        float score = 0.0;
        
        // Territory estimation
        score += territory_weight * estimate_territory(board, color);
        
        // Group strength
        score += group_weight * evaluate_groups(board, color);
        
        // Pattern matching
        score += pattern_weight * match_patterns(board, color);
        
        // Prisoners
        score += prisoner_weight * (board.prisoners(color) - board.prisoners(opponent(color)));
        
        return score;
    }
};
```

### 3.4. MCTS Engine

#### 3.4.1. Core Components
- **MCTSEngine:** Main MCTS algorithm
- **MCTSNode:** Tree node structure
- **UCB1:** Selection policy
- **Simulation:** Random playouts

#### 3.4.2. MCTS Process
1. **Selection:** Traverse tree using UCB1
2. **Expansion:** Add new node
3. **Simulation:** Random playout
4. **Backpropagation:** Update statistics

### 3.5. Opening Book

- **Format:** Pre-computed opening sequences
- **Usage:** Levels 2-4 sử dụng opening book
- **Storage:** Embedded trong code hoặc file

---

## PHẦN 4 — MACHINE LEARNING

### 4.1. Tổng quan

ML models được sử dụng cho:
- **Premium Features:** Hint, Analysis, Review
- **Position Evaluation:** Win probability, territory estimation
- **Tactical Analysis:** Threat detection, attack opportunities, intent recognition

### 4.2. Model Architecture

#### 4.2.1. Multi-Task Learning Model
```
Input: 17 planes × board_size × board_size
  ↓
Shared Backbone (ResNet-like)
  - 4 residual blocks
  - 64 base channels
  ↓
┌─────────────┬─────────────┬─────────────┐
│ Threat Head │ Attack Head │ Intent Head │
│ (Heatmap)   │ (Heatmap)   │ (Classification)│
└─────────────┴─────────────┴─────────────┘
```

#### 4.2.2. Policy Network
- **Input:** Board state (17 planes)
- **Output:** Move probability distribution
- **Usage:** Move suggestion, MCTS guidance

#### 4.2.3. Value Network
- **Input:** Board state (17 planes)
- **Output:** Win probability (0-1)
- **Usage:** Position evaluation

### 4.3. Training Pipeline

#### 4.3.1. Data Collection
- **Source:** SGF files từ professional games
- **Processing:** Parse SGF → extract positions → generate labels
- **Format:** Chunked datasets (9x9, 13x13, 19x19)

#### 4.3.2. Training Process
1. **Data Preprocessing:** Normalize, augment
2. **Model Training:** Supervised learning với multi-task loss
3. **Evaluation:** Test trên held-out games
4. **Deployment:** Save checkpoint → load in backend

### 4.4. ML Service Integration

```python
# backend/app/services/ml_service.py
class MLService:
    def analyze_position(self, board_state, analysis_type):
        # Load model
        model = self.load_model()
        
        # Preprocess input
        features = self.preprocess(board_state)
        
        # Run inference
        results = model(features)
        
        # Post-process
        return self.postprocess(results)
```

---

## PHẦN 5 — API DESIGN

### 5.1. Authentication Endpoints

```
POST   /auth/register          # Đăng ký
POST   /auth/login             # Đăng nhập
POST   /auth/refresh           # Refresh token
POST   /auth/logout            # Đăng xuất
```

### 5.2. Match Endpoints

```
POST   /matches/create         # Tạo match (PvP hoặc PvAI)
GET    /matches/{match_id}     # Lấy thông tin match
POST   /matches/{match_id}/move # Đánh nước cờ
POST   /matches/{match_id}/resign # Đầu hàng
GET    /matches/{match_id}/history # Lịch sử nước đi
```

### 5.3. Matchmaking Endpoints

```
POST   /matchmaking/queue/join    # Tham gia queue
GET    /matchmaking/queue/status  # Trạng thái queue
GET    /matchmaking/queue/match   # Kiểm tra match found
POST   /matchmaking/queue/leave   # Rời queue
```

### 5.4. Premium Endpoints

```
POST   /premium/hint             # Gợi ý nước đi (10 coins)
POST   /premium/analysis          # Phân tích vị trí (20 coins)
POST   /premium/review            # Review ván đấu (30 coins)
```

### 5.5. Coin Endpoints

```
GET    /coins/balance            # Số dư coins
POST   /coins/daily-bonus        # Nhận daily bonus
GET    /coins/transactions       # Lịch sử giao dịch
```

### 5.6. Statistics Endpoints

```
GET    /statistics/profile       # Thống kê cá nhân
GET    /statistics/leaderboard   # Bảng xếp hạng
GET    /statistics/matches       # Lịch sử matches
```

### 5.7. ML Endpoints

```
POST   /ml/analyze               # Phân tích vị trí (ML)
GET    /ml/models                # Danh sách models
```

---

## PHẦN 6 — FEATURES

### 6.1. Core Features (Đã implement)

#### 6.1.1. Game Modes
- ✅ **PvP (Player vs Player):** Chơi với người chơi khác
- ✅ **PvAI (Player vs AI):** Chơi với AI (4 levels)
- ✅ **Room Code:** Tạo phòng và tham gia bằng mã

#### 6.1.2. Matchmaking
- ✅ **Automatic Matching:** Ghép trận tự động dựa trên ELO và board size
- ✅ **ELO-based:** Tìm đối thủ có ELO tương đồng
- ✅ **Ready System:** Cả 2 players phải ready mới bắt đầu

#### 6.1.3. Ranking System
- ✅ **ELO Rating:** Hệ thống xếp hạng ELO (khởi đầu 1500)
- ✅ **Leaderboard:** Top 100 người chơi
- ✅ **Statistics:** Win rate, total matches, etc.

#### 6.1.4. Time Control
- ✅ **Time Limits:** Giới hạn thời gian cho mỗi người chơi
- ✅ **Time Tracking:** Theo dõi thời gian còn lại
- ✅ **Timeout Detection:** Tự động kết thúc khi hết thời gian

#### 6.1.5. Premium Features
- ✅ **Hint:** Gợi ý nước đi tốt (10 coins)
- ✅ **Analysis:** Phân tích vị trí chi tiết (20 coins)
- ✅ **Review:** Review toàn bộ ván đấu (30 coins)

#### 6.1.6. Coin System
- ✅ **Earning:** Daily login, win game, complete game
- ✅ **Spending:** Premium features
- ✅ **Transaction History:** Lịch sử giao dịch

### 6.2. UI/UX Features

#### 6.2.1. Game Board
- ✅ **Interactive Board:** Click để đánh cờ
- ✅ **Move History:** Hiển thị lịch sử nước đi
- ✅ **Undo:** Hoàn tác nước đi (trong game)
- ✅ **Ko Detection:** Cảnh báo khi có Ko

#### 6.2.2. Visualizations
- ✅ **ML Analysis Overlay:** Hiển thị phân tích ML
- ✅ **Threat Visualization:** Hiển thị mối đe dọa
- ✅ **Attack Visualization:** Hiển thị cơ hội tấn công
- ✅ **Intent Display:** Hiển thị ý định đối thủ

#### 6.2.3. Dialogs
- ✅ **Login/Register:** Đăng nhập/Đăng ký
- ✅ **Matchmaking:** Tìm đối thủ
- ✅ **Settings:** Cài đặt
- ✅ **Shop:** Mua coins
- ✅ **Premium:** Premium features

---

## PHẦN 7 — DEPLOYMENT

### 7.1. Backend Deployment

#### 7.1.1. Local Development
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

#### 7.1.2. Production (Fly.io)
- **Platform:** Fly.io
- **Database:** PostgreSQL (Fly Postgres)
- **MongoDB:** MongoDB Atlas hoặc Fly.io
- **Environment Variables:** `.env` file

### 7.2. Frontend Deployment

#### 7.2.1. Local Development
```bash
cd frontend-web
npm install
npm run dev
```

#### 7.2.2. Production Build
```bash
npm run build
# Deploy dist/ folder to static hosting
```

### 7.3. C++ AI Module Build

#### 7.3.1. Windows (MSYS2)
```bash
# Trong MSYS2 MinGW 64-bit shell
cd build
cmake .. -G "Ninja" -DCMAKE_BUILD_TYPE=Release
cmake --build . --target gogame_py
```

#### 7.3.2. Linux/Mac
```bash
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --target gogame_py
```

---

## PHẦN 8 — HƯỚNG PHÁT TRIỂN TƯƠNG LAI

### 8.1. Cải tiến AI Engine với Heuristic Chuyên Nghiệp

#### 8.1.1. Tổng quan

Mục tiêu là nâng cấp AI từ các heuristic cơ bản sang một hệ thống AI có khả năng suy nghĩ như một kỳ thủ chuyên nghiệp, hiểu sâu về các khái niệm chiến thuật và chiến lược trong Cờ Vây.

#### 8.1.2. Các Yếu Tố Chuyên Nghiệp Cần Cải Tiến

**A. Fuseki (Khai Cuộc) - Opening Strategy**

```cpp
class FusekiHeuristic {
    // Các nguyên tắc khai cuộc chuyên nghiệp
    float evaluate_fuseki(const Board& board, Color color) {
        float score = 0.0;
        
        // 1. Corner-first principle (Góc trước)
        score += evaluate_corner_occupation(board, color) * 0.3;
        
        // 2. Side expansion (Mở rộng cạnh)
        score += evaluate_side_expansion(board, color) * 0.25;
        
        // 3. Balance (Cân bằng)
        score += evaluate_board_balance(board, color) * 0.2;
        
        // 4. Influence (Ảnh hưởng)
        score += evaluate_influence_spread(board, color) * 0.15;
        
        // 5. Avoid overconcentration (Tránh tập trung quá mức)
        score -= evaluate_overconcentration_penalty(board, color) * 0.1;
        
        return score;
    }
    
    // Đánh giá việc chiếm góc
    float evaluate_corner_occupation(const Board& board, Color color) {
        // Góc có giá trị cao nhất trong khai cuộc
        // 3-3, 3-4, 4-4 là các điểm khai cuộc chuẩn
        float corner_value = 0.0;
        for (auto corner : {Point(2,2), Point(2, board.size-3), 
                           Point(board.size-3, 2), 
                           Point(board.size-3, board.size-3)}) {
            if (board.get(corner) == color) {
                corner_value += 1.0;
            }
        }
        return corner_value / 4.0;  // Normalize
    }
};
```

**B. Joseki (Định Thức Góc) - Corner Patterns**

```cpp
class JosekiHeuristic {
    // Database các joseki patterns phổ biến
    std::map<PatternHash, JosekiSequence> joseki_database;
    
    float evaluate_joseki(const Board& board, const Move& move) {
        // 1. Kiểm tra xem move có phải là joseki move không
        PatternHash pattern = extract_corner_pattern(board, move.position);
        
        if (joseki_database.count(pattern)) {
            JosekiSequence sequence = joseki_database[pattern];
            
            // 2. Đánh giá theo joseki
            float joseki_score = evaluate_joseki_sequence(
                board, move, sequence
            );
            
            // 3. Điều chỉnh theo context (có thể không nên chơi joseki trong một số tình huống)
            float context_adjustment = evaluate_joseki_context(
                board, move, sequence
            );
            
            return joseki_score * context_adjustment;
        }
        
        return 0.0;  // Không phải joseki move
    }
    
    // Đánh giá context: có nên chơi joseki không?
    float evaluate_joseki_context(const Board& board, const Move& move, 
                                  const JosekiSequence& sequence) {
        // Nếu đối thủ đã có quân ở gần → có thể không nên chơi joseki
        // Nếu có nhóm yếu cần cứu → ưu tiên cứu nhóm
        // Nếu đang trong endgame → joseki không còn quan trọng
        
        float context = 1.0;
        
        // Kiểm tra có nhóm yếu không
        if (has_weak_group_nearby(board, move.position)) {
            context *= 0.3;  // Giảm giá trị joseki
        }
        
        // Kiểm tra giai đoạn ván đấu
        int move_count = board.get_move_count();
        if (move_count > 100) {  // Endgame
            context *= 0.5;  // Joseki ít quan trọng hơn
        }
        
        return context;
    }
};
```

**C. Tesuji (Nước Đi Tinh Tế) - Tactical Moves**

```cpp
class TesujiHeuristic {
    // Các tesuji patterns phổ biến
    float evaluate_tesuji(const Board& board, const Move& move) {
        float score = 0.0;
        
        // 1. Atari (Đe dọa bắt quân)
        if (is_atari_move(board, move)) {
            score += 50.0;
            
            // Atari với mục đích bắt quân
            if (can_capture_after_atari(board, move)) {
                score += 30.0;
            }
            
            // Atari với mục đích cứu quân
            if (saves_group_after_atari(board, move)) {
                score += 40.0;
            }
        }
        
        // 2. Hane (Gập)
        if (is_hane_move(board, move)) {
            score += evaluate_hane_value(board, move) * 25.0;
        }
        
        // 3. Cut (Cắt)
        if (is_cut_move(board, move)) {
            score += evaluate_cut_value(board, move) * 35.0;
        }
        
        // 4. Connection (Kết nối)
        if (is_connection_move(board, move)) {
            score += evaluate_connection_value(board, move) * 30.0;
        }
        
        // 5. Peep (Chọc)
        if (is_peep_move(board, move)) {
            score += evaluate_peep_value(board, move) * 20.0;
        }
        
        // 6. Attachment (Dán)
        if (is_attachment_move(board, move)) {
            score += evaluate_attachment_value(board, move) * 25.0;
        }
        
        // 7. Shoulder hit (Đánh vai)
        if (is_shoulder_hit(board, move)) {
            score += evaluate_shoulder_hit_value(board, move) * 30.0;
        }
        
        return score;
    }
    
    // Đánh giá giá trị của cut
    float evaluate_cut_value(const Board& board, const Move& move) {
        // Cut có giá trị cao nếu:
        // - Tách được nhóm đối thủ thành 2 phần yếu
        // - Tạo được 2 nhóm có thể tấn công
        // - Ngăn đối thủ kết nối
        
        float value = 0.0;
        
        // Kiểm tra xem cut có tách nhóm không
        auto groups = find_separated_groups(board, move);
        if (groups.size() >= 2) {
            // Đánh giá độ yếu của các nhóm bị tách
            for (const auto& group : groups) {
                int liberties = count_liberties(board, group);
                if (liberties <= 2) {
                    value += 1.0;  // Nhóm rất yếu
                } else if (liberties <= 4) {
                    value += 0.5;  // Nhóm yếu
                }
            }
        }
        
        return value;
    }
};
```

**D. Life and Death (Sống Chết) - Critical Evaluation**

```cpp
class LifeAndDeathHeuristic {
    float evaluate_life_and_death(const Board& board, Color color) {
        float score = 0.0;
        
        // 1. Kiểm tra các nhóm có sống không
        auto groups = find_all_groups(board, color);
        for (const auto& group : groups) {
            LifeStatus status = determine_life_status(board, group);
            
            switch (status) {
                case LifeStatus::ALIVE:
                    score += 100.0;  // Nhóm sống rất quan trọng
                    break;
                case LifeStatus::ALIVE_WITH_TWO_EYES:
                    score += 150.0;  // Sống với 2 mắt = sống vĩnh viễn
                    break;
                case LifeStatus::UNSETTLED:
                    score += evaluate_settlement_potential(board, group) * 50.0;
                    break;
                case LifeStatus::DEAD:
                    score -= 200.0;  // Nhóm chết = mất rất nhiều điểm
                    break;
                case LifeStatus::IN_ATARI:
                    score -= 150.0;  // Sắp chết = rất nguy hiểm
                    break;
            }
        }
        
        // 2. Đánh giá khả năng giết nhóm đối thủ
        auto opponent_groups = find_all_groups(board, opponent(color));
        for (const auto& group : opponent_groups) {
            LifeStatus status = determine_life_status(board, group);
            
            if (status == LifeStatus::UNSETTLED || status == LifeStatus::IN_ATARI) {
                // Có thể giết nhóm đối thủ
                float kill_potential = evaluate_kill_potential(board, group);
                score += kill_potential * 80.0;
            }
        }
        
        return score;
    }
    
    LifeStatus determine_life_status(const Board& board, const Group& group) {
        // 1. Kiểm tra có 2 mắt không (sống vĩnh viễn)
        if (has_two_eyes(board, group)) {
            return LifeStatus::ALIVE_WITH_TWO_EYES;
        }
        
        // 2. Kiểm tra số liberties
        int liberties = count_liberties(board, group);
        if (liberties == 1) {
            return LifeStatus::IN_ATARI;
        }
        
        // 3. Kiểm tra có thể tạo mắt không
        if (can_make_eye(board, group)) {
            return LifeStatus::UNSETTLED;
        }
        
        // 4. Kiểm tra có thể bị giết không
        if (can_be_killed(board, group)) {
            return LifeStatus::DEAD;
        }
        
        // 5. Kiểm tra có thể sống không
        if (can_live(board, group)) {
            return LifeStatus::ALIVE;
        }
        
        return LifeStatus::UNSETTLED;
    }
};
```

**E. Ko Fights (Chiến Đấu Ko) - Strategic Ko**

```cpp
class KoFightHeuristic {
    float evaluate_ko_fight(const Board& board, const Move& move) {
        if (!is_ko_move(board, move)) {
            return 0.0;
        }
        
        float score = 0.0;
        
        // 1. Đánh giá giá trị của ko
        float ko_value = evaluate_ko_value(board, move);
        
        // 2. Đánh giá ko threats (đe dọa ko)
        float ko_threats = count_ko_threats(board, move);
        
        // 3. Đánh giá sente/gote trong ko
        bool is_sente = is_sente_ko(board, move);
        
        // 4. Tính toán
        score = ko_value * (1.0 + ko_threats * 0.1);
        if (is_sente) {
            score *= 1.5;  // Sente ko có giá trị cao hơn
        }
        
        return score;
    }
    
    float evaluate_ko_value(const Board& board, const Move& move) {
        // Ko có giá trị cao nếu:
        // - Liên quan đến nhóm lớn
        // - Liên quan đến territory lớn
        // - Quyết định sống/chết của nhóm
        
        float value = 0.0;
        
        // Kiểm tra nhóm liên quan
        auto related_groups = find_ko_related_groups(board, move);
        for (const auto& group : related_groups) {
            int group_size = group.size();
            value += group_size * 10.0;  // Nhóm càng lớn, ko càng quan trọng
        }
        
        // Kiểm tra territory
        float territory_affected = estimate_ko_territory(board, move);
        value += territory_affected * 5.0;
        
        return value;
    }
};
```

**F. Sente/Gote (Chủ Động/Bị Động) - Initiative**

```cpp
class SenteGoteHeuristic {
    float evaluate_sente_gote(const Board& board, const Move& move) {
        float score = 0.0;
        
        // 1. Kiểm tra move có phải sente không
        if (is_sente_move(board, move)) {
            score += 40.0;  // Sente moves có giá trị cao
            
            // Sente với mục đích cụ thể
            if (is_sente_to_save_group(board, move)) {
                score += 30.0;
            }
            if (is_sente_to_attack(board, move)) {
                score += 25.0;
            }
            if (is_sente_to_expand(board, move)) {
                score += 20.0;
            }
        } else if (is_gote_move(board, move)) {
            // Gote moves có giá trị thấp hơn
            score -= 20.0;
            
            // Nhưng nếu gote move là bắt buộc (phải đáp)
            if (is_forced_gote(board, move)) {
                score += 15.0;  // Bù lại một phần
            }
        }
        
        // 2. Đánh giá khả năng tạo sente cho nước tiếp theo
        float future_sente = evaluate_future_sente_potential(board, move);
        score += future_sente * 15.0;
        
        return score;
    }
    
    bool is_sente_move(const Board& board, const Move& move) {
        // Move là sente nếu đối thủ phải đáp lại
        Board test_board = board;
        test_board.make_move(move);
        
        // Kiểm tra xem có move nào bắt buộc đối thủ phải đáp không
        return has_urgent_response(test_board, opponent(move.color));
    }
};
```

**G. Aji (Tiềm Năng) - Potential**

```cpp
class AjiHeuristic {
    float evaluate_aji(const Board& board, const Move& move) {
        float score = 0.0;
        
        // 1. Đánh giá aji của các quân cờ đã đánh
        float existing_aji = evaluate_existing_aji(board, move);
        score += existing_aji * 20.0;
        
        // 2. Đánh giá aji được tạo ra bởi move
        float created_aji = evaluate_created_aji(board, move);
        score += created_aji * 25.0;
        
        // 3. Đánh giá aji bị phá hủy
        float destroyed_aji = evaluate_destroyed_aji(board, move);
        score -= destroyed_aji * 30.0;  // Phá hủy aji là xấu
        
        return score;
    }
    
    float evaluate_existing_aji(const Board& board, const Move& move) {
        // Aji là tiềm năng của các quân cờ có thể được sử dụng sau này
        float aji_value = 0.0;
        
        // Kiểm tra các quân cờ gần move
        auto nearby_stones = find_nearby_stones(board, move.position, 3);
        for (const auto& stone : nearby_stones) {
            // Đánh giá tiềm năng của quân cờ này
            float stone_aji = evaluate_stone_aji(board, stone);
            aji_value += stone_aji;
        }
        
        return aji_value;
    }
};
```

**H. Thickness/Thinness (Dày/Mỏng) - Group Strength**

```cpp
class ThicknessHeuristic {
    float evaluate_thickness(const Board& board, Color color) {
        float score = 0.0;
        
        // 1. Đánh giá độ dày của các nhóm
        auto groups = find_all_groups(board, color);
        for (const auto& group : groups) {
            float thickness = evaluate_group_thickness(board, group);
            score += thickness * 15.0;
        }
        
        // 2. Đánh giá độ mỏng (thinness) - nhóm mỏng dễ bị tấn công
        for (const auto& group : groups) {
            float thinness = evaluate_group_thinness(board, group);
            score -= thinness * 20.0;  // Mỏng = xấu
        }
        
        return score;
    }
    
    float evaluate_group_thickness(const Board& board, const Group& group) {
        // Nhóm dày có:
        // - Nhiều liberties
        // - Kết nối chắc chắn
        // - Khó bị tấn công
        
        float thickness = 0.0;
        
        // 1. Số liberties
        int liberties = count_liberties(board, group);
        thickness += liberties * 2.0;
        
        // 2. Độ kết nối
        float connectivity = evaluate_connectivity(board, group);
        thickness += connectivity * 3.0;
        
        // 3. Khả năng phòng thủ
        float defensive_strength = evaluate_defensive_strength(board, group);
        thickness += defensive_strength * 2.5;
        
        return thickness;
    }
};
```

**I. Endgame (Yose) - Endgame Play**

```cpp
class EndgameHeuristic {
    float evaluate_endgame(const Board& board, const Move& move) {
        // Endgame bắt đầu từ khoảng move 100-150
        int move_count = board.get_move_count();
        if (move_count < 100) {
            return 0.0;  // Chưa đến endgame
        }
        
        float score = 0.0;
        
        // 1. Đánh giá sente moves trong endgame
        if (is_sente_move(board, move)) {
            score += 30.0;
        }
        
        // 2. Đánh giá territory trong endgame
        float territory_gain = evaluate_endgame_territory(board, move);
        score += territory_gain * 25.0;
        
        // 3. Đánh giá reduction (giảm territory đối thủ)
        float reduction = evaluate_territory_reduction(board, move);
        score += reduction * 20.0;
        
        // 4. Đánh giá invasion (xâm nhập)
        if (is_invasion_move(board, move)) {
            float invasion_value = evaluate_invasion_value(board, move);
            score += invasion_value * 15.0;
        }
        
        return score;
    }
};
```

**J. Shape Judgment (Đánh Giá Hình Cờ) - Shape Quality**

```cpp
class ShapeHeuristic {
    float evaluate_shape(const Board& board, const Move& move) {
        float score = 0.0;
        
        // 1. Kiểm tra shape tốt
        if (is_good_shape(board, move)) {
            score += 25.0;
        }
        
        // 2. Kiểm tra shape xấu (cần tránh)
        if (is_bad_shape(board, move)) {
            score -= 40.0;
        }
        
        // 3. Đánh giá các shape patterns cụ thể
        if (is_empty_triangle(board, move)) {
            score -= 15.0;  // Empty triangle thường là shape xấu
        }
        if (is_bamboo_joint(board, move)) {
            score += 20.0;  // Bamboo joint là shape tốt
        }
        if (is_tiger_mouth(board, move)) {
            score += 18.0;  // Tiger mouth là shape tốt
        }
        
        return score;
    }
    
    bool is_good_shape(const Board& board, const Move& move) {
        // Shape tốt có:
        // - Kết nối chắc chắn
        // - Không có điểm yếu
        // - Linh hoạt
        
        return has_strong_connection(board, move) &&
               !has_weak_point(board, move) &&
               is_flexible(board, move);
    }
};
```

#### 8.1.3. Combined Professional Heuristic

```cpp
class ProfessionalHeuristic {
    FusekiHeuristic fuseki;
    JosekiHeuristic joseki;
    TesujiHeuristic tesuji;
    LifeAndDeathHeuristic life_death;
    KoFightHeuristic ko_fight;
    SenteGoteHeuristic sente_gote;
    AjiHeuristic aji;
    ThicknessHeuristic thickness;
    EndgameHeuristic endgame;
    ShapeHeuristic shape;
    
    float evaluate_position(const Board& board, const Move& move, Color color) {
        float total_score = 0.0;
        
        // 1. Xác định giai đoạn ván đấu
        GamePhase phase = determine_game_phase(board);
        
        // 2. Tính toán các heuristic theo giai đoạn
        switch (phase) {
            case GamePhase::FUSEKI:
                total_score += fuseki.evaluate_fuseki(board, color) * 0.3;
                total_score += joseki.evaluate_joseki(board, move) * 0.25;
                total_score += sente_gote.evaluate_sente_gote(board, move) * 0.2;
                total_score += aji.evaluate_aji(board, move) * 0.15;
                total_score += shape.evaluate_shape(board, move) * 0.1;
                break;
                
            case GamePhase::MIDDLE_GAME:
                total_score += life_death.evaluate_life_and_death(board, color) * 0.3;
                total_score += tesuji.evaluate_tesuji(board, move) * 0.25;
                total_score += ko_fight.evaluate_ko_fight(board, move) * 0.2;
                total_score += thickness.evaluate_thickness(board, color) * 0.15;
                total_score += sente_gote.evaluate_sente_gote(board, move) * 0.1;
                break;
                
            case GamePhase::ENDGAME:
                total_score += endgame.evaluate_endgame(board, move) * 0.4;
                total_score += sente_gote.evaluate_sente_gote(board, move) * 0.3;
                total_score += life_death.evaluate_life_and_death(board, color) * 0.2;
                total_score += tesuji.evaluate_tesuji(board, move) * 0.1;
                break;
        }
        
        // 3. Điều chỉnh theo context
        float context_adjustment = evaluate_context(board, move, phase);
        total_score *= context_adjustment;
        
        return total_score;
    }
    
    GamePhase determine_game_phase(const Board& board) {
        int move_count = board.get_move_count();
        int board_size = board.get_size();
        
        // Fuseki: 0-30 moves (9x9), 0-50 moves (19x19)
        // Middle game: 30-100 (9x9), 50-150 (19x19)
        // Endgame: 100+ (9x9), 150+ (19x19)
        
        int fuseki_end = board_size == 9 ? 30 : 50;
        int middle_end = board_size == 9 ? 100 : 150;
        
        if (move_count < fuseki_end) {
            return GamePhase::FUSEKI;
        } else if (move_count < middle_end) {
            return GamePhase::MIDDLE_GAME;
        } else {
            return GamePhase::ENDGAME;
        }
    }
};
```

### 8.2. Hệ thống Chat với Tích Hợp Công Nghệ Bên Thứ Ba

#### 8.2.1. Tổng quan

Thay vì tự xây dựng chat system từ đầu, tích hợp các dịch vụ chat chuyên nghiệp để có trải nghiệm tốt hơn, tính năng phong phú hơn, và giảm chi phí phát triển.

#### 8.2.2. Các Lựa Chọn Công Nghệ

**A. Firebase Realtime Database + Cloud Messaging**

**Ưu điểm:**
- ✅ Real-time synchronization tự động
- ✅ Offline support
- ✅ Scalable (Google infrastructure)
- ✅ Dễ tích hợp với React
- ✅ Free tier hào phóng

**Implementation:**
```javascript
// frontend-web/src/services/firebaseChat.js
import { initializeApp } from 'firebase/app';
import { getDatabase, ref, push, onValue, off } from 'firebase/database';
import { getMessaging, getToken, onMessage } from 'firebase/messaging';

class FirebaseChatService {
    constructor(config) {
        this.app = initializeApp(config);
        this.db = getDatabase(this.app);
        this.messaging = getMessaging(this.app);
    }
    
    // Gửi tin nhắn
    async sendMessage(roomId, userId, message) {
        const messagesRef = ref(this.db, `chat_rooms/${roomId}/messages`);
        await push(messagesRef, {
            userId,
            message,
            timestamp: Date.now(),
            type: 'text'
        });
    }
    
    // Lắng nghe tin nhắn mới
    subscribeToMessages(roomId, callback) {
        const messagesRef = ref(this.db, `chat_rooms/${roomId}/messages`);
        onValue(messagesRef, (snapshot) => {
            const messages = [];
            snapshot.forEach((child) => {
                messages.push({
                    id: child.key,
                    ...child.val()
                });
            });
            callback(messages);
        });
    }
    
    // Push notifications
    async setupPushNotifications() {
        const token = await getToken(this.messaging);
        // Gửi token lên backend để lưu
        return token;
    }
}
```

**B. SendBird (Professional Chat Platform)**

**Ưu điểm:**
- ✅ Feature-rich (typing indicators, read receipts, file sharing)
- ✅ Scalable và reliable
- ✅ Good documentation
- ✅ Support nhiều platforms

**Implementation:**
```javascript
// frontend-web/src/services/sendbirdChat.js
import SendBird from 'sendbird';

class SendBirdChatService {
    constructor(appId) {
        this.sb = SendBird.getInstance();
        this.sb.init(appId);
    }
    
    async connect(userId, accessToken) {
        return new Promise((resolve, reject) => {
            this.sb.connect(userId, accessToken, (user, error) => {
                if (error) reject(error);
                else resolve(user);
            });
        });
    }
    
    async sendMessage(channelUrl, message) {
        const channel = await this.sb.OpenChannel.getChannel(channelUrl);
        return channel.sendUserMessage(message);
    }
    
    // Typing indicators
    startTyping(channelUrl) {
        const channel = this.sb.OpenChannel.getChannel(channelUrl);
        channel.startTyping();
    }
    
    // Read receipts
    markAsRead(channelUrl) {
        const channel = this.sb.OpenChannel.getChannel(channelUrl);
        channel.markAsRead();
    }
}
```

**C. Socket.io với Redis Adapter (Self-hosted)**

**Ưu điểm:**
- ✅ Full control
- ✅ Customizable
- ✅ Không phụ thuộc bên thứ ba
- ✅ Có thể tích hợp với backend hiện tại

**Implementation:**
```python
# backend/app/websocket/chat_handler.py
from socketio import AsyncServer
import redis.asyncio as redis

class ChatService:
    def __init__(self):
        self.sio = AsyncServer(
            cors_allowed_origins="*",
            async_mode='asgi'
        )
        self.redis = redis.from_url("redis://localhost:6379")
        self.setup_handlers()
    
    def setup_handlers(self):
        @self.sio.on('connect')
        async def on_connect(sid, environ, auth):
            user_id = auth.get('user_id')
            await self.sio.save_session(sid, {'user_id': user_id})
            await self.sio.enter_room(sid, f'user_{user_id}')
        
        @self.sio.on('join_game_chat')
        async def on_join_game_chat(sid, data):
            game_id = data['game_id']
            await self.sio.enter_room(sid, f'game_{game_id}')
        
        @self.sio.on('send_message')
        async def on_send_message(sid, data):
            session = await self.sio.get_session(sid)
            user_id = session['user_id']
            room = data['room']
            message = data['message']
            
            # Lưu vào database
            await self.save_message(user_id, room, message)
            
            # Broadcast
            await self.sio.emit('new_message', {
                'user_id': user_id,
                'message': message,
                'timestamp': datetime.now().isoformat()
            }, room=room)
```

#### 8.2.3. Kiến Trúc Tích Hợp

```
┌─────────────────────────────────────────┐
│         Frontend (React)                 │
│  - Chat UI Components                   │
│  - Message List                          │
│  - Input Box                             │
└──────────────┬──────────────────────────┘
               │
               │ SDK/API
               │
┌──────────────▼──────────────────────────┐
│    Chat Service (Firebase/SendBird)     │
│  - Real-time messaging                   │
│  - Presence                              │
│  - Typing indicators                     │
│  - Read receipts                         │
└──────────────┬──────────────────────────┘
               │
               │ Webhook/Events
               │
┌──────────────▼──────────────────────────┐
│      Backend API (FastAPI)              │
│  - User authentication                  │
│  - Chat room management                 │
│  - Message persistence                  │
│  - Notification triggers                 │
└─────────────────────────────────────────┘
```

#### 8.2.4. Database Schema cho Chat

```sql
-- Chat rooms (reference to external chat service)
CREATE TABLE chat_rooms (
    id UUID PRIMARY KEY,
    room_type VARCHAR(20) NOT NULL, -- 'private', 'game', 'group'
    external_room_id VARCHAR(255), -- ID từ Firebase/SendBird
    game_id UUID REFERENCES matches(id), -- NULL nếu không phải game chat
    created_at TIMESTAMP DEFAULT NOW()
);

-- Chat participants
CREATE TABLE chat_participants (
    room_id UUID REFERENCES chat_rooms(id),
    user_id UUID REFERENCES users(id),
    joined_at TIMESTAMP DEFAULT NOW(),
    last_read_at TIMESTAMP,
    PRIMARY KEY(room_id, user_id)
);

-- Message metadata (actual messages lưu trong external service)
CREATE TABLE chat_messages_meta (
    id UUID PRIMARY KEY,
    room_id UUID REFERENCES chat_rooms(id),
    user_id UUID REFERENCES users(id),
    external_message_id VARCHAR(255), -- ID từ Firebase/SendBird
    message_type VARCHAR(20), -- 'text', 'emoji', 'system'
    created_at TIMESTAMP DEFAULT NOW()
);
```

#### 8.2.5. So Sánh Các Lựa Chọn

| Tiêu chí | Firebase | SendBird | Socket.io |
|----------|----------|----------|-----------|
| **Cost** | Free tier tốt | Paid | Free (self-hosted) |
| **Setup** | Dễ | Trung bình | Phức tạp hơn |
| **Features** | Cơ bản | Phong phú | Tùy chỉnh |
| **Scalability** | Tự động | Tốt | Phụ thuộc setup |
| **Control** | Ít | Trung bình | Full control |

**Khuyến nghị:**
- **MVP/Development:** Firebase (dễ setup, free)
- **Production Scale:** SendBird (features phong phú, reliable)
- **Full Control:** Socket.io (nếu cần customize nhiều)

### 8.3. Giải đấu với Coin Đặt Cược
- Tournament system
- Betting system
- Prize distribution

### 8.4. Hệ thống Cài đặt
- User settings (game, AI, UI, sound, notifications)
- Privacy settings
- Language settings

### 8.5. Chức năng Quên Mật khẩu
- Password reset via email
- Token-based reset

### 8.6. Cải tiến Hệ thống Alert
- Smart notifications
- Notification grouping
- Priority system

---

## PHẦN 9 — APPENDIX

### A. Technology Versions

```
Backend:
- Python: 3.10+
- FastAPI: 0.111.0
- SQLAlchemy: 2.0.30
- PostgreSQL: 14+
- MongoDB: 6.0+

Frontend:
- Node.js: 18+
- React: 18.2.0
- Vite: 7.2.4

AI Engine:
- C++: 20
- CMake: 3.20+
- pybind11: 2.11+

ML:
- PyTorch: 2.0+
- NumPy: 1.24+
```

### B. Key Files Reference

- **Backend Design:** `docs/BackendDesign.md`
- **AI Levels:** `docs/AI_LEVELS_EXPLAINED.md`
- **ELO System:** `docs/ELO_RANKING_SYSTEM.md`
- **Matchmaking:** `docs/MATCHMAKING_SYSTEM.md`
- **ML Guide:** `docs/ML_COMPREHENSIVE_GUIDE.md`
- **Setup Guide:** `README.md`, `SETUP.md`

### C. API Documentation

API documentation có thể được xem tại:
- **Swagger UI:** `http://localhost:8000/docs` (khi chạy backend)
- **ReDoc:** `http://localhost:8000/redoc`

---

## ✅ KẾT LUẬN

Tài liệu **SystemSpec.md** này mô tả toàn bộ hệ thống GoGame dựa trên **implementation thực tế** của dự án, bao gồm:

1. ✅ **System Overview** - Kiến trúc và công nghệ
2. ✅ **Database Design** - Schema PostgreSQL và MongoDB
3. ✅ **AI Engine Design** - Minimax và MCTS implementation
4. ✅ **Machine Learning** - ML models và training pipeline
5. ✅ **API Design** - REST API endpoints
6. ✅ **Features** - Các tính năng đã implement
7. ✅ **Deployment** - Hướng dẫn deploy
8. ✅ **Future Development** - Hướng phát triển tương lai

**Tài liệu này được cập nhật dựa trên code thực tế của dự án, không phải design document ban đầu.**

---

**DOCUMENT VERSION:** 2.0  
**DATE:** December 2024  
**STATUS:** ✅ Complete - Based on Current Implementation

---

**Kết thúc SystemSpec.md**
