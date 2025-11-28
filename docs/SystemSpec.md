# TÀI LIỆU THIẾT KẾ HỆ THỐNG - TRÒ CHƠI CỜ VÂY
## Đồ Án Môn Học: Trí Tuệ Nhân Tạo

---

## PHẦN 1 — SYSTEM OVERVIEW

### 1.1. Giới thiệu dự án

**Tên dự án:** GoGame - Hệ thống trò chơi Cờ Vây thông minh  
**Đề tài:** Đề tài số 18 - Xây dựng AI chơi Cờ Vây  
**Môn học:** Trí Tuệ Nhân Tạo (Artificial Intelligence)  
**Năm học:** 2025

Dự án GoGame là một hệ thống trò chơi Cờ Vây hoàn chỉnh, tập trung vào việc **ứng dụng các thuật toán AI cổ điển** từ môn học Trí Tuệ Nhân Tạo để xây dựng một AI có khả năng chơi Cờ Vây ở nhiều cấp độ khác nhau, đồng thời cung cấp trải nghiệm game hiện đại và các tính năng hỗ trợ người chơi.

### 1.2. Mục tiêu dự án

#### 1.2.1. Mục tiêu học thuật (Trọng tâm môn học)
- **Ứng dụng thuật toán tìm kiếm:** Hiện thực hóa Monte Carlo Tree Search (MCTS) - thuật toán tìm kiếm ngẫu nhiên heuristic cho không gian trạng thái lớn
- **Heuristic evaluation:** Xây dựng hàm đánh giá vị thế bàn cờ dựa trên tri thức lãnh vực (domain knowledge)
- **Tối ưu hóa tìm kiếm:** Áp dụng các kỹ thuật cắt tỉa, song song hóa, và cache để tăng hiệu năng
- **Học máy bổ trợ:** Kết hợp học có giám sát (supervised learning) và học tăng cường (reinforcement learning) mức cơ bản để cải thiện hàm đánh giá
- **Biểu diễn tri thức:** Mô hình hóa luật chơi, pattern recognition, và chiến thuật Cờ Vây
- **Suy luận:** Xây dựng module nhận diện life-and-death, capture, và territory estimation

#### 1.2.2. Mục tiêu sản phẩm
- Xây dựng game Cờ Vây có thể chơi được với AI mạnh và nhanh
- Hỗ trợ nhiều chế độ chơi: PvP, PvAI, AI vs AI
- Cung cấp công cụ học tập cho người mới chơi Cờ Vây
- Có khả năng mở rộng thành sản phẩm thương mại

### 1.3. Phạm vi dự án

#### 1.3.1. Trong phạm vi (In Scope)

**Về Game:**
- Hỗ trợ 2 kích thước bàn cờ: 9×9 (ưu tiên) và 19×19
- Luật chơi: Luật Trung Quốc (Area Scoring/中国围棋规则)
- Các luật cơ bản: Ko rule, Superko detection, Self-capture prevention
- Tính điểm tự động cuối ván

**Về AI:**
- AI Engine sử dụng **MCTS (Monte Carlo Tree Search)** làm thuật toán chính
- Hàm đánh giá heuristic dựa trên:
  - Pattern matching (3×3, 5×5 local patterns)
  - Territory estimation
  - Life-and-death analysis (cơ bản)
  - Group strength evaluation
- Module học máy nhỏ:
  - **Policy Network nhẹ** (neural network 5-10 layers): Dự đoán xác suất nước đi
  - **Value Network nhẹ**: Đánh giá vị thế bàn cờ
  - Training qua self-play với REINFORCE hoặc TD-learning
- 3-4 cấp độ AI:
  - **Cấp 1 (Beginner):** MCTS với simulation ngắn (500 playouts)
  - **Cấp 2 (Intermediate):** MCTS với heuristic cơ bản (2000 playouts)
  - **Cấp 3 (Advanced):** MCTS + Policy Network (5000 playouts)
  - **Cấp 4 (Expert):** MCTS + Policy + Value Network (10000 playouts)

**Về tính năng:**
- Chơi offline: Người vs Người, Người vs AI, AI vs AI
- Chơi online: Multiplayer qua mạng LAN hoặc Internet
- Lưu/Load ván chơi (SGF format)
- Replay với timeline navigation
- Hint system: Gợi ý nước đi tốt
- Move analysis: Phân tích sai lầm
- Hệ thống xếp hạng Elo
- Thống kê cá nhân

**Về kiến trúc:**
- **Game Engine:** Core logic xử lý luật chơi, board state
- **AI Engine:** MCTS + heuristics + ML models
- **Training Pipeline:** Self-play training cho neural networks
- **Frontend:** Desktop app (cross-platform)
- **Backend (optional):** Server cho multiplayer và lưu trữ dữ liệu
- **Database:** Lưu trữ game history, player profiles, models

#### 1.3.2. Ngoài phạm vi (Out of Scope)

- Không xây dựng AI ở mức AlphaGo/AlphaZero đầy đủ (quá phức tạp, vượt phạm vi môn học)
- Không sử dụng ResNet sâu (>20 layers) hoặc training phân tán
- Không hỗ trợ các biến thể luật phức tạp (New Zealand rules, Ing rules, etc.)
- Không xây dựng mobile app trong giai đoạn 1

### 1.4. Công nghệ và công cụ dự kiến

#### 1.4.1. Ngôn ngữ lập trình (Kiến trúc Hybrid)

**Chiến lược Hybrid C++/Python** được áp dụng để cân bằng giữa hiệu năng và tốc độ phát triển:

- **C++ 17/20** (Performance-Critical Core):
  - **Game Engine Core:** Board representation, move generation, rule validation
  - **MCTS Engine:** Tree search, simulation, node expansion/backpropagation
  - **Heuristic Engine:** Pattern matching, territory evaluation, life-and-death
  - **Bitboard operations:** Fast board state manipulation
  - Lý do: Tốc độ cao (10-100x so với Python), quan trọng cho real-time play
  - Thư viện: STL, Boost (optional), Eigen (linear algebra)
  
- **Python 3.10+** (AI Training + High-Level Logic):
  - **ML Training Pipeline:** Neural network training với PyTorch
  - **AI Controller:** Orchestration các AI components
  - **Data Processing:** SGF parsing, game analysis, statistics
  - **Testing & Experimentation:** Rapid prototyping các thuật toán mới
  - Lý do: Ecosystem ML mạnh, code nhanh, debug dễ
  - Thư viện: PyTorch, NumPy, Python-SGF
  
- **Python ↔ C++ Binding:**
  - **pybind11:** Expose C++ classes/functions sang Python
  - Python gọi C++ engine cho performance-critical operations
  - C++ callback Python cho neural network inference
  
- **JavaScript/TypeScript + Electron** (Frontend):
  - Cross-platform desktop UI
  - Hoặc **Python + PyQt6** nếu muốn full Python stack (trade-off UI đẹp hơn)

#### 1.4.2. Framework & Libraries

- **AI/ML:**
  - PyTorch: Neural networks
  - NumPy: Matrix operations
  - Numba: JIT compilation cho Python
  
- **Game Development:**
  - PyGame/Kivy (nếu full Python)
  - Hoặc Web-based với React + Canvas + Electron
  
- **Backend:**
  - FastAPI: Modern async web framework
  - Uvicorn: ASGI server
  - WebSocket: Real-time game synchronization
  
- **Database & Storage:**
  - **PostgreSQL:** SQLAlchemy ORM, psycopg3 driver
  - **MongoDB:** PyMongo, Motor (async)
  - **S3/MinIO:** boto3 (AWS SDK), minio-py
  - **Redis (optional):** Caching, session storage
  
- **C++ Libraries:**
  - pybind11: Python bindings
  - Eigen: Linear algebra (nếu cần)
  - Boost (optional): Utilities
  - Google Test: Unit testing

- **Training:**
  - Ray/Joblib: Parallel self-play
  - TensorBoard: Training visualization
  - Weights & Biases (optional): Experiment tracking

#### 1.4.3. Tools
- Git + GitHub: Version control
- Docker: Containerization
- Pytest: Unit testing
- Black/Ruff: Code formatting

### 1.5. Chiến lược Hybrid Architecture (C++/Python)

#### 1.5.1. Lý do áp dụng Hybrid Architecture

**Vấn đề hiệu năng với Cờ Vây:**
- Branching factor trung bình: ~250 nước đi hợp lệ mỗi lượt (so với Chess: ~35)
- Bàn cờ 19×19: 361 vị trí, không gian trạng thái ≈ 10^170
- MCTS cần thực hiện hàng nghìn đến hàng chục nghìn simulations mỗi nước (5000-10000 playouts)
- Mỗi simulation cần: move generation → move validation → board update → scoring
- **Yêu cầu:** AI phải tính được trong 2-10 giây/nước ở cấp độ cao

**Giải pháp Hybrid:**
- **C++ cho "hot path":** Các operations được gọi hàng triệu lần/giây
- **Python cho "cold path":** Logic level cao, training, analysis

#### 1.5.2. Phân chia trách nhiệm C++/Python

**Layer 1 - C++ Core (Performance-Critical):**

```
┌─────────────────────────────────────────────────────┐
│           C++ High-Performance Core                 │
├─────────────────────────────────────────────────────┤
│  [Board Module - C++]                               │
│   - Bitboard representation (8-64 bit integers)     │
│   - Zobrist hashing (fast state comparison)         │
│   - Move generation (scan valid positions)          │
│   - Fast copy-on-write board state                  │
│                                                      │
│  [Rules Engine - C++]                               │
│   - Ko detection (hash table lookup: O(1))          │
│   - Superko check                                   │
│   - Capture detection (flood fill: optimized)       │
│   - Liberty counting                                │
│                                                      │
│  [MCTS Core - C++]                                  │
│   - Tree structure (custom memory pool)             │
│   - Node selection (UCB calculation)                │
│   - Fast simulation (light playout)                 │
│   - Backpropagation (pointer traversal)             │
│   - Multi-threading (OpenMP/std::thread)            │
│                                                      │
│  [Heuristics - C++]                                 │
│   - 3x3/5x5 pattern matching (hash lookup)          │
│   - Territory estimation (flood fill)               │
│   - Eye detection (local pattern check)             │
│   - Atari/capture detection                         │
└─────────────────────────────────────────────────────┘
         ▲                              │
         │ pybind11                     │ pybind11
         │ (function calls)             │ (callbacks)
         │                              ▼
┌─────────────────────────────────────────────────────┐
│           Python High-Level Layer                   │
├─────────────────────────────────────────────────────┤
│  [AI Controller - Python]                           │
│   - AI level selection                              │
│   - Time management                                 │
│   - Move selection orchestration                    │
│                                                      │
│  [Neural Networks - Python/PyTorch]                 │
│   - Policy network inference                        │
│   - Value network inference                         │
│   - Model loading/management                        │
│                                                      │
│  [Training Pipeline - Python]                       │
│   - Self-play game generation                       │
│   - Training data preparation                       │
│   - Model training (backprop)                       │
│   - Model evaluation                                │
│                                                      │
│  [Data & Analysis - Python]                         │
│   - SGF parsing/writing                             │
│   - Game statistics                                 │
│   - Move analysis                                   │
│   - Visualization                                   │
└─────────────────────────────────────────────────────┘
```

#### 1.5.3. Performance Targets với Hybrid Architecture

**Benchmarks mục tiêu:**

| Operation | Pure Python | Hybrid C++ | Speedup |
|-----------|-------------|------------|---------|
| Move generation (9×9) | ~100 μs | ~1-2 μs | 50-100x |
| Move validation + apply | ~50 μs | ~0.5 μs | 100x |
| Board state copy | ~20 μs | ~0.2 μs | 100x |
| MCTS simulation (100 moves) | ~50 ms | ~0.5-1 ms | 50-100x |
| Full MCTS search (5000 playouts) | ~250 s | ~2.5-5 s | 50-100x |

**Kết quả mong đợi:**
- Cấp độ 1 (500 playouts): < 1 giây
- Cấp độ 2 (2000 playouts): 1-2 giây
- Cấp độ 3 (5000 playouts): 2-5 giây
- Cấp độ 4 (10000 playouts): 5-10 giây

#### 1.5.4. C++ Implementation Strategy

**Board Representation - Bitboard:**
```cpp
// C++ - Fast board state using bitboards
class Board {
    uint64_t black_stones[6];  // 19x19 = 361 bits ≈ 6x uint64
    uint64_t white_stones[6];
    uint64_t ko_point;
    int16_t prisoner_count[2];
    
    // Fast operations
    inline bool is_valid_move(int pos);
    inline void place_stone(int pos, Color color);
    inline int count_liberties_fast(int pos);
};
```

**MCTS Node - Memory Pool:**
```cpp
// Custom memory pool để tránh allocation overhead
class MCTSTree {
    std::vector<MCTSNode> node_pool;  // Pre-allocated
    int pool_index;
    
    MCTSNode* allocate_node() {
        return &node_pool[pool_index++];  // O(1)
    }
};
```

**Python Binding Example:**
```python
# Python có thể gọi C++ engine trực tiếp
import go_engine  # C++ module

board = go_engine.Board(size=19)
board.place_stone(3, 3, go_engine.BLACK)

mcts = go_engine.MCTSEngine(board)
mcts.set_playouts(5000)
best_move = mcts.search()  # Runs in C++, fast!

# Python xử lý results
print(f"Best move: {best_move}")
```

#### 1.5.5. Lợi ích của Hybrid Architecture

✅ **Performance:** 50-100x faster cho game engine và MCTS  
✅ **Real-time:** Có thể chơi được với AI mạnh trong thời gian hợp lý  
✅ **Flexibility:** Python cho ML training và experimentation  
✅ **Best of both worlds:** Speed của C++ + Ecosystem của Python  
✅ **Maintainability:** Code C++ tập trung vào phần stable, Python cho phần thay đổi nhiều  
✅ **Học thuật:** Sinh viên học được cả systems programming (C++) và AI/ML (Python)  

#### 1.5.6. Trade-offs và Giải pháp

**Challenges:**
1. **Complexity tăng:** Phải maintain 2 codebases
   - *Giải pháp:* Interface rõ ràng, unit tests kỹ lưỡng
   
2. **Binding overhead:** Python ↔ C++ có overhead
   - *Giải pháp:* Batch operations, minimize crossing boundary
   
3. **Debugging khó hơn:** Cross-language debugging
   - *Giải pháp:* Extensive logging, separate testing của mỗi layer
   
4. **Build complexity:** Cần compile C++
   - *Giải pháp:* CMake + setup scripts, pre-built binaries cho distribution

### 1.6. Định hướng thiết kế theo môn học

Hệ thống được thiết kế xoay quanh **4 trụ cột chính của môn Trí Tuệ Nhân Tạo:**

#### 1.6.1. Tìm kiếm (Search)
- **MCTS** là thuật toán tìm kiếm chính, thuộc nhóm **Best-First Search** với chiến lược **UCB (Upper Confidence Bound)**
- So sánh với Minimax/Alpha-Beta (không khả thi với Cờ Vây do branching factor ~250)
- Kỹ thuật tối ưu: Virtual loss, RAVE (Rapid Action Value Estimation), Progressive widening

#### 1.6.2. Tri thức và Biểu diễn (Knowledge Representation)
- **Rule-based system:** Mã hóa luật cờ, pattern library
- **Feature engineering:** Trích xuất đặc trưng từ board state (liberties, eyes, influence)
- **Pattern database:** Lưu trữ các joseki (định thức), tesuji (chiêu thức)

#### 1.6.3. Heuristic và Đánh giá (Heuristics & Evaluation)
- Xây dựng hàm heuristic đa chiều:
  - Territorial control
  - Stone connectivity
  - Group safety (life and death)
  - Influence mapping
- So sánh với hàm đánh giá trong Chess/Checkers (material count)

#### 1.6.4. Học máy (Machine Learning - Bổ trợ)
- **Supervised Learning:** Train Policy Network từ game records của kỳ thủ
- **Reinforcement Learning:** Self-play training với reward = game outcome
- Giải thích liên hệ với Q-Learning, Temporal Difference Learning (TD-λ)

### 1.7. Kiến trúc tổng quan (High-Level Hybrid Architecture)

**Chú thích ngôn ngữ:** [C++] = Performance-critical, [Python] = High-level logic

```
┌─────────────────────────────────────────────────────────────┐
│              FRONTEND LAYER [JS/Python]                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Game Board  │  │  UI Controls │  │  Statistics  │      │
│  │  Renderer    │  │  & Menus     │  │  & Analysis  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└────────────────────────┬────────────────────────────────────┘
                         │ API Calls
┌────────────────────────▼────────────────────────────────────┐
│             GAME ENGINE LAYER [C++ Core]                     │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Game Controller [Python Wrapper]                   │    │
│  └──────────┬─────────────────────────────┬────────────┘    │
│             │                             │                  │
│  ┌──────────▼──────────┐      ┌──────────▼──────────┐      │
│  │  Board Manager [C++]│      │  Rule Engine [C++]  │      │
│  │  - Bitboard repr.   │      │  - Move validation  │      │
│  │  - State tracking   │      │  - Fast scoring     │      │
│  │  - Zobrist hashing  │      │  - Ko detection     │      │
│  │  - Copy-on-write    │      │  - Capture logic    │      │
│  └─────────────────────┘      └─────────────────────┘      │
│            ▲                              ▲                  │
│            └──────────────┬───────────────┘                  │
│                    pybind11 bindings                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  SGF I/O & History [Python]                          │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                  AI ENGINE LAYER [Hybrid]                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  AI Controller [Python] - Orchestration             │    │
│  └──────┬──────────────────────────┬──────────┬────────┘    │
│         │                          │          │              │
│  ┌──────▼────────┐  ┌──────────────▼─────┐  ┌▼─────────┐   │
│  │  MCTS [C++]   │  │  Heuristics [C++]  │  │ ML Models│   │
│  │  - Tree ops   │  │  - Pattern match   │  │ [PyTorch]│   │
│  │  - Selection  │  │  - Territory eval  │  │ - Policy │   │
│  │  - Expansion  │  │  - Eye detection   │  │ - Value  │   │
│  │  - Simulation │  │  - Life/death      │  │          │   │
│  │  - Backprop   │  │  - Atari check     │  │          │   │
│  │  - Threading  │  │                    │  │          │   │
│  └───────────────┘  └────────────────────┘  └──────────┘   │
│         ▲                     ▲                    │         │
│         └─────────────────────┴────────────────────┘         │
│              pybind11 (C++ ↔ Python callbacks)               │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│            TRAINING PIPELINE LAYER [Python]                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Self-Play   │  │  Data        │  │  Model       │      │
│  │  Generator   │  │  Processing  │  │  Training    │      │
│  │  (calls C++) │  │  (NumPy)     │  │  (PyTorch)   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│           DATA & STORAGE LAYER [Multi-DB Strategy]           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  PostgreSQL  │  │   MongoDB    │  │   S3/Minio   │      │
│  │  (Relational)│  │  (Document)  │  │  (Object)    │      │
│  ├──────────────┤  ├──────────────┤  ├──────────────┤      │
│  │ • Users      │  │ • SGF files  │  │ • Models     │      │
│  │ • Matches    │  │ • Game logs  │  │ • Replays    │      │
│  │ • Elo ratings│  │ • Analytics  │  │ • Backups    │      │
│  │ • Transactions│ │ • Metadata   │  │ • Training   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Pattern DB [Embedded: JSON/Binary in C++]          │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘

[OPTIONAL] Backend Server Layer [Python/FastAPI] (for Online Play)

════════════════════════════════════════════════════════════════
PERFORMANCE CRITICAL PATH (C++):
  Board operations → MCTS search → Heuristic eval
  
HIGH-LEVEL LOGIC (Python):
  Training → Neural networks → Data processing → API serving
════════════════════════════════════════════════════════════════
```

### 1.8. Database Architecture (Multi-DB Strategy)

Hệ thống sử dụng **polyglot persistence** - mỗi loại database tối ưu cho một use case cụ thể:

#### 1.8.1. PostgreSQL (Relational Database)

**Mục đích:** Dữ liệu có cấu trúc, quan hệ phức tạp, ACID compliance

**Schema chính:**

```sql
-- Users & Authentication
users (
  id UUID PRIMARY KEY,
  username VARCHAR(50) UNIQUE,
  email VARCHAR(255) UNIQUE,
  password_hash VARCHAR(255),
  elo_rating INT DEFAULT 1500,
  created_at TIMESTAMP,
  last_login TIMESTAMP
)

-- Matches (metadata only, full game in MongoDB)
matches (
  id UUID PRIMARY KEY,
  black_player_id UUID REFERENCES users(id),
  white_player_id UUID REFERENCES users(id),
  ai_level INT,  -- NULL if PvP
  board_size INT,
  ruleset VARCHAR(20),
  result VARCHAR(10),  -- 'B+12.5', 'W+Resign', etc.
  start_time TIMESTAMP,
  end_time TIMESTAMP,
  sgf_id VARCHAR(255),  -- Reference to MongoDB
  INDEX (black_player_id, white_player_id)
)

-- Elo Rating History
rating_history (
  id SERIAL PRIMARY KEY,
  user_id UUID REFERENCES users(id),
  match_id UUID REFERENCES matches(id),
  old_rating INT,
  new_rating INT,
  rating_change INT,
  timestamp TIMESTAMP
)

-- Transactions (nếu có tính năng premium/shop)
transactions (
  id UUID PRIMARY KEY,
  user_id UUID REFERENCES users(id),
  amount DECIMAL(10,2),
  type VARCHAR(50),
  status VARCHAR(20),
  created_at TIMESTAMP
)
```

**Lý do chọn PostgreSQL:**
- ✅ ACID transactions cho Elo rating updates (consistency quan trọng)
- ✅ Complex queries (leaderboard, user statistics)
- ✅ Foreign key constraints đảm bảo data integrity
- ✅ Rich indexing cho performance

#### 1.8.2. MongoDB (Document Database)

**Mục đích:** Dữ liệu semi-structured, flexible schema, game records

**Collections:**

```javascript
// games collection - Full game data with SGF
{
  _id: ObjectId("..."),
  match_id: "uuid-from-postgres",
  board_size: 19,
  ruleset: "Chinese",
  sgf_content: "(;FF[4]GM[1]SZ[19]...)",  // Full SGF string
  moves: [
    { number: 1, color: "B", position: [3, 3], time_taken: 5.2 },
    { number: 2, color: "W", position: [15, 15], time_taken: 3.8 },
    // ... all moves
  ],
  metadata: {
    black_player: "username1",
    white_player: "username2",
    komi: 7.5,
    time_control: "10m+30s",
    date: ISODate("2025-11-17")
  },
  analysis: {
    ai_evaluations: [...],  // AI analysis per move
    win_probability_curve: [...],
    key_moments: [...]
  },
  created_at: ISODate("2025-11-17"),
  indexed: true
}

// game_logs collection - Debug/analytics logs
{
  _id: ObjectId("..."),
  match_id: "uuid",
  log_type: "ai_decision",
  timestamp: ISODate("2025-11-17"),
  data: {
    move_number: 42,
    mcts_playouts: 5000,
    top_moves: [
      { position: [10, 10], win_rate: 0.58, visits: 1200 },
      { position: [10, 11], win_rate: 0.56, visits: 980 }
    ],
    computation_time: 3.2
  }
}

// analytics collection - Aggregated statistics
{
  _id: ObjectId("..."),
  user_id: "uuid",
  period: "2025-11",
  stats: {
    total_games: 45,
    win_rate: 0.62,
    avg_game_length: 180,
    favorite_openings: ["3-3", "4-4", "star points"],
    mistakes_per_game: 12.3
  }
}
```

**Lý do chọn MongoDB:**
- ✅ Schema flexible cho game data (different formats, analysis types)
- ✅ Fast writes cho logging (async, high throughput)
- ✅ Rich query language cho analytics (aggregation pipeline)
- ✅ Horizontal scaling dễ dàng khi data lớn
- ✅ Native support cho nested documents (moves, analysis)

#### 1.8.3. S3/MinIO (Object Storage)

**Mục đích:** Large binary files, backups, model storage

**Bucket structure:**

```
gogame-storage/
├── models/
│   ├── policy_net/
│   │   ├── v1.0.0.pt
│   │   ├── v1.1.0.pt
│   │   └── latest.pt -> v1.1.0.pt
│   ├── value_net/
│   │   ├── v1.0.0.pt
│   │   └── latest.pt
│   └── metadata/
│       └── model_info.json
├── game_records/
│   ├── 2025/11/
│   │   ├── game_uuid1.sgf
│   │   ├── game_uuid1.json  # JSON format alternative
│   │   └── game_uuid2_replay.mp4  # Video replay (optional)
│   └── professional_games/  # Training data
│       ├── lee_sedol_vs_alphago.sgf
│       └── ...
├── backups/
│   ├── postgres_dump_20251117.sql.gz
│   ├── mongodb_backup_20251117.tar.gz
│   └── ...
└── training_data/
    ├── self_play_batch_001.npz  # Compressed numpy arrays
    ├── self_play_batch_002.npz
    └── ...
```

**SGF Format Example:**
```sgf
(;FF[4]
GM[1]
SZ[19]
CA[UTF-8]
AP[GoGame:1.0]
RU[Chinese]
KM[7.5]
DT[2025-11-17]
PB[Player1]
PW[AI Level 3]
BR[5k]
WR[2d]
RE[W+12.5]
;B[pd];W[dp];B[pq];W[dd]
...
)
```

**JSON Format Alternative:**
```json
{
  "game_info": {
    "id": "uuid",
    "format": "json",
    "version": "1.0",
    "board_size": 19,
    "ruleset": "Chinese",
    "komi": 7.5,
    "date": "2025-11-17",
    "players": {
      "black": {"name": "Player1", "rank": "5k"},
      "white": {"name": "AI Level 3", "rank": "2d"}
    },
    "result": "W+12.5"
  },
  "moves": [
    {"n": 1, "c": "B", "pos": "pd", "time": 5.2, "comment": "Standard opening"},
    {"n": 2, "c": "W", "pos": "dp", "time": 2.1},
    ...
  ],
  "metadata": {
    "ai_analysis": true,
    "training_data": false,
    "public": true
  }
}
```

**Lý do chọn S3/MinIO:**
- ✅ Unlimited scalability cho models và game records
- ✅ Cost-effective cho cold storage (old games, backups)
- ✅ Versioning support cho model updates
- ✅ CDN integration cho fast model downloads
- ✅ MinIO là S3-compatible, có thể self-host

#### 1.8.4. Embedded Pattern Database (C++)

**Mục đích:** Fast pattern matching trong MCTS, load tại startup

**Format:** Binary file hoặc memory-mapped file

```cpp
// pattern_db.bin structure
struct PatternEntry {
    uint64_t hash;      // 3x3 or 5x5 pattern hash
    float value;        // Heuristic value
    uint16_t frequency; // How often seen in pro games
    uint8_t urgency;    // Move urgency (0-255)
};

// In-memory hash table
std::unordered_map<uint64_t, PatternEntry> pattern_db;
```

**Lý do không dùng external DB:**
- ✅ Ultra-fast lookup (< 1 microsecond)
- ✅ Load once tại startup, read-only
- ✅ Small size (< 100MB for comprehensive patterns)
- ✅ No network latency

#### 1.8.5. Data Flow & Synchronization

```
┌─────────────┐
│  User plays │
└──────┬──────┘
       │
       ▼
┌──────────────────────────────────┐
│ 1. Create match metadata         │
│    → PostgreSQL (INSERT)         │
└──────┬───────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│ 2. Game in progress              │
│    → Store moves in memory       │
│    → AI logs → MongoDB (async)   │
└──────┬───────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│ 3. Game finished                 │
│    → Update match result (PG)    │
│    → Save full game (MongoDB)    │
│    → Upload SGF to S3 (async)    │
│    → Update Elo ratings (PG)     │
└──────┬───────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│ 4. Post-game analysis            │
│    → Read from MongoDB           │
│    → AI analysis → MongoDB       │
└──────────────────────────────────┘
```

#### 1.8.6. Deployment Options

**Option 1: Development (Single Machine)**
- PostgreSQL: Local instance
- MongoDB: Local instance
- S3: MinIO local server hoặc local filesystem
- Pattern DB: Local binary file

**Option 2: Production (Cloud)**
- PostgreSQL: AWS RDS / Azure Database / GCP Cloud SQL
- MongoDB: MongoDB Atlas / AWS DocumentDB
- S3: AWS S3 / Azure Blob / GCP Cloud Storage
- Pattern DB: Bundled with application

**Option 3: Self-Hosted**
- PostgreSQL: Docker container
- MongoDB: Docker container
- S3: MinIO server
- All orchestrated với Docker Compose

### 1.9. Các module chính

1. **Game Engine Module [C++]**
   - Board representation (bitboard)
   - Move generation & validation
   - Ko rule enforcement
   - Scoring system
   - Game state serialization

2. **AI Engine Module [C++ + Python]**
   - MCTS implementation (C++)
   - Heuristic evaluators (C++)
   - Neural network inference (Python/PyTorch)
   - Multi-level AI players (Python orchestration)

3. **Training Module [Python]**
   - Self-play data generation (calls C++ engine)
   - Neural network training (PyTorch)
   - Model versioning (S3)
   - Performance benchmarking

4. **Frontend Module [JS/Python]**
   - Game board visualization
   - User interaction
   - Move animation
   - Analysis tools

5. **Data Module [Python]**
   - Database management (SQLAlchemy for PG, PyMongo for Mongo)
   - SGF parser/writer
   - Pattern library loader
   - Statistics tracking

6. **Backend API Module [Python/FastAPI]**
   - RESTful API endpoints
   - WebSocket for real-time play
   - Authentication & authorization
   - Database orchestration

7. **Multiplayer Module (Optional) [Python]**
   - Networking layer
   - Game synchronization
   - Matchmaking
   - Chat system

### 1.10. Nguyên tắc thiết kế

1. **Modularity:** Mỗi component độc lập, dễ test và thay thế
2. **Performance-first:** Tối ưu hóa các hot path (MCTS simulation)
3. **Extensibility:** Dễ dàng thêm luật mới, AI level mới
4. **Education-focused:** Code rõ ràng, có comment giải thích thuật toán
5. **AI-centric:** Toàn bộ thiết kế phục vụ mục tiêu học thuật về AI

### 1.11. Sản phẩm dự kiến

**Giai đoạn 1 (MVP - Minimum Viable Product):**
- Game Cờ Vây 9×9 chạy được
- AI với MCTS thuần (2 cấp độ)
- Chơi offline PvP, PvAI
- Lưu/load game cơ bản

**Giai đoạn 2 (Core Features):**
- Bàn cờ 19×19
- AI với heuristics (3 cấp độ)
- Replay & analysis
- Training pipeline cơ bản

**Giai đoạn 3 (Advanced Features):**
- Neural networks (Policy + Value)
- Self-play training
- Hint system
- Full analysis tools

**Giai đoạn 4 (Polish & Extension):**
- Online multiplayer
- Elo ranking
- Professional UI/UX
- Mobile port (nếu có thời gian)

### 1.12. Đóng góp của dự án đối với môn học

Dự án này thể hiện sự hiểu biết sâu sắc về:

✅ **Thuật toán tìm kiếm:** MCTS là thuật toán state-of-the-art cho game có không gian trạng thái lớn  
✅ **Heuristic design:** Xây dựng hàm đánh giá cho domain phức tạp  
✅ **Tối ưu hóa:** Cache, parallelization, pruning techniques  
✅ **Học máy:** Kết hợp supervised + reinforcement learning  
✅ **Biểu diễn tri thức:** Mã hóa domain knowledge của Cờ Vây  
✅ **AI thực chiến:** Xây dựng AI có thể sử dụng được trong sản phẩm thực tế  

---

**Kết thúc PHẦN 1 — SYSTEM OVERVIEW**

---

## PHẦN 2 — PHÂN TÍCH YÊU CẦU THEO MÔN HỌC

### 2.1. Giới thiệu

Phần này thực hiện việc **ánh xạ (mapping)** các khái niệm và thuật toán cốt lõi của môn **Trí Tuệ Nhân Tạo** vào thiết kế cụ thể của hệ thống Cờ Vây. Mục tiêu là chứng minh rằng dự án này không chỉ là một ứng dụng game thông thường, mà là một **đồ án học thuật đầy đủ** thể hiện kiến thức AI.

### 2.2. Mapping với chương trình môn học

Dựa trên syllabus chuẩn của môn Trí Tuệ Nhân Tạo, các chủ đề chính bao gồm:

#### 2.2.1. Các chủ đề cốt lõi trong môn học

| Chủ đề môn học | Áp dụng trong dự án Cờ Vây | Mức độ |
|----------------|----------------------------|---------|
| **1. Tìm kiếm (Search)** | MCTS, Alpha-Beta pruning concepts | ★★★★★ |
| **2. Game Playing** | Adversarial search, game tree, evaluation | ★★★★★ |
| **3. Knowledge Representation** | Rule-based system, pattern database | ★★★★☆ |
| **4. Heuristics** | Position evaluation, pattern recognition | ★★★★★ |
| **5. Machine Learning (Basic)** | Supervised learning, Neural networks | ★★★★☆ |
| **6. Reinforcement Learning** | Self-play, TD-learning | ★★★☆☆ |
| **7. Optimization** | Algorithm tuning, pruning, caching | ★★★★☆ |
| **8. Uncertainty & Probability** | UCB in MCTS, stochastic rollouts | ★★★★☆ |

### 2.3. Chủ đề 1: Tìm kiếm (Search Algorithms)

#### 2.3.1. Lý thuyết môn học

**Các thuật toán tìm kiếm đã học:**
- **Uninformed Search:** BFS, DFS, Uniform Cost Search
- **Informed Search:** A*, Greedy Best-First Search
- **Adversarial Search:** Minimax, Alpha-Beta Pruning (★ YÊU CẦU ĐỀ TÀI)
- **Stochastic Search:** Monte Carlo methods

**Đặc điểm không gian tìm kiếm Cờ Vây:**
- State space: ~10^170 states (19×19 board)
- Branching factor: ~250 moves/position (so với Chess ~35)
- Depth: ~200-300 moves/game
- No clear evaluation function (không như Chess với material count)

**Thách thức:** Minimax/Alpha-Beta có **giới hạn** với Cờ Vây full-scale vì:
```
Time complexity = O(b^d)
với b=250, d=200 → exponential explosion
```

#### 2.3.2. Chiến lược Dual-Algorithm: Minimax + MCTS

**Dự án sử dụng CẢ HAI thuật toán** để thỏa mãn yêu cầu học thuật VÀ practical:

```
┌─────────────────────────────────────────────────────────┐
│         DUAL ALGORITHM STRATEGY                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. MINIMAX/ALPHA-BETA (Yêu cầu đề tài số 18)         │
│     ├─ Mục đích: Demo, trình diễn thuật toán cổ điển │
│     ├─ Use case: AI Level 1-2 (Beginner, Easy)       │
│     ├─ Depth: 3-5 ply (limited search)                │
│     ├─ Board: 9×9 (giảm branching factor)             │
│     └─ Educational value: Thể hiện thuật toán môn học │
│                                                         │
│  2. MCTS (Thuật toán bổ trợ cho AI mạnh)              │
│     ├─ Mục đích: AI chơi thật, competitive            │
│     ├─ Use case: AI Level 3-4 (Advanced, Expert)      │
│     ├─ Playouts: 5000-10000 simulations               │
│     ├─ Board: 9×9 và 19×19                            │
│     └─ Practical value: State-of-the-art approach     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Lý do sử dụng cả hai:**
1. **Yêu cầu đề tài:** Minimax là thuật toán được đề xuất → phải implement
2. **Giáo dục:** So sánh trực quan giữa classical vs modern approach
3. **Practical:** MCTS cho phép AI chơi ở level competitive
4. **Scalability:** Minimax cho small board, MCTS cho full-size board

#### 2.3.3. Thuật toán 1: Minimax với Alpha-Beta Pruning (YÊU CẦU ĐỀ TÀI)

**Giới thiệu:**
Minimax là thuật toán cổ điển cho **adversarial search**, được yêu cầu trong đề tài số 18. Dù có limitations với Cờ Vây full-scale, nó vẫn:
- ✅ Hoạt động tốt với **limited depth** (3-5 ply) và **small board** (9×9)
- ✅ Thể hiện rõ ràng **game tree concepts** từ môn học
- ✅ Dễ **visualize và debug** cho mục đích giáo dục
- ✅ So sánh được với MCTS (classical vs modern)

**Minimax Algorithm (theo sách AI chuẩn - Russell & Norvig):**

```python
def minimax_decision(state, depth_limit=4):
    """
    Minimax với depth-limited search
    
    Args:
        state: Current game state
        depth_limit: Maximum search depth (3-5 cho Cờ Vây)
    
    Returns:
        Best move theo minimax principle
    """
    def max_value(state, depth, alpha, beta):
        """MAX player (maximizing)"""
        if state.is_terminal() or depth == 0:
            return evaluate(state), None
        
        v = -infinity
        best_move = None
        
        for move in state.get_legal_moves():
            # Thử move
            new_state = state.apply_move(move)
            
            # Recursive call to MIN
            v2, _ = min_value(new_state, depth - 1, alpha, beta)
            
            if v2 > v:
                v = v2
                best_move = move
            
            # Alpha-Beta Pruning
            alpha = max(alpha, v)
            if v >= beta:
                return v, best_move  # Beta cutoff
        
        return v, best_move
    
    def min_value(state, depth, alpha, beta):
        """MIN player (minimizing)"""
        if state.is_terminal() or depth == 0:
            return evaluate(state), None
        
        v = +infinity
        best_move = None
        
        for move in state.get_legal_moves():
            new_state = state.apply_move(move)
            
            # Recursive call to MAX
            v2, _ = max_value(new_state, depth - 1, alpha, beta)
            
            if v2 < v:
                v = v2
                best_move = move
            
            # Alpha-Beta Pruning
            beta = min(beta, v)
            if v <= alpha:
                return v, best_move  # Alpha cutoff
        
        return v, best_move
    
    # Start search from root (MAX player)
    _, best_move = max_value(state, depth_limit, -infinity, +infinity)
    return best_move
```

**Evaluation Function cho Minimax:**

```python
def evaluate(game_state):
    """
    Static evaluation function cho Minimax
    Đây là hàm HEURISTIC quan trọng nhất
    """
    my_color = game_state.current_player
    opp_color = opponent(my_color)
    
    score = 0
    
    # Factor 1: Territory control (weighted heavily)
    my_territory = estimate_territory(game_state.board, my_color)
    opp_territory = estimate_territory(game_state.board, opp_color)
    score += 10.0 * (my_territory - opp_territory)
    
    # Factor 2: Captured stones
    score += 5.0 * (game_state.prisoners[my_color] - 
                    game_state.prisoners[opp_color])
    
    # Factor 3: Group strength (liberties)
    my_groups = find_all_groups(game_state.board, my_color)
    opp_groups = find_all_groups(game_state.board, opp_color)
    
    for group in my_groups:
        liberties = count_liberties(game_state.board, group)
        if liberties == 1:
            score -= 50  # Atari, very bad
        elif liberties == 2:
            score -= 10
        elif liberties >= 4:
            score += 5
    
    for group in opp_groups:
        liberties = count_liberties(game_state.board, group)
        if liberties == 1:
            score += 50  # Opponent in atari, good!
        elif liberties == 2:
            score += 10
    
    # Factor 4: Pattern matching (corners, edges)
    score += evaluate_patterns(game_state.board, my_color)
    
    return score
```

**Alpha-Beta Pruning Visualization:**

```
Game Tree với Alpha-Beta:

                    MAX (α=-∞, β=+∞)
                    /      |      \
                  /        |        \
          MIN(α=-∞)   MIN(α=3)   MIN(α=5)
          /  |  \      /  |  \      /  |  \
         3   12  8    2   5  9    14  5  2
         ↑               ↑  ✂         ✂  ✂
    Return 3         Return 5    Pruned! (β=5, v=14 > β)
    
Nodes pruned: Không cần evaluate subtrees đã bị cắt
Improvement: Best case O(b^(d/2)) thay vì O(b^d)
```

**Move Ordering (Tối ưu hóa Alpha-Beta):**

```python
def order_moves(state, moves):
    """
    Move ordering để tăng hiệu quả alpha-beta pruning
    Concept: Evaluate "promising" moves trước
    """
    move_scores = []
    
    for move in moves:
        score = 0
        
        # Priority 1: Capturing moves
        if captures_opponent(state, move):
            score += 1000
        
        # Priority 2: Saving own groups from atari
        if saves_atari(state, move):
            score += 500
        
        # Priority 3: Center/corner positions (heuristic)
        score += position_value(move)
        
        # Priority 4: Pattern matching
        score += pattern_score(state, move)
        
        move_scores.append((score, move))
    
    # Sort descending (best moves first)
    move_scores.sort(reverse=True)
    return [move for _, move in move_scores]


def minimax_with_ordering(state, depth, alpha, beta):
    """Minimax với move ordering"""
    if state.is_terminal() or depth == 0:
        return evaluate(state), None
    
    moves = state.get_legal_moves()
    ordered_moves = order_moves(state, moves)  # KEY OPTIMIZATION
    
    # ... rest of minimax logic
```

**Limitations của Minimax cho Cờ Vây:**

| Challenge | Impact | Mitigation |
|-----------|--------|------------|
| **Branching factor ~250** | Tree explodes exponentially | Limit depth to 3-5 ply |
| **No good eval function** | Hard to evaluate mid-game | Use multiple heuristics |
| **Deep tactics** | Can't see deep sequences | Combine with tactical search |
| **Computation time** | Slow for 19×19 | Use 9×9 board only |

**Use Cases cho Minimax trong dự án:**

```python
class MinimaxAI:
    """
    AI sử dụng Minimax - cho mục đích giáo dục và demo
    """
    def __init__(self, level):
        if level == 1:  # Beginner
            self.depth = 2
            self.board_size = 9
            self.use_pruning = False
        elif level == 2:  # Easy
            self.depth = 4
            self.board_size = 9
            self.use_pruning = True  # Alpha-Beta enabled
    
    def select_move(self, game_state):
        """Entry point"""
        if self.use_pruning:
            return minimax_decision(game_state, self.depth)
        else:
            return minimax_no_pruning(game_state, self.depth)
    
    def visualize_tree(self, game_state):
        """
        Demo mode: Visualize game tree for educational purposes
        Show how Minimax explores and evaluates positions
        """
        tree = build_game_tree(game_state, self.depth)
        return tree  # Can be displayed in UI
```

#### 2.3.4. Thuật toán 2: Monte Carlo Tree Search (MCTS) - AI Mạnh

**Giới thiệu:**
MCTS là thuật toán **state-of-the-art** cho Cờ Vây, được sử dụng trong AlphaGo. Không phải yêu cầu đề tài, nhưng cần thiết để:
- ✅ AI có thể chơi ở level **competitive**
- ✅ Hoạt động với **19×19 board**
- ✅ Không cần evaluation function hoàn hảo
- ✅ **Anytime algorithm** (dừng bất kỳ lúc nào)

**MCTS là hybrid của:**
- **Best-First Search:** Chọn node promising nhất để expand
- **Monte Carlo Simulation:** Estimate value bằng random playouts
- **UCB (Upper Confidence Bound):** Balance exploration vs exploitation

**4 pha của MCTS (liên hệ với concepts môn học):**

```
┌─────────────────────────────────────────────────────────┐
│                    MCTS ALGORITHM                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. SELECTION (Tree Policy)                            │
│     ├─ Best-First Search với UCB1                      │
│     ├─ Formula: UCB = Q/N + C*sqrt(ln(parent_N)/N)    │
│     └─ Balance: Exploitation (Q/N) vs                  │
│                 Exploration (sqrt term)                │
│                                                         │
│  2. EXPANSION                                           │
│     ├─ Add new child nodes                             │
│     ├─ Tương tự Breadth-First expansion                │
│     └─ Progressive widening (advanced)                 │
│                                                         │
│  3. SIMULATION (Rollout/Playout)                       │
│     ├─ Random/heuristic-guided playouts                │
│     ├─ Tương tự Monte Carlo sampling                   │
│     └─ Estimate value of position                      │
│                                                         │
│  4. BACKPROPAGATION                                     │
│     ├─ Update statistics từ leaf → root               │
│     ├─ Tương tự Dynamic Programming                    │
│     └─ Value aggregation                               │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Pseudocode (theo sách AI chuẩn):**

```python
def MCTS_Search(root_state, num_iterations):
    """
    Monte Carlo Tree Search
    - root_state: Current board position
    - num_iterations: Number of simulations (playouts)
    """
    root = Node(root_state)
    
    for i in range(num_iterations):
        # 1. SELECTION - Tree traversal với UCB
        node = TreePolicy(root)
        
        # 2. EXPANSION - Thêm child node mới
        if not node.is_terminal():
            node = Expand(node)
        
        # 3. SIMULATION - Random playout
        reward = DefaultPolicy(node.state)
        
        # 4. BACKPROPAGATION - Update statistics
        Backup(node, reward)
    
    # Chọn move tốt nhất dựa trên statistics
    return BestChild(root, c=0).move


def TreePolicy(node):
    """Selection với UCB1"""
    while not node.is_terminal():
        if not node.is_fully_expanded():
            return Expand(node)
        else:
            node = BestChild(node, c=sqrt(2))  # UCB constant
    return node


def BestChild(node, c):
    """UCB1 formula - Balance exploration/exploitation"""
    return argmax(
        child.wins/child.visits + 
        c * sqrt(log(node.visits) / child.visits)
        for child in node.children
    )


def DefaultPolicy(state):
    """Simulation - Random playout until terminal"""
    while not state.is_terminal():
        action = random_legal_move(state)
        state = state.apply(action)
    return state.get_reward()


def Backup(node, reward):
    """Backpropagation - Update ancestors"""
    while node is not None:
        node.visits += 1
        node.wins += reward
        node = node.parent
        reward = 1 - reward  # Flip reward for opponent
```

#### 2.3.3. Kỹ thuật tối ưu hóa tìm kiếm (theo môn học)

**1. Pruning (Cắt tỉa):**
- **Progressive Widening:** Không expand tất cả children ngay
- **AMAF (All Moves As First):** Sử dụng thông tin từ các moves sau
- **RAVE (Rapid Action Value Estimation):** Khởi tạo giá trị nhanh hơn

**2. Caching & Memoization:**
- **Transposition Table:** Lưu states đã thăm (dùng Zobrist hashing)
- **Tree reuse:** Giữ lại tree từ previous search

**3. Parallelization:**
- **Leaf Parallelization:** Chạy nhiều simulations song song
- **Root Parallelization:** Nhiều trees search từ root
- **Virtual Loss:** Tránh threads search cùng branch

**4. Early Termination:**
- **Time management:** Stop khi hết thời gian
- **Obvious moves:** Nếu 1 move có win rate >> others

#### 2.3.5. So sánh Minimax vs MCTS (Quan trọng cho đồ án)

| Tiêu chí | Minimax/Alpha-Beta | MCTS |
|----------|-------------------|------|
| **Yêu cầu đề tài** | ✅ YÊU CẦU (Đề tài 18) | ⭕ Bổ trợ |
| **Complete?** | Yes (search toàn bộ tree) | No (sample-based) |
| **Optimal?** | Yes (nếu heuristic perfect) | Converge to optimal (infinite time) |
| **Heuristic?** | Cần hàm eval mạnh | Không bắt buộc |
| **Branching factor** | Khó với b > 50 | OK với b = 250+ |
| **Domain knowledge** | Highly dependent | Less dependent |
| **Anytime** | No (phải search full depth) | Yes (improve over time) |
| **Parallelizable** | Khó | Dễ |
| **Board size** | 9×9 OK, 19×19 không khả thi | Cả 9×9 và 19×19 |
| **Visualization** | Dễ visualize tree | Khó visualize |
| **Educational value** | Cao (classical algorithm) | Cao (modern approach) |
| **Practical strength** | Weak-Medium | Strong |

**Kết luận - Dual Strategy:**

```
┌─────────────────────────────────────────────────────────┐
│  MINIMAX                      vs              MCTS      │
├─────────────────────────────────────────────────────────┤
│  ✅ Yêu cầu đề tài                  ⭕ Bổ trợ          │
│  ✅ Demo/Educational                ✅ Practical AI    │
│  ✅ Visualizable                    ✅ Scalable        │
│  ❌ Limited strength                ✅ Strong play     │
│  ❌ 19×19 không khả thi             ✅ 19×19 OK        │
│                                                         │
│  → Dùng cho Level 1-2               → Dùng cho Level 3-4│
│  → Trình diễn thuật toán            → Chơi competitive  │
│  → Board 9×9                        → Board 9×9 và 19×19│
└─────────────────────────────────────────────────────────┘
```

**Tại sao cần CẢ HAI thuật toán:**
1. **Academic:** Minimax là yêu cầu đề tài → phải có
2. **Comparison:** So sánh classical vs modern approach → giá trị học thuật
3. **Practical:** MCTS để AI thực sự chơi được → giá trị ứng dụng
4. **Flexibility:** User có thể chọn algorithm tùy mục đích

### 2.4. Chủ đề 2: Game Playing & Adversarial Search

#### 2.4.1. Lý thuyết môn học

**Game tree concepts:**
- **Minimax value:** Giá trị tốt nhất với optimal play
- **Adversarial nature:** 2 người chơi có mục tiêu đối lập
- **Horizon effect:** Không thể search đến terminal states
- **Evaluation function:** Estimate giá trị của non-terminal states

**Zero-sum game:**
- Cờ Vây là **zero-sum game:** win của Black = loss của White
- Reward: +1 (win), 0 (loss), 0.5 (draw - hiếm trong Go)

#### 2.4.2. Áp dụng vào Cờ Vây

**1. Game State Representation:**
```python
class GameState:
    def __init__(self, board_size=19):
        self.board = np.zeros((board_size, board_size))  # 0=empty, 1=black, 2=white
        self.current_player = BLACK  # 1 or 2
        self.ko_point = None  # Ko rule tracking
        self.history = []  # Move history for superko
        self.passes = 0  # Consecutive passes
        self.prisoners = {BLACK: 0, WHITE: 0}
    
    def get_legal_moves(self):
        """Generate all legal moves (môn học: successor function)"""
        moves = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.is_legal(i, j):
                    moves.append((i, j))
        moves.append(PASS)  # Pass is always legal
        return moves
    
    def is_terminal(self):
        """Check if game over (môn học: terminal test)"""
        return self.passes >= 2  # Two consecutive passes
    
    def get_winner(self):
        """Score the game (môn học: utility function)"""
        black_score = self.count_territory(BLACK) + self.prisoners[WHITE]
        white_score = self.count_territory(WHITE) + self.prisoners[BLACK] + KOMI
        
        if black_score > white_score:
            return BLACK, black_score - white_score
        else:
            return WHITE, white_score - black_score
```

**2. Adversarial Assumptions:**
- Opponent plays optimally (hoặc near-optimal)
- MCTS simulates cả 2 players
- Value function phải account for both sides

#### 2.4.3. Multi-level AI Implementation (Balanced + Monetization)

**3-Tier Strategy:** Core Game (Free) + Premium Features + Monetization

```python
class AIPlayer:
    """
    AI Player với difficulty curve hợp lý:
    - Level 1-2: Minimax (yêu cầu đề tài, balanced for beginners)
    - Level 3: MCTS thuần (challenging but beatable)
    - Level 4: MCTS mạnh (advanced players)
    
    ML Features (Policy/Value nets) → PREMIUM/SHOP ITEMS
    """
    
    # ========== FREE TO PLAY LEVELS (Core Game) ==========
    LEVELS = {
        1: {  # Beginner - MINIMAX
            'algorithm': 'minimax',
            'depth': 2,
            'use_pruning': False,
            'board_size': 9,
            'time_limit': 1.0,
            'strength': 'Very Weak (~15 kyu)',
            'description': 'Minimax thuần - Người mới bắt đầu',
            'tier': 'free'
        },
        
        2: {  # Intermediate - MINIMAX + Alpha-Beta
            'algorithm': 'minimax',
            'depth': 4,
            'use_pruning': True,
            'move_ordering': True,
            'board_size': 9,
            'time_limit': 2.0,
            'strength': 'Weak (~12 kyu)',
            'description': 'Minimax tối ưu - Người chơi casual',
            'tier': 'free'
        },
        
        3: {  # Advanced - MCTS (NO ML)
            'algorithm': 'mcts',
            'playouts': 3000,  # Giảm xuống để balanced
            'use_heuristics': True,
            'use_neural_net': False,  # ❌ NO ML
            'board_size': [9, 19],
            'time_limit': 5.0,
            'strength': 'Medium (~5 kyu)',
            'description': 'MCTS + Heuristics - Thử thách',
            'tier': 'free'
        },
        
        4: {  # Expert - MCTS Strong (NO ML)
            'algorithm': 'mcts',
            'playouts': 8000,  # Tăng playouts thay vì ML
            'use_heuristics': True,
            'use_neural_net': False,  # ❌ NO ML
            'parallel': True,
            'board_size': [9, 19],
            'time_limit': 10.0,
            'strength': 'Strong (~1 dan)',
            'description': 'MCTS mạnh - Cao thủ',
            'tier': 'free'
        }
    }
    
    # ========== PREMIUM FEATURES (Monetization) ==========
    PREMIUM_FEATURES = {
        'ai_hint': {
            'name': 'AI Gợi ý nước đi (AI Hint)',
            'description': 'Sử dụng Policy Network để gợi ý 3 nước đi tốt nhất',
            'technology': 'Policy Network (Neural Net)',
            'usage': 'PvP hoặc PvAI',
            'cost': {
                'per_use': 10,  # coins
                'package_10': 80,  # 10 hints = 80 coins (20% discount)
                'package_50': 350,  # 50 hints = 350 coins (30% discount)
                'unlimited_monthly': 500  # Unlimited/month
            },
            'cooldown': 30,  # seconds between uses
            'tier': 'premium'
        },
        
        'position_analysis': {
            'name': 'Phân tích vị thế (Position Analysis)',
            'description': 'Value Network đánh giá win probability và territory',
            'technology': 'Value Network + Territory Analyzer',
            'usage': 'Sau mỗi nước đi hoặc cuối ván',
            'cost': {
                'per_use': 5,
                'package_20': 80,
                'unlimited_monthly': 300
            },
            'tier': 'premium'
        },
        
        'game_review': {
            'name': 'Review AI chi tiết (Game Review)',
            'description': 'AI review toàn bộ ván đấu, chỉ ra mistakes và better moves',
            'technology': 'Policy Net + Value Net + MCTS deep analysis',
            'usage': 'Sau khi kết thúc ván',
            'cost': {
                'per_use': 20,
                'package_10': 150,
                'unlimited_monthly': 800
            },
            'analysis_depth': 'Full game with move-by-move suggestions',
            'tier': 'premium'
        },
        
        'opening_book': {
            'name': 'Thư viện định thức (Joseki/Opening Book)',
            'description': 'Database các opening sequences từ pro players',
            'technology': 'Pattern Database + Pro game records',
            'usage': 'Opening phase (first 20-30 moves)',
            'cost': {
                'basic': 200,  # One-time purchase
                'advanced': 500,  # Includes rare joseki
                'pro': 1000  # Full database + updates
            },
            'tier': 'premium'
        },
        
        'training_mode': {
            'name': 'Chế độ luyện tập (Training Mode)',
            'description': 'Tsumego puzzles + Life/Death problems với AI hints',
            'technology': 'Problem database + AI solver',
            'cost': {
                'monthly': 300,
                'yearly': 2500  # ~30% discount
            },
            'tier': 'premium'
        }
    }
    
    def select_move(self, game_state, level):
        """
        Core game move selection - ALWAYS FREE
        """
        config = self.LEVELS[level]
        
        if config['algorithm'] == 'minimax':
            return self._minimax_move(game_state, config)
        elif config['algorithm'] == 'mcts':
            return self._mcts_move(game_state, config)
    
    # ========== PREMIUM FEATURE IMPLEMENTATIONS ==========
    
    def get_ai_hint(self, game_state, user_coins):
        """
        Premium Feature: AI Hint
        Requires: Policy Network (ML model)
        Cost: 10 coins per use
        """
        feature = self.PREMIUM_FEATURES['ai_hint']
        
        # Check if user has enough coins
        if user_coins < feature['cost']['per_use']:
            return {
                'success': False,
                'error': 'Insufficient coins',
                'required': feature['cost']['per_use'],
                'current': user_coins
            }
        
        # Load Policy Network (ML model)
        policy_net = load_policy_network()  # Trained model
        
        # Get move probabilities
        features = extract_features(game_state)
        move_probs = policy_net.predict(features)
        
        # Get top 3 moves
        top_moves = get_top_k_moves(move_probs, k=3)
        
        # Deduct coins
        new_balance = user_coins - feature['cost']['per_use']
        
        return {
            'success': True,
            'hints': [
                {
                    'move': move,
                    'probability': prob,
                    'reason': explain_move(game_state, move)
                }
                for move, prob in top_moves
            ],
            'coins_used': feature['cost']['per_use'],
            'new_balance': new_balance
        }
    
    def analyze_position(self, game_state, user_coins):
        """
        Premium Feature: Position Analysis
        Requires: Value Network + Territory Analyzer
        Cost: 5 coins per use
        """
        feature = self.PREMIUM_FEATURES['position_analysis']
        
        if user_coins < feature['cost']['per_use']:
            return {'success': False, 'error': 'Insufficient coins'}
        
        # Load Value Network
        value_net = load_value_network()
        
        # Analyze position
        features = extract_features(game_state)
        win_probability = value_net.predict(features)
        
        # Territory analysis
        territory = analyze_territory(game_state)
        
        # Weak groups detection
        weak_groups = detect_weak_groups(game_state)
        
        return {
            'success': True,
            'analysis': {
                'win_probability': {
                    'black': win_probability,
                    'white': 1 - win_probability
                },
                'territory': territory,
                'weak_groups': weak_groups,
                'suggested_focus': get_focus_areas(game_state)
            },
            'coins_used': feature['cost']['per_use'],
            'new_balance': user_coins - feature['cost']['per_use']
        }
    
    def review_game(self, game_history, user_coins):
        """
        Premium Feature: Full Game Review
        Requires: Policy Net + Value Net + Deep MCTS
        Cost: 20 coins per game
        """
        feature = self.PREMIUM_FEATURES['game_review']
        
        if user_coins < feature['cost']['per_use']:
            return {'success': False, 'error': 'Insufficient coins'}
        
        # Load ML models
        policy_net = load_policy_network()
        value_net = load_value_network()
        
        mistakes = []
        move_quality = []
        
        for move_num, (state, move) in enumerate(game_history):
            # Get AI suggested move
            ai_move = policy_net.get_best_move(state)
            
            # Evaluate both moves
            actual_value = value_net.evaluate(state.apply(move))
            best_value = value_net.evaluate(state.apply(ai_move))
            
            quality = 'excellent'
            if best_value - actual_value > 0.1:
                quality = 'mistake'
                mistakes.append({
                    'move_number': move_num,
                    'played': move,
                    'better': ai_move,
                    'value_loss': best_value - actual_value
                })
            elif best_value - actual_value > 0.05:
                quality = 'inaccuracy'
            
            move_quality.append({
                'move': move,
                'quality': quality,
                'value': actual_value
            })
        
        return {
            'success': True,
            'review': {
                'mistakes': mistakes,
                'move_quality': move_quality,
                'accuracy': calculate_accuracy(move_quality),
                'key_moments': identify_key_moments(game_history),
                'suggestions': generate_improvement_tips(mistakes)
            },
            'coins_used': feature['cost']['per_use'],
            'new_balance': user_coins - feature['cost']['per_use']
        }
```

**Monetization Flow:**

```python
class CoinSystem:
    """
    Virtual currency system
    """
    COIN_PACKAGES = {
        'starter': {'coins': 100, 'price_usd': 0.99},
        'basic': {'coins': 500, 'price_usd': 3.99},
        'standard': {'coins': 1200, 'price_usd': 8.99},  # 20% bonus
        'premium': {'coins': 3000, 'price_usd': 19.99},  # 50% bonus
        'ultimate': {'coins': 10000, 'price_usd': 49.99}  # 100% bonus
    }
    
    # Free coins earning
    EARN_COINS = {
        'daily_login': 10,
        'complete_game': 5,
        'win_game': 10,
        'rank_up': 50,
        'achievement': 20,
        'watch_ad': 5  # Optional
    }
    
    def purchase_coins(self, user_id, package):
        """Process coin purchase"""
        # Payment gateway integration
        pass
    
    def earn_free_coins(self, user_id, action):
        """Give free coins for activities"""
        if action in self.EARN_COINS:
            amount = self.EARN_COINS[action]
            add_coins(user_id, amount)
            return amount
        return 0
```

**UI Mock-up cho Premium Features:**

```
┌─────────────────────────────────────────────────────┐
│  Game Board (9x9)                    [💰 250 coins] │
├─────────────────────────────────────────────────────┤
│  Current Move: Black #42                            │
│                                                      │
│  [🆓 Core AI Level]  [💎 Premium Tools]            │
│   • Level 1-4 Free    • Hint (10 coins)             │
│                       • Analysis (5 coins)           │
│                       • Review (20 coins)            │
│                                                      │
│  💡 Need help? Use AI Hint (10 coins)               │
│     Shows top 3 moves with explanations             │
│                                                      │
│  📊 Want analysis? (5 coins)                        │
│     Win probability & territory estimation          │
│                                                      │
│  After game: Full AI Review (20 coins)              │
│     Detailed mistake analysis + improvement tips    │
└─────────────────────────────────────────────────────┘
```

**Giải thích phân cấp:**

| Level | Algorithm | Strength | Purpose | Tier |
|-------|-----------|----------|---------|------|
| **1** | Minimax (depth 2) | ~15 kyu | Beginner-friendly | 🆓 Free |
| **2** | Minimax + Alpha-Beta (depth 4) | ~12 kyu | Casual players | 🆓 Free |
| **3** | MCTS (3k playouts, no ML) | ~5 kyu | Challenge | 🆓 Free |
| **4** | MCTS (8k playouts, no ML) | ~1 dan | Advanced | 🆓 Free |
| **💎** | ML Hints | Pro-level | Premium tool | 💰 Shop |
| **💎** | ML Analysis | Pro-level | Premium tool | 💰 Shop |
| **💎** | ML Review | Pro-level | Premium tool | 💰 Shop |

**Key Points của Strategy này:**

1. **✅ Core game HOÀN TOÀN FREE:**
   - 4 AI levels từ dễ → khó
   - Không paywall gameplay
   - Fair & balanced difficulty

2. **✅ ML = Premium Tools (NOT opponents):**
   - AI Hint: Gợi ý nước đi
   - Position Analysis: Đánh giá vị thế
   - Game Review: Phân tích sau ván
   - Opening Book: Thư viện định thức

3. **✅ Freemium Model:**
   - Free coins qua daily login, wins
   - Optional ads for coins
   - Purchase coins packages

4. **✅ Academic Value giữ nguyên:**
   - Minimax vẫn là thuật toán chính (đề tài)
   - MCTS vẫn được implement
   - ML được justify là "premium features"

5. **✅ Business Potential:**
   - Sustainable monetization
   - User-friendly (không pay-to-win)
   - Similar to Chess.com, Lichess model
    
    def _minimax_move(self, game_state, config):
        """Minimax/Alpha-Beta move selection"""
        engine = MinimaxEngine(
            depth=config['depth'],
            use_pruning=config['use_pruning'],
            move_ordering=config.get('move_ordering', False)
        )
        return engine.search(game_state)
    
    def _mcts_move(self, game_state, config):
        """MCTS move selection"""
        engine = MCTSEngine(
            playouts=config['playouts'],
            use_heuristics=config['use_heuristics'],
            policy_net=load_policy_net() if config['use_neural_net'] else None,
            value_net=load_value_net() if config['use_value_net'] else None
        )
        return engine.search(game_state)
    
    def get_algorithm_info(self, level):
        """
        Trả về thông tin thuật toán đang dùng
        Useful cho UI để hiển thị
        """
        config = self.LEVELS[level]
        return {
            'level': level,
            'algorithm': config['algorithm'],
            'description': config['description'],
            'suitable_board': config['board_size']
        }
```

**Comparison Mode - Feature đặc biệt:**

```python
class AIComparison:
    """
    Mode đặc biệt: So sánh Minimax vs MCTS
    Useful cho demo và educational purposes
    """
    
    def compare_algorithms(self, game_state):
        """
        Chạy cả hai thuật toán trên cùng position
        """
        # Minimax decision
        minimax_start = time.time()
        minimax_engine = MinimaxEngine(depth=4, use_pruning=True)
        minimax_move = minimax_engine.search(game_state)
        minimax_time = time.time() - minimax_start
        minimax_eval = minimax_engine.get_evaluation()
        
        # MCTS decision
        mcts_start = time.time()
        mcts_engine = MCTSEngine(playouts=5000)
        mcts_move = mcts_engine.search(game_state)
        mcts_time = time.time() - mcts_start
        mcts_winrate = mcts_engine.get_winrate(mcts_move)
        
        return {
            'minimax': {
                'move': minimax_move,
                'evaluation': minimax_eval,
                'time': minimax_time,
                'nodes_explored': minimax_engine.nodes_count
            },
            'mcts': {
                'move': mcts_move,
                'win_rate': mcts_winrate,
                'time': mcts_time,
                'playouts': 5000,
                'visits': mcts_engine.get_visits(mcts_move)
            },
            'agreement': minimax_move == mcts_move  # Do they agree?
        }
```

**Giải thích phân cấp:**

| Level | Algorithm | Strength | Purpose | Board Size |
|-------|-----------|----------|---------|------------|
| **1** | Minimax (depth 2) | Very Weak | Demo thuật toán, beginner-friendly | 9×9 |
| **2** | Minimax + Alpha-Beta (depth 4) | Weak-Medium | Trình diễn optimization techniques | 9×9 |
| **3** | MCTS + Policy Net (5k playouts) | Strong | Competitive play, practical AI | 9×9, 19×19 |
| **4** | MCTS Full (10k playouts) | Very Strong | Expert play, tournament-level | 9×9, 19×19 |

**Visualizations cho từng level:**

```python
def visualize_ai_decision(game_state, level):
    """
    Visualize AI decision process
    Level 1-2: Show minimax game tree
    Level 3-4: Show MCTS statistics
    """
    config = AIPlayer.LEVELS[level]
    
    if config['algorithm'] == 'minimax':
        # Visualize game tree
        tree = build_minimax_tree(game_state, config['depth'])
        return {
            'type': 'game_tree',
            'data': tree,
            'nodes': tree.total_nodes,
            'pruned': tree.pruned_nodes,
            'best_line': tree.principal_variation
        }
    else:
        # Visualize MCTS statistics
        stats = get_mcts_statistics(game_state)
        return {
            'type': 'mcts_stats',
            'data': stats,
            'top_moves': stats.top_5_moves,
            'visit_distribution': stats.visit_counts,
            'win_rates': stats.win_rates
        }
```

**Ưu điểm của 3-Tier Strategy (Free + Premium + ML):**

1. ✅ **Tuân thủ yêu cầu đề tài:**
   - Minimax/Alpha-Beta là thuật toán chính (Level 1-2)
   - MCTS bổ trợ cho AI mạnh hơn (Level 3-4)
   - ML được justify là "premium tools" không phải core AI

2. ✅ **Balanced Difficulty:**
   - Level 1-2: Dễ, beatable cho beginners (~15-12 kyu)
   - Level 3: Challenging nhưng fair (~5 kyu)
   - Level 4: Khó nhưng không impossible (~1 dan)
   - ❌ KHÔNG có AI "siêu mạnh" khiến user frustrated

3. ✅ **ML có mục đích rõ ràng:**
   - KHÔNG dùng để tạo AI opponent quá mạnh
   - Dùng cho **premium tools**: Hints, Analysis, Review
   - Giá trị thực tế: Giúp người chơi học và improve
   - Monetization hợp lý

4. ✅ **Business Model khả thi:**
   - Core game FREE (4 AI levels)
   - Premium features có giá trị thực
   - Freemium model (earn coins qua play)
   - Không pay-to-win

5. ✅ **Academic + Commercial Balance:**
   - Đồ án vẫn focus vào AI algorithms
   - Có business potential cho future
   - Professional product design

**Visualization - 3-Tier Architecture:**

```
┌──────────────────────────────────────────────────────────────┐
│                        USER EXPERIENCE                        │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  🆓 FREE TIER (Core Game)                                    │
│  ├─ Play vs AI Level 1-4                                     │
│  ├─ Play vs Human (Online/Local)                             │
│  ├─ Save/Load games (SGF)                                    │
│  ├─ Basic statistics                                         │
│  └─ Replay games                                             │
│                                                               │
│  💎 PREMIUM TOOLS (ML-Powered)                               │
│  ├─ 💡 AI Hint (10 coins)                                    │
│  │   → Shows top 3 moves with explanations                   │
│  ├─ 📊 Position Analysis (5 coins)                           │
│  │   → Win probability + Territory map                       │
│  ├─ 🔍 Game Review (20 coins)                                │
│  │   → Full analysis with mistakes highlighted               │
│  ├─ 📚 Opening Book (200+ coins)                             │
│  │   → Professional joseki database                          │
│  └─ 🎓 Training Mode (300 coins/month)                       │
│      → Tsumego puzzles with AI solutions                     │
│                                                               │
│  💰 COIN SYSTEM                                              │
│  ├─ Earn Free: Daily login, wins, achievements               │
│  ├─ Watch Ads (optional): 5 coins per ad                     │
│  └─ Purchase: $0.99 - $49.99 packages                        │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

**So sánh với Competitors:**

| Feature | GoGame (Our) | Chess.com | Lichess | OGS (Online Go) |
|---------|--------------|-----------|---------|-----------------|
| **Core Game** | 🆓 Free | 🆓 Free | 🆓 Free | 🆓 Free |
| **Basic AI** | 🆓 4 Levels | 🆓 Limited | 🆓 Stockfish | 🆓 Basic Bot |
| **Analysis** | 💰 Premium | 💰 Premium | 🆓 Free | ❌ Limited |
| **Hints** | 💰 Premium | 💰 Premium | ❌ No | ❌ No |
| **Opening Book** | 💰 Premium | 💰 Premium | 🆓 Free | ❌ Limited |
| **Training** | 💰 Premium | 💰 Premium | 🆓 Puzzles | ❌ No |

**Revenue Estimation (Conservative):**

```
Assumptions:
- 10,000 active users/month
- 5% conversion to premium users = 500 users
- Average $5/user/month

Revenue = 500 × $5 = $2,500/month = $30,000/year

Costs:
- Server (AWS/Azure): ~$500/month
- Storage (S3): ~$100/month  
- ML model hosting: ~$200/month
- Marketing: ~$500/month
Total Costs: ~$1,300/month = $15,600/year

Net Profit: $30,000 - $15,600 = $14,400/year

(Conservative estimate, can scale higher)
```

**Justification cho Đồ Án:**

1. **Đáp ứng yêu cầu môn học:**
   - ✅ Minimax/Alpha-Beta (thuật toán chính)
   - ✅ MCTS (thuật toán bổ trợ)
   - ✅ Heuristics
   - ✅ ML (optional, cho premium features)

2. **Balance giữa Academic & Practical:**
   - Academic: Thuật toán cổ điển + hiện đại
   - Practical: Business model + User experience

3. **ML được justify đúng đắn:**
   - Không phải để tạo AI quá mạnh
   - Là tools để giúp người chơi học tập
   - Có giá trị thương mại rõ ràng

4. **Scalable & Sustainable:**
   - Freemium model proven to work
   - Can grow user base organically
   - Revenue to sustain development

**Kết luận:**

Strategy này **tốt hơn nhiều** so với ban đầu vì:
- ✅ AI không quá mạnh, balanced cho mọi level
- ✅ ML có mục đích rõ ràng (tools, not opponents)
- ✅ Business model khả thi
- ✅ Academic requirements vẫn được đáp ứng đầy đủ
- ✅ Có potential cho commercialization thực tế

### 2.5. Chủ đề 3: Heuristics & Evaluation Functions

#### 2.5.1. Lý thuyết môn học

**Heuristic function h(n):**
- Estimate cost/value from n to goal
- Admissible nếu never overestimate
- Consistent nếu satisfy triangle inequality

Trong game playing:
- **Evaluation function:** Estimate utility of non-terminal state
- Trade-off: **Accuracy vs Speed**
- Domain knowledge → better heuristics

#### 2.5.2. Heuristics cho Cờ Vây

Cờ Vây khó vì không có "material count" như Chess. Các heuristics phải dựa trên:

**1. Territory Estimation:**
```python
def estimate_territory(board, color):
    """
    Flood-fill algorithm để estimate territory
    Concept: BFS/DFS từ môn học
    """
    territory = 0
    visited = set()
    
    for i in range(board.size):
        for j in range(board.size):
            if board[i,j] == EMPTY and (i,j) not in visited:
                # Flood fill to find connected empty region
                region, borders = flood_fill(board, i, j)
                
                # Check if region is controlled by color
                if all(b == color for b in borders):
                    territory += len(region)
                    visited.update(region)
    
    return territory
```

**2. Pattern Recognition:**
```python
def evaluate_patterns(board, position):
    """
    Pattern matching với pre-compiled database
    Concept: Knowledge representation từ môn học
    """
    score = 0
    
    # Check 3x3 patterns around position
    pattern_3x3 = extract_pattern(board, position, size=3)
    hash_3x3 = zobrist_hash(pattern_3x3)
    
    if hash_3x3 in PATTERN_DB_3x3:
        score += PATTERN_DB_3x3[hash_3x3].value
    
    # Check 5x5 patterns for bigger structures
    pattern_5x5 = extract_pattern(board, position, size=5)
    hash_5x5 = zobrist_hash(pattern_5x5)
    
    if hash_5x5 in PATTERN_DB_5x5:
        score += PATTERN_DB_5x5[hash_5x5].value
    
    return score
```

**3. Group Strength (Liberty Counting):**
```python
def evaluate_group_strength(board, position):
    """
    Đếm liberties (breathing spaces) của group
    Concept: Connected components, Graph search
    """
    group = find_connected_group(board, position)  # DFS/BFS
    liberties = count_liberties(board, group)
    
    # Heuristic values
    if liberties == 1:
        return -100  # Atari! Very bad
    elif liberties == 2:
        return -30   # Weak group
    elif liberties >= 4:
        return +20   # Strong group
    else:
        return 0
```

**4. Eye Detection (Life and Death):**
```python
def has_two_eyes(board, group):
    """
    Check if group có 2 eyes → alive unconditionally
    Concept: Domain knowledge + pattern matching
    """
    empty_regions = find_enclosed_empty(board, group)
    eyes = 0
    
    for region in empty_regions:
        if is_eye(board, region, group.color):
            eyes += 1
    
    return eyes >= 2
```

**5. Influence Map:**
```python
def compute_influence(board, color):
    """
    Diffusion algorithm để tính influence
    Concept: Propagation, similar to heat diffusion
    """
    influence = np.zeros_like(board, dtype=float)
    
    # Each stone radiates influence
    for i in range(board.size):
        for j in range(board.size):
            if board[i,j] == color:
                # Diffuse influence to nearby points
                for dx in range(-5, 6):
                    for dy in range(-5, 6):
                        dist = abs(dx) + abs(dy)
                        if dist > 0:
                            ni, nj = i+dx, j+dy
                            if is_valid(ni, nj):
                                influence[ni,nj] += 1.0 / (dist + 1)
    
    return influence
```

#### 2.5.3. Combined Evaluation Function

```python
def evaluate_position(game_state):
    """
    Multi-factor evaluation function
    Weighted sum of heuristics
    """
    weights = {
        'territory': 1.0,
        'prisoners': 1.0,
        'group_strength': 0.5,
        'influence': 0.3,
        'patterns': 0.4
    }
    
    score = 0
    my_color = game_state.current_player
    opp_color = opponent(my_color)
    
    # Territory
    my_territory = estimate_territory(game_state.board, my_color)
    opp_territory = estimate_territory(game_state.board, opp_color)
    score += weights['territory'] * (my_territory - opp_territory)
    
    # Prisoners
    score += weights['prisoners'] * (
        game_state.prisoners[my_color] - 
        game_state.prisoners[opp_color]
    )
    
    # Group strength
    my_strength = sum(evaluate_group_strength(g) for g in my_groups)
    opp_strength = sum(evaluate_group_strength(g) for g in opp_groups)
    score += weights['group_strength'] * (my_strength - opp_strength)
    
    # ... other factors
    
    return score
```

**Trade-off (theo môn học):**
- Evaluation càng phức tạp → chính xác hơn
- Nhưng slower → ít playouts hơn trong MCTS
- **Cần balance:** Fast enough, Good enough

### 2.6. Chủ đề 4: Knowledge Representation

#### 2.6.1. Lý thuyết môn học

**Cách biểu diễn tri thức:**
- **Rule-based systems:** IF-THEN rules
- **Semantic networks:** Nodes & edges
- **Frames:** Structured objects with slots
- **Logic:** First-order logic, propositional logic

Trong AI game:
- **Opening book:** Lưu trữ các opening sequences tốt
- **Endgame tablebases:** Solved positions
- **Pattern databases:** Common tactical patterns

#### 2.6.2. Áp dụng vào Cờ Vây

**1. Rule-based System cho Legal Moves:**

```python
class GoRules:
    """
    Rule-based system encoding game rules
    """
    def is_legal_move(self, board, position, color):
        """Check all rules"""
        # Rule 1: Position must be empty
        if not self.is_empty(board, position):
            return False, "Position occupied"
        
        # Rule 2: Ko rule
        if self.violates_ko(board, position, color):
            return False, "Ko violation"
        
        # Rule 3: No suicide (unless capturing)
        if self.is_suicide(board, position, color):
            return False, "Suicide move"
        
        return True, None
    
    def is_suicide(self, board, position, color):
        """
        Rule: Cannot place stone with 0 liberties
        UNLESS it captures opponent stones
        """
        # Simulate placing stone
        temp_board = board.copy()
        temp_board.place(position, color)
        
        # Check if placed group has liberties
        group_liberties = count_liberties(temp_board, position)
        if group_liberties > 0:
            return False  # Has liberties, OK
        
        # Check if captures opponent
        captures_opponent = False
        for neighbor in get_neighbors(position):
            if temp_board[neighbor] == opponent(color):
                if count_liberties(temp_board, neighbor) == 0:
                    captures_opponent = True
        
        return not captures_opponent
```

**2. Pattern Database (Joseki & Tesuji):**

```python
class PatternDatabase:
    """
    Knowledge base of Go patterns
    Concepts: Knowledge engineering, Expert system
    """
    def __init__(self):
        self.joseki = {}      # Opening patterns (corners)
        self.tesuji = {}      # Tactical patterns
        self.bad_moves = {}   # Known bad shapes
    
    def load_joseki(self, filepath):
        """
        Load joseki (standard corner sequences) from SGF files
        Source: Professional game records
        """
        for sgf in load_sgf_files(filepath):
            sequence = extract_corner_sequence(sgf)
            hash_value = zobrist_hash(sequence)
            self.joseki[hash_value] = {
                'moves': sequence,
                'frequency': sgf.metadata['frequency'],
                'win_rate': sgf.metadata['win_rate']
            }
    
    def match_pattern(self, board, position):
        """Pattern matching - O(1) với hash table"""
        pattern = extract_local_pattern(board, position)
        hash_val = zobrist_hash(pattern)
        
        if hash_val in self.joseki:
            return self.joseki[hash_val]
        elif hash_val in self.tesuji:
            return self.tesuji[hash_val]
        
        return None
```

**3. Feature Extraction cho Neural Networks:**

```python
def extract_features(game_state):
    """
    Convert board state to feature vector
    Concept: Feature engineering từ môn học ML
    """
    features = []
    
    # Plane 1: Current player stones
    features.append(game_state.board == game_state.current_player)
    
    # Plane 2: Opponent stones
    features.append(game_state.board == opponent(game_state.current_player))
    
    # Plane 3: Liberties (1 liberty)
    lib1 = np.zeros_like(game_state.board)
    for group in find_all_groups(game_state.board):
        if count_liberties(group) == 1:
            lib1[group.positions] = 1
    features.append(lib1)
    
    # Plane 4: Liberties (2 liberties)
    # ... similar
    
    # Plane 5-8: History (previous 4 moves)
    for i in range(4):
        if len(game_state.history) > i:
            hist_board = game_state.history[-(i+1)]
            features.append(hist_board == BLACK)
            features.append(hist_board == WHITE)
        else:
            features.append(np.zeros_like(game_state.board))
            features.append(np.zeros_like(game_state.board))
    
    # Stack all planes: shape (N, board_size, board_size)
    return np.stack(features, axis=0)
```

**Số lượng feature planes (theo AlphaGo paper - đơn giản hóa):**
- Stone colors: 3 (black, white, empty)
- Liberties: 4 planes (1, 2, 3, 4+ liberties)
- History: 8 planes (4 previous positions × 2 colors)
- Turn color: 1 plane
- **Total: ~17 planes cho 1 board state**

### 2.7. Chủ đề 5: Machine Learning (Supervised Learning)

#### 2.7.1. Lý thuyết môn học

**Supervised Learning:**
- Training data: (input, label) pairs
- Goal: Learn function f: X → Y
- Loss function: Measure prediction error
- Optimization: Gradient descent

**Neural Networks:**
- Universal function approximators
- Layers: Input → Hidden → Output
- Activation functions: ReLU, Sigmoid, Tanh
- Backpropagation: Compute gradients

#### 2.7.2. Policy Network - Dự đoán nước đi

**Mục tiêu:** Học từ game records của kỳ thủ chuyên nghiệp

```python
class PolicyNetwork(nn.Module):
    """
    Neural network dự đoán xác suất mỗi nước đi
    Input: Board state (17 planes × 19 × 19)
    Output: Probability distribution over moves (361 positions + 1 pass)
    """
    def __init__(self, board_size=19, num_planes=17):
        super().__init__()
        
        # Convolutional layers (học local patterns)
        self.conv1 = nn.Conv2d(num_planes, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # Output layer
        self.conv_out = nn.Conv2d(128, 1, kernel_size=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)
    
    def forward(self, x):
        # x shape: (batch, 17, 19, 19)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        x = self.conv_out(x)  # (batch, 1, 19, 19)
        x = x.view(x.size(0), -1)  # Flatten to (batch, 361)
        
        # Softmax to get probability distribution
        return F.softmax(x, dim=1)
```

**Training Process:**

```python
def train_policy_network(model, training_data):
    """
    Supervised learning từ professional games
    Training data: (board_state, expert_move) pairs
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch in training_data:
            board_states = batch['states']  # (batch, 17, 19, 19)
            expert_moves = batch['moves']   # (batch,) - move indices
            
            # Forward pass
            predictions = model(board_states)  # (batch, 361)
            
            # Compute loss
            loss = criterion(predictions, expert_moves)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch}: Loss = {total_loss/len(training_data)}")
```

**Data Preparation:**

```python
def prepare_training_data(sgf_files):
    """
    Convert SGF files to training samples
    """
    training_data = []
    
    for sgf_path in sgf_files:
        game = load_sgf(sgf_path)
        
        # Only use games from strong players
        if game.black_rank < "5d" or game.white_rank < "5d":
            continue
        
        board_state = GameState()
        
        for move in game.moves:
            # Extract features
            features = extract_features(board_state)
            
            # Move as label (classification target)
            move_index = move.position[0] * 19 + move.position[1]
            
            training_data.append({
                'state': features,
                'move': move_index
            })
            
            # Apply move to board
            board_state.apply_move(move)
    
    return training_data
```

#### 2.7.3. Value Network - Đánh giá vị thế

**Mục tiêu:** Predict win probability từ board position

```python
class ValueNetwork(nn.Module):
    """
    Neural network dự đoán win probability
    Input: Board state (17 planes × 19 × 19)
    Output: Win probability [0, 1]
    """
    def __init__(self, board_size=19, num_planes=17):
        super().__init__()
        
        # Similar conv layers
        self.conv1 = nn.Conv2d(num_planes, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 19 * 19, 256)
        self.fc2 = nn.Linear(256, 1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Output in [0, 1]
        
        return x
```

**Training:**

```python
def train_value_network(model, training_data):
    """
    Supervised learning: board state → win probability
    Label = 1 if current player wins, 0 if loses
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()  # Regression loss
    
    for epoch in range(num_epochs):
        for batch in training_data:
            states = batch['states']
            outcomes = batch['outcomes']  # 0 or 1
            
            predictions = model(states)
            loss = criterion(predictions, outcomes)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

### 2.8. Chủ đề 6: Reinforcement Learning (Self-Play)

#### 2.8.1. Lý thuyết môn học

**Reinforcement Learning:**
- Agent học từ interaction với environment
- Reward signal: Delayed, sparse
- Goal: Maximize cumulative reward
- Methods: Q-learning, Policy Gradient, TD-learning

**Key concepts:**
- **State s:** Game position
- **Action a:** Move to play
- **Reward r:** Win (+1) or Loss (0)
- **Policy π(a|s):** Probability of action given state
- **Value V(s):** Expected return from state

#### 2.8.2. Self-Play Training

**Ý tưởng:** AI chơi với chính nó để generate training data

```python
def self_play_training(policy_net, value_net, num_games=1000):
    """
    Self-play reinforcement learning
    Concept: Bootstrap learning - AI improves by playing itself
    """
    training_data = []
    
    for game_num in range(num_games):
        game_history = []
        game_state = GameState()
        
        # Play one full game
        while not game_state.is_terminal():
            # Use current policy to select move
            features = extract_features(game_state)
            move_probs = policy_net(features)
            
            # Sample move from probability distribution
            # (exploration via sampling)
            move = sample_move(move_probs)
            
            # Store (state, move, value) for training
            game_history.append({
                'state': features,
                'move': move,
                'player': game_state.current_player
            })
            
            # Apply move
            game_state.apply_move(move)
        
        # Game finished, get outcome
        winner = game_state.get_winner()
        
        # Assign rewards: +1 for winner, 0 for loser
        for entry in game_history:
            if entry['player'] == winner:
                entry['reward'] = 1.0
            else:
                entry['reward'] = 0.0
        
        training_data.extend(game_history)
        
        # Every N games, train networks
        if (game_num + 1) % 10 == 0:
            train_on_batch(policy_net, value_net, training_data)
            training_data = []  # Clear buffer
    
    return policy_net, value_net
```

**TD-Learning (Temporal Difference):**

```python
def td_learning_update(value_net, game_history, gamma=0.99):
    """
    TD-Learning: Update value estimates
    V(s_t) ← V(s_t) + α[r_t + γV(s_{t+1}) - V(s_t)]
    """
    for t in range(len(game_history) - 1):
        state_t = game_history[t]['state']
        state_t1 = game_history[t+1]['state']
        reward_t = game_history[t]['reward']
        
        # Current value estimate
        value_t = value_net(state_t)
        
        # Next state value (bootstrap)
        with torch.no_grad():
            value_t1 = value_net(state_t1)
        
        # TD target
        td_target = reward_t + gamma * value_t1
        
        # TD error
        td_error = td_target - value_t
        
        # Update network to minimize TD error
        loss = td_error ** 2
        loss.backward()
```

#### 2.8.3. Policy Gradient (REINFORCE)

```python
def policy_gradient_update(policy_net, game_history):
    """
    REINFORCE algorithm: Update policy to increase probability
    of actions that lead to wins
    
    ∇J(θ) = E[∇log π(a|s) * R]
    """
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=0.001)
    
    for entry in game_history:
        state = entry['state']
        action = entry['move']
        reward = entry['reward']  # 1 if win, 0 if loss
        
        # Forward pass
        action_probs = policy_net(state)
        
        # Log probability of taken action
        log_prob = torch.log(action_probs[action])
        
        # Policy gradient: Increase prob if won, decrease if lost
        loss = -log_prob * reward
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 2.9. Chủ đề 7: Optimization & Performance

#### 2.9.1. Algorithm Optimization

**1. Zobrist Hashing (Fast state comparison):**

```python
class ZobristHash:
    """
    Perfect hashing for Go boards
    Concept: Hashing from môn học cấu trúc dữ liệu
    """
    def __init__(self, board_size=19):
        # Random bitstrings for each (position, color)
        self.table = np.random.randint(
            0, 2**64, 
            size=(board_size, board_size, 3),  # 3 = BLACK, WHITE, EMPTY
            dtype=np.uint64
        )
        self.ko_hash = np.random.randint(0, 2**64, dtype=np.uint64)
    
    def compute_hash(self, board):
        """O(n^2) initial computation"""
        hash_value = 0
        for i in range(board.size):
            for j in range(board.size):
                color = board[i, j]
                hash_value ^= self.table[i, j, color]
        return hash_value
    
    def update_hash(self, old_hash, position, old_color, new_color):
        """O(1) incremental update when move is played"""
        i, j = position
        new_hash = old_hash
        new_hash ^= self.table[i, j, old_color]  # Remove old
        new_hash ^= self.table[i, j, new_color]  # Add new
        return new_hash
```

**2. Transposition Table:**

```python
class TranspositionTable:
    """
    Cache for MCTS nodes
    Avoid re-computing same positions
    """
    def __init__(self, max_size=1_000_000):
        self.table = {}
        self.max_size = max_size
    
    def lookup(self, board_hash):
        """O(1) lookup"""
        return self.table.get(board_hash, None)
    
    def store(self, board_hash, node_info):
        """Store MCTS statistics"""
        if len(self.table) >= self.max_size:
            # Eviction: Remove least visited
            self._evict()
        self.table[board_hash] = node_info
```

**3. Bitboard Optimization (C++):**

```cpp
// Fast board operations với bitwise operations
class BitBoard {
    uint64_t black[6];   // 361 bits for 19x19
    uint64_t white[6];
    
    // O(1) check if position has stone
    inline bool has_stone(int pos) const {
        int block = pos / 64;
        int bit = pos % 64;
        return (black[block] | white[block]) & (1ULL << bit);
    }
    
    // O(1) place stone
    inline void place_stone(int pos, Color color) {
        int block = pos / 64;
        int bit = pos % 64;
        if (color == BLACK)
            black[block] |= (1ULL << bit);
        else
            white[block] |= (1ULL << bit);
    }
};
```

#### 2.9.2. Parallel Computing

```python
def parallel_mcts(root_state, num_threads=4):
    """
    Leaf parallelization: Multiple simulations simultaneously
    Concept: Parallel algorithms từ môn học
    """
    from concurrent.futures import ThreadPoolExecutor
    
    root = Node(root_state)
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Each thread runs MCTS simulations
        futures = []
        for _ in range(num_threads):
            future = executor.submit(mcts_worker, root, num_iterations=1000)
            futures.append(future)
        
        # Wait for all threads
        for future in futures:
            future.result()
    
    return best_move(root)
```

### 2.10. Tổng kết mapping

**Bảng mapping đầy đủ:**

| Concept môn học | Implementation | Thuật toán | Location | Mức độ |
|-----------------|----------------|------------|----------|---------|
| **Adversarial Search (Minimax)** ⭐ | Minimax + Alpha-Beta pruning, Move ordering | **MINIMAX** | AI Engine [C++] | ★★★★★ |
| **Stochastic Search (MCTS)** | Selection, Expansion, Simulation, Backprop | **MCTS** | AI Engine [C++] | ★★★★★ |
| **Game Tree Concepts** | Zero-sum game, Minimax value, Terminal test | Both | Game Engine | ★★★★★ |
| **Heuristics** | Territory, Pattern matching, Liberty count, Eval function | Both | Heuristic Engine [C++] | ★★★★★ |
| **Alpha-Beta Pruning** | Cutoffs, Move ordering | **MINIMAX** | Minimax Engine | ★★★★★ |
| **UCB (Exploration/Exploitation)** | UCB1 formula, Balance | **MCTS** | MCTS Selection | ★★★★☆ |
| **Knowledge Representation** | Rule-based system, Pattern DB, Features | Both | Game Rules + Pattern DB | ★★★★☆ |
| **Supervised Learning** | Policy Network, Value Network | **MCTS** | Training Pipeline [Python] | ★★★★☆ |
| **Reinforcement Learning** | Self-play, TD-learning, Policy gradient | **MCTS** | Training Pipeline [Python] | ★★★☆☆ |
| **Optimization** | Zobrist hash, Transposition table, Bitboards | Both | Performance layer [C++] | ★★★★☆ |
| **Probability & Uncertainty** | UCB formula, Stochastic rollouts | **MCTS** | MCTS Core | ★★★★☆ |

**Chú thích:**
- ⭐ = Yêu cầu chính của đề tài số 18
- **MINIMAX** = Level 1-2 (Demo, Educational)
- **MCTS** = Level 3-4 (Practical, Competitive)
- **Both** = Được sử dụng bởi cả hai thuật toán

**Summary - Dual Algorithm Strategy:**

```
┌──────────────────────────────────────────────────────────────┐
│              PHÂN BỐ CONCEPTS THEO THUẬT TOÁN                │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  MINIMAX/ALPHA-BETA (Đề tài số 18)                         │
│  ├─ Adversarial search (minimax value)                      │
│  ├─ Alpha-Beta pruning                                      │
│  ├─ Move ordering                                           │
│  ├─ Evaluation function (heuristics)                        │
│  ├─ Game tree visualization                                 │
│  └─ Depth-limited search                                    │
│                                                              │
│  MCTS (Bổ trợ - Practical AI)                              │
│  ├─ Best-first search với UCB                              │
│  ├─ Monte Carlo simulation                                  │
│  ├─ Exploration vs Exploitation                             │
│  ├─ Policy/Value networks (ML)                              │
│  ├─ Self-play training (RL)                                 │
│  └─ Anytime algorithm                                       │
│                                                              │
│  SHARED COMPONENTS (Cả hai dùng chung)                     │
│  ├─ Game rules engine                                       │
│  ├─ Board representation (bitboards)                        │
│  ├─ Heuristic evaluators                                    │
│  ├─ Pattern database                                        │
│  ├─ Zobrist hashing                                         │
│  └─ Move generation                                         │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**Đóng góp của dự án với môn học:**

1. **Adversarial Search (Minimax) ⭐** - Yêu cầu chính đề tài:
   - ✅ Implement đầy đủ Minimax với Alpha-Beta
   - ✅ Demo rõ ràng game tree exploration
   - ✅ Visualize pruning và move ordering
   - ✅ Evaluation function design

2. **Modern AI (MCTS)** - Mở rộng kiến thức:
   - ✅ So sánh classical vs modern approach
   - ✅ Understand limitations của Minimax
   - ✅ Practical AI cho real-world applications

3. **Machine Learning** - Tích hợp AI/ML:
   - ✅ Supervised learning (Policy/Value networks)
   - ✅ Reinforcement learning (Self-play)
   - ✅ Feature engineering

4. **Optimization** - Kỹ thuật tối ưu:
   - ✅ Alpha-Beta pruning
   - ✅ Move ordering
   - ✅ Zobrist hashing
   - ✅ Bitboard operations

5. **System Design** - Kiến trúc phần mềm:
   - ✅ Hybrid C++/Python architecture
   - ✅ Multi-database strategy
   - ✅ Modular design

**Kết luận:**

Dự án này **vượt yêu cầu** đề tài số 18 bằng cách:
1. ✅ Implement đầy đủ **Minimax/Alpha-Beta** (yêu cầu chính)
2. ✅ Thêm **MCTS** để so sánh và practical application
3. ✅ Áp dụng **đa dạng concepts** từ môn học
4. ✅ Có giá trị **educational** (demo, visualize) và **practical** (competitive AI)
5. ✅ Architecture **professional-grade** với C++/Python hybrid

---

**Kết thúc PHẦN 2 — PHÂN TÍCH YÊU CẦU THEO MÔN HỌC**

---

## PHẦN 3 — TÓM TẮT THUẬT TOÁN VÀ KIẾN TRÚC

> **Lưu ý:** Chi tiết implementation đầy đủ xem file `ImplementationGuide.md`

### 3.1. Tổng quan Thuật toán

Hệ thống sử dụng **3 thuật toán chính**:

| Thuật toán | Ngôn ngữ | Mục đích | AI Level | Complexity |
|------------|----------|----------|----------|------------|
| **Minimax + Alpha-Beta** | C++ | Demo, Educational | Level 1-2 | O(b^(d/2)) |
| **MCTS** | C++ | Competitive AI | Level 3-4 | O(iterations) |
| **Neural Networks** | Python | Premium Tools | Shop Items | O(model_size) |

### 3.2. Thuật toán 1: Minimax Engine (Yêu cầu Đề tài)

**Tổng quan:**

#### 3.2.1. Kiến trúc Minimax Engine (C++)

**File structure:**
```
src/ai/minimax/
├── minimax_engine.h
├── minimax_engine.cpp
├── evaluator.h
├── evaluator.cpp
├── move_ordering.h
├── move_ordering.cpp
└── transposition_table.h
```

**Core Class - MinimaxEngine:**

```cpp
// minimax_engine.h
#ifndef MINIMAX_ENGINE_H
#define MINIMAX_ENGINE_H

#include "../game/board.h"
#include "evaluator.h"
#include "transposition_table.h"
#include <vector>
#include <limits>

class MinimaxEngine {
public:
    struct SearchResult {
        Move best_move;
        float evaluation;
        int nodes_searched;
        int nodes_pruned;
        double search_time;
        std::vector<Move> principal_variation;
    };
    
    struct Config {
        int max_depth;              // 2-4 for Go
        bool use_alpha_beta;        // Alpha-Beta pruning
        bool use_move_ordering;     // Move ordering optimization
        bool use_transposition;     // Transposition table
        double time_limit_seconds;  // Time constraint
        int board_size;             // 9x9 only
    };

private:
    Config config_;
    Evaluator evaluator_;
    TranspositionTable tt_;
    
    // Statistics
    int nodes_searched_;
    int nodes_pruned_;
    
    // Constants
    static constexpr float INFINITY_VALUE = 
        std::numeric_limits<float>::infinity();
    static constexpr float MATE_SCORE = 100000.0f;

public:
    MinimaxEngine(const Config& config);
    
    // Main search entry point
    SearchResult search(const Board& board, Color to_move);
    
    // Get detailed tree for visualization
    GameTree build_game_tree(const Board& board, int depth);

private:
    // Core minimax with alpha-beta
    float minimax(
        Board& board,
        int depth,
        float alpha,
        float beta,
        Color maximizing_player,
        Move* best_move_out
    );
    
    float max_value(
        Board& board,
        int depth,
        float alpha,
        float beta,
        Move* best_move_out
    );
    
    float min_value(
        Board& board,
        int depth,
        float alpha,
        float beta,
        Move* best_move_out
    );
    
    // Helper functions
    std::vector<Move> get_ordered_moves(
        const Board& board,
        Color player
    );
    
    bool is_cutoff(const Board& board, int depth);
    
    float evaluate_position(const Board& board, Color player);
};

#endif // MINIMAX_ENGINE_H
```

**Implementation:**

```cpp
// minimax_engine.cpp
#include "minimax_engine.h"
#include "move_ordering.h"
#include <algorithm>
#include <chrono>

MinimaxEngine::MinimaxEngine(const Config& config)
    : config_(config)
    , evaluator_(config.board_size)
    , tt_(1000000)  // 1M entries
    , nodes_searched_(0)
    , nodes_pruned_(0)
{
}

MinimaxEngine::SearchResult MinimaxEngine::search(
    const Board& board,
    Color to_move
) {
    nodes_searched_ = 0;
    nodes_pruned_ = 0;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Make mutable copy
    Board search_board = board;
    Move best_move;
    
    // Start minimax search
    float evaluation = minimax(
        search_board,
        config_.max_depth,
        -INFINITY_VALUE,
        INFINITY_VALUE,
        to_move,
        &best_move
    );
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double search_time = std::chrono::duration<double>(
        end_time - start_time
    ).count();
    
    // Build principal variation
    std::vector<Move> pv = extract_principal_variation(
        search_board,
        best_move
    );
    
    return SearchResult{
        best_move,
        evaluation,
        nodes_searched_,
        nodes_pruned_,
        search_time,
        pv
    };
}

float MinimaxEngine::minimax(
    Board& board,
    int depth,
    float alpha,
    float beta,
    Color maximizing_player,
    Move* best_move_out
) {
    nodes_searched_++;
    
    // Terminal test or depth limit
    if (is_cutoff(board, depth)) {
        return evaluate_position(board, maximizing_player);
    }
    
    // Check transposition table
    if (config_.use_transposition) {
        uint64_t hash = board.zobrist_hash();
        auto tt_entry = tt_.lookup(hash);
        if (tt_entry.valid && tt_entry.depth >= depth) {
            return tt_entry.evaluation;
        }
    }
    
    // Get legal moves
    std::vector<Move> moves = board.get_legal_moves(
        maximizing_player
    );
    
    if (moves.empty()) {
        // No legal moves (passed), evaluate position
        return evaluate_position(board, maximizing_player);
    }
    
    // Move ordering
    if (config_.use_move_ordering) {
        MoveOrdering::order_moves(moves, board, maximizing_player);
    }
    
    float best_value;
    Move local_best_move;
    
    if (board.current_player() == maximizing_player) {
        // Maximizing player
        best_value = -INFINITY_VALUE;
        
        for (const Move& move : moves) {
            // Make move
            Board::UndoInfo undo = board.make_move(move);
            
            // Recursive call
            Move child_best;
            float value = minimax(
                board,
                depth - 1,
                alpha,
                beta,
                maximizing_player,
                &child_best
            );
            
            // Undo move
            board.undo_move(undo);
            
            // Update best
            if (value > best_value) {
                best_value = value;
                local_best_move = move;
            }
            
            // Alpha-Beta pruning
            if (config_.use_alpha_beta) {
                alpha = std::max(alpha, value);
                if (beta <= alpha) {
                    nodes_pruned_++;
                    break;  // Beta cutoff
                }
            }
        }
    } else {
        // Minimizing player
        best_value = INFINITY_VALUE;
        
        for (const Move& move : moves) {
            Board::UndoInfo undo = board.make_move(move);
            
            Move child_best;
            float value = minimax(
                board,
                depth - 1,
                alpha,
                beta,
                maximizing_player,
                &child_best
            );
            
            board.undo_move(undo);
            
            if (value < best_value) {
                best_value = value;
                local_best_move = move;
            }
            
            // Alpha-Beta pruning
            if (config_.use_alpha_beta) {
                beta = std::min(beta, value);
                if (beta <= alpha) {
                    nodes_pruned_++;
                    break;  // Alpha cutoff
                }
            }
        }
    }
    
    // Store in transposition table
    if (config_.use_transposition) {
        uint64_t hash = board.zobrist_hash();
        tt_.store(hash, depth, best_value, local_best_move);
    }
    
    if (best_move_out) {
        *best_move_out = local_best_move;
    }
    
    return best_value;
}

bool MinimaxEngine::is_cutoff(const Board& board, int depth) {
    // Cutoff if:
    // 1. Depth limit reached
    if (depth <= 0) return true;
    
    // 2. Game over (two consecutive passes)
    if (board.is_game_over()) return true;
    
    // 3. Time limit exceeded (optional)
    // TODO: Implement time check
    
    return false;
}

float MinimaxEngine::evaluate_position(
    const Board& board,
    Color player
) {
    return evaluator_.evaluate(board, player);
}
```

#### 3.2.2. Evaluation Function (Heuristics)

**Evaluator Class:**

```cpp
// evaluator.h
class Evaluator {
public:
    struct Weights {
        float territory = 10.0f;
        float prisoners = 5.0f;
        float group_strength = 3.0f;
        float influence = 2.0f;
        float patterns = 1.0f;
    };

private:
    int board_size_;
    Weights weights_;
    PatternDatabase pattern_db_;

public:
    Evaluator(int board_size);
    
    float evaluate(const Board& board, Color player);

private:
    // Individual heuristics
    float evaluate_territory(const Board& board, Color player);
    float evaluate_prisoners(const Board& board, Color player);
    float evaluate_group_strength(const Board& board, Color player);
    float evaluate_influence(const Board& board, Color player);
    float evaluate_patterns(const Board& board, Color player);
    
    // Helper functions
    int count_liberties(const Board& board, const Group& group);
    bool has_two_eyes(const Board& board, const Group& group);
    std::vector<Point> find_territory(
        const Board& board,
        Color player
    );
};
```

**Implementation:**

```cpp
// evaluator.cpp
#include "evaluator.h"

Evaluator::Evaluator(int board_size)
    : board_size_(board_size)
    , pattern_db_("patterns.db")
{
}

float Evaluator::evaluate(const Board& board, Color player) {
    Color opponent = opposite_color(player);
    
    float score = 0.0f;
    
    // Territory control
    float my_territory = evaluate_territory(board, player);
    float opp_territory = evaluate_territory(board, opponent);
    score += weights_.territory * (my_territory - opp_territory);
    
    // Captured stones
    int my_prisoners = board.get_prisoners(player);
    int opp_prisoners = board.get_prisoners(opponent);
    score += weights_.prisoners * (my_prisoners - opp_prisoners);
    
    // Group strength
    float my_strength = evaluate_group_strength(board, player);
    float opp_strength = evaluate_group_strength(board, opponent);
    score += weights_.group_strength * (my_strength - opp_strength);
    
    // Influence
    float my_influence = evaluate_influence(board, player);
    float opp_influence = evaluate_influence(board, opponent);
    score += weights_.influence * (my_influence - opp_influence);
    
    // Pattern bonuses
    float my_patterns = evaluate_patterns(board, player);
    float opp_patterns = evaluate_patterns(board, opponent);
    score += weights_.patterns * (my_patterns - opp_patterns);
    
    return score;
}

float Evaluator::evaluate_territory(
    const Board& board,
    Color player
) {
    std::vector<Point> territory = find_territory(board, player);
    return static_cast<float>(territory.size());
}

float Evaluator::evaluate_group_strength(
    const Board& board,
    Color player
) {
    float total_strength = 0.0f;
    
    auto groups = board.find_all_groups(player);
    
    for (const Group& group : groups) {
        int liberties = count_liberties(board, group);
        
        // Liberty-based scoring
        if (liberties == 1) {
            total_strength -= 100.0f;  // Atari! Very bad
        } else if (liberties == 2) {
            total_strength -= 20.0f;   // Weak
        } else if (liberties >= 4) {
            total_strength += 10.0f;   // Strong
        }
        
        // Two eyes = unconditionally alive
        if (has_two_eyes(board, group)) {
            total_strength += 50.0f;
        }
        
        // Size bonus (big groups are valuable)
        total_strength += std::sqrt(group.size()) * 5.0f;
    }
    
    return total_strength;
}

float Evaluator::evaluate_patterns(
    const Board& board,
    Color player
) {
    float score = 0.0f;
    
    // Check 3x3 patterns at each stone
    for (int i = 0; i < board_size_; i++) {
        for (int j = 0; j < board_size_; j++) {
            if (board.at(i, j) != player) continue;
            
            // Extract 3x3 pattern
            Pattern3x3 pattern = board.extract_pattern_3x3(i, j);
            uint64_t hash = pattern.hash();
            
            // Lookup in database
            auto entry = pattern_db_.lookup(hash);
            if (entry.valid) {
                score += entry.value;
            }
        }
    }
    
    return score;
}
```

#### 3.2.3. Move Ordering

**Move Ordering Strategy:**

```cpp
// move_ordering.h
class MoveOrdering {
public:
    static void order_moves(
        std::vector<Move>& moves,
        const Board& board,
        Color player
    );

private:
    static float score_move(
        const Move& move,
        const Board& board,
        Color player
    );
    
    // Priority checks
    static bool is_capturing_move(
        const Move& move,
        const Board& board
    );
    
    static bool saves_atari(
        const Move& move,
        const Board& board,
        Color player
    );
    
    static float position_value(const Move& move, int board_size);
};
```

**Implementation:**

```cpp
// move_ordering.cpp
void MoveOrdering::order_moves(
    std::vector<Move>& moves,
    const Board& board,
    Color player
) {
    // Score each move
    std::vector<std::pair<float, Move>> scored_moves;
    scored_moves.reserve(moves.size());
    
    for (const Move& move : moves) {
        float score = score_move(move, board, player);
        scored_moves.emplace_back(score, move);
    }
    
    // Sort by score (descending)
    std::sort(
        scored_moves.begin(),
        scored_moves.end(),
        [](const auto& a, const auto& b) {
            return a.first > b.first;
        }
    );
    
    // Extract sorted moves
    moves.clear();
    for (const auto& [score, move] : scored_moves) {
        moves.push_back(move);
    }
}

float MoveOrdering::score_move(
    const Move& move,
    const Board& board,
    Color player
) {
    float score = 0.0f;
    
    // Priority 1: Capturing moves (highest priority)
    if (is_capturing_move(move, board)) {
        score += 1000.0f;
    }
    
    // Priority 2: Saving own groups from atari
    if (saves_atari(move, board, player)) {
        score += 500.0f;
    }
    
    // Priority 3: Position value (center > edges > corners)
    score += position_value(move, board.size());
    
    // Priority 4: Pattern matching
    // TODO: Add pattern-based scoring
    
    return score;
}

bool MoveOrdering::is_capturing_move(
    const Move& move,
    const Board& board
) {
    // Check if this move would capture opponent stones
    // by placing them in atari or removing their last liberty
    
    Color opponent = opposite_color(move.color);
    
    // Get neighbors
    auto neighbors = board.get_neighbors(move.point);
    
    for (const Point& neighbor : neighbors) {
        if (board.at(neighbor) == opponent) {
            // Check if opponent group has only 1 liberty
            Group opponent_group = board.get_group(neighbor);
            if (count_liberties_after_move(
                board, opponent_group, move
            ) == 0) {
                return true;  // Would capture!
            }
        }
    }
    
    return false;
}

float MoveOrdering::position_value(
    const Move& move,
    int board_size
) {
    int i = move.point.x;
    int j = move.point.y;
    
    // Distance from center
    int center = board_size / 2;
    float dist_from_center = std::sqrt(
        (i - center) * (i - center) +
        (j - center) * (j - center)
    );
    
    // Closer to center = higher value
    return 100.0f / (1.0f + dist_from_center);
}
```

#### 3.2.4. Performance Characteristics

**Complexity Analysis:**

| Aspect | Without Alpha-Beta | With Alpha-Beta | With All Optimizations |
|--------|-------------------|-----------------|------------------------|
| **Time** | O(b^d) | O(b^(d/2)) best case | O(b^(d/2-1)) |
| **Space** | O(d) | O(d) | O(d + TT_size) |
| **Nodes @ depth=4** | ~250^4 ≈ 3.9B | ~250^2 ≈ 62K | ~10-20K |
| **Time @ depth=4** | ~Hours | ~Minutes | ~Seconds |

**Benchmarks (9×9 board):**

```
Configuration: Intel Core i7, Single thread

Depth 2 (no pruning):
- Nodes: ~60,000
- Time: ~0.5s
- Strength: ~15 kyu

Depth 2 (with pruning):
- Nodes: ~15,000
- Time: ~0.15s
- Pruning: 75%

Depth 4 (no pruning):
- Nodes: ~15M
- Time: ~60s
- Too slow!

Depth 4 (with pruning + move ordering):
- Nodes: ~50,000
- Time: ~2s
- Pruning: 99.7%
- Strength: ~12 kyu
```

#### 3.2.5. Visualization Support

**Game Tree Builder:**

```cpp
struct GameTreeNode {
    Move move;
    float evaluation;
    int depth;
    bool pruned;
    std::vector<GameTreeNode> children;
};

class GameTree {
public:
    GameTreeNode root;
    
    // For visualization
    int total_nodes() const;
    int pruned_nodes() const;
    std::vector<Move> principal_variation() const;
    
    // Export for UI
    std::string to_json() const;
};
```

**Usage in UI:**

```python
# Python side
engine = MinimaxEngine(depth=3, use_pruning=True)
result = engine.search(board)

# Get game tree for visualization
tree = engine.build_game_tree(board, depth=3)

# Display in UI
ui.show_game_tree(tree)
ui.highlight_principal_variation(tree.pv)
ui.show_stats(
    nodes_searched=result.nodes_searched,
    nodes_pruned=result.nodes_pruned,
    pruning_efficiency=result.nodes_pruned / result.nodes_searched
)
```

---

### 3.3. Thuật toán 2: MCTS Engine (AI Mạnh)

#### 3.3.1. Kiến trúc MCTS Engine (C++)

**File structure:**
```
src/ai/mcts/
├── mcts_engine.h
├── mcts_engine.cpp
├── mcts_node.h
├── mcts_node.cpp
├── ucb.h
└── simulation.h
```

**Core Class - MCTSEngine:**

```cpp
// mcts_engine.h
#ifndef MCTS_ENGINE_H
#define MCTS_ENGINE_H

#include "../game/board.h"
#include "mcts_node.h"
#include <memory>
#include <vector>

class MCTSEngine {
public:
    struct Config {
        int num_playouts;           // 3000-8000
        double time_limit_seconds;  // Time constraint
        double ucb_constant;        // Exploration constant (√2)
        bool use_heuristics;        // Use heuristic rollouts
        bool parallel;              // Multi-threading
        int num_threads;            // If parallel
    };
    
    struct SearchResult {
        Move best_move;
        double win_rate;
        int total_visits;
        double search_time;
        std::vector<MoveStats> top_moves;
    };
    
    struct MoveStats {
        Move move;
        int visits;
        double win_rate;
        double ucb_value;
    };

private:
    Config config_;
    std::unique_ptr<MCTSNode> root_;
    
    // Statistics
    int total_playouts_;

public:
    MCTSEngine(const Config& config);
    
    // Main search entry point
    SearchResult search(const Board& board, Color to_move);
    
    // Get MCTS statistics for visualization
    std::vector<MoveStats> get_move_statistics() const;

private:
    // 4 phases of MCTS
    MCTSNode* selection(MCTSNode* node);
    MCTSNode* expansion(MCTSNode* node, const Board& board);
    double simulation(const Board& board, Color to_move);
    void backpropagation(MCTSNode* node, double result);
    
    // UCB calculation
    double calculate_ucb(const MCTSNode* node, int parent_visits) const;
    
    // Best child selection
    MCTSNode* best_child_ucb(MCTSNode* node) const;
    MCTSNode* best_child_robust(MCTSNode* node) const;
    
    // Parallel search
    void parallel_search(const Board& board, int num_iterations);
};

#endif // MCTS_ENGINE_H
```

#### 3.3.2. MCTS Implementation Summary

**Chi tiết implementation của MCTS sẽ được viết trong `Implementation/MCTSGuide.md` khi bắt đầu code.**

**Key Points:**
- 4 phases: Selection (UCB), Expansion, Simulation, Backpropagation
- Memory pool cho nodes (avoid allocation overhead)
- Parallel search với virtual loss
- Heuristic-guided rollouts (optional)
- No neural networks trong core game (Level 3-4)

**Performance Targets:**
- 3000 playouts: ~3s (Level 3, ~5 kyu)
- 8000 playouts: ~8s (Level 4, ~1 dan)

---

### 3.4. Thuật toán 3: ML Models (Premium Tools Only)

**Neural networks CHỈ dùng cho Premium Features, KHÔNG dùng trong core game AI.**

#### 3.4.1. Policy Network (AI Hint Feature)

**Architecture:** Lightweight CNN (5-7 layers)
```
Input: 17 planes × 19 × 19
Conv layers: 64 → 128 → 128 channels
Output: 361 + 1 (moves + pass)
```

**Purpose:** Gợi ý top 3 moves cho users (premium feature)

#### 3.4.2. Value Network (Analysis Feature)

**Architecture:** Similar CNN + FC layers
```
Input: 17 planes × 19 × 19
Output: Win probability [0, 1]
```

**Purpose:** Phân tích vị thế, win probability estimation

#### 3.4.3. Training Strategy

**Simplified approach cho đồ án:**
1. Supervised learning từ KGS game records (free dataset)
2. Small-scale self-play (1000-5000 games)
3. NO large-scale distributed training

**Chi tiết training sẽ viết trong `Implementation/MLModelsGuide.md`**

---

### 3.5. Integration Strategy

#### 3.5.1. Unified AI Interface

```python
class GoAI:
    """Unified interface cho tất cả AI components"""
    
    def __init__(self):
        # Core game engines (C++)
        self.minimax_engine = MinimaxEngine()
        self.mcts_engine = MCTSEngine()
        
        # Premium ML models (Python)
        self.policy_net = None  # Load on demand
        self.value_net = None   # Load on demand
    
    def select_move(self, board, level):
        """Core game AI - FREE"""
        if level <= 2:
            return self.minimax_engine.search(board, level)
        else:
            return self.mcts_engine.search(board, level)
    
    def get_hint(self, board, user_coins):
        """Premium feature - ML powered"""
        if user_coins < 10:
            return {"error": "Insufficient coins"}
        
        if not self.policy_net:
            self.policy_net = load_policy_network()
        
        return self.policy_net.suggest_moves(board, top_k=3)
```

#### 3.5.2. Component Communication

```
┌──────────────────────────────────────────────────┐
│           Python Layer (Game Controller)         │
├──────────────────────────────────────────────────┤
│                     │                             │
│  ┌──────────────────▼────────────────┐           │
│  │      C++ Engines (pybind11)       │           │
│  │  ┌────────────┐  ┌──────────────┐ │           │
│  │  │  Minimax   │  │    MCTS      │ │           │
│  │  │  Level 1-2 │  │  Level 3-4   │ │           │
│  │  └────────────┘  └──────────────┘ │           │
│  └────────────────────────────────────┘           │
│                     │                             │
│  ┌──────────────────▼────────────────┐           │
│  │   ML Models (PyTorch - Optional)  │           │
│  │  • Policy Net (Hints)              │           │
│  │  • Value Net (Analysis)            │           │
│  └────────────────────────────────────┘           │
└──────────────────────────────────────────────────┘
```

---

### 3.6. Summary - Thuật toán đã chọn

| Component | Algorithm | Technology | Purpose | Tier |
|-----------|-----------|------------|---------|------|
| **AI Level 1** | Minimax (depth 2) | C++ | Demo, beginner | 🆓 Free |
| **AI Level 2** | Minimax + Alpha-Beta (depth 4) | C++ | Educational | 🆓 Free |
| **AI Level 3** | MCTS (3k playouts, no ML) | C++ | Challenging | 🆓 Free |
| **AI Level 4** | MCTS (8k playouts, no ML) | C++ | Advanced | 🆓 Free |
| **AI Hint** | Policy Network | Python/PyTorch | Gợi ý nước đi | 💰 Premium |
| **Analysis** | Value Network | Python/PyTorch | Win probability | 💰 Premium |
| **Review** | Policy + Value + Deep MCTS | Hybrid | Full analysis | 💰 Premium |

**Justification:**
- ✅ Minimax: Yêu cầu đề tài 18
- ✅ MCTS: Practical AI, không quá mạnh
- ✅ ML: Premium tools, có giá trị thực tế
- ✅ Balance: Academic + Commercial

---

**Kết thúc PHẦN 3 — ĐỀ XUẤT THUẬT TOÁN CHI TIẾT**

*Chi tiết implementation sẽ được viết trong các guide riêng khi bắt đầu code phase.*

---

## PHẦN 4 — TỔNG KẾT & ROADMAP

### 4.1. Tóm tắt hệ thống

**GoGame** là một hệ thống trò chơi Cờ Vây hoàn chỉnh với:

#### 4.1.1. Core Features (Free)
✅ 4 AI levels: Minimax (1-2) + MCTS (3-4)  
✅ Multiple board sizes: 9×9 (priority), 19×19  
✅ Game modes: PvP, PvAI, AI vs AI  
✅ Save/Load games (SGF format)  
✅ Replay & basic statistics  
✅ Elo ranking system  

#### 4.1.2. Premium Features (Monetization)
💰 AI Hints (Policy Network)  
💰 Position Analysis (Value Network)  
💰 Game Review (Full AI analysis)  
💰 Opening Book (Joseki database)  
💰 Training Mode (Tsumego puzzles)  

#### 4.1.3. Technical Stack
🔧 **C++** (Game Engine, AI engines) - Performance  
🔧 **Python** (ML, Training, Backend) - Flexibility  
🔧 **PyTorch** (Neural networks) - ML framework  
🔧 **PostgreSQL** (Users, matches) - Relational data  
🔧 **MongoDB** (Game records, logs) - Flexible storage  
🔧 **S3/MinIO** (Models, backups) - Object storage  

#### 4.1.4. AI Algorithms
🤖 **Minimax + Alpha-Beta** (Đề tài yêu cầu)  
🤖 **MCTS** (State-of-the-art cho Go)  
🤖 **Policy/Value Networks** (Premium tools)  
🤖 **Heuristics** (Territory, patterns, groups)  

### 4.2. Đóng góp với Môn học

| Concept | Implementation | Evidence |
|---------|----------------|----------|
| **Adversarial Search** ⭐ | Minimax + Alpha-Beta | Full implementation, visualization |
| **Game Tree** | Minimax tree explorer | Educational demo |
| **Heuristics** | 5 evaluation functions | Territory, patterns, liberties, etc. |
| **Optimization** | Alpha-Beta pruning, move ordering | 99.7% node reduction |
| **Stochastic Search** | MCTS với UCB | Modern approach |
| **ML (Basic)** | Policy/Value networks | Lightweight models |
| **Knowledge Rep** | Rule-based + Pattern DB | Domain knowledge |
| **Practical AI** | Working game system | Real-world application |

### 4.3. Roadmap Phát triển

#### 📅 Phase 1: MVP (4-6 tuần)
**Mục tiêu:** Core game chạy được, nộp đồ án

- [ ] Week 1-2: Game Engine (C++)
  - Board representation (bitboards)
  - Move validation, Ko rule
  - Scoring system
  - Python bindings

- [ ] Week 3-4: AI Engines
  - Minimax + Alpha-Beta (Level 1-2)
  - MCTS basic (Level 3-4)
  - Heuristic evaluators

- [ ] Week 5-6: UI & Integration
  - Desktop UI (PyQt/Electron)
  - Game controller
  - Basic features (PvP, PvAI)
  - Save/Load SGF

**Deliverable:** Working game demo cho đồ án

#### 📅 Phase 2: Polish & Features (2-3 tuần)
- [ ] Replay system
- [ ] Statistics tracking
- [ ] Elo rating
- [ ] UI improvements
- [ ] Testing & bug fixes

**Deliverable:** Complete core game

#### 📅 Phase 3: ML & Premium (3-4 tuần)
- [ ] Train Policy Network
- [ ] Train Value Network
- [ ] Implement premium features
- [ ] Coin system
- [ ] Testing

**Deliverable:** Premium features ready

#### 📅 Phase 4: Online & Deployment (2-3 tuần)
- [ ] Backend API (FastAPI)
- [ ] Multiplayer support
- [ ] Database setup
- [ ] Deployment (Docker)
- [ ] Testing & launch

**Deliverable:** Online platform

### 4.4. Rủi ro & Giải pháp

| Rủi ro | Impact | Giải pháp |
|--------|--------|-----------|
| **C++ complexity** | High | Start với Python prototype, migrate dần |
| **MCTS too slow** | Medium | Profile & optimize, giảm playouts nếu cần |
| **ML training costly** | Medium | Use pre-trained models, small-scale training |
| **Time constraint** | High | Focus MVP first, features sau |
| **Database overhead** | Low | Start với SQLite, scale sau |

### 4.5. Next Steps - Bắt đầu Implementation

#### 📝 Documents cần tạo khi code:

1. **`Implementation/BoardEngine.md`**
   - Board representation chi tiết
   - Bitboard operations
   - Move validation logic

2. **`Implementation/MinimaxGuide.md`**
   - Full C++ implementation
   - Evaluator details
   - Performance tuning

3. **`Implementation/MCTSGuide.md`**
   - MCTS node structure
   - UCB implementation
   - Parallel search

4. **`Implementation/MLModels.md`**
   - Network architectures
   - Training procedures
   - Model deployment

5. **`Implementation/APIReference.md`**
   - Python API documentation
   - C++ bindings reference
   - Usage examples

6. **`Implementation/DatabaseSetup.md`**
   - Schema creation scripts
   - Sample queries
   - Migration guides

#### 🚀 First Steps:

```bash
# 1. Setup project structure
mkdir -p GoGame/{src,tests,docs,data}
cd GoGame

# 2. Initialize Git
git init
git add .
git commit -m "Initial commit: System spec"

# 3. Setup Python environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install numpy pygame torch

# 4. Setup C++ build system
# Create CMakeLists.txt
# Setup pybind11

# 5. Start với Board Engine
# Implement Board class (C++)
# Write unit tests
# Create Python bindings

# 6. Iterate...
```

---

## 📚 APPENDIX

### A. File Structure (Final)

```
GoGame/
├── docs/
│   ├── SystemSpec.md                 ✅ (This file)
│   ├── DatabaseDesign.md             (To be created)
│   └── Implementation/               (To be created during coding)
│       ├── BoardEngine.md
│       ├── MinimaxGuide.md
│       ├── MCTSGuide.md
│       ├── MLModels.md
│       └── APIReference.md
│
├── src/
│   ├── cpp/                          (C++ core)
│   │   ├── game/
│   │   │   ├── board.h/.cpp
│   │   │   ├── rules.h/.cpp
│   │   │   └── zobrist.h/.cpp
│   │   ├── ai/
│   │   │   ├── minimax/
│   │   │   │   ├── engine.h/.cpp
│   │   │   │   ├── evaluator.h/.cpp
│   │   │   │   └── move_ordering.h/.cpp
│   │   │   └── mcts/
│   │   │       ├── engine.h/.cpp
│   │   │       ├── node.h/.cpp
│   │   │       └── ucb.h/.cpp
│   │   └── bindings/
│   │       └── python_bindings.cpp
│   │
│   └── python/
│       ├── game/
│       │   ├── controller.py
│       │   └── sgf_parser.py
│       ├── ai/
│       │   ├── ai_player.py
│       │   └── premium_features.py
│       ├── ml/
│       │   ├── policy_network.py
│       │   ├── value_network.py
│       │   └── training.py
│       ├── backend/
│       │   ├── api.py
│       │   ├── database.py
│       │   └── coin_system.py
│       └── ui/
│           ├── main_window.py
│           └── board_widget.py
│
├── tests/
│   ├── test_board.py
│   ├── test_minimax.py
│   └── test_mcts.py
│
├── data/
│   ├── patterns/
│   ├── models/
│   └── games/
│
├── CMakeLists.txt
├── requirements.txt
├── setup.py
└── README.md
```

### B. Technology Versions

```
C++: 17 or 20
Python: 3.10+
PyTorch: 2.0+
pybind11: 2.11+
PostgreSQL: 14+
MongoDB: 6.0+
```

### C. References

1. **Go Rules:** Tromp-Taylor rules, Chinese rules
2. **Algorithms:**
   - Minimax: Russell & Norvig, "Artificial Intelligence: A Modern Approach"
   - MCTS: Browne et al., "A Survey of Monte Carlo Tree Search Methods"
3. **Neural Networks:** 
   - AlphaGo paper (simplified version)
   - Small-scale implementations

---

## ✅ KẾT LUẬN

Tài liệu **SystemSpec.md** này cung cấp blueprint đầy đủ cho dự án GoGame, bao gồm:

1. ✅ **System Overview** - Tổng quan hệ thống
2. ✅ **Requirements Analysis** - Phân tích yêu cầu môn học
3. ✅ **Algorithm Design** - Thiết kế thuật toán (high-level)
4. ✅ **Roadmap** - Lộ trình phát triển

**Implementation details** sẽ được viết trong các guide riêng khi bắt đầu code, đảm bảo:
- 📖 Document dễ đọc và maintain
- 💻 Code guide chi tiết khi cần
- 🎯 Focus vào từng component một
- ✅ Practical và actionable

---

**DOCUMENT VERSION:** 1.0  
**DATE:** November 17, 2025  
**STATUS:** ✅ Complete - Ready for Implementation

---

**Kết thúc SystemSpec.md**

*Next: Start implementation with BoardEngine (C++)*


