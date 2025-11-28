# 🧠 HƯỚNG DẪN TOÀN DIỆN: MACHINE LEARNING CHO GOGAME

## 📑 MỤC LỤC

1. [Tổng quan và Chiến lược](#1-tổng-quan-và-chiến-lược)
2. [Kiến trúc ML Chi tiết](#2-kiến-trúc-ml-chi-tiết)
3. [Dữ liệu Training](#3-dữ-liệu-training)
4. [Roadmap Triển khai](#4-roadmap-triển-khai)
5. [Technical Implementation](#5-technical-implementation)
6. [UI/UX Design](#6-uiux-design)
7. [Monetization Strategy](#7-monetization-strategy)
8. [Best Practices](#8-best-practices)

---

## 1. TỔNG QUAN VÀ CHIẾN LƯỢC

### 1.1. Tại sao KHÔNG chỉ làm "Gợi ý nước đi tốt nhất"?

#### ❌ Vấn đề với Simple Move Suggestion

| Khía cạnh | Vấn đề | Giải thích |
|-----------|--------|------------|
| **Thuật toán có sẵn** | ✅ MCTS đã làm tốt | Không cần ML để làm điều này |
| **Educational value** | ❌ Thiếu | Chỉ biết "đi đây" không biết "tại sao" |
| **Showcase ML** | ❌ Không đủ | ML có khả năng phân tích sâu hơn nhiều |
| **User value** | ❌ Thấp | Giống "cheating" hơn là "learning tool" |

#### ✅ Điều ML làm được mà thuật toán truyền thống KHÔNG THỂ

| Khả năng | Thuật toán truyền thống | Machine Learning | Giá trị |
|----------|------------------------|------------------|---------|
| **Gợi ý nước đi** | ✔️ MCTS làm tốt | ✔️ ML cũng làm được | ⭐ Không đặc biệt |
| **Win probability** | ❌ Khó, không chính xác | ✔️ Value Network rất tốt | ⭐⭐⭐⭐ |
| **Nhận diện patterns** | ❌ Phải hard-code | ✔️ Tự học từ data | ⭐⭐⭐⭐⭐ |
| **Phân tích ý đồ** | ❌ Không thể | ✔️ Có thể (attention) | ⭐⭐⭐⭐⭐ |
| **Territory prediction** | ❌ Heuristic sơ sài | ✔️ Chính xác cao | ⭐⭐⭐⭐⭐ |
| **Life/Death analysis** | ⚠️ Cases đơn giản | ✔️ Phức tạp cũng được | ⭐⭐⭐⭐ |
| **Visualize thinking** | ❌ Không có gì show | ✔️ Attention/heatmaps | ⭐⭐⭐⭐⭐ |

### 1.2. Concept: "AI TACTICAL VISION SYSTEM"

**Ý tưởng chính**: Cho người dùng "nhìn thấy" những gì AI nhìn thấy, không chỉ nói "đi đây".

#### Các Vision Modes

```
┌─────────────────────────────────────────────┐
│  👁️ VISION MODE: Territory Analysis        │
├─────────────────────────────────────────────┤
│  🟦 Black territory (85% confidence)        │
│  🟥 White territory (90% confidence)        │
│  🟨 Contested area (50-50)                  │
│  🟩 Influence zone                          │
│                                             │
│  Score Estimate:                            │
│  • Black: 65±5 points                       │
│  • White: 58±4 points                       │
│  • Win probability: 68%                     │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  ⚔️ VISION MODE: Tactical Threats          │
├─────────────────────────────────────────────┤
│  🔴 URGENT (1 move to capture)              │
│     └─ White group at Q15 (4 stones)       │
│                                             │
│  🟠 WEAK GROUPS (2-3 liberties)             │
│     └─ Black group at D4 (6 stones)        │
│                                             │
│  🟢 STRONG GROUPS (alive)                   │
│     └─ Black corner group (2 eyes)         │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  🎯 VISION MODE: Strategic Intent          │
├─────────────────────────────────────────────┤
│  AI predicts opponent will:                 │
│  1. Invade upper-right (70%)                │
│  2. Attack weak group at D4 (55%)           │
│  3. Build territory on left (40%)           │
│                                             │
│  Recommended counters:                      │
│  • Strengthen D4 first                      │
│  • Prepare defense in upper-right          │
└─────────────────────────────────────────────┘
```

### 1.3. Core Value Proposition

#### Giá trị mang lại cho người dùng

1. **Educational**: Hiểu sâu hơn về cờ vây, cải thiện kỹ năng
2. **Unique**: Tính năng độc đáo, chưa có ở competitor
3. **Visual**: Dễ hiểu qua visualization, không cần đọc nhiều text
4. **Actionable**: Không chỉ phân tích mà còn đưa ra hướng giải quyết
5. **Premium worth**: Đáng giá để trả tiền, không phải "nice to have"

#### Khác biệt với competitor

| Feature | Chess.com | Lichess | OGS | **GoGame (Ours)** |
|---------|-----------|---------|-----|-------------------|
| Move hints | ✅ | ✅ | ✅ | ✅ |
| Territory map | ❌ | ❌ | ⚠️ Basic | ✅ **AI-powered** |
| Threat detection | ❌ | ❌ | ❌ | ✅ **Real-time** |
| Intent analysis | ❌ | ❌ | ❌ | ✅ **Unique** |
| Visual heatmaps | ❌ | ❌ | ❌ | ✅ **Beautiful** |
| Post-game review | ⚠️ Basic | ⚠️ Basic | ⚠️ Basic | ✅ **AI-detailed** |

---

## 2. KIẾN TRÚC ML CHI TIẾT

### 2.1. Multi-Task Learning Architecture (Tối ưu)

```
┌────────────────────────────────────────────────────────┐
│           INPUT LAYER: Board State Features            │
│  • 17 planes × board_size × board_size                 │
│  • Stone positions, liberties, history, etc.           │
└────────────────────────────────────────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────┐
        │    Shared Feature Extractor     │
        │    (Lightweight ResNet-like)    │
        │  • 4 residual blocks             │
        │  • 64 base channels              │
        │  • BatchNorm + ReLU              │
        └─────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Threat     │  │   Attack     │  │   Intent     │
│  Detection   │  │ Opportunity  │  │ Recognition  │
│    Head      │  │    Head      │  │    Head      │
│              │  │              │  │              │
│ CNN → Conv1  │  │ CNN → Conv1  │  │ CNN → FC     │
│ Output:      │  │ Output:      │  │ Outputs:     │
│ • Heatmap    │  │ • Heatmap    │  │ • Class (5)  │
│ • Regions    │  │ • Regions    │  │ • Heatmap    │
└──────────────┘  └──────────────┘  └──────────────┘
        │                 │                 │
        └─────────────────┴─────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   Post-Processing     │
              │  • Region extraction  │
              │  • Confidence scoring │
              │  • Description gen    │
              └───────────────────────┘
```

### 2.2. Các Model Components Chi tiết

#### 2.2.1. Shared Backbone (✅ Đã implement)

**File**: `src/ml/models/shared_backbone.py`

**Kiến trúc**:
- Input: (batch, 17, board_size, board_size)
- Conv1: 17→64 channels, 5×5 kernel
- 4× Residual Blocks (64 channels)
- Output: (batch, 64, board_size, board_size)

**Đặc điểm**:
- ✅ Lightweight: ~500K parameters
- ✅ Fast inference: ~50ms trên CPU
- ✅ Reusable: Dùng chung cho tất cả tasks

**Code snippet**:
```python
class SharedBackbone(nn.Module):
    def __init__(self, input_planes=17, base_channels=64, num_res_blocks=4):
        super().__init__()
        self.conv1 = nn.Conv2d(input_planes, base_channels, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(base_channels) for _ in range(num_res_blocks)
        ])
```

#### 2.2.2. Threat Detection Head (✅ Đã implement)

**File**: `src/ml/models/threat_head.py`

**Mục đích**: Phát hiện các vùng bị đe dọa

**Output**:
```python
{
    "threat_map": Tensor[board_size, board_size],  # Values: 0-1
    "regions": [
        {
            "type": "weak_group" | "atari" | "false_eye" | "cutting_point",
            "positions": [[x1, y1], [x2, y2], ...],
            "severity": 0.0-1.0,
            "description": "Nhóm quân yếu, thiếu mắt",
            "recommendation": "Strengthen at [x, y]"
        }
    ]
}
```

**Cách hoạt động**:
1. Nhận features từ backbone (64 channels)
2. Conv layers để giảm xuống 1 channel
3. Sigmoid để normalize về [0, 1]
4. Post-processing để extract regions

#### 2.2.3. Attack Opportunity Head (✅ Đã implement)

**File**: `src/ml/models/attack_head.py`

**Mục đích**: Tìm cơ hội tấn công

**Output**:
```python
{
    "attack_map": Tensor[board_size, board_size],
    "opportunities": [
        {
            "type": "capture" | "cut" | "invasion" | "ladder",
            "position": [x, y],
            "confidence": 0.0-1.0,
            "target": "white_group_3",
            "expected_gain": 15,  # points
            "description": "Có thể bắt 3 quân trắng",
            "sequence": [[x1,y1], [x2,y2], ...]  # Suggested moves
        }
    ]
}
```

#### 2.2.4. Intent Recognition Head (✅ Đã implement)

**File**: `src/ml/models/intent_head.py`

**Mục đích**: Nhận biết ý định chiến lược

**Intent Classes**:
1. `territory` - Xây dựng lãnh thổ
2. `attack` - Tấn công đối thủ
3. `defense` - Phòng thủ
4. `connection` - Kết nối nhóm quân
5. `cut` - Cắt đứt đối thủ

**Output**:
```python
{
    "intent_logits": Tensor[5],  # Raw logits
    "intent_heatmap": Tensor[board_size, board_size],
    "predictions": {
        "primary_intent": "attack",
        "confidence": 0.85,
        "all_probabilities": {
            "territory": 0.10,
            "attack": 0.85,
            "defense": 0.03,
            "connection": 0.01,
            "cut": 0.01
        },
        "related_regions": [[x1,y1], [x2,y2], ...]
    }
}
```

#### 2.2.5. Position Evaluation Head (⏳ TODO)

**File**: `src/ml/models/evaluation_head.py` (cần implement)

**Mục đích**: Đánh giá tổng thể vị thế

**Output**:
```python
{
    "win_probability": 0.68,  # Black's chance to win
    "territory_map": Tensor[board_size, board_size],
    "influence_map": Tensor[board_size, board_size],
    "score_estimate": {
        "black": 65.5,
        "white": 58.0,
        "confidence": 0.85
    },
    "game_phase": "middle" | "opening" | "endgame"
}
```

### 2.3. Model Size & Performance

| Model Component | Parameters | Size (MB) | Inference Time (CPU) |
|----------------|------------|-----------|---------------------|
| Shared Backbone | ~500K | 2.0 | 50ms |
| Threat Head | ~150K | 0.6 | 10ms |
| Attack Head | ~150K | 0.6 | 10ms |
| Intent Head | ~200K | 0.8 | 15ms |
| Evaluation Head | ~250K | 1.0 | 20ms |
| **Total** | **~1.25M** | **~5MB** | **~105ms** |

**Với caching**: Response time < 500ms (bao gồm cả post-processing)

---

## 3. DỮ LIỆU TRAINING

### 3.1. Data Sources (Chi tiết)

#### 3.1.1. Self-Play Games (PRIMARY - 70% data)

**Ưu điểm**:
- ✅ Unlimited data generation
- ✅ Controlled quality
- ✅ Đa dạng positions
- ✅ Fresh data (không outdated)

**Cách thu thập**:
```bash
# Script đã có: src/ml/training/data_collector.py
python src/ml/training/data_collector.py \
  --board-size 9 \
  --num-games 1000 \
  --output data/training/self_play_9x9_1000.pt
```

**Expected output**:
- 1000 games × ~80 moves/game = 80,000 positions
- Mỗi position có: board_state, move, outcome

#### 3.1.2. Professional Games (SECONDARY - 20% data)

**Sources**:

1. **KGS Game Archive** (FREE) ⭐ RECOMMENDED
   - URL: https://u-go.net/gamerecords/
   - ~70,000 professional games
   - Format: SGF
   - Ranks: 1d - 9d professional
   - Download script:
   ```bash
   python scripts/download_kgs_games.py \
     --output data/raw/kgs/ \
     --min-rank 5d \
     --max-games 5000
   ```

2. **OGS API** (FREE)
   - URL: https://online-go.com/api/v1/games/
   - Recent games, various skill levels
   - Can filter by rank

3. **GoGoD Database** (PAID - Optional)
   - ~100,000 historical games
   - Very high quality
   - Cost: ~$40 one-time

#### 3.1.3. Annotated Positions (10% data - For validation)

**Cách tạo**:
1. Human annotation (slow but accurate)
2. MCTS evaluation (fast but approximate)
3. Hybrid approach (recommended)

**Script**:
```python
# scripts/create_annotations.py
python scripts/create_annotations.py \
  --input data/processed/positions.pt \
  --method hybrid \
  --output data/annotated/
```

### 3.2. Data Format (Standardized)

#### Training Sample Structure

```python
{
    # Core data
    "board_state": Tensor[17, board_size, board_size],  # Feature planes
    "metadata": {
        "game_id": "kgs_2023_001",
        "move_number": 42,
        "board_size": 19,
        "current_player": "B",
        "timestamp": "2023-01-15"
    },
    
    # Labels cho từng task
    "labels": {
        "threat_map": Tensor[board_size, board_size],      # 0-1 values
        "attack_map": Tensor[board_size, board_size],      # 0-1 values
        "intent": {
            "type": "attack",                               # One of 5 classes
            "confidence": 0.85,
            "region": [[x1,y1], [x2,y2], ...]
        },
        "evaluation": {
            "win_probability": 0.68,
            "territory_map": Tensor[board_size, board_size],
            "influence_map": Tensor[board_size, board_size]
        }
    },
    
    # Additional info
    "game_outcome": "B+12.5",
    "player_ranks": {"black": "5d", "white": "4d"}
}
```

### 3.3. Feature Engineering (17 Planes)

**Feature planes breakdown**:

| Plane | Description | Values |
|-------|-------------|--------|
| 0-1 | Current stones (Black, White) | 0 or 1 |
| 2-3 | Stones with 1 liberty | 0 or 1 |
| 4-5 | Stones with 2 liberties | 0 or 1 |
| 6-7 | Stones with 3+ liberties | 0 or 1 |
| 8-15 | History (last 4 moves, 2 planes each) | 0 or 1 |
| 16 | Turn indicator (1=Black, 0=White) | 0 or 1 |

**Code**:
```python
# Đã có trong: src/ml/features.py
def board_to_tensor(board, current_player):
    """Convert C++ Board to 17-plane tensor"""
    features = torch.zeros((17, board_size, board_size))
    
    # Plane 0-1: Current stones
    features[0] = (board == BLACK).float()
    features[1] = (board == WHITE).float()
    
    # Plane 2-7: Liberty counts
    # ... (implementation details)
    
    # Plane 8-15: History
    # ... (from move history)
    
    # Plane 16: Turn
    features[16] = 1.0 if current_player == BLACK else 0.0
    
    return features
```

### 3.4. Data Augmentation

**Techniques**:
1. **Rotation**: 90°, 180°, 270° → 4× data
2. **Reflection**: Horizontal, vertical → 2× data
3. **Color swap**: Black ↔ White → 2× data
4. **Combined**: Up to 16× augmentation

**Implementation**:
```python
def augment_position(board_tensor, labels):
    """Random augmentation"""
    # Rotation
    k = random.randint(0, 3)
    board_tensor = torch.rot90(board_tensor, k, dims=[1, 2])
    labels = rotate_labels(labels, k)
    
    # Reflection
    if random.random() > 0.5:
        board_tensor = torch.flip(board_tensor, dims=[2])
        labels = flip_labels(labels)
    
    # Color swap
    if random.random() > 0.5:
        board_tensor = swap_colors(board_tensor)
        labels = swap_label_colors(labels)
    
    return board_tensor, labels
```

**Khi nào dùng**:
- ✅ Training time: Luôn luôn
- ❌ Validation/Test: KHÔNG BAO GIỜ
- ⚠️ Production inference: Tùy trường hợp

---

# 🧠 HƯỚNG DẪN TOÀN DIỆN ML - PHẦN 2: ROADMAP & IMPLEMENTATION

## 4. ROADMAP TRIỂN KHAI CHI TIẾT

### 4.1. Timeline Overview

```
┌────────────────────────────────────────────────────────────┐
│              FULL ROADMAP (12-16 tuần)                     │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  PHASE 1: Data Collection & Prep (2-3 tuần)              │
│  ├─ Week 1: Data collection pipeline                      │
│  ├─ Week 2: Ground truth generation                       │
│  └─ Week 3: Dataset creation & validation                 │
│  Status: ✅ 40% Complete (data collector ready)           │
│                                                            │
│  PHASE 2: Model Architecture (1 tuần)                     │
│  ├─ Week 4: Architecture design & testing                 │
│  Status: ✅ 100% Complete (all models ready)              │
│                                                            │
│  PHASE 3: Training Pipeline (2-3 tuần)                    │
│  ├─ Week 5: Training infrastructure                       │
│  ├─ Week 6-7: Model training & tuning                     │
│  Status: ⏳ 0% (Next phase)                               │
│                                                            │
│  PHASE 4: Inference Service (1 tuần)                      │
│  ├─ Week 8: Model serving & API                           │
│  Status: ✅ 50% (service skeleton ready)                  │
│                                                            │
│  PHASE 5: Frontend Integration (2-3 tuần)                 │
│  ├─ Week 9-10: UI components                              │
│  ├─ Week 11: User interaction                             │
│  Status: ⏳ 0% (Pending)                                  │
│                                                            │
│  PHASE 6: Testing & Launch (2 tuần)                       │
│  ├─ Week 12-13: Beta testing & polish                     │
│  └─ Week 14: Soft launch                                  │
│  Status: ⏳ 0% (Pending)                                  │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

### 4.2. PHASE 1: Data Collection & Preparation (Chi tiết)

#### Week 1: Data Collection Pipeline

**Mục tiêu**: Có 10,000+ training samples từ self-play và professional games

**Tasks**:

1. **Self-Play Data Collection** (3 days)
   ```bash
   # Đã có script: src/ml/training/data_collector.py
   
   # Step 1: Generate games với board size khác nhau
   python src/ml/training/data_collector.py \
     --board-size 9 \
     --num-games 500 \
     --output data/training/self_play_9x9.pt
   
   python src/ml/training/data_collector.py \
     --board-size 13 \
     --num-games 300 \
     --output data/training/self_play_13x13.pt
   
   python src/ml/training/data_collector.py \
     --board-size 19 \
     --num-games 200 \
     --output data/training/self_play_19x19.pt
   
   # Expected: ~80,000 positions từ self-play
   ```

2. **Download Professional Games** (2 days)
   ```bash
   # Cần tạo script mới: scripts/download_kgs_games.py
   python scripts/download_kgs_games.py \
     --source kgs \
     --min-rank 5d \
     --max-games 5000 \
     --output data/raw/kgs/
   
   # Expected: 5,000 games × 200 moves = 1,000,000 positions
   ```

3. **Parse SGF Files** (2 days)
   ```bash
   # Cần tạo: scripts/parse_sgf.py
   python scripts/parse_sgf.py \
     --input data/raw/kgs/ \
     --output data/parsed/ \
     --board-sizes 9,13,19
   
   # Output format: PyTorch tensors
   ```

**Deliverables**:
- ✅ 500 games × 3 board sizes = 1,500 self-play games
- ✅ 5,000 professional games parsed
- ✅ Total: ~1M+ training positions

---

#### Week 2: Ground Truth Label Generation

**Mục tiêu**: Tạo labels cho threat detection, attack detection, intent

**Tasks**:

1. **Implement Label Generators** (4 days)

   **A. Threat Label Generator**
   ```python
   # File: src/ml/training/label_generator.py
   
   class ThreatLabelGenerator:
       """Generate threat labels using rule-based heuristics"""
       
       def generate_threat_map(self, board_state):
           """
           Returns: (board_size, board_size) tensor
           Values: 0.0 (safe) to 1.0 (critical threat)
           """
           threat_map = np.zeros((board_size, board_size))
           
           # Rule 1: Groups with 1 liberty → 1.0 (atari)
           for group in self.find_groups(board_state):
               if group.liberties == 1:
                   threat_map[group.positions] = 1.0
           
           # Rule 2: Groups with 2 liberties → 0.7
           for group in self.find_groups(board_state):
               if group.liberties == 2:
                   threat_map[group.positions] = 0.7
           
           # Rule 3: False eyes → 0.6
           false_eyes = self.detect_false_eyes(board_state)
           threat_map[false_eyes] = 0.6
           
           # Rule 4: Cutting points → 0.5
           cuts = self.detect_cutting_points(board_state)
           threat_map[cuts] = 0.5
           
           return threat_map
   ```

   **B. Attack Label Generator**
   ```python
   class AttackLabelGenerator:
       """Generate attack opportunity labels"""
       
       def generate_attack_map(self, board_state, current_player):
           attack_map = np.zeros((board_size, board_size))
           
           # Rule 1: Opponent in atari → 1.0
           opponent_groups = self.find_opponent_groups(board_state, current_player)
           for group in opponent_groups:
               if group.liberties == 1:
                   # Mark the capturing move
                   capture_point = group.last_liberty
                   attack_map[capture_point] = 1.0
           
           # Rule 2: Can cut → 0.8
           cut_points = self.find_cut_opportunities(board_state, current_player)
           attack_map[cut_points] = 0.8
           
           # Rule 3: Invasion points → 0.6
           invasions = self.find_invasion_points(board_state, current_player)
           attack_map[invasions] = 0.6
           
           # Rule 4: Ladder works → 0.7
           ladders = self.find_working_ladders(board_state, current_player)
           attack_map[ladders] = 0.7
           
           return attack_map
   ```

   **C. Intent Label Generator**
   ```python
   class IntentLabelGenerator:
       """Generate intent labels from move sequences"""
       
       INTENT_PATTERNS = {
           'territory': [
               'enclosure_3_3',
               'side_extension',
               'shimari'
           ],
           'attack': [
               'attach',
               'cut',
               'peep',
               'atari'
           ],
           'defense': [
               'connect',
               'add_eye_space',
               'escape'
           ],
           'connection': [
               'kosumi',
               'keima',
               'one_point_jump'
           ],
           'cut': [
               'wedge',
               'diagonal_cut',
               'contact_cut'
           ]
       }
       
       def generate_intent_label(self, board_state, move, prev_moves):
           """
           Analyze move pattern to determine intent
           """
           # Check against known patterns
           for intent_type, patterns in self.INTENT_PATTERNS.items():
               for pattern in patterns:
                   if self.matches_pattern(move, prev_moves, pattern):
                       return intent_type, 0.9  # High confidence
           
           # Fallback: heuristic analysis
           return self.heuristic_intent_analysis(board_state, move)
   ```

2. **Generate Labels for Dataset** (2 days)
   ```bash
   python src/ml/training/label_generator.py \
     --input data/parsed/ \
     --output data/labels/ \
     --tasks threat,attack,intent
   
   # This will process all positions and create labels
   ```

3. **Validate Labels** (1 day)
   ```python
   # Manual validation on sample
   python scripts/validate_labels.py \
     --input data/labels/ \
     --sample-size 100 \
     --visualize
   
   # Opens UI to manually check if labels make sense
   ```

**Deliverables**:
- ✅ Label generators cho 3 tasks
- ✅ Labeled dataset: 1M+ positions
- ✅ Validation report

---

#### Week 3: Dataset Creation & Validation

**Tasks**:

1. **Create PyTorch Dataset** (2 days)
   ```python
   # File: src/ml/training/dataset.py
   
   class GoPositionDataset(Dataset):
       """Dataset for Go position analysis"""
       
       def __init__(self, data_dir, board_size=19, augment=True):
           self.data_dir = data_dir
           self.board_size = board_size
           self.augment = augment
           
           # Load all data
           self.positions = []
           self.labels = []
           
           for file in Path(data_dir).glob("*.pt"):
               data = torch.load(file)
               self.positions.extend(data['positions'])
               self.labels.extend(data['labels'])
       
       def __len__(self):
           return len(self.positions)
       
       def __getitem__(self, idx):
           position = self.positions[idx]
           label = self.labels[idx]
           
           # Convert to tensor
           features = board_to_tensor(position)
           
           # Augmentation
           if self.augment:
               features, label = self.augment_data(features, label)
           
           return {
               'features': features,
               'threat_map': label['threat_map'],
               'attack_map': label['attack_map'],
               'intent': label['intent'],
           }
   ```

2. **Split Dataset** (1 day)
   ```python
   # Train/Val/Test: 80/10/10
   from sklearn.model_selection import train_test_split
   
   train_indices, temp_indices = train_test_split(
       range(len(dataset)), 
       test_size=0.2, 
       random_state=42
   )
   val_indices, test_indices = train_test_split(
       temp_indices, 
       test_size=0.5, 
       random_state=42
   )
   
   # Save splits
   torch.save({
       'train': train_indices,
       'val': val_indices,
       'test': test_indices
   }, 'data/splits.pt')
   ```

3. **Data Statistics & Validation** (2 days)
   ```bash
   python scripts/analyze_dataset.py \
     --input data/training/ \
     --output reports/dataset_analysis.html
   
   # Generates:
   # - Distribution plots
   # - Sample visualizations
   # - Quality metrics
   ```

**Deliverables**:
- ✅ PyTorch Dataset class
- ✅ Train/Val/Test splits
- ✅ Dataset analysis report
- ✅ ~800K train, ~100K val, ~100K test samples

---

### 4.3. PHASE 2: Model Architecture (✅ HOÀN THÀNH)

**Status**: Đã implement xong tất cả components

**Completed**:
- ✅ Shared Backbone (`src/ml/models/shared_backbone.py`)
- ✅ Threat Head (`src/ml/models/threat_head.py`)
- ✅ Attack Head (`src/ml/models/attack_head.py`)
- ✅ Intent Head (`src/ml/models/intent_head.py`)
- ✅ Multi-Task Model (`src/ml/models/multi_task_model.py`)

**Remaining**:
- ⏳ Evaluation Head (optional, có thể dùng Value Network hiện có)

**Tests**:
```bash
# Test all models
python src/ml/models/shared_backbone.py  # ✅ Pass
python src/ml/models/threat_head.py      # ✅ Pass
python src/ml/models/attack_head.py      # ✅ Pass
python src/ml/models/intent_head.py      # ✅ Pass
python src/ml/models/multi_task_model.py # ✅ Pass
```

---

### 4.4. PHASE 3: Training Pipeline (NEXT - ⏳ QUAN TRỌNG)

#### Week 5: Training Infrastructure Setup

**Tasks**:

1. **Training Script** (2 days)
   ```python
   # File: src/ml/training/train_multi_task.py
   
   import torch
   import torch.nn as nn
   from torch.utils.data import DataLoader
   from torch.utils.tensorboard import SummaryWriter
   from tqdm import tqdm
   
   class MultiTaskTrainer:
       def __init__(self, config):
           self.config = config
           self.device = torch.device(config.device)
           
           # Model
           self.model = MultiTaskModel(config.model_config).to(self.device)
           
           # Optimizer
           self.optimizer = torch.optim.AdamW(
               self.model.parameters(),
               lr=config.learning_rate,
               weight_decay=config.weight_decay
           )
           
           # Learning rate scheduler
           self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
               self.optimizer,
               T_max=config.num_epochs
           )
           
           # Loss functions với weights
           self.loss_weights = {
               'threat': 1.0,
               'attack': 1.0,
               'intent_class': 0.5,
               'intent_map': 0.5
           }
           
           # TensorBoard
           self.writer = SummaryWriter(config.log_dir)
       
       def compute_loss(self, outputs, labels):
           """Multi-task loss"""
           losses = {}
           
           # Threat detection loss (MSE)
           losses['threat'] = nn.MSELoss()(
               outputs['threat_map'],
               labels['threat_map']
           )
           
           # Attack detection loss (MSE)
           losses['attack'] = nn.MSELoss()(
               outputs['attack_map'],
               labels['attack_map']
           )
           
           # Intent classification loss (CrossEntropy)
           losses['intent_class'] = nn.CrossEntropyLoss()(
               outputs['intent_logits'],
               labels['intent_class']
           )
           
           # Intent heatmap loss (MSE)
           losses['intent_map'] = nn.MSELoss()(
               outputs['intent_heatmap'],
               labels['intent_heatmap']
           )
           
           # Total weighted loss
           total_loss = sum(
               self.loss_weights[k] * v 
               for k, v in losses.items()
           )
           
           return total_loss, losses
       
       def train_epoch(self, train_loader):
           self.model.train()
           total_loss = 0
           losses_dict = {k: 0 for k in self.loss_weights.keys()}
           
           pbar = tqdm(train_loader, desc='Training')
           for batch in pbar:
               # Move to device
               features = batch['features'].to(self.device)
               labels = {k: v.to(self.device) for k, v in batch.items() if k != 'features'}
               
               # Forward
               self.optimizer.zero_grad()
               outputs = self.model(features)
               loss, losses = self.compute_loss(outputs, labels)
               
               # Backward
               loss.backward()
               self.optimizer.step()
               
               # Track
               total_loss += loss.item()
               for k, v in losses.items():
                   losses_dict[k] += v.item()
               
               pbar.set_postfix({'loss': f'{loss.item():.4f}'})
           
           # Average
           avg_loss = total_loss / len(train_loader)
           avg_losses = {k: v/len(train_loader) for k, v in losses_dict.items()}
           
           return avg_loss, avg_losses
       
       def validate(self, val_loader):
           self.model.eval()
           total_loss = 0
           losses_dict = {k: 0 for k in self.loss_weights.keys()}
           
           with torch.no_grad():
               for batch in tqdm(val_loader, desc='Validating'):
                   features = batch['features'].to(self.device)
                   labels = {k: v.to(self.device) for k, v in batch.items() if k != 'features'}
                   
                   outputs = self.model(features)
                   loss, losses = self.compute_loss(outputs, labels)
                   
                   total_loss += loss.item()
                   for k, v in losses.items():
                       losses_dict[k] += v.item()
           
           avg_loss = total_loss / len(val_loader)
           avg_losses = {k: v/len(val_loader) for k, v in losses_dict.items()}
           
           return avg_loss, avg_losses
       
       def fit(self, train_loader, val_loader):
           best_val_loss = float('inf')
           patience_counter = 0
           
           for epoch in range(self.config.num_epochs):
               print(f'\n=== Epoch {epoch+1}/{self.config.num_epochs} ===')
               
               # Train
               train_loss, train_losses = self.train_epoch(train_loader)
               
               # Validate
               val_loss, val_losses = self.validate(val_loader)
               
               # Learning rate step
               self.scheduler.step()
               
               # Log
               self.writer.add_scalar('epoch/train_loss', train_loss, epoch)
               self.writer.add_scalar('epoch/val_loss', val_loss, epoch)
               self.writer.add_scalar('epoch/lr', self.scheduler.get_last_lr()[0], epoch)
               
               for k in self.loss_weights.keys():
                   self.writer.add_scalar(f'train/{k}_loss', train_losses[k], epoch)
                   self.writer.add_scalar(f'val/{k}_loss', val_losses[k], epoch)
               
               print(f'Train Loss: {train_loss:.4f}')
               print(f'Val Loss: {val_loss:.4f}')
               
               # Save best model
               if val_loss < best_val_loss - self.config.min_delta:
                   best_val_loss = val_loss
                   patience_counter = 0
                   self.save_checkpoint(f'best_model_epoch_{epoch}.pt')
                   print(f'✓ Saved best model (val_loss: {val_loss:.4f})')
               else:
                   patience_counter += 1
               
               # Early stopping
               if patience_counter >= self.config.patience:
                   print(f'Early stopping at epoch {epoch}')
                   break
               
               # Save periodic checkpoint
               if (epoch + 1) % 10 == 0:
                   self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
           
           self.writer.close()
       
       def save_checkpoint(self, filename):
           checkpoint = {
               'model_state': self.model.state_dict(),
               'optimizer_state': self.optimizer.state_dict(),
               'config': self.config.__dict__,
               'model_config': self.model.config.__dict__
           }
           torch.save(checkpoint, Path(self.config.checkpoint_dir) / filename)
   ```

2. **Config File** (1 day)
   ```python
   # File: src/ml/training/config.py
   
   from dataclasses import dataclass
   from pathlib import Path
   
   @dataclass
   class TrainingConfig:
       # Model
       model_config: MultiTaskConfig = MultiTaskConfig()
       
       # Training
       batch_size: int = 32
       num_epochs: int = 50
       learning_rate: float = 1e-3
       weight_decay: float = 1e-4
       
       # Device
       device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
       num_workers: int = 4
       
       # Paths
       data_dir: Path = Path('data/training')
       checkpoint_dir: Path = Path('checkpoints')
       log_dir: Path = Path('logs')
       
       # Early stopping
       patience: int = 10
       min_delta: float = 1e-4
       
       # Data
       board_sizes: list = (9, 13, 19)
       augment_train: bool = True
       augment_val: bool = False
   ```

3. **Run Training** (remaining days)
   ```bash
   # Start training
   python src/ml/training/train_multi_task.py \
     --config configs/training_config.yaml \
     --board-size 9
   
   # Monitor with TensorBoard
   tensorboard --logdir logs/
   ```

**Deliverables**:
- ✅ Training pipeline complete
- ✅ TensorBoard logging
- ✅ Checkpoint management

---

#### Week 6-7: Model Training & Hyperparameter Tuning

**Training Plan**:

1. **Phase 1: 9×9 Board** (3 days)
   - Fast iteration
   - Validate concept
   - Expected: 70%+ validation accuracy

2. **Phase 2: 13×13 Board** (2 days)
   - Scale up
   - Expected: 65%+ validation accuracy

3. **Phase 3: 19×19 Board** (5 days)
   - Full training
   - Expected: 60%+ validation accuracy
   - Most time-consuming

**Hyperparameter Tuning**:
```python
# Grid search / Random search
hyperparams = {
    'learning_rate': [1e-3, 5e-4, 1e-4],
    'batch_size': [16, 32, 64],
    'loss_weights': {
        'threat': [0.5, 1.0, 2.0],
        'attack': [0.5, 1.0, 2.0],
        'intent': [0.5, 1.0, 2.0]
    }
}

# Use Optuna or Ray Tune for auto-tuning
import optuna

def objective(trial):
    lr = trial.suggest_float('lr', 1e-4, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    # ... train model
    return val_loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)
```

**Deliverables**:
- ✅ Trained models cho 3 board sizes
- ✅ Training logs và metrics
- ✅ Best hyperparameters documented

---

# 🧠 HƯỚNG DẪN TOÀN DIỆN ML - PHẦN 3: FRONTEND & MONETIZATION

## 5. PHASE 4 & 5: INFERENCE SERVICE & FRONTEND

### 5.1. PHASE 4: Inference Service (Week 8)

#### 5.1.1. Backend API Endpoints

**File**: Update `backend/app/routers/ml.py`

```python
from fastapi import APIRouter, Depends, HTTPException
from typing import Annotated
from uuid import UUID

router = APIRouter()

@router.post("/analyze-position")
async def analyze_position(
    match_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    ml_service: Annotated[MLAnalysisService, Depends(get_ml_analysis_service)],
    match_service: Annotated[MatchService, Depends(get_match_service)],
    premium_service: Annotated[PremiumService, Depends(get_premium_service)]
):
    """
    Comprehensive ML analysis của vị trí hiện tại
    
    Cost: 50 coins
    """
    # Check coins
    if current_user.coins < 50:
        raise HTTPException(status_code=402, detail="Không đủ coins")
    
    # Get match
    match = match_service.get_match(match_id)
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    
    # Check permission
    if current_user.id not in [match.black_player_id, match.white_player_id]:
        raise HTTPException(status_code=403, detail="Not your match")
    
    # Load board
    board = await match_service._get_or_create_board(match)
    current_player = board.current_player()
    
    # Run ML analysis
    analysis = await ml_service.analyze_position(board, current_player)
    
    # Deduct coins
    coin_service.add_transaction(
        current_user, 
        -50, 
        "spend", 
        source="ml_analysis"
    )
    
    # Save to database
    await save_analysis_result(match_id, current_user.id, analysis)
    
    return {
        "status": "success",
        "coins_spent": 50,
        "coins_remaining": current_user.coins - 50,
        "analysis": analysis
    }


@router.get("/analysis-history/{match_id}")
async def get_analysis_history(
    match_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
):
    """Lấy lịch sử phân tích của match"""
    analyses = await get_saved_analyses(match_id, current_user.id)
    return {"analyses": analyses}


@router.post("/quick-threats")
async def quick_threats(
    match_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    ml_service: Annotated[MLAnalysisService, Depends(get_ml_analysis_service)],
    match_service: Annotated[MatchService, Depends(get_match_service)],
):
    """
    Quick threat detection (cheaper)
    
    Cost: 10 coins
    """
    # Similar flow but only return threat analysis
    # ...
    
    return {
        "threats": analysis["threats"],
        "coins_spent": 10
    }
```

#### 5.1.2. ML Analysis Service Enhancement

**File**: Update `backend/app/services/ml_analysis_service.py`

```python
class MLAnalysisService:
    """Enhanced ML Analysis Service"""
    
    def __init__(self, model_path: Optional[Path] = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = None
        self.model_loaded = False
        self.cache = {}  # In-memory cache
        
        if model_path and model_path.exists():
            self.load_model(model_path)
    
    async def analyze_position(
        self, 
        board: "go.Board", 
        current_player: "go.Color",
        tasks: list = ["threats", "attacks", "intent", "evaluation"]
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis with caching
        """
        # Generate cache key
        cache_key = self._get_cache_key(board, current_player)
        
        # Check cache
        if cache_key in self.cache:
            logger.debug(f"Cache hit: {cache_key}")
            return self.cache[cache_key]
        
        # Run ML inference
        if self.model_loaded:
            result = await self._ml_inference(board, current_player, tasks)
        else:
            result = await self._fallback_analysis(board, current_player)
        
        # Add descriptions
        result = self._add_descriptions(result, board)
        
        # Cache result (TTL: 1 hour)
        self.cache[cache_key] = result
        asyncio.create_task(self._expire_cache(cache_key, ttl=3600))
        
        return result
    
    async def _ml_inference(
        self, 
        board: "go.Board", 
        current_player: "go.Color",
        tasks: list
    ) -> Dict[str, Any]:
        """Run ML model inference"""
        # Convert board to tensor
        board_tensor = board_to_tensor(board, current_player)
        board_tensor = board_tensor.unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            predictions = self.model.predict(board_tensor)
        
        # Post-process
        result = {}
        
        if "threats" in tasks:
            result["threats"] = self._process_threats(
                predictions["threat_map"][0],
                board
            )
        
        if "attacks" in tasks:
            result["attacks"] = self._process_attacks(
                predictions["attack_map"][0],
                board
            )
        
        if "intent" in tasks:
            result["intent"] = self._process_intent(
                predictions["intent"],
                board
            )
        
        return result
    
    def _process_threats(
        self, 
        threat_map: torch.Tensor, 
        board: "go.Board"
    ) -> Dict[str, Any]:
        """Process threat detection output"""
        threat_map_np = threat_map.cpu().numpy()
        board_size = board.size()
        
        # Extract regions
        regions = self._extract_regions(
            threat_map_np, 
            threshold=0.6, 
            board_size=board_size
        )
        
        # Classify threats
        classified_regions = []
        for region in regions:
            threat_type = self._classify_threat(region, board)
            classified_regions.append({
                "type": threat_type,
                "positions": region["positions"],
                "severity": region["severity"],
                "description": self._generate_threat_description(
                    threat_type, 
                    region, 
                    board
                ),
                "recommendation": self._generate_threat_recommendation(
                    threat_type, 
                    region, 
                    board
                )
            })
        
        return {
            "heatmap": threat_map_np.tolist(),
            "regions": classified_regions,
            "summary": {
                "critical": len([r for r in classified_regions if r["severity"] > 0.8]),
                "moderate": len([r for r in classified_regions if 0.5 < r["severity"] <= 0.8]),
                "low": len([r for r in classified_regions if r["severity"] <= 0.5])
            }
        }
    
    def _classify_threat(self, region: Dict, board: "go.Board") -> str:
        """Classify type of threat"""
        positions = region["positions"]
        severity = region["severity"]
        
        # Check if it's a group
        if len(positions) > 1:
            # Check liberties
            liberties = self._count_group_liberties(positions, board)
            
            if liberties == 1:
                return "atari"
            elif liberties == 2:
                return "weak_group"
            elif self._is_false_eye(positions, board):
                return "false_eye"
            else:
                return "weak_group"
        else:
            # Single point threat
            if self._is_cutting_point(positions[0], board):
                return "cutting_point"
            else:
                return "weak_point"
    
    def _generate_threat_description(
        self, 
        threat_type: str, 
        region: Dict, 
        board: "go.Board"
    ) -> str:
        """Generate human-readable description"""
        positions = region["positions"]
        severity = region["severity"]
        
        descriptions = {
            "atari": f"Nhóm quân đang bị atari tại {self._format_positions(positions)}. Cần xử lý ngay!",
            "weak_group": f"Nhóm quân yếu ({len(positions)} quân) với {self._count_group_liberties(positions, board)} liberties. Cần củng cố.",
            "false_eye": f"Mắt giả tại {self._format_positions(positions)}. Không phải mắt thật.",
            "cutting_point": f"Điểm cắt nguy hiểm tại {self._format_positions(positions)}. Đối thủ có thể tấn công.",
            "weak_point": f"Điểm yếu tại {self._format_positions(positions)}. Cần chú ý."
        }
        
        return descriptions.get(threat_type, "Mối đe dọa được phát hiện")
    
    def _generate_threat_recommendation(
        self, 
        threat_type: str, 
        region: Dict, 
        board: "go.Board"
    ) -> str:
        """Generate recommendation"""
        positions = region["positions"]
        
        recommendations = {
            "atari": "Chạy ngay hoặc nối để thoát atari",
            "weak_group": "Tăng liberties bằng cách mở rộng hoặc kết nối với nhóm khác",
            "false_eye": "Tạo mắt thật ở vị trí khác",
            "cutting_point": "Bảo vệ bằng cách nối hoặc tạo hình mạnh hơn",
            "weak_point": "Củng cố vùng này hoặc chấp nhận rủi ro"
        }
        
        return recommendations.get(threat_type, "Xem xét kỹ vùng này")
    
    def _process_attacks(
        self, 
        attack_map: torch.Tensor, 
        board: "go.Board"
    ) -> Dict[str, Any]:
        """Process attack opportunity output"""
        attack_map_np = attack_map.cpu().numpy()
        board_size = board.size()
        
        # Find hot spots
        opportunities = []
        
        # Method 1: Peak detection
        peaks = self._find_peaks(attack_map_np, threshold=0.7)
        
        for peak in peaks:
            x, y = peak
            confidence = float(attack_map_np[y, x])
            
            # Analyze why this is an attack opportunity
            attack_type, target = self._analyze_attack(peak, board)
            
            opportunities.append({
                "type": attack_type,
                "position": [x, y],
                "confidence": confidence,
                "target": target,
                "expected_gain": self._estimate_attack_value(peak, board),
                "description": self._generate_attack_description(
                    attack_type, 
                    peak, 
                    target, 
                    board
                ),
                "sequence": self._generate_attack_sequence(peak, board)
            })
        
        # Sort by expected gain
        opportunities.sort(key=lambda x: x["expected_gain"], reverse=True)
        
        return {
            "heatmap": attack_map_np.tolist(),
            "opportunities": opportunities[:10],  # Top 10
            "summary": {
                "total": len(opportunities),
                "high_value": len([o for o in opportunities if o["expected_gain"] > 10]),
                "medium_value": len([o for o in opportunities if 5 < o["expected_gain"] <= 10])
            }
        }
    
    def _analyze_attack(self, position: tuple, board: "go.Board") -> tuple:
        """Analyze type of attack and target"""
        x, y = position
        
        # Check if it's a capture
        if self._is_capture_move(position, board):
            target = self._find_capturable_group(position, board)
            return "capture", target
        
        # Check if it's a cut
        if self._is_cutting_move(position, board):
            return "cut", "opponent_connection"
        
        # Check if it's an invasion
        if self._is_invasion_point(position, board):
            return "invasion", "opponent_territory"
        
        # Check if it's a ladder
        if self._is_ladder_move(position, board):
            target = self._find_ladder_target(position, board)
            return "ladder", target
        
        return "attack", "unknown"
    
    def _process_intent(
        self, 
        intent_data: Dict, 
        board: "go.Board"
    ) -> Dict[str, Any]:
        """Process intent recognition output"""
        return {
            "primary_intent": intent_data["names"][0],
            "confidence": float(intent_data["probabilities"][0].max()),
            "all_intents": [
                {
                    "type": intent_data["names"][0],
                    "probability": float(prob),
                    "description": self._get_intent_description(
                        intent_data["names"][0]
                    )
                }
                for prob in intent_data["probabilities"][0]
            ],
            "heatmap": intent_data["heatmap"][0].cpu().numpy().tolist(),
            "strategic_advice": self._generate_strategic_advice(
                intent_data["names"][0],
                board
            )
        }
    
    def _get_intent_description(self, intent_type: str) -> str:
        """Get description for intent type"""
        descriptions = {
            "territory": "Xây dựng và mở rộng lãnh thổ",
            "attack": "Tấn công nhóm quân đối thủ",
            "defense": "Bảo vệ nhóm quân của mình",
            "connection": "Kết nối các nhóm quân",
            "cut": "Cắt đứt kết nối của đối thủ"
        }
        return descriptions.get(intent_type, "Không xác định")
    
    def _generate_strategic_advice(
        self, 
        intent_type: str, 
        board: "go.Board"
    ) -> str:
        """Generate strategic advice based on detected intent"""
        advice = {
            "territory": "Đối thủ đang xây dựng lãnh thổ. Xem xét xâm nhập hoặc giảm ảnh hưởng.",
            "attack": "Đối thủ đang chuẩn bị tấn công. Củng cố phòng thủ hoặc phản công.",
            "defense": "Đối thủ đang phòng thủ. Cơ hội để mở rộng ảnh hưởng ở vùng khác.",
            "connection": "Đối thủ đang cố kết nối. Xem xét cắt đứt nếu có lợi.",
            "cut": "Đối thủ đang cố cắt. Bảo vệ kết nối của bạn."
        }
        return advice.get(intent_type, "Theo dõi tình hình")
    
    def _get_cache_key(self, board: "go.Board", current_player: "go.Color") -> str:
        """Generate cache key from board state"""
        board_hash = board.zobrist_hash()
        player_char = "B" if current_player == go.Color.Black else "W"
        return f"{board_hash}_{player_char}"
    
    async def _expire_cache(self, key: str, ttl: int):
        """Expire cache after TTL seconds"""
        await asyncio.sleep(ttl)
        if key in self.cache:
            del self.cache[key]
```

#### 5.1.3. Response Format Example

```json
{
  "status": "success",
  "coins_spent": 50,
  "coins_remaining": 150,
  "analysis": {
    "threats": {
      "heatmap": [[0.1, 0.3, ...], ...],
      "regions": [
        {
          "type": "atari",
          "positions": [[5, 5], [5, 6]],
          "severity": 0.95,
          "description": "Nhóm quân đang bị atari tại F4-F5. Cần xử lý ngay!",
          "recommendation": "Chạy ngay hoặc nối để thoát atari"
        },
        {
          "type": "weak_group",
          "positions": [[3, 3], [3, 4], [4, 3]],
          "severity": 0.7,
          "description": "Nhóm quân yếu (3 quân) với 2 liberties. Cần củng cố.",
          "recommendation": "Tăng liberties bằng cách mở rộng"
        }
      ],
      "summary": {
        "critical": 1,
        "moderate": 2,
        "low": 0
      }
    },
    "attacks": {
      "heatmap": [[0.2, 0.5, ...], ...],
      "opportunities": [
        {
          "type": "capture",
          "position": [5, 7],
          "confidence": 0.92,
          "target": "white_group_3",
          "expected_gain": 15,
          "description": "Có thể bắt 4 quân trắng tại F6",
          "sequence": [[5, 7], [6, 7], [5, 8]]
        },
        {
          "type": "cut",
          "position": [8, 9],
          "confidence": 0.85,
          "target": "opponent_connection",
          "expected_gain": 12,
          "description": "Cắt đứt kết nối giữa 2 nhóm trắng",
          "sequence": [[8, 9]]
        }
      ],
      "summary": {
        "total": 5,
        "high_value": 2,
        "medium_value": 3
      }
    },
    "intent": {
      "primary_intent": "attack",
      "confidence": 0.85,
      "all_intents": [
        {
          "type": "attack",
          "probability": 0.85,
          "description": "Tấn công nhóm quân đối thủ"
        },
        {
          "type": "territory",
          "probability": 0.10,
          "description": "Xây dựng và mở rộng lãnh thổ"
        }
      ],
      "heatmap": [[0.1, 0.2, ...], ...],
      "strategic_advice": "Đối thủ đang chuẩn bị tấn công. Củng cố phòng thủ hoặc phản công."
    }
  }
}
```

---

### 5.2. PHASE 5: Frontend Integration (Week 9-11)

#### 5.2.1. Main Analysis Component

**File**: `frontend-web/src/components/MLAnalysisPanel.jsx`

```jsx
import React, { useState, useEffect } from 'react';
import { Canvas } from 'react-konva';
import { Button } from './ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Alert, AlertDescription } from './ui/alert';

export const MLAnalysisPanel = ({ matchId, onAnalyze }) => {
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [userCoins, setUserCoins] = useState(0);
  const [visualMode, setVisualMode] = useState('threats');
  
  const ANALYSIS_COST = 50;
  
  useEffect(() => {
    fetchUserCoins();
  }, []);
  
  const fetchUserCoins = async () => {
    const response = await fetch('/api/users/me');
    const data = await response.json();
    setUserCoins(data.coins);
  };
  
  const handleAnalyze = async () => {
    if (userCoins < ANALYSIS_COST) {
      setError('Không đủ coins! Cần 50 coins để phân tích.');
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`/api/ml/analyze-position`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ match_id: matchId })
      });
      
      if (!response.ok) {
        throw new Error('Analysis failed');
      }
      
      const data = await response.json();
      setAnalysis(data.analysis);
      setUserCoins(data.coins_remaining);
      
      if (onAnalyze) {
        onAnalyze(data.analysis);
      }
    } catch (err) {
      setError('Phân tích thất bại. Vui lòng thử lại.');
      console.error('Analysis error:', err);
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div className="ml-analysis-panel bg-white rounded-lg shadow-lg p-4">
      {/* Header */}
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-bold">🧠 AI Analysis</h3>
        <div className="flex items-center gap-4">
          <div className="text-sm">
            💰 <span className="font-bold">{userCoins}</span> coins
          </div>
          <Button
            onClick={handleAnalyze}
            disabled={loading || userCoins < ANALYSIS_COST}
            className="bg-blue-500 hover:bg-blue-600"
          >
            {loading ? (
              <>
                <span className="animate-spin mr-2">⏳</span>
                Đang phân tích...
              </>
            ) : (
              `🧠 Phân tích AI (${ANALYSIS_COST} coins)`
            )}
          </Button>
        </div>
      </div>
      
      {/* Error */}
      {error && (
        <Alert variant="destructive" className="mb-4">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}
      
      {/* Analysis Results */}
      {analysis ? (
        <Tabs value={visualMode} onValueChange={setVisualMode}>
          <TabsList className="grid w-full grid-cols-3 mb-4">
            <TabsTrigger value="threats">
              🔴 Mối đe dọa ({analysis.threats?.summary?.critical || 0})
            </TabsTrigger>
            <TabsTrigger value="attacks">
              ⚔️ Cơ hội tấn công ({analysis.attacks?.summary?.high_value || 0})
            </TabsTrigger>
            <TabsTrigger value="intent">
              🎯 Ý định ({analysis.intent?.primary_intent || 'N/A'})
            </TabsTrigger>
          </TabsList>
          
          <TabsContent value="threats">
            <ThreatAnalysisView threats={analysis.threats} />
          </TabsContent>
          
          <TabsContent value="attacks">
            <AttackAnalysisView attacks={analysis.attacks} />
          </TabsContent>
          
          <TabsContent value="intent">
            <IntentAnalysisView intent={analysis.intent} />
          </TabsContent>
        </Tabs>
      ) : (
        <div className="text-center py-8 text-gray-500">
          <p className="mb-4">Nhấn "Phân tích AI" để xem phân tích chi tiết</p>
          <ul className="text-sm text-left inline-block space-y-2">
            <li>✓ Phát hiện mối đe dọa nguy hiểm</li>
            <li>✓ Tìm cơ hội tấn công</li>
            <li>✓ Nhận biết ý định đối thủ</li>
            <li>✓ Gợi ý chiến lược</li>
          </ul>
        </div>
      )}
    </div>
  );
};
```

#### 5.2.2. Threat Visualization Component

**File**: `frontend-web/src/components/ThreatVisualization.jsx`

```jsx
import React from 'react';
import { Layer, Rect, Circle, Text, Group } from 'react-konva';

export const ThreatVisualization = ({ threatData, boardSize = 19, cellSize = 30 }) => {
  const { heatmap, regions } = threatData;
  
  return (
    <Layer>
      {/* Heatmap overlay */}
      {heatmap.map((row, y) =>
        row.map((value, x) => {
          if (value < 0.3) return null;  // Skip low values
          
          const alpha = Math.min(value, 0.7);  // Max 70% opacity
          const color = getSeverityColor(value);
          
          return (
            <Rect
              key={`threat-${x}-${y}`}
              x={x * cellSize}
              y={y * cellSize}
              width={cellSize}
              height={cellSize}
              fill={color}
              opacity={alpha}
            />
          );
        })
      )}
      
      {/* Region highlights */}
      {regions.map((region, idx) => (
        <RegionHighlight
          key={`region-${idx}`}
          region={region}
          cellSize={cellSize}
          boardSize={boardSize}
        />
      ))}
    </Layer>
  );
};

const RegionHighlight = ({ region, cellSize, boardSize }) => {
  const { type, positions, severity, description } = region;
  
  // Calculate bounding box
  const xs = positions.map(p => p[0]);
  const ys = positions.map(p => p[1]);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  
  const boxX = minX * cellSize - 5;
  const boxY = minY * cellSize - 5;
  const boxWidth = (maxX - minX + 1) * cellSize + 10;
  const boxHeight = (maxY - minY + 1) * cellSize + 10;
  
  const color = getSeverityColor(severity);
  const icon = getTypeIcon(type);
  
  return (
    <Group>
      {/* Bounding box */}
      <Rect
        x={boxX}
        y={boxY}
        width={boxWidth}
        height={boxHeight}
        stroke={color}
        strokeWidth={3}
        dash={[5, 5]}
        listening={false}
      />
      
      {/* Icon/Label */}
      <Circle
        x={boxX + boxWidth / 2}
        y={boxY - 15}
        radius={12}
        fill={color}
      />
      <Text
        x={boxX + boxWidth / 2 - 8}
        y={boxY - 20}
        text={icon}
        fontSize={16}
      />
      
      {/* Tooltip on hover */}
      <Rect
        x={boxX}
        y={boxY}
        width={boxWidth}
        height={boxHeight}
        fill="transparent"
        onMouseEnter={(e) => {
          const container = e.target.getStage().container();
          container.style.cursor = 'pointer';
          // Show tooltip
          showTooltip(description, e.evt.clientX, e.evt.clientY);
        }}
        onMouseLeave={(e) => {
          const container = e.target.getStage().container();
          container.style.cursor = 'default';
          hideTooltip();
        }}
      />
    </Group>
  );
};

const getSeverityColor = (severity) => {
  if (severity > 0.8) return '#ef4444';  // Red
  if (severity > 0.5) return '#f97316';  // Orange
  return '#eab308';  // Yellow
};

const getTypeIcon = (type) => {
  const icons = {
    'atari': '🚨',
    'weak_group': '⚠️',
    'false_eye': '👁️',
    'cutting_point': '✂️',
    'weak_point': '⚡'
  };
  return icons[type] || '❗';
};
```

#### 5.2.3. Threat Analysis Detail View

**File**: Thêm vào `ThreatVisualization.jsx`

```jsx
export const ThreatAnalysisView = ({ threats }) => {
  const { regions, summary } = threats;
  
  // Group by severity
  const critical = regions.filter(r => r.severity > 0.8);
  const moderate = regions.filter(r => r.severity > 0.5 && r.severity <= 0.8);
  const low = regions.filter(r => r.severity <= 0.5);
  
  return (
    <div className="threat-analysis-view space-y-4">
      {/* Summary */}
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-red-50 p-3 rounded border border-red-200">
          <div className="text-red-600 font-bold text-2xl">{summary.critical}</div>
          <div className="text-sm text-red-700">Khẩn cấp</div>
        </div>
        <div className="bg-orange-50 p-3 rounded border border-orange-200">
          <div className="text-orange-600 font-bold text-2xl">{summary.moderate}</div>
          <div className="text-sm text-orange-700">Trung bình</div>
        </div>
        <div className="bg-yellow-50 p-3 rounded border border-yellow-200">
          <div className="text-yellow-600 font-bold text-2xl">{summary.low}</div>
          <div className="text-sm text-yellow-700">Thấp</div>
        </div>
      </div>
      
      {/* Critical Threats */}
      {critical.length > 0 && (
        <div>
          <h4 className="font-bold text-red-600 mb-2 flex items-center gap-2">
            🚨 Mối đe dọa khẩn cấp
          </h4>
          <div className="space-y-2">
            {critical.map((threat, idx) => (
              <ThreatCard key={idx} threat={threat} severity="critical" />
            ))}
          </div>
        </div>
      )}
      
      {/* Moderate Threats */}
      {moderate.length > 0 && (
        <div>
          <h4 className="font-bold text-orange-600 mb-2 flex items-center gap-2">
            ⚠️ Mối đe dọa trung bình
          </h4>
          <div className="space-y-2">
            {moderate.map((threat, idx) => (
              <ThreatCard key={idx} threat={threat} severity="moderate" />
            ))}
          </div>
        </div>
      )}
      
      {/* Low Threats */}
      {low.length > 0 && (
        <details className="mt-4">
          <summary className="cursor-pointer font-semibold text-yellow-600">
            ⚡ Mối đe dọa thấp ({low.length})
          </summary>
          <div className="mt-2 space-y-2">
            {low.map((threat, idx) => (
              <ThreatCard key={idx} threat={threat} severity="low" />
            ))}
          </div>
        </details>
      )}
    </div>
  );
};

const ThreatCard = ({ threat, severity }) => {
  const { type, positions, description, recommendation } = threat;
  
  const severityColors = {
    critical: 'border-red-300 bg-red-50',
    moderate: 'border-orange-300 bg-orange-50',
    low: 'border-yellow-300 bg-yellow-50'
  };
  
  const icon = getTypeIcon(type);
  
  return (
    <div className={`p-3 rounded border ${severityColors[severity]}`}>
      <div className="flex items-start gap-3">
        <div className="text-2xl">{icon}</div>
        <div className="flex-1">
          <div className="font-medium text-sm mb-1">
            {formatThreatType(type)}
          </div>
          <p className="text-sm text-gray-700 mb-2">
            {description}
          </p>
          <div className="bg-white bg-opacity-50 p-2 rounded text-sm">
            <span className="font-medium">💡 Gợi ý:</span> {recommendation}
          </div>
          <div className="text-xs text-gray-500 mt-1">
            Vị trí: {formatPositions(positions)}
          </div>
        </div>
      </div>
    </div>
  );
};

const formatThreatType = (type) => {
  const names = {
    'atari': 'Atari (nguy cấp)',
    'weak_group': 'Nhóm quân yếu',
    'false_eye': 'Mắt giả',
    'cutting_point': 'Điểm cắt',
    'weak_point': 'Điểm yếu'
  };
  return names[type] || type;
};

const formatPositions = (positions) => {
  const letters = 'ABCDEFGHJKLMNOPQRST';
  return positions
    .slice(0, 3)
    .map(([x, y]) => `${letters[x]}${19-y}`)
    .join(', ') + (positions.length > 3 ? '...' : '');
};
```

# 🧠 HƯỚNG DẪN TOÀN DIỆN ML - PHẦN 4: MONETIZATION & BEST PRACTICES

## 6. MONETIZATION STRATEGY (Chi tiết)

### 6.1. Pricing Model

#### 6.1.1. Coin Packages

```javascript
const COIN_PACKAGES = {
  starter: {
    coins: 50,
    price: 10000,  // VND (~$0.40)
    bonus: 0,
    best_for: "Thử nghiệm"
  },
  basic: {
    coins: 150,
    price: 25000,  // ~$1
    bonus: 10,  // +10 coins free
    best_for: "Casual players",
    popular: false
  },
  standard: {
    coins: 500,
    price: 75000,  // ~$3
    bonus: 50,  // +50 coins (10% bonus)
    best_for: "Regular players",
    popular: true  // Most popular
  },
  premium: {
    coins: 1500,
    price: 200000,  // ~$8
    bonus: 300,  // +300 coins (20% bonus)
    best_for: "Serious players",
    popular: false
  },
  ultimate: {
    coins: 5000,
    price: 600000,  // ~$24
    bonus: 1500,  // +1500 coins (30% bonus)
    best_for: "Professionals",
    popular: false
  }
};
```

#### 6.1.2. Feature Pricing

| Feature | Cost (coins) | Description | Estimated usage |
|---------|--------------|-------------|-----------------|
| **Quick Threats** | 10 | Chỉ xem threats, không có attacks/intent | High (50% users) |
| **Full Analysis** | 50 | Threats + Attacks + Intent | Medium (30% users) |
| **Post-Game Review** | 30 | Phân tích toàn bộ ván đấu | Medium (25% users) |
| **Training Mode** | 5/puzzle | Giải tsumego problems | High (40% users) |
| **Opening Book** | 100 | Unlock 1 lần (permanent) | Low (10% users) |
| **AI Hint** | 15 | Gợi ý nước đi + explanation | Medium (35% users) |
| **Territory Analysis** | 20 | Chi tiết về territory prediction | Medium (20% users) |
| **Influence Map** | 20 | Visualize influence | Low (15% users) |

### 6.2. Subscription Tiers

```javascript
const SUBSCRIPTIONS = {
  free: {
    name: "Free",
    price: 0,
    price_yearly: 0,
    features: {
      games_per_day: "unlimited",
      basic_stats: true,
      ai_opponents: [1, 2, 3, 4],  // All AI levels
      multiplayer: true,
      
      // ML Features (limited)
      quick_threats: 0,  // 0 free per day
      full_analysis: 0,
      post_game_review: 0,
      training_mode: 0,
      ai_hint: 0,
      
      // Other
      save_games: 10,  // Max 10 saved games
      replay: true,
      undo: 3,  // Max 3 undos per game
    },
    color: "gray"
  },
  
  silver: {
    name: "Silver",
    price: 50000,  // ~$2/month
    price_yearly: 500000,  // ~$20/year (save 17%)
    features: {
      games_per_day: "unlimited",
      basic_stats: true,
      ai_opponents: [1, 2, 3, 4],
      multiplayer: true,
      
      // ML Features
      quick_threats: 10,  // 10 free per day
      full_analysis: 3,   // 3 free per day
      post_game_review: 2,
      training_mode: 20,
      ai_hint: 5,
      
      // Bonuses
      coin_discount: 0.1,  // 10% off coin purchases
      save_games: 50,
      undo: 5,
      no_ads: true,
    },
    color: "gray-400",
    badge: "🥈"
  },
  
  gold: {
    name: "Gold",
    price: 150000,  // ~$6/month
    price_yearly: 1500000,  // ~$60/year (save 17%)
    features: {
      games_per_day: "unlimited",
      basic_stats: true,
      advanced_stats: true,
      ai_opponents: [1, 2, 3, 4],
      multiplayer: true,
      
      // ML Features (unlimited)
      quick_threats: "unlimited",
      full_analysis: 20,  // 20 per day
      post_game_review: 10,
      training_mode: "unlimited",
      ai_hint: 20,
      territory_analysis: 10,
      influence_map: 10,
      
      // Premium bonuses
      opening_book_access: true,  // Permanent
      coin_discount: 0.2,  // 20% off
      save_games: "unlimited",
      undo: "unlimited",
      no_ads: true,
      priority_support: true,
      early_access: true,  // New features
      custom_board_themes: true,
    },
    color: "yellow-500",
    badge: "🥇",
    popular: true
  },
  
  platinum: {
    name: "Platinum",
    price: 300000,  // ~$12/month
    price_yearly: 3000000,  // ~$120/year (save 17%)
    features: {
      // All Gold features +
      full_analysis: "unlimited",
      post_game_review: "unlimited",
      territory_analysis: "unlimited",
      influence_map: "unlimited",
      
      // Exclusive
      ai_commentary: true,  // AI explains moves in real-time
      personalized_training: true,  // AI tạo bài tập riêng
      rank_prediction: true,  // Dự đoán rank improvement
      coaching_insights: true,  // Deep analysis of playing style
      
      coin_discount: 0.3,  // 30% off
      priority_matchmaking: true,
      exclusive_tournaments: true,
    },
    color: "purple-500",
    badge: "💎"
  }
};
```

### 6.3. Monetization Tactics

#### 6.3.1. Freemium Funnel

```
┌─────────────────────────────────────────────┐
│         FREE USERS (100%)                   │
│  • Play games unlimited                     │
│  • Basic AI opponents                       │
│  • No ML features (unless buy coins)        │
└─────────────────────────────────────────────┘
                    │
                    │ Show ML analysis teaser
                    │ "Unlock AI insights!"
                    ▼
        ┌───────────────────────┐
        │   COIN PURCHASE (10%) │
        │  • Buy coins for ML   │
        │  • Try features       │
        └───────────────────────┘
                    │
                    │ After 3-5 uses:
                    │ "Save 40% with subscription!"
                    ▼
        ┌───────────────────────┐
        │   SILVER SUB (3%)     │
        │  • Regular features   │
        │  • Save money         │
        └───────────────────────┘
                    │
                    │ After 2-3 months:
                    │ "Upgrade for unlimited!"
                    ▼
        ┌───────────────────────┐
        │   GOLD SUB (1%)       │
        │  • Power users        │
        │  • Best value         │
        └───────────────────────┘
```

#### 6.3.2. Revenue Projections

**Conservative Scenario** (5,000 MAU):

| Revenue Source | Users | ARPU | Monthly | Yearly |
|----------------|-------|------|---------|--------|
| Coin purchases | 500 (10%) | 40,000 VND | 20M VND | 240M VND |
| Silver subs | 150 (3%) | 50,000 VND | 7.5M VND | 90M VND |
| Gold subs | 50 (1%) | 150,000 VND | 7.5M VND | 90M VND |
| Platinum subs | 10 (0.2%) | 300,000 VND | 3M VND | 36M VND |
| **Total** | | | **38M VND** | **456M VND** |
| | | | **~$1,520** | **~$18,240** |

**Optimistic Scenario** (20,000 MAU):

| Revenue Source | Users | ARPU | Monthly | Yearly |
|----------------|-------|------|---------|--------|
| Coin purchases | 2,000 (10%) | 40,000 VND | 80M VND | 960M VND |
| Silver subs | 600 (3%) | 50,000 VND | 30M VND | 360M VND |
| Gold subs | 200 (1%) | 150,000 VND | 30M VND | 360M VND |
| Platinum subs | 40 (0.2%) | 300,000 VND | 12M VND | 144M VND |
| **Total** | | | **152M VND** | **1,824M VND** |
| | | | **~$6,080** | **~$72,960** |

**Success Scenario** (50,000 MAU):

| Revenue Source | Users | ARPU | Monthly | Yearly |
|----------------|-------|------|---------|--------|
| Coin purchases | 5,000 (10%) | 40,000 VND | 200M VND | 2,400M VND |
| Silver subs | 1,500 (3%) | 50,000 VND | 75M VND | 900M VND |
| Gold subs | 500 (1%) | 150,000 VND | 75M VND | 900M VND |
| Platinum subs | 100 (0.2%) | 300,000 VND | 30M VND | 360M VND |
| **Total** | | | **380M VND** | **4,560M VND** |
| | | | **~$15,200** | **~$182,400** |

#### 6.3.3. Growth Tactics

1. **Viral Loop**
   - Share analysis results on social media
   - "Challenge your friends to beat this position"
   - Referral program: 50 coins for each friend who signs up

2. **Limited-Time Offers**
   - First purchase: 50% bonus coins
   - Weekend deals: 2× analysis for 1 price
   - Holiday promotions: Special packages

3. **Gamification**
   - Daily login rewards (coins)
   - Achievement system with coin rewards
   - Streak bonuses
   - Leaderboard prizes

4. **Content Marketing**
   - Blog posts about Go strategy (with ML insights)
   - YouTube tutorials featuring ML analysis
   - Twitch streams with pro players using ML features

---

## 7. BEST PRACTICES & TIPS

### 7.1. Development Best Practices

#### 7.1.1. Start Small, Iterate Fast

✅ **DO**:
```python
# Phase 1: Start with 9×9 board
model = MultiTaskModel(board_size=9, base_channels=64)

# Test thoroughly
test_accuracy = evaluate_model(model, test_dataset_9x9)
if test_accuracy > 0.70:
    print("✅ 9×9 model ready!")
    # Now move to 13×13

# Phase 2: Scale to 13×13
model_13 = MultiTaskModel(board_size=13, base_channels=64)
# Train and test

# Phase 3: Finally 19×19
model_19 = MultiTaskModel(board_size=19, base_channels=64)
```

❌ **DON'T**:
```python
# Don't start with 19×19 and 256 channels
model = HugeModel(board_size=19, base_channels=256, num_layers=50)
# This will take weeks to train and might not work
```

#### 7.1.2. Data Quality > Quantity

✅ **DO**:
```python
# Filter high-quality games
def filter_quality_games(games):
    return [
        game for game in games
        if game.player_rank >= "5d"  # Only strong players
        and game.moves > 100  # Complete games
        and not game.has_obvious_mistakes()
    ]

# 5,000 high-quality games > 50,000 random games
```

❌ **DON'T**:
```python
# Don't use all data blindly
all_games = download_all_games()  # Including beginners
# This will teach model bad habits
```

#### 7.1.3. Monitor Training Closely

✅ **DO**:
```python
# Use TensorBoard
writer = SummaryWriter('logs/')

# Log everything
writer.add_scalar('train/loss', loss, step)
writer.add_scalar('train/threat_loss', threat_loss, step)
writer.add_scalar('val/accuracy', accuracy, epoch)

# Visualize predictions
writer.add_image('predictions/threat_map', threat_heatmap, epoch)

# Save checkpoints frequently
if epoch % 5 == 0:
    save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
```

#### 7.1.4. Cache Aggressively

✅ **DO**:
```python
class MLAnalysisService:
    def __init__(self):
        self.cache = LRUCache(maxsize=1000)
    
    async def analyze_position(self, board):
        cache_key = board.zobrist_hash()
        
        # Check cache first
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Run inference
        result = await self._ml_inference(board)
        
        # Cache result
        self.cache[cache_key] = result
        return result
```

**Cache hit rate target**: >70%

#### 7.1.5. A/B Testing

```python
# Test different model versions
class ModelExperiment:
    def __init__(self):
        self.models = {
            'baseline': load_model('baseline_v1.pt'),
            'experimental': load_model('experimental_v2.pt')
        }
        self.user_assignments = {}  # user_id -> model_version
    
    def get_analysis(self, user_id, board):
        # Assign user to experiment group (50/50 split)
        if user_id not in self.user_assignments:
            self.user_assignments[user_id] = random.choice(['baseline', 'experimental'])
        
        model_version = self.user_assignments[user_id]
        model = self.models[model_version]
        
        result = model.analyze(board)
        
        # Track metrics
        self.track_metric(user_id, model_version, result)
        
        return result
```

### 7.2. Common Pitfalls to Avoid

#### 7.2.1. Over-engineering

❌ **DON'T**:
```python
# Don't build ResNet-50 with attention mechanisms and transformers
class ComplexModel(nn.Module):
    def __init__(self):
        self.resnet = ResNet50()
        self.attention = MultiHeadAttention(heads=16)
        self.transformer = Transformer(layers=12)
        # 100M+ parameters!
```

✅ **DO**:
```python
# Simple CNN is enough
class SimpleModel(nn.Module):
    def __init__(self):
        self.conv_layers = nn.Sequential(
            nn.Conv2d(17, 64, 5, padding=2),
            nn.ReLU(),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
        )
        # ~1M parameters, works well!
```

#### 7.2.2. Training Without Validation

❌ **DON'T**:
```python
# Train on all data
for epoch in range(100):
    for batch in all_data:
        loss = train_step(batch)
    # No validation → overfitting!
```

✅ **DO**:
```python
# Always validate
train_data, val_data, test_data = split_dataset(all_data, [0.8, 0.1, 0.1])

for epoch in range(100):
    train_loss = train_epoch(train_data)
    val_loss = validate(val_data)
    
    if val_loss > prev_val_loss:
        print("⚠️ Overfitting detected!")
        break
```

#### 7.2.3. Ignoring Production Constraints

❌ **DON'T**:
```python
# Model that takes 5 seconds to run
class SlowModel(nn.Module):
    def forward(self, x):
        # 50 layers of computation
        # Users will leave!
```

✅ **DO**:
```python
# Optimize for speed
class FastModel(nn.Module):
    def forward(self, x):
        # 5-7 layers, optimized
        # Target: <100ms inference time
```

**Production targets**:
- Inference time: <100ms (CPU), <10ms (GPU)
- Model size: <50MB
- Memory usage: <500MB

#### 7.2.4. Not Testing on Real Users

❌ **DON'T**:
```python
# Just look at metrics
print(f"Validation accuracy: 85%")
# Deploy!
```

✅ **DO**:
```python
# Beta test with real users
beta_users = select_beta_testers(count=50)

for user in beta_users:
    feedback = collect_feedback(user)
    
    if feedback.rating < 4:
        print("⚠️ Users not happy, need improvements")

# Only deploy when users are satisfied
```

### 7.3. Performance Optimization

#### 7.3.1. Model Optimization

```python
# 1. Quantization (reduce model size)
import torch.quantization

model_fp32 = load_model('model.pt')
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32,
    {nn.Linear, nn.Conv2d},
    dtype=torch.qint8
)
# Size: 50MB → 12MB
# Speed: 2× faster

# 2. ONNX conversion (for production)
import torch.onnx

dummy_input = torch.randn(1, 17, 19, 19)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=11
)

# Use ONNX Runtime for faster inference
import onnxruntime

session = onnxruntime.InferenceSession("model.onnx")
output = session.run(None, {"input": input_data})
# 1.5× faster than PyTorch
```

#### 7.3.2. Caching Strategy

```python
from functools import lru_cache
import hashlib

class SmartCache:
    def __init__(self, max_size=1000, ttl=3600):
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl
        self.access_count = {}
        self.hit_count = 0
        self.miss_count = 0
    
    def get_key(self, board_state):
        """Generate cache key"""
        state_bytes = board_state.tobytes()
        return hashlib.md5(state_bytes).hexdigest()
    
    def get(self, key):
        """Get from cache"""
        if key in self.cache:
            entry = self.cache[key]
            
            # Check expiry
            if time.time() - entry['timestamp'] < self.ttl:
                self.hit_count += 1
                self.access_count[key] = self.access_count.get(key, 0) + 1
                return entry['value']
            else:
                # Expired
                del self.cache[key]
        
        self.miss_count += 1
        return None
    
    def set(self, key, value):
        """Set cache"""
        # Evict if full (LRU)
        if len(self.cache) >= self.max_size:
            # Remove least accessed
            lru_key = min(self.access_count, key=self.access_count.get)
            del self.cache[lru_key]
            del self.access_count[lru_key]
        
        self.cache[key] = {
            'value': value,
            'timestamp': time.time()
        }
        self.access_count[key] = 0
    
    def stats(self):
        """Cache statistics"""
        total = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total if total > 0 else 0
        
        return {
            'hit_rate': hit_rate,
            'size': len(self.cache),
            'hits': self.hit_count,
            'misses': self.miss_count
        }
```

#### 7.3.3. Batch Processing

```python
class BatchedMLService:
    def __init__(self, batch_size=8, max_wait_ms=100):
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        self.queue = []
        self.processing = False
    
    async def analyze(self, board):
        """Add to queue and wait for batch processing"""
        future = asyncio.Future()
        self.queue.append((board, future))
        
        # Start processing if batch is full or timeout
        if len(self.queue) >= self.batch_size:
            asyncio.create_task(self._process_batch())
        elif not self.processing:
            asyncio.create_task(self._process_after_timeout())
        
        return await future
    
    async def _process_batch(self):
        """Process queued requests in batch"""
        if self.processing or not self.queue:
            return
        
        self.processing = True
        batch = self.queue[:self.batch_size]
        self.queue = self.queue[self.batch_size:]
        
        # Extract boards
        boards = [item[0] for item in batch]
        futures = [item[1] for item in batch]
        
        # Batch inference (much faster)
        board_tensors = torch.stack([board_to_tensor(b) for b in boards])
        
        with torch.no_grad():
            results = self.model(board_tensors)
        
        # Return results
        for i, future in enumerate(futures):
            future.set_result(results[i])
        
        self.processing = False
    
    async def _process_after_timeout(self):
        """Process if timeout reached"""
        await asyncio.sleep(self.max_wait_ms / 1000)
        await self._process_batch()
```

---

## 8. CONCLUSION & NEXT STEPS

### 8.1. Summary

Tài liệu này đã trình bày **chiến lược toàn diện** để phát triển ML cho GoGame:

#### ✅ Completed
1. **Architecture** - Multi-task model với 4 heads
2. **Code** - Backbone, heads, và integration service
3. **Strategy** - Monetization và feature plan
4. **Documentation** - Complete guide

#### ⏳ In Progress
1. **Data Collection** - Cần thu thập thêm data
2. **Training** - Cần train models
3. **Testing** - Cần test với real users

#### 📋 TODO (Priority order)
1. **Week 1-2**: Complete data collection pipeline
2. **Week 3**: Generate ground truth labels
3. **Week 4-6**: Train models (9×9 → 13×13 → 19×19)
4. **Week 7**: Integration testing
5. **Week 8-9**: Frontend implementation
6. **Week 10**: Beta testing
7. **Week 11-12**: Polish and launch

### 8.2. Key Success Factors

1. **Data Quality**: High-quality training data là foundation
2. **Simple Architecture**: Lightweight models → fast inference
3. **Great UX**: Beautiful visualization > raw numbers
4. **Fair Pricing**: Value phải xứng với giá
5. **Iterative Development**: Ship fast, learn, improve

### 8.3. Immediate Next Steps (This Week)

**Option A: Training với Self-Play (Local)**
```bash
# 1. Test data collector
python src/ml/training/data_collector.py --board-size 9 --num-games 100

# 2. Review collected data
python scripts/visualize_training_data.py

# 3. Start label generation
python scripts/create_label_generator.py

# 4. Set up training environment
pip install torch tensorboard optuna

# 5. Create training config
cp configs/training_config.example.yaml configs/my_training.yaml
```

**Option B: Training với Professional Games (Colab/Kaggle) ⭐ RECOMMENDED**
```bash
# 1. Download professional games
python scripts/download_kgs_games.py --min-rank 5 --max-games 10000

# 2. Parse SGF files
python scripts/parse_sgf_to_positions.py --input data/raw/kgs --output data/processed

# 3. Generate labels
python scripts/generate_labels.py --input data/processed/positions_9x9.pt --output data/labeled/labels_9x9.pt

# 4. Upload to Colab/Kaggle và train
# Xem chi tiết: docs/ML_TRAINING_COLAB_GUIDE.md
```

### 8.4. Resources

**Documentation**:
- `docs/ML_COMPREHENSIVE_GUIDE.md` - This document
- `docs/ML_QUICK_START.md` - Quick start guide
- `docs/ML_TRAINING_ROADMAP.md` - Original roadmap
- `docs/ML_TRAINING_COLAB_GUIDE.md` - **NEW**: Training trên Colab/Kaggle với dữ liệu chuyên nghiệp

**Code**:
- `src/ml/models/` - Model architecture
- `src/ml/training/` - Training scripts
- `backend/app/services/ml_analysis_service.py` - API service

**Scripts**:
- `src/ml/training/data_collector.py` - Data collection
- Training script (TODO): `src/ml/training/train_multi_task.py`

### 8.5. Support & Questions

**Common Questions**:

Q: Tôi nên train trên CPU hay GPU?
A: GPU nhanh hơn 10-100×. Nếu không có GPU local, dùng Google Colab (free GPU).

Q: Dataset cần bao nhiêu data?
A: Minimum 10,000 games (1M+ positions). More is better.

Q: Training mất bao lâu?
A: 9×9: ~2-3 hours, 13×13: ~6-8 hours, 19×19: ~24-48 hours (on GPU).

Q: Làm sao biết model đã đủ tốt?
A: Validation accuracy >70% AND real user feedback positive.

Q: Có thể dùng pre-trained model không?
A: Có thể transfer learning từ AlphaGo-like models, nhưng cần fine-tune.

---

**🎉 Chúc bạn thành công với dự án ML!**

*Remember: Start small, iterate fast, focus on user value! 🚀*

---

## APPENDIX: Code Templates

### A. Training Script Template

```python
# src/ml/training/train_multi_task.py
# (See Part 2 for full implementation)
```

### B. Dataset Template

```python
# src/ml/training/dataset.py
# (See Part 2 for full implementation)
```

### C. Evaluation Script Template

```python
# scripts/evaluate_model.py

import torch
from src.ml.models.multi_task_model import MultiTaskModel
from src.ml.training.dataset import GoPositionDataset

def evaluate_model(model_path, test_dataset_path):
    """Evaluate trained model"""
    # Load model
    model = MultiTaskModel()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    # Load test data
    test_dataset = GoPositionDataset(test_dataset_path, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Evaluate
    metrics = {
        'threat_accuracy': 0,
        'attack_accuracy': 0,
        'intent_accuracy': 0
    }
    
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(batch['features'])
            
            # Calculate metrics
            # ... (implementation)
    
    return metrics

if __name__ == '__main__':
    metrics = evaluate_model(
        'checkpoints/best_model.pt',
        'data/test/'
    )
    print(f"Test Metrics: {metrics}")
```

### D. Deployment Checklist

- [ ] Model size < 50MB
- [ ] Inference time < 100ms (CPU)
- [ ] Validation accuracy > 70%
- [ ] Cache hit rate > 70%
- [ ] API response time < 500ms
- [ ] Beta test with 50+ users
- [ ] User satisfaction > 4/5
- [ ] Documentation complete
- [ ] Monitoring setup (logs, metrics)
- [ ] Rollback plan ready

---

**END OF COMPREHENSIVE GUIDE**

*Last updated: 2025-01-27*







