TÀI LIỆU PHÁT TRIỂN MACHINE LEARNING CHO GAME CỜ VÂY
🎯 TỔNG QUAN DỰ ÁN ML
Dựa trên phân tích của bạn và hiện trạng dự án, tôi đề xuất phát triển "AI TACTICAL VISION & POSITION ANALYZER" - một hệ thống ML phân tích thế cờ toàn diện, không chỉ đơn thuần gợi ý nước đi.

📋 MỤC LỤC

Tại sao không chỉ "Gợi ý nước đi tốt nhất"?
Đề xuất: AI Tactical Vision System
Chi tiết các tính năng ML
Roadmap triển khai từng bước
Technical Implementation
UI/UX Design
Monetization Strategy


1. TẠI SAO KHÔNG CHỈ "GỢI Ý NƯỚC ĐI TỐT NHẤT"?
❌ Vấn đề với "Simple Move Suggestion"
Bạn đã nhận định đúng! Lý do:

Thuật toán có sẵn làm được:

MCTS đã cho ra nước đi tốt ở level cao
Không cần ML để làm điều này
User không thấy "value add" rõ ràng


Thiếu educational value:

Chỉ biết "đi đây" nhưng không biết "tại sao"
Không giúp người chơi improve
Giống như "cheating" hơn là "learning tool"


Không showcase được sức mạnh ML:

ML có khả năng phân tích sâu hơn nhiều
Pattern recognition, strategic understanding
Visualization của "AI thinking process"



## ✔️ Điều ML làm được mà thuật toán truyền thống **không thể**

|        **Khả năng**         |   **Thuật toán truyền thống**  |          **Machine Learning**              |
|-----------------------------|--------------------------------|--------------------------------------------|
| Gợi ý nước đi               | ✔️ MCTS làm tốt                | ✔️ ML cũng làm được (nhưng không đặc biệt)|
| Đánh giá win probability    | ❌ Khó, không chính xác        | ✔️ Value Network rất tốt                  |
| Nhận diện tactical patterns | ❌ Phải hard-code từng pattern | ✔️ Tự học từ data                         |
| Phân tích ý đồ chiến lược   | ❌ Không thể                   | ✔️ Có thể (attention mechanism)           |
| Territory prediction        | ❌ Heuristic sơ sài            | ✔️ Chính xác cao                          |
| Life/Death analysis         | ⚠️ Chỉ cases đơn giản          | ✔️ Phức tạp cũng được                     |
| Visualize “thinking”        | ❌ Không có gì để show         | ✔️ Attention maps, heatmaps               |


2. ĐỀ XUẤT: AI TACTICAL VISION SYSTEM
🎨 Concept: "Nhìn qua mắt AI"
Thay vì chỉ nói "đi đây", hệ thống sẽ cho người dùng "nhìn thấy" những gì AI nhìn thấy:
🧠 AI TACTICAL VISION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─────────────────────────────────────────────┐
│  👁️ VISION MODE: Territory Analysis        │
├─────────────────────────────────────────────┤
│                                             │
│     🟦🟦🟦 = Black's territory (85% sure)  │
│     🟥🟥🟥 = White's territory (90% sure)  │
│     🟨🟨🟨 = Contested area (50-50)        │
│     🟩 = Influence zone                     │
│                                             │
│  Current Score Estimate:                    │
│  Black: 65±5 points                         │
│  White: 58±4 points                         │
│  Black winning probability: 68%             │
│                                             │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  ⚔️ VISION MODE: Tactical Threats          │
├─────────────────────────────────────────────┤
│                                             │
│  🔴 URGENT (1 move to capture)              │
│     └─ White group at Q15 (4 stones)       │
│        Missing liberty at R14               │
│                                             │
│  🟠 WEAK GROUPS (2-3 liberties)             │
│     └─ Black group at D4 (6 stones)        │
│        Can be attacked at C3 or E3          │
│                                             │
│  🟢 STRONG GROUPS (alive)                   │
│     └─ Black corner group (has 2 eyes)     │
│                                             │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  🎯 VISION MODE: Strategic Intentions      │
├─────────────────────────────────────────────┤
│                                             │
│  AI thinks opponent is planning:            │
│  1. Invade upper-right corner (70%)         │
│  2. Attack weak group at D4 (55%)           │
│  3. Build territory on left side (40%)      │
│                                             │
│  Recommended counter-strategies:            │
│  • Strengthen D4 group first                │
│  • Prepare defense in upper-right          │
│                                             │
└─────────────────────────────────────────────┘

3. CHI TIẾT CÁC TÍNH NĂNG ML
🔍 Feature 1: SMART POSITION ANALYZER (Core Feature)
Mô tả: Phân tích toàn diện vị thế hiện tại của bàn cờ
Sub-features:
A. Territory Prediction
Input: Board state
Model: CNN-based Territory Classifier
Output: 
{
  "territory_map": [
    [0.9, 0.85, 0.7, ...],  // Probability each point belongs to Black
    ...
  ],
  "territory_estimate": {
    "black": 65,
    "white": 58,
    "neutral": 5,
    "confidence": 0.85
  },
  "visualization": {
    "black_zones": [[x1,y1], [x2,y2], ...],
    "white_zones": [...],
    "contested_zones": [...],
    "intensity": [...]  // Độ chắc chắn từng vùng
  }
}
Cách hiển thị trên UI:

Overlay màu mờ trên bàn cờ
Độ đậm = độ chắc chắn (opacity)
Màu: Xanh (Black), Đỏ (White), Vàng (Contested)


B. Win Probability Tracker
Input: Board state + Move history
Model: Value Network (ResNet-style)
Output:
{
  "current_win_probability": {
    "black": 0.68,
    "white": 0.32
  },
  "probability_over_time": [
    {"move": 1, "black_wp": 0.50},
    {"move": 2, "black_wp": 0.52},
    ...
    {"move": 42, "black_wp": 0.68}
  ],
  "key_turning_points": [
    {
      "move_number": 28,
      "before": 0.55,
      "after": 0.45,
      "change": -0.10,
      "reason": "Failed ladder at R15"
    }
  ]
}
Cách hiển thị:

Line chart theo thời gian
Highlight các điểm chuyển giao quan trọng
Tooltip explain tại sao thay đổi


C. Group Strength Analyzer
Input: Board state
Model: CNN + Graph Neural Network
Output:
{
  "groups": [
    {
      "id": "black_group_1",
      "stones": [[3,3], [3,4], [4,3], [4,4]],
      "health_score": 0.95,  // 0-1
      "status": "ALIVE",
      "liberties": 8,
      "eyes": 2,
      "threats": [],
      "visualization": {
        "color": "green",
        "border_thickness": 2,
        "label": "Strong (2 eyes)"
      }
    },
    {
      "id": "white_group_3",
      "stones": [[10,10], [10,11], [11,10]],
      "health_score": 0.25,
      "status": "CRITICAL",
      "liberties": 2,
      "eyes": 0,
      "threats": [
        {
          "type": "ATARI",
          "attack_point": [11,11],
          "severity": "HIGH"
        }
      ],
      "visualization": {
        "color": "red",
        "border_thickness": 4,
        "label": "Atari! (2 liberties)",
        "pulse_animation": true
      }
    }
  ]
}
Cách hiển thị:

Border quanh các nhóm với màu theo health
Nhấp nháy nhóm nguy hiểm
Tooltip hiện chi tiết khi hover


D. Influence Map
Input: Board state
Model: CNN-based Influence Predictor
Output:
{
  "influence_map": [
    [0.7, 0.6, 0.4, ...],  // Black influence at each point
    ...
  ],
  "strong_influence_zones": {
    "black": [
      {
        "center": [4, 4],
        "radius": 5,
        "strength": 0.85
      }
    ],
    "white": [...]
  },
  "weak_points": [
    {
      "position": [10, 10],
      "weakness_score": 0.9,
      "reason": "Low influence from both sides"
    }
  ]
}
Cách hiển thị:

Heat map gradient
Hoặc vẽ "vùng ảnh hưởng" như game chiến thuật


⚔️ Feature 2: TACTICAL THREAT DETECTOR
Mô tả: Phát hiện các mối đe dọa chiến thuật ngay lập tức
Input: Board state
Model: Pattern Recognition CNN + Attention Mechanism
Output:
{
  "urgent_threats": [
    {
      "type": "ATARI",
      "target_group": "white_group_3",
      "attack_move": [11, 11],
      "consequence": "Capture 4 stones",
      "priority": 1,
      "visualization": {
        "highlight_group": "white_group_3",
        "mark_attack_point": [11, 11],
        "animation": "pulse_red",
        "label": "🚨 URGENT: Atari!"
      }
    }
  ],
  
  "tactical_opportunities": [
    {
      "type": "CUT",
      "move": [8, 9],
      "effect": "Separate two white groups",
      "value_gain": 15,  // points
      "priority": 2,
      "visualization": {
        "mark_cut_point": [8, 9],
        "show_separation_line": true,
        "label": "✂️ Cut opportunity"
      }
    },
    {
      "type": "LADDER",
      "move": [12, 5],
      "success_probability": 0.85,
      "captured_stones": 3,
      "ladder_path": [[12,5], [12,6], [13,6], ...],
      "visualization": {
        "show_ladder_sequence": true,
        "animation": "step_by_step"
      }
    }
  ],
  
  "defensive_weaknesses": [
    {
      "your_group": "black_group_5",
      "threat_type": "CAN_BE_CUT",
      "opponent_can_play": [7, 8],
      "recommendation": "Connect at [7, 7]",
      "priority": 2
    }
  ]
}
Cách hiển thị:

Màn hình chia làm 3 phần:

🔴 Urgent (cần xử lý ngay)
🟡 Opportunities (cơ hội tấn công)
🟢 Defenses (điểm yếu cần bảo vệ)


Click vào từng item để xem visualization trên bàn cờ


🧠 Feature 3: STRATEGIC INTENT ANALYZER (Advanced)
Mô tả: Dự đoán ý định chiến lược của đối thủ (hoặc AI)
Input: Board state + Recent 10 moves
Model: Transformer-based Sequential Model
Output:
{
  "opponent_likely_plans": [
    {
      "plan": "INVADE_CORNER",
      "target_area": "upper_right_corner",
      "probability": 0.70,
      "typical_invasion_points": [[15,15], [16,16]],
      "reasoning": "Large open territory, weak defense",
      "counter_strategy": "Play protective move at [16, 14]"
    },
    {
      "plan": "ATTACK_WEAK_GROUP",
      "target_group": "black_group_5",
      "probability": 0.55,
      "attack_sequence": [[7,8], [8,7]],
      "reasoning": "Only 3 liberties, no eyes formed"
    }
  ],
  
  "your_strategic_options": [
    {
      "strategy": "SOLIDIFY_TERRITORY",
      "area": "left_side",
      "expected_gain": 20,
      "risk": "low",
      "recommended_moves": [[3,10], [3,12]]
    },
    {
      "strategy": "AGGRESSIVE_INVASION",
      "area": "opponent_bottom_territory",
      "expected_gain": 25,
      "risk": "high",
      "recommended_moves": [[10,3]]
    }
  ]
}
Cách hiển thị:

Tab "🧠 Strategy Analysis"
List các kế hoạch có thể của đối thủ
Recommendation panel ở bên cạnh


📊 Feature 4: POST-GAME REVIEW SYSTEM
Mô tả: Phân tích toàn bộ ván đấu sau khi kết thúc
Input: Complete game record (SGF)
Model: Combined (Value Net + Policy Net + Tactical Detector)
Output:
{
  "game_summary": {
    "winner": "Black",
    "score": "+12.5",
    "game_length": 185,
    "opening_style": "Chinese Opening",
    "overall_quality": {
      "black": 0.78,  // 0-1
      "white": 0.72
    }
  },
  
  "mistakes": [
    {
      "player": "white",
      "move_number": 42,
      "played_move": [10, 10],
      "evaluation": -0.15,  // Loss in win probability
      "severity": "MISTAKE",
      "better_move": [10, 11],
      "alternative_eval": -0.05,
      "explanation": "Missed opportunity to connect groups. Playing at [10,11] would maintain position."
    },
    {
      "move_number": 78,
      "severity": "BLUNDER",
      "evaluation": -0.35,
      "explanation": "Allowed opponent to capture corner group"
    }
  ],
  
  "good_moves": [
    {
      "player": "black",
      "move_number": 65,
      "played_move": [8, 15],
      "evaluation": +0.12,
      "quality": "EXCELLENT",
      "explanation": "Perfect timing to invade, leading to significant territory gain"
    }
  ],
  
  "key_moments": [
    {
      "move_number": 28,
      "before_wp": 0.55,
      "after_wp": 0.45,
      "title": "Failed ladder attempt",
      "description": "White tried ladder but it didn't work, losing 4 stones"
    }
  ],
  
  "learning_points": [
    "Work on life and death reading - missed several tactical opportunities",
    "Good opening understanding - maintained advantage early game",
    "Consider timing of invasions more carefully"
  ]
}
Cách hiển thị:

Timeline với mistakes/good moves marked
Click để xem chi tiết từng nước
Win probability graph
Learning tips ở cuối


4. ROADMAP TRIỂN KHAI TỪNG BƯỚC
🗺️ GIAI ĐOẠN 1: CHUẨN BỊ DỮ LIỆU (2-3 tuần)
Bước 1.1: Thu thập dữ liệu
Mục tiêu: Có dataset để train models
Nguồn dữ liệu:

KGS Game Archive (FREE)

Link: https://u-go.net/gamerecords/
~70,000 professional games
Format: SGF
Ranks: 1d - 9d professional


GoGoD Database (Paid, optional)

~100,000 historical games
High quality professional games


OGS API (FREE)

Recent games from online players
Various skill levels



Công việc:
bash# Download script
python scripts/download_kgs_games.py --output data/raw/kgs/

# Expected output:
# data/raw/kgs/
#   ├── 2023-01/
#   │   ├── game_001.sgf
#   │   ├── game_002.sgf
#   │   └── ...
#   ├── 2023-02/
#   └── ...
Dataset mục tiêu:

10,000 games (bàn 19x19) để bắt đầu
Có thể expand sau
Mix cả pro games và strong amateur (5d+)


Bước 1.2: Preprocessing & Labeling
Mục tiêu: Convert SGF → Training data format
Tasks:
A. Parse SGF files
python# scripts/preprocess/parse_sgf.py

from sgfmill import sgf
import numpy as np

def parse_sgf_to_states(sgf_path):
    """
    Convert SGF file to list of (board_state, metadata) pairs
    """
    with open(sgf_path, 'rb') as f:
        game = sgf.Sgf_game.from_bytes(f.read())
    
    board_size = game.get_size()
    states = []
    
    # Replay game move by move
    board = Board(board_size)
    for node in game.get_main_sequence():
        move = node.get_move()
        if move[1] is not None:  # Not a pass
            states.append({
                'board': board.to_numpy(),
                'move': move,
                'player': move[0],
                'move_number': len(states) + 1
            })
            board.play_move(move)
    
    return states, game.get_winner()
B. Tạo labels cho từng task
Territory Labels:
pythondef create_territory_labels(final_board_state):
    """
    Label: Từng điểm thuộc về ai ở cuối game
    
    Returns: 19x19 array với values:
      1.0 = Black territory
      0.0 = White territory  
      0.5 = Neutral
    """
    territory_map = np.zeros((19, 19))
    
    # Sử dụng Tromp-Taylor scoring
    for i in range(19):
        for j in range(19):
            owner = determine_point_owner(final_board_state, i, j)
            if owner == BLACK:
                territory_map[i,j] = 1.0
            elif owner == WHITE:
                territory_map[i,j] = 0.0
            else:
                territory_map[i,j] = 0.5
    
    return territory_map
Win Probability Labels:
pythondef create_win_labels(game_states, winner):
    """
    Label: Win probability tại mỗi move
    
    Simplified approach: Linear interpolation
      - At start: 50-50
      - At end: 100-0 (winner gets 1.0)
    """
    num_moves = len(game_states)
    labels = []
    
    for i, state in enumerate(game_states):
        # Linear from 0.5 to 1.0 (if winner) or 0.0 (if loser)
        progress = i / num_moves
        
        if state['player'] == winner:
            win_prob = 0.5 + 0.5 * progress
        else:
            win_prob = 0.5 - 0.5 * progress
        
        labels.append(win_prob)
    
    return labels
Tactical Pattern Labels:
pythondef create_tactical_labels(board_state, next_move):
    """
    Label: Tactical patterns
    
    Detect if move is:
      - Atari (attacking)
      - Escape from atari
      - Cut
      - Connection
      - Capture
    """
    patterns = {
        'is_atari': False,
        'is_escape': False,
        'is_cut': False,
        'is_connection': False,
        'is_capture': False
    }
    
    # Check each pattern
    if causes_atari(board_state, next_move):
        patterns['is_atari'] = True
    
    if escapes_atari(board_state, next_move):
        patterns['is_escape'] = True
    
    # ... similar checks
    
    return patterns

Bước 1.3: Feature Engineering
Mục tiêu: Biến board state thành input features cho NN
Feature Planes (theo AlphaGo paper nhưng đơn giản hơn):
pythondef extract_features(board_state, history=8):
    """
    Convert board state to feature tensor
    
    Returns: (N, 19, 19) tensor where N = number of feature planes
    """
    features = []
    
    # Plane 1-2: Current stones
    features.append(board_state == BLACK)  # Shape: (19,19)
    features.append(board_state == WHITE)
    
    # Plane 3-4: Stones with 1 liberty
    lib1_black = get_stones_with_n_liberties(board_state, BLACK, 1)
    lib1_white = get_stones_with_n_liberties(board_state, WHITE, 1)
    features.append(lib1_black)
    features.append(lib1_white)
    
    # Plane 5-6: Stones with 2 liberties
    lib2_black = get_stones_with_n_liberties(board_state, BLACK, 2)
    lib2_white = get_stones_with_n_liberties(board_state, WHITE, 2)
    features.append(lib2_black)
    features.append(lib2_white)
    
    # Plane 7-8: Stones with 3+ liberties
    lib3_black = get_stones_with_n_liberties(board_state, BLACK, 3, or_more=True)
    lib3_white = get_stones_with_n_liberties(board_state, WHITE, 3, or_more=True)
    features.append(lib3_black)
    features.append(lib3_white)
    
    # Plane 9-16: History (previous 4 moves, 2 planes per move)
    for i in range(4):
        if len(history) > i:
            hist_board = history[-(i+1)]
            features.append(hist_board == BLACK)
            features.append(hist_board == WHITE)
        else:
            # Padding with zeros if not enough history
            features.append(np.zeros((19,19)))
            features.append(np.zeros((19,19)))
    
    # Plane 17: Turn indicator (all 1s if Black to play, all 0s if White)
    turn = np.ones((19,19)) if current_player == BLACK else np.zeros((19,19))
    features.append(turn)
    
    # Stack all planes: (17, 19, 19)
    return np.stack(features, axis=0)
```

**Output format:**
```
data/processed/
  ├── features/
  │   ├── batch_001.npy  # (1000, 17, 19, 19)
  │   ├── batch_002.npy
  │   └── ...
  ├── labels/
  │   ├── territory/
  │   │   ├── batch_001.npy  # (1000, 19, 19)
  │   │   └── ...
  │   ├── win_prob/
  │   │   ├── batch_001.npy  # (1000, 1)
  │   │   └── ...
  │   └── tactical/
  │       ├── batch_001.json
  │       └── ...
  └── metadata.json

🏗️ GIAI ĐOẠN 2: THIẾT KẾ MODELS (1-2 tuần)
Bước 2.1: Architecture Design
Model 1: Territory Predictor
pythonimport torch
import torch.nn as nn

class TerritoryPredictor(nn.Module):
    """
    Input: (batch, 17, 19, 19) - Feature planes
    Output: (batch, 19, 19) - Territory ownership probability
    """
    def __init__(self):
        super().__init__()
        
        # Encoder: Extract features
        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv2d(17, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Block 3
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Decoder: Output territory map
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Output layer
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, x):
        # x: (batch, 17, 19, 19)
        features = self.encoder(x)  # (batch, 256, 19, 19)
        territory = self.decoder(features)  # (batch, 1, 19, 19)
        return territory.squeeze(1)  # (batch, 19, 19)
Model 2: Value Network (Win Probability)
pythonclass ValueNetwork(nn.Module):
    """
    Input: (batch, 17, 19, 19)
    Output: (batch, 1) - Win probability
    """
    def __init__(self):
        super().__init__()
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(17, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: (batch, 17, 19, 19)
        features = self.conv_layers(x)  # (batch, 256, 19, 19)
        pooled = self.gap(features)  # (batch, 256, 1, 1)
        flattened = pooled.view(pooled.size(0), -1)  # (batch, 256)
        win_prob = self.fc(flattened)  # (batch, 1)
        return win_prob
Model 3: Tactical Pattern Detector
pythonclass TacticalDetector(nn.Module):
    """
    Multi-task classifier for tactical patterns
    
    Output: Dictionary of pattern probabilities
    """
    def __init__(self, num_patterns=5):
        super().__init__()
        
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(17, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 19x19 -> 9x9
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 9x9 -> 4x4
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Task-specific heads
        self.atari_head = self._make_head(256, 19*19)  # Atari location
        self.weak_group_head = self._make_head(256, 19*19)  # Weak groups
        self.cut_head = self._make_head(256, 19*19)  # Cut points
        
    def _make_head(self, in_channels, out_size):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, out_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.backbone(x)
        
        return {
            'atari': self.atari_head(features).view(-1, 19, 19),
            'weak_groups': self.weak_group_head(features).view(-1, 19, 19),
            'cuts': self.cut_head(features).view(-1, 19, 19)
        }

Bước 2.2: Loss Functions
python# Territory Prediction Loss
def territory_loss(pred, target):
    """
    pred: (batch, 19, 19) - predicted territory
    target: (batch, 19, 19) - actual territory (0/0.5/1)
    """
    # Weighted MSE (punish more on borders)
    weights = create_importance_weights()  # Corners/edges important
    mse = (pred - target) ** 2
    weighted_mse = mse * weights
    return weighted_mse.mean()

# Value Network Loss  
def value_loss(pred, target):
    """
    pred: (batch, 1) - predicted win prob
    target: (batch, 1) - actual outcome (0 or 1)
    """
    return nn.BCELoss()(pred, target)

# Tactical Detection Loss
def tactical_loss(pred_dict, target_dict):
    """
    Multi-task loss
    """
    total_loss = 0
    
    for task_name in pred_dict.keys():
        pred = pred_dict[task_name]
        target = target_dict[task_name]
        
        # Binary cross-entropy for each task
        task_loss = nn.BCELoss()(pred, target)
        total_loss += task_loss
    
    return total_loss

🎓 GIAI ĐOẠN 3: TRAINING (3-4 tuần)
Bước 3.1: Training Infrastructure Setup
python# training/config.py

class TrainingConfig:
    # Hardware
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = 4
    
    # Hyperparameters
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 50
    
    # Paths
    data_dir = 'data/processed'
    checkpoint_dir = 'checkpoints'
    log_dir = 'logs'
    
    # Early stopping
    patience = 10
    min_delta = 0.001
python# training/trainer.py

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class ModelTrainer:
    def __init__(self, model, config):
        self.model = model.to(config.device)
        self.config = config
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate
        )
        self.writer = SummaryWriter(config.log_dir)
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        for batch_idx, (features, labels) in enumerate(train_loader):
            features = features.to(self.config.device)
            labels = labels.to(self.config.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Log every 100 batches
            if batch_idx % 100 == 0:
                self.writer.add_scalar(
                    'train/loss',
                    loss.item(),
                    self.global_step
                )
                self.global_step += 1
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(self.config.device)
                labels = labels.to(self.config.device)
                
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def fit(self, train_loader, val_loader):
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            print(f"Epoch {epoch+1}/{self.config.num_epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            
            # Log to tensorboard
            self.writer.add_scalar('epoch/train_loss', train_loss, epoch)
            self.writer.add_scalar('epoch/val_loss', val_loss, epoch)
            
            # Save checkpoint
            if val_loss < best_val_loss - self.config.min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint(f'best_model_epoch_{epoch}.pth')
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config.patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        self.writer.close()

Bước 3.2: Training Plan
Phase 1: Train Territory Predictor (1 tuần)
bashpython train.py \
  --model territory \
  --data data/processed/territory \
  --epochs 50 \
  --batch-size 32 \
  --lr 0.001
Metrics theo dõi:

Training loss (MSE)
Validation loss
Accuracy (% điểm dự đoán đúng)
F1 score (Black/White/Neutral classification)

Target performance:

Validation accuracy > 85%
Converge trong ~30-40 epochs


Phase 2: Train Value Network (1 tuần)
bashpython train.py \
  --model value \
  --data data/processed/win_prob \
  --epochs 50 \
  --batch-size 64 \
  --lr 0.001
Metrics:

Binary cross-entropy loss
Accuracy (win prediction)
Calibration (predicted prob vs actual)

Target:

Validation accuracy > 70%


Phase 3: Train Tactical Detector (1-2 tuần)
bashpython train.py \
  --model tactical \
  --data data/processed/tactical \
  --epochs 50 \
  --batch-size 32 \
  --lr 0.001
Metrics (per task):

Precision/Recall for each pattern
mAP (mean Average Precision)

Target:

Atari detection: Precision > 90%
Weak group detection: Precision > 80%
Cut detection: Precision > 75%


🔌 GIAI ĐOẠN 4: INTEGRATION VỚI BACKEND (2 tuần)
Bước 4.1: Model Serving Setup
Tạo API endpoint:
python# backend/ml_service.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np

app = FastAPI()

# Load models at startup
territory_model = torch.load('models/territory_predictor.pth')
value_model = torch.load('models/value_network.pth')
tactical_model = torch.load('models/tactical_detector.pth')

territory_model.eval()
value_model.eval()
tactical_model.eval()

class BoardState(BaseModel):
    board: list[list[int]]  # 19x19 array
    history: list[list[list[int]]]  # Recent moves
    current_player: int  # 1=Black, 2=White

@app.post("/api/ml/analyze-position")
async def analyze_position(state: BoardState):
    """
    Comprehensive position analysis
    """
    try:
        # Convert to features
        features = extract_features(
            state.board,
            state.history,
            state.current_player
        )
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            # Territory prediction
            territory_pred = territory_model(features_tensor)
            territory_map = territory_pred.squeeze().cpu().numpy()
            
            # Win probability
            win_prob = value_model(features_tensor)
            win_prob_value = win_prob.item()
            
            # Tactical patterns
            tactical_pred = tactical_model(features_tensor)
            atari_map = tactical_pred['atari'].squeeze().cpu().numpy()
            weak_groups_map = tactical_pred['weak_groups'].squeeze().cpu().numpy()
            cuts_map = tactical_pred['cuts'].squeeze().cpu().numpy()
        
        # Post-process results
        result = {
            "territory_analysis": {
                "map": territory_map.tolist(),
                "black_territory": int((territory_map > 0.7).sum()),
                "white_territory": int((territory_map < 0.3).sum()),
                "contested": int(((territory_map >= 0.3) & (territory_map <= 0.7)).sum())
            },
            
            "win_probability": {
                "black": win_prob_value,
                "white": 1 - win_prob_value
            },
            
            "tactical_threats": {
                "atari_positions": find_hot_spots(atari_map, threshold=0.8),
                "weak_groups": find_hot_spots(weak_groups_map, threshold=0.7),
                "cut_points": find_hot_spots(cuts_map, threshold=0.75)
            },
            
            "visualization_data": {
                "territory_heatmap": territory_map.tolist(),
                "threat_heatmap": atari_map.tolist()
            }
        }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def find_hot_spots(heatmap, threshold=0.8):
    """
    Find positions with high probability
    """
    hot_spots = []
    rows, cols = np.where(heatmap > threshold)
    
    for i, j in zip(rows, cols):
        hot_spots.append({
            "position": [int(i), int(j)],
            "confidence": float(heatmap[i, j])
        })
    
    return hot_spots

Bước 4.2: Caching Strategy
python# backend/ml_cache.py

from redis import Redis
import json
import hashlib

redis_client = Redis(host='localhost', port=6379, db=0)

def get_board_hash(board_state):
    """
    Tạo unique hash cho board state
    """
    board_str = json.dumps(board_state, sort_keys=True)
    return hashlib.md5(board_str.encode()).hexdigest()

def cache_ml_result(board_state, result, ttl=3600):
    """
    Cache kết quả ML (TTL = 1 hour)
    """
    cache_key = f"ml:analysis:{get_board_hash(board_state)}"
    redis_client.setex(
        cache_key,
        ttl,
        json.dumps(result)
    )

def get_cached_result(board_state):
    """
    Lấy kết quả từ cache
    """
    cache_key = f"ml:analysis:{get_board_hash(board_state)}"
    cached = redis_client.get(cache_key)
    
    if cached:
        return json.loads(cached)
    return None

# Sử dụng trong endpoint
@app.post("/api/ml/analyze-position")
async def analyze_position(state: BoardState):
    # Check cache first
    cached = get_cached_result(state.dict())
    if cached:
        return cached
    
    # Run inference
    result = run_ml_inference(state)
    
    # Cache result
    cache_ml_result(state.dict(), result)
    
    return result

🎨 GIAI ĐOẠN 5: UI/UX IMPLEMENTATION (2-3 tuần)
Bước 5.1: React Components
Component 1: Territory Visualization
jsx// components/TerritoryOverlay.jsx

import React from 'react';
import { Canvas } from 'react-konva';

export const TerritoryOverlay = ({ territoryData, boardSize = 19 }) => {
  const cellSize = 30; // pixels
  
  const getColorForValue = (value) => {
    // value từ 0 (White) đến 1 (Black)
    if (value > 0.7) {
      // Black territory
      const alpha = (value - 0.7) / 0.3;  // 0.7-1.0 → 0-1
      return `rgba(0, 0, 0, ${alpha * 0.3})`;
    } else if (value < 0.3) {
      // White territory
      const alpha = (0.3 - value) / 0.3;
      return `rgba(255, 255, 255, ${alpha * 0.3})`;
    } else {
      // Contested
      return `rgba(255, 255, 0, 0.2)`;
    }
  };
  
  return (
    <div className="territory-overlay" style={{ position: 'absolute', top: 0, left: 0 }}>
      {territoryData.map((row, i) => (
        row.map((value, j) => (
          <div
            key={`${i}-RéessayerHContinuerOrchestrated continuation of ML documentation for Go game project.Người dùng muốn tôi tiếp tục với file tài liệu markdown. Họ đã yêu cầu tạo tài liệu đầy đủ về cách phát triển ML cho game cờ vây của họ. Tôi đã bắt đầu viết về các tính năng ML, roadmap, và đang ở phần UI/UX implementation.${j}`}style={{position: 'absolute',left: j * cellSize,top: i * cellSize,width: cellSize,height: cellSize,backgroundColor: getColorForValue(value),pointerEvents: 'none'}}/>))))}</div>);};

**Component 2: Win Probability Chart**
```jsx
// components/WinProbabilityChart.jsx

import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ReferenceLine } from 'recharts';

export const WinProbabilityChart = ({ gameHistory, currentMove }) => {
  const data = gameHistory.map((move, index) => ({
    move: index + 1,
    blackWinProb: move.win_probability.black,
    whiteWinProb: move.win_probability.white
  }));
  
  return (
    <div className="win-probability-chart">
      <h3>Biểu đồ xác suất thắng</h3>
      <LineChart width={600} height={300} data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="move" label={{ value: 'Nước đi', position: 'insideBottom', offset: -5 }} />
        <YAxis label={{ value: 'Xác suất', angle: -90, position: 'insideLeft' }} />
        <Tooltip />
        <Legend />
        <ReferenceLine y={0.5} stroke="gray" strokeDasharray="3 3" />
        <ReferenceLine x={currentMove} stroke="red" strokeDasharray="5 5" />
        <Line type="monotone" dataKey="blackWinProb" stroke="#000000" name="Black" strokeWidth={2} />
        <Line type="monotone" dataKey="whiteWinProb" stroke="#FFFFFF" name="White" strokeWidth={2} />
      </LineChart>
    </div>
  );
};
```

**Component 3: Tactical Threats Panel**
```jsx
// components/TacticalThreatsPanel.jsx

import React, { useState } from 'react';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';

export const TacticalThreatsPanel = ({ threats, onThreatClick }) => {
  const [selectedThreat, setSelectedThreat] = useState(null);
  
  const getPriorityColor = (priority) => {
    if (priority === 1) return 'bg-red-100 border-red-500';
    if (priority === 2) return 'bg-orange-100 border-orange-500';
    return 'bg-yellow-100 border-yellow-500';
  };
  
  const getPriorityIcon = (priority) => {
    if (priority === 1) return '🚨';
    if (priority === 2) return '⚠️';
    return 'ℹ️';
  };
  
  return (
    <div className="tactical-threats-panel p-4 bg-white rounded-lg shadow">
      <h3 className="text-lg font-bold mb-4">🎯 Phân tích chiến thuật</h3>
      
      {/* Urgent Threats */}
      <div className="mb-6">
        <h4 className="font-semibold text-red-600 mb-2">🔴 MỐI ĐE DỌA KHẨN CẤP</h4>
        {threats.urgent.map((threat, index) => (
          <Alert 
            key={index} 
            className={`mb-2 cursor-pointer ${getPriorityColor(threat.priority)}`}
            onClick={() => {
              setSelectedThreat(threat);
              onThreatClick(threat);
            }}
          >
            <AlertTitle>
              {getPriorityIcon(threat.priority)} {threat.type}
            </AlertTitle>
            <AlertDescription>
              <p className="font-medium">{threat.consequence}</p>
              <p className="text-sm mt-1">
                {threat.target_group && `Nhóm: ${threat.target_group}`}
                {threat.attack_move && ` • Nước tấn công: ${formatPosition(threat.attack_move)}`}
              </p>
            </AlertDescription>
          </Alert>
        ))}
      </div>
      
      {/* Tactical Opportunities */}
      <div className="mb-6">
        <h4 className="font-semibold text-orange-600 mb-2">🟡 CƠ HỘI CHIẾN THUẬT</h4>
        {threats.opportunities.map((opp, index) => (
          <div 
            key={index}
            className="border border-orange-300 rounded p-3 mb-2 cursor-pointer hover:bg-orange-50"
            onClick={() => {
              setSelectedThreat(opp);
              onThreatClick(opp);
            }}
          >
            <div className="flex justify-between items-start">
              <div>
                <p className="font-medium">✂️ {opp.type}</p>
                <p className="text-sm text-gray-600">{opp.effect}</p>
              </div>
              <div className="text-right">
                <p className="text-sm font-bold text-green-600">+{opp.value_gain} điểm</p>
              </div>
            </div>
            {opp.move && (
              <p className="text-xs mt-2 text-gray-500">
                Đánh tại: {formatPosition(opp.move)}
              </p>
            )}
          </div>
        ))}
      </div>
      
      {/* Defensive Needs */}
      <div>
        <h4 className="font-semibold text-blue-600 mb-2">🛡️ ĐIỂM YẾU CẦN BẢO VỆ</h4>
        {threats.defensive.map((def, index) => (
          <div 
            key={index}
            className="border border-blue-300 rounded p-3 mb-2 cursor-pointer hover:bg-blue-50"
            onClick={() => {
              setSelectedThreat(def);
              onThreatClick(def);
            }}
          >
            <p className="font-medium">{def.threat_type}</p>
            <p className="text-sm text-gray-600">Nhóm của bạn: {def.your_group}</p>
            <p className="text-sm mt-1 text-blue-600">
              💡 Gợi ý: {def.recommendation}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
};

const formatPosition = (pos) => {
  const [x, y] = pos;
  const letters = 'ABCDEFGHJKLMNOPQRST'; // Skip 'I'
  return `${letters[x]}${19 - y}`;
};
```

**Component 4: Strategic Intent Display**
```jsx
// components/StrategicIntentPanel.jsx

import React from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

export const StrategicIntentPanel = ({ opponentPlans, yourOptions }) => {
  return (
    <div className="strategic-intent-panel p-4 bg-white rounded-lg shadow">
      <h3 className="text-lg font-bold mb-4">🧠 Phân tích chiến lược</h3>
      
      <Tabs defaultValue="opponent">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="opponent">Ý đồ đối thủ</TabsTrigger>
          <TabsTrigger value="your">Lựa chọn của bạn</TabsTrigger>
        </TabsList>
        
        <TabsContent value="opponent" className="mt-4">
          <h4 className="font-semibold mb-3">AI dự đoán đối thủ sẽ:</h4>
          
          {opponentPlans.map((plan, index) => (
            <div key={index} className="mb-4 p-3 border rounded">
              <div className="flex justify-between items-start mb-2">
                <h5 className="font-medium">{plan.plan}</h5>
                <span className="text-sm bg-gray-200 px-2 py-1 rounded">
                  {Math.round(plan.probability * 100)}%
                </span>
              </div>
              
              <p className="text-sm text-gray-600 mb-2">{plan.reasoning}</p>
              
              {plan.counter_strategy && (
                <div className="bg-blue-50 p-2 rounded text-sm">
                  <p className="font-medium text-blue-700">💡 Cách đối phó:</p>
                  <p className="text-blue-600">{plan.counter_strategy}</p>
                </div>
              )}
            </div>
          ))}
        </TabsContent>
        
        <TabsContent value="your" className="mt-4">
          <h4 className="font-semibold mb-3">Các chiến lược bạn có thể chọn:</h4>
          
          {yourOptions.map((option, index) => (
            <div key={index} className="mb-4 p-3 border rounded">
              <div className="flex justify-between items-start mb-2">
                <h5 className="font-medium">{option.strategy}</h5>
                <div className="text-right">
                  <p className="text-sm text-green-600 font-bold">
                    +{option.expected_gain} điểm
                  </p>
                  <p className="text-xs text-gray-500">
                    Rủi ro: {option.risk}
                  </p>
                </div>
              </div>
              
              <p className="text-sm text-gray-600 mb-2">
                Vùng: {option.area}
              </p>
              
              {option.recommended_moves && (
                <div className="flex gap-2 mt-2">
                  <span className="text-xs text-gray-500">Nước đi đề xuất:</span>
                  {option.recommended_moves.map((move, i) => (
                    <span key={i} className="text-xs bg-green-100 px-2 py-1 rounded">
                      {formatPosition(move)}
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))}
        </TabsContent>
      </Tabs>
    </div>
  );
};
```

---

#### Bước 5.2: Main Analysis Page
```jsx
// pages/GameAnalysis.jsx

import React, { useState, useEffect } from 'react';
import { BoardDisplay } from '@/components/BoardDisplay';
import { TerritoryOverlay } from '@/components/TerritoryOverlay';
import { WinProbabilityChart } from '@/components/WinProbabilityChart';
import { TacticalThreatsPanel } from '@/components/TacticalThreatsPanel';
import { StrategicIntentPanel } from '@/components/StrategicIntentPanel';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

export const GameAnalysisPage = ({ gameId }) => {
  const [gameData, setGameData] = useState(null);
  const [analysisData, setAnalysisData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [visualizationMode, setVisualizationMode] = useState('territory');
  const [userCoins, setUserCoins] = useState(0);
  
  const ANALYSIS_COST = 10; // coins
  
  useEffect(() => {
    fetchGameData();
    fetchUserCoins();
  }, [gameId]);
  
  const fetchGameData = async () => {
    const response = await fetch(`/api/games/${gameId}`);
    const data = await response.json();
    setGameData(data);
  };
  
  const fetchUserCoins = async () => {
    const response = await fetch('/api/user/coins');
    const data = await response.json();
    setUserCoins(data.coins);
  };
  
  const runAnalysis = async () => {
    if (userCoins < ANALYSIS_COST) {
      alert('Không đủ coins! Bạn cần 10 coins để phân tích.');
      return;
    }
    
    setLoading(true);
    
    try {
      const response = await fetch('/api/ml/analyze-position', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          board: gameData.current_board,
          history: gameData.move_history,
          current_player: gameData.current_player
        })
      });
      
      const data = await response.json();
      setAnalysisData(data);
      
      // Deduct coins
      await fetch('/api/user/deduct-coins', {
        method: 'POST',
        body: JSON.stringify({ amount: ANALYSIS_COST })
      });
      
      setUserCoins(userCoins - ANALYSIS_COST);
      
    } catch (error) {
      console.error('Analysis failed:', error);
      alert('Phân tích thất bại. Vui lòng thử lại.');
    } finally {
      setLoading(false);
    }
  };
  
  if (!gameData) return <div>Loading...</div>;
  
  return (
    <div className="game-analysis-page p-6">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold">🔍 Phân tích bàn cờ bằng AI</h1>
        <div className="flex gap-4 items-center">
          <div className="text-sm">
            Coins: <span className="font-bold">{userCoins}</span>
          </div>
          <Button 
            onClick={runAnalysis}
            disabled={loading || userCoins < ANALYSIS_COST}
            className="bg-blue-500 hover:bg-blue-600"
          >
            {loading ? 'Đang phân tích...' : `🧠 Phân tích AI (${ANALYSIS_COST} coins)`}
          </Button>
        </div>
      </div>
      
      <div className="grid grid-cols-3 gap-6">
        {/* Left: Board with overlays */}
        <div className="col-span-2">
          <div className="relative">
            <BoardDisplay board={gameData.current_board} />
            
            {analysisData && (
              <>
                {visualizationMode === 'territory' && (
                  <TerritoryOverlay 
                    territoryData={analysisData.territory_analysis.map}
                  />
                )}
                
                {visualizationMode === 'threats' && (
                  <ThreatOverlay 
                    threatsData={analysisData.tactical_threats}
                  />
                )}
              </>
            )}
          </div>
          
          {analysisData && (
            <div className="mt-4">
              <Tabs value={visualizationMode} onValueChange={setVisualizationMode}>
                <TabsList className="grid w-full grid-cols-3">
                  <TabsTrigger value="territory">🗺️ Lãnh thổ</TabsTrigger>
                  <TabsTrigger value="threats">⚔️ Mối đe dọa</TabsTrigger>
                  <TabsTrigger value="influence">🌊 Ảnh hưởng</TabsTrigger>
                </TabsList>
              </Tabs>
            </div>
          )}
          
          {analysisData && (
            <div className="mt-6">
              <WinProbabilityChart 
                gameHistory={gameData.move_history}
                currentMove={gameData.current_move}
              />
            </div>
          )}
        </div>
        
        {/* Right: Analysis panels */}
        <div className="col-span-1 space-y-6">
          {analysisData ? (
            <>
              {/* Territory Stats */}
              <div className="bg-white p-4 rounded-lg shadow">
                <h3 className="font-bold mb-3">📊 Thống kê lãnh thổ</h3>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span>⚫ Lãnh thổ Black:</span>
                    <span className="font-bold">
                      {analysisData.territory_analysis.black_territory} điểm
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>⚪ Lãnh thổ White:</span>
                    <span className="font-bold">
                      {analysisData.territory_analysis.white_territory} điểm
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>🟡 Vùng tranh chấp:</span>
                    <span className="font-bold">
                      {analysisData.territory_analysis.contested} điểm
                    </span>
                  </div>
                  <div className="border-t pt-2 mt-2">
                    <div className="flex justify-between font-bold">
                      <span>Xác suất thắng:</span>
                      <span className={
                        analysisData.win_probability.black > 0.5 
                          ? 'text-green-600' 
                          : 'text-red-600'
                      }>
                        {Math.round(analysisData.win_probability.black * 100)}%
                      </span>
                    </div>
                  </div>
                </div>
              </div>
              
              {/* Tactical Threats */}
              <TacticalThreatsPanel 
                threats={analysisData.tactical_threats}
                onThreatClick={(threat) => {
                  // Highlight threat on board
                  console.log('Show threat:', threat);
                }}
              />
              
              {/* Strategic Intent (if available) */}
              {analysisData.strategic_intent && (
                <StrategicIntentPanel 
                  opponentPlans={analysisData.strategic_intent.opponent_plans}
                  yourOptions={analysisData.strategic_intent.your_options}
                />
              )}
            </>
          ) : (
            <div className="bg-gray-50 p-8 rounded-lg text-center">
              <p className="text-gray-600 mb-4">
                Nhấn nút "Phân tích AI" để xem phân tích chi tiết về vị thế hiện tại
              </p>
              <ul className="text-sm text-left text-gray-500 space-y-2">
                <li>✓ Dự đoán lãnh thổ chính xác</li>
                <li>✓ Phát hiện mối đe dọa chiến thuật</li>
                <li>✓ Xác suất thắng theo thời gian</li>
                <li>✓ Phân tích ý đồ chiến lược</li>
              </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
```

---

## 7. MONETIZATION STRATEGY

### 💰 Pricing Model
```javascript
// Coin packages
const COIN_PACKAGES = {
  starter: {
    coins: 50,
    price: 9000,  // VND (~ $0.38)
    bonus: 0
  },
  basic: {
    coins: 150,
    price: 25000,  // ~ $1
    bonus: 10  // +10 coins
  },
  standard: {
    coins: 500,
    price: 75000,  // ~ $3
    bonus: 50  // +50 coins (10% bonus)
  },
  premium: {
    coins: 1500,
    price: 200000,  // ~ $8
    bonus: 300  // +300 coins (20% bonus)
  }
};

// Feature pricing
const FEATURE_COSTS = {
  position_analysis: 10,        // Phân tích cơ bản
  full_analysis: 20,            // Phân tích đầy đủ + strategic intent
  post_game_review: 30,         // Review toàn bộ ván đấu
  training_mode: 5,             // Mỗi bài tập tsumego
  opening_book_access: 100,     // Mua 1 lần (permanent)
  ai_hint: 15                   // Gợi ý nước đi tốt
};

// Subscription tiers (monthly)
const SUBSCRIPTIONS = {
  free: {
    price: 0,
    features: {
      games_per_day: 'unlimited',
      basic_stats: true,
      position_analysis: 0,  // 0 free uses/day
      post_game_review: 0
    }
  },
  
  silver: {
    price: 50000,  // ~ $2/month
    features: {
      games_per_day: 'unlimited',
      basic_stats: true,
      position_analysis: 5,  // 5 free uses/day
      post_game_review: 2,   // 2 free/day
      discount: 0.1          // 10% off coin purchases
    }
  },
  
  gold: {
    price: 150000,  // ~ $6/month
    features: {
      games_per_day: 'unlimited',
      basic_stats: true,
      position_analysis: 'unlimited',
      post_game_review: 10,  // 10/day
      training_mode: 'unlimited',
      opening_book_access: true,
      discount: 0.2,  // 20% off
      priority_support: true
    }
  }
};
```

### 📈 Revenue Projection (Conservative)
Assumptions:

5,000 monthly active users
3% conversion to paying (Silver)
1% conversion to Gold
Average non-subscriber spends 20,000 VND/month on coins

Monthly Revenue:
Silver subs:   5000 × 3% × 50,000    = 7,500,000 VND
Gold subs:     5000 × 1% × 150,000   = 7,500,000 VND
Coin sales:    5000 × 96% × 20,000   = 9,600,000 VND
Total:  24,600,000 VND/month  (~$1,000 USD)
Annual: 295,200,000 VND       ($12,000 USD)
With growth to 20,000 users:
Annual: 1,180,800,000 VND     ($48,000 USD)

---

## 8. BEST PRACTICES & TIPS

### ✅ Những điều NÊN LÀM

1. **Start Small, Iterate Fast**
   - Bắt đầu với 1 model đơn giản (Territory Predictor)
   - Test trên 9x9 board trước
   - Có working prototype trước khi scale

2. **Data Quality > Quantity**
   - 5,000 high-quality games > 50,000 low-quality games
   - Filter games: chỉ lấy từ players >5d
   - Clean data: remove obvious mistakes

3. **Monitor Training Closely**
   - Use TensorBoard để visualize
   - Save checkpoints thường xuyên
   - Validate trên data thật (không phải training set)

4. **User Testing Early**
   - Beta test với nhóm nhỏ users
   - Gather feedback về UI/UX
   - Iterate based on real usage

5. **Cache Aggressively**
   - ML inference expensive
   - Cache results với Redis
   - Pre-compute common positions

### ❌ Những điều TRÁNH

1. **Đừng over-engineer lúc đầu**
   - Không cần ResNet-50 với 100 layers
   - Lightweight CNN (5-7 layers) đủ
   - Scale khi có evidence cần thiết

2. **Đừng train trên toàn bộ 19x19 ngay**
   - Bắt đầu với 9x9 (faster training)
   - Validate concept
   - Mở rộng sau

3. **Đừng promise quá nhiều**
   - ML không phải magic
   - Set expectations đúng với users
   - Underpromise, overdeliver

4. **Đừng skip validation**
   - Test trên positions thật
   - So sánh với human experts
   - Measure real accuracy

5. **Đừng hardcode thresholds**
   - Make parameters configurable
   - A/B test different values
   - Let data guide decisions

---

## 9. TIMELINE TỔNG THỂ
┌─────────────────────────────────────────────────────────────┐
│              FULL ROADMAP (12-16 tuần)                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  PHASE 1: Data & Prep (3 tuần)                             │
│  ├─ Week 1: Download & parse SGF files                     │
│  ├─ Week 2: Create labels & features                       │
│  └─ Week 3: Setup training infrastructure                  │
│                                                             │
│  PHASE 2: Model Development (4 tuần)                       │
│  ├─ Week 4: Design architectures                           │
│  ├─ Week 5: Train Territory Predictor                      │
│  ├─ Week 6: Train Value Network                            │
│  └─ Week 7: Train Tactical Detector                        │
│                                                             │
│  PHASE 3: Backend Integration (2 tuần)                     │
│  ├─ Week 8: Setup ML serving API                           │
│  └─ Week 9: Implement caching & optimization               │
│                                                             │
│  PHASE 4: Frontend Implementation (3 tuần)                 │
│  ├─ Week 10: Build visualization components                │
│  ├─ Week 11: Create analysis page                          │
│  └─ Week 12: Implement monetization UI                     │
│                                                             │
│  PHASE 5: Testing & Polish (2 tuần)                        │
│  ├─ Week 13: Beta testing with users                       │
│  ├─ Week 14: Bug fixes & improvements                      │
│  └─ Week 15-16: Soft launch & iteration                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘

---

## 10. KẾT LUẬN & NEXT STEPS

### 🎯 Tóm tắt

Dự án ML này sẽ tạo ra một **hệ thống phân tích cờ vây thông minh** với 4 tính năng chính:

1. **Territory Prediction** - Dự đoán lãnh thổ với visualization rõ ràng
2. **Win Probability** - Xác suất thắng theo thời gian thực
3. **Tactical Detection** - Phát hiện atari, cuts, weak groups
4. **Strategic Intent** - Phân tích ý đồ đối thủ (advanced)

### 🚀 Bắt đầu ngay

**Bước đầu tiên (Tuần này):**
```bash
# 1. Setup môi trường
cd your-project
python -m venv ml_env
source ml_env/bin/activate
pip install torch torchvision numpy pandas sgfmill

# 2. Download data
mkdir -p data/raw/kgs
python scripts/download_kgs.py --limit 1000  # Start small

# 3. Parse SGF
python scripts/parse_sgf.py --input data/raw/kgs --output data/parsed

# 4. Create first training batch
python scripts/create_features.py --input data/parsed --output data/processed
```

### 📚 Resources

**Tutorials & Papers:**
- AlphaGo Paper (simplified): https://www.nature.com/articles/nature16961
- PyTorch CNN Tutorial: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
- Go datasets: https://u-go.net/gamerecords/

**Tools:**
- TensorBoard: `pip install tensorboard`
- Weights & Biases (optional): https://wandb.ai

### 💬 Support

Khi gặp khó khăn:
1. Check documentation ở trên
2. Test với subset nhỏ trước
3. Visualize intermediate results
4. Ask for help với specific error messages

---

## PHẦN PHỤ LỤC

### A. Sample Training Script
```python
# train_territory_predictor.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm

class GoDataset(Dataset):
    def __init__(self, features_dir, labels_dir):
        self.features = np.load(f'{features_dir}/features.npy')
        self.labels = np.load(f'{labels_dir}/territory_labels.npy')
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.features[idx]),
            torch.FloatTensor(self.labels[idx])
        )

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc='Training')
    for features, labels in pbar:
        features = features.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for features, labels in tqdm(loader, desc='Validating'):
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    
    return total_loss / len(loader)

def main():
    # Config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 50
    
    # Data
    train_dataset = GoDataset('data/processed/train', 'data/labels/train')
    val_dataset = GoDataset('data/processed/val', 'data/labels/val')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Model
    model = TerritoryPredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'checkpoints/best_territory_model.pth')
            print('✓ Saved best model')

if __name__ == '__main__':
    main()
```

---

### B. Sample API Endpoint Test
```python
# test_ml_api.py

import requests
import numpy as np
import json

def test_position_analysis():
    """
    Test ML API endpoint
    """
    # Sample board state (9x9 for testing)
    board = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 2, 2, 0],
        [0, 1, 0, 1, 0, 2, 0, 0, 2],
        # ... rest of board
    ]
    
    payload = {
        "board": board,
        "history": [],
        "current_player": 1
    }
    
    response = requests.post(
        'http://localhost:8000/api/ml/analyze-position',
        json=payload
    )
    
    if response.status_code == 200:
        result = response.json()
        
        print("✓ Analysis successful!")
        print(f"Black territory: {result['territory_analysis']['black_territory']}")
        print(f"White territory: {result['territory_analysis']['white_territory']}")
        print(f"Win probability: {result['win_probability']['black']:.2%}")
        
        if result['tactical_threats']['atari_positions']:
            print(f"⚠️ Found {len(result['tactical_threats']['atari_positions'])} atari threats")
    else:
        print(f"✗ Error: {response.status_code}")
        print(response.text)

if __name__ == '__main__':
    test_position_analysis()
```

---

**HẾT TÀI LIỆU**

Chúc bạn thành công với dự án ML! 🚀

*Lưu ý: Đây là roadmap đầy đủ nhưng flexible - bạn có thể adjust theo tình hình thực tế. Quan trọng là start small và iterate dựa trên feedback thật.*