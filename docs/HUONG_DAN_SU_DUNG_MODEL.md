# 📖 HƯỚNG DẪN SỬ DỤNG MODEL ĐÃ TRAIN

## 📋 Mục Lục

1. [Tổng Quan](#1-tổng-quan)
2. [Cài Đặt và Yêu Cầu](#2-cài-đặt-và-yêu-cầu)
3. [Load Model](#3-load-model)
4. [Chuẩn Bị Dữ Liệu Đầu Vào](#4-chuẩn-bị-dữ-liệu-đầu-vào)
5. [Thực Hiện Dự Đoán](#5-thực-hiện-dự-đoán)
6. [Tích Hợp Vào Game](#6-tích-hợp-vào-game)
7. [Ví Dụ Hoàn Chỉnh](#7-ví-dụ-hoàn-chỉnh)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Tổng Quan

Sau khi training xong, bạn sẽ có các checkpoint files:
- `best_model.pt` - Model tốt nhất (validation loss thấp nhất) ⭐ **Khuyến nghị dùng**
- `final_model.pt` - Model sau epoch cuối cùng
- `checkpoint_epoch_X.pt` - Checkpoints định kỳ

Model bao gồm 2 networks:
- **Policy Network**: Dự đoán xác suất cho mỗi nước đi (move probabilities)
- **Value Network**: Dự đoán xác suất thắng của người chơi hiện tại (win probability)

### 📍 Vị Trí Đặt File Model

**Quan trọng**: Sau khi tải file model (ví dụ: `final_model.pt`), bạn cần đặt nó vào thư mục `checkpoints/` ở **root của project**:

```
GoGame-master/
├── checkpoints/              ← Đặt file model ở đây
│   ├── final_model.pt       ← Copy file của bạn vào đây
│   └── README.md
├── docs/
├── scripts/
└── ...
```

**Các bước:**
1. Tìm thư mục `GoGame-master` (thư mục gốc của project)
2. Vào thư mục `checkpoints/` (nếu chưa có, sẽ được tạo tự động)
3. Copy file `final_model.pt` vào thư mục này
4. Đường dẫn đầy đủ sẽ là: `GoGame-master/checkpoints/final_model.pt`

**Ví dụ đường dẫn trên Windows:**
```
C:\Users\Gigabyte\OneDrive - Ho Chi Minh city University of Industry and Trade\Máy tính\lamphuocthuan\Python\GoGame-master\checkpoints\final_model.pt
```

Sau khi đặt file, bạn có thể load model bằng đường dẫn tương đối:
```python
checkpoint_path = 'checkpoints/final_model.pt'
```

---

## 2. Cài Đặt và Yêu Cầu

### Dependencies

```bash
pip install torch torchvision torchaudio
pip install numpy
```

### Import cần thiết

```python
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# Import model classes (từ src/ml hoặc copy vào project)
from policy_network import PolicyNetwork, PolicyConfig
from value_network import ValueNetwork, ValueConfig
```

---

## 3. Load Model

### 3.1. Load từ `best_model.pt` (Khuyến nghị)

```python
def load_trained_model(checkpoint_path: str, device: str = 'cpu'):
    """
    Load trained model từ checkpoint.
    
    Args:
        checkpoint_path: Đường dẫn đến checkpoint file (ví dụ: 'checkpoints/best_model.pt')
        device: 'cpu' hoặc 'cuda'
    
    Returns:
        policy_net: PolicyNetwork instance
        value_net: ValueNetwork instance
        board_size: Kích thước bàn cờ
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Lấy config từ checkpoint
    policy_config = PolicyConfig(**checkpoint['policy_config'])
    value_config = ValueConfig(**checkpoint['value_config'])
    board_size = checkpoint['board_size']
    
    # Khởi tạo models
    policy_net = PolicyNetwork(policy_config)
    value_net = ValueNetwork(value_config)
    
    # Load weights
    policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    value_net.load_state_dict(checkpoint['value_net_state_dict'])
    
    # Chuyển sang device và set eval mode
    policy_net = policy_net.to(device)
    value_net = value_net.to(device)
    policy_net.eval()
    value_net.eval()
    
    return policy_net, value_net, board_size

# Sử dụng
device = 'cuda' if torch.cuda.is_available() else 'cpu'
policy_net, value_net, board_size = load_trained_model(
    'checkpoints/best_model.pt',
    device=device
)
```

### 3.2. Load từ checkpoint epoch cụ thể

```python
# Load từ checkpoint epoch 5
policy_net, value_net, board_size = load_trained_model(
    'checkpoints/dataset_2019_checkpoint_epoch_5.pt',
    device='cpu'
)
```

### 3.3. Lưu ý quan trọng

- **Luôn dùng `eval()` mode**: Tắt dropout và batch normalization trong inference
- **Dùng `torch.no_grad()`**: Tắt gradient computation để tiết kiệm memory và tăng tốc
- **`map_location`**: Dùng `'cpu'` nếu load trên CPU, hoặc `'cuda:0'` nếu load trên GPU

---

## 4. Chuẩn Bị Dữ Liệu Đầu Vào

Model yêu cầu input là **17-plane features** với shape `[1, 17, board_size, board_size]`.

### 4.1. 17-Plane Features Format

| Plane | Mô tả |
|-------|-------|
| 0 | Quân cờ của người chơi hiện tại |
| 1 | Quân cờ của đối thủ |
| 2 | Quân cờ của người chơi hiện tại có 1 liberty |
| 3 | Quân cờ của đối thủ có 1 liberty |
| 4 | Quân cờ của người chơi hiện tại có 2 liberties |
| 5 | Quân cờ của đối thủ có 2 liberties |
| 6 | Quân cờ của người chơi hiện tại có 3+ liberties |
| 7 | Quân cờ của đối thủ có 3+ liberties |
| 8-15 | Lịch sử nước đi (4 nước gần nhất, mỗi nước = 2 planes) |
| 16 | Chỉ số lượt chơi (1.0 nếu Black, 0.0 nếu White) |

### 4.2. Hàm tạo 17-plane features

```python
def get_liberties_simple(board_state: np.ndarray, x: int, y: int, board_size: int) -> int:
    """Tính số liberties của một quân cờ tại (x, y)."""
    if board_state[y, x] == 0:
        return 0
    
    color = board_state[y, x]
    liberties = 0
    
    # Check 4 neighbors
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < board_size and 0 <= ny < board_size:
            if board_state[ny, nx] == 0:
                liberties += 1
    
    return liberties


def board_to_features_17_planes(
    board_state: np.ndarray,
    current_player: str,
    move_history: list = None,
    board_size: int = 9
) -> torch.Tensor:
    """
    Convert board state thành 17-plane tensor.
    
    Args:
        board_state: numpy array [board_size, board_size]
                    0 = empty, 1 = black, 2 = white
        current_player: 'B' hoặc 'W'
        move_history: List of (x, y) tuples cho last 4 moves
        board_size: Kích thước bàn cờ
    
    Returns:
        Tensor [17, board_size, board_size]
    """
    features = torch.zeros((17, board_size, board_size), dtype=torch.float32)
    
    # Plane 0: Current player stones
    # Plane 1: Opponent stones
    if current_player == 'B':
        features[0] = torch.from_numpy((board_state == 1).astype(np.float32))
        features[1] = torch.from_numpy((board_state == 2).astype(np.float32))
    else:  # White
        features[0] = torch.from_numpy((board_state == 2).astype(np.float32))
        features[1] = torch.from_numpy((board_state == 1).astype(np.float32))
    
    # Plane 2-7: Liberty counts
    for y in range(board_size):
        for x in range(board_size):
            if board_state[y, x] == 0:
                continue
            
            # Determine if this is current player's stone
            is_current = (
                (current_player == 'B' and board_state[y, x] == 1) or
                (current_player == 'W' and board_state[y, x] == 2)
            )
            
            if is_current:
                liberties = get_liberties_simple(board_state, x, y, board_size)
                if liberties == 1:
                    features[2, y, x] = 1.0
                elif liberties == 2:
                    features[4, y, x] = 1.0
                elif liberties >= 3:
                    features[6, y, x] = 1.0
            else:
                # Opponent stones
                liberties = get_liberties_simple(board_state, x, y, board_size)
                if liberties == 1:
                    features[3, y, x] = 1.0
                elif liberties == 2:
                    features[5, y, x] = 1.0
                elif liberties >= 3:
                    features[7, y, x] = 1.0
    
    # Plane 8-15: Move history (last 4 moves)
    if move_history is None:
        move_history = []
    
    # Chỉ lấy 4 nước gần nhất
    move_history = move_history[-4:]
    
    for i, (mx, my) in enumerate(move_history):
        if i >= 4:
            break
        # Mỗi move = 2 planes (x và y)
        plane_x = 8 + i * 2
        plane_y = 9 + i * 2
        if 0 <= mx < board_size and 0 <= my < board_size:
            features[plane_x, my, mx] = 1.0
            features[plane_y, my, mx] = 1.0
    
    # Plane 16: Turn indicator
    features[16].fill_(1.0 if current_player == 'B' else 0.0)
    
    return features
```

### 4.3. Ví dụ tạo features từ board state

```python
# Giả sử bạn có board state từ game
board_state = np.array([
    [0, 0, 1, 0, 0],
    [0, 1, 2, 1, 0],
    [1, 2, 0, 2, 1],
    [0, 1, 2, 1, 0],
    [0, 0, 1, 0, 0]
])  # 5x5 board, 1=black, 2=white, 0=empty

current_player = 'B'  # Black's turn
move_history = [(2, 2), (1, 1)]  # Last 2 moves

# Tạo features
features = board_to_features_17_planes(
    board_state=board_state,
    current_player=current_player,
    move_history=move_history,
    board_size=5
)

# Thêm batch dimension: [1, 17, board_size, board_size]
features = features.unsqueeze(0)  # Shape: [1, 17, 5, 5]
```

---

## 5. Thực Hiện Dự Đoán

### 5.1. Dự đoán Policy (Move Probabilities)

```python
def predict_move(policy_net, features, board_size):
    """
    Dự đoán xác suất cho mỗi nước đi.
    
    Args:
        policy_net: PolicyNetwork instance
        features: Tensor [1, 17, board_size, board_size]
        board_size: Kích thước bàn cờ
    
    Returns:
        policy_probs: Tensor [board_size * board_size] - xác suất cho mỗi move
        best_move: (x, y) - nước đi có xác suất cao nhất
    """
    policy_net.eval()
    
    with torch.no_grad():
        # Forward pass
        policy_logits = policy_net(features)  # Shape: [1, board_size * board_size]
        
        # Convert logits to probabilities
        policy_probs = torch.exp(policy_logits[0])  # Shape: [board_size * board_size]
    
    # Tìm nước đi tốt nhất
    best_move_idx = torch.argmax(policy_probs).item()
    best_move_y = best_move_idx // board_size
    best_move_x = best_move_idx % board_size
    
    return policy_probs, (best_move_x, best_move_y)

# Sử dụng
policy_probs, best_move = predict_move(policy_net, features, board_size)
print(f"Best move: {best_move}")
print(f"Probability: {policy_probs[best_move[1] * board_size + best_move[0]]:.4f}")
```

### 5.2. Dự đoán Value (Win Probability)

```python
def predict_value(value_net, features):
    """
    Dự đoán xác suất thắng của người chơi hiện tại.
    
    Args:
        value_net: ValueNetwork instance
        features: Tensor [1, 17, board_size, board_size]
    
    Returns:
        win_probability: float trong khoảng [0, 1]
    """
    value_net.eval()
    
    with torch.no_grad():
        value_pred = value_net(features)  # Shape: [1, 1]
        win_prob = value_pred[0, 0].item()
    
    return win_prob

# Sử dụng
win_prob = predict_value(value_net, features)
print(f"Win probability: {win_prob:.4f} ({win_prob * 100:.2f}%)")
```

### 5.3. Dự đoán kết hợp (Policy + Value)

```python
def predict(policy_net, value_net, features, board_size):
    """
    Dự đoán cả policy và value cùng lúc.
    
    Returns:
        policy_probs: Tensor [board_size * board_size]
        best_move: (x, y)
        win_prob: float
    """
    policy_net.eval()
    value_net.eval()
    
    with torch.no_grad():
        # Policy prediction
        policy_logits = policy_net(features)
        policy_probs = torch.exp(policy_logits[0])
        
        # Value prediction
        value_pred = value_net(features)
        win_prob = value_pred[0, 0].item()
    
    # Best move
    best_move_idx = torch.argmax(policy_probs).item()
    best_move_y = best_move_idx // board_size
    best_move_x = best_move_idx % board_size
    
    return policy_probs, (best_move_x, best_move_y), win_prob

# Sử dụng
policy_probs, best_move, win_prob = predict(policy_net, value_net, features, board_size)
print(f"Best move: {best_move}")
print(f"Win probability: {win_prob:.4f}")
```

---

## 6. Tích Hợp Vào Game

### 6.1. Class wrapper cho dễ sử dụng

```python
class GoAIModel:
    """
    Wrapper class để sử dụng trained model trong game.
    """
    
    def __init__(self, checkpoint_path: str, device: str = 'cpu'):
        """
        Args:
            checkpoint_path: Đường dẫn đến checkpoint file
            device: 'cpu' hoặc 'cuda'
        """
        self.device = torch.device(device)
        self.policy_net, self.value_net, self.board_size = load_trained_model(
            checkpoint_path, device=device
        )
    
    def predict_move(self, board_state: np.ndarray, current_player: str, 
                     move_history: list = None) -> tuple:
        """
        Predict move từ board state.
        
        Args:
            board_state: numpy array [board_size, board_size]
                        0 = empty, 1 = black, 2 = white
            current_player: 'B' hoặc 'W'
            move_history: List of (x, y) tuples cho last 4 moves
        
        Returns:
            best_move: (x, y) - nước đi tốt nhất
            policy_probs: numpy array [board_size * board_size]
            win_prob: float - xác suất thắng
        """
        # Tạo features
        features = board_to_features_17_planes(
            board_state, current_player, move_history, self.board_size
        )
        features = features.unsqueeze(0).to(self.device)  # [1, 17, board_size, board_size]
        
        # Predict
        policy_probs, best_move, win_prob = predict(
            self.policy_net, self.value_net, features, self.board_size
        )
        
        # Convert to numpy
        policy_probs_np = policy_probs.cpu().numpy()
        
        return best_move, policy_probs_np, win_prob
    
    def get_top_moves(self, board_state: np.ndarray, current_player: str,
                      move_history: list = None, top_k: int = 5) -> list:
        """
        Lấy top K nước đi tốt nhất.
        
        Returns:
            List of tuples: [(x, y, probability), ...]
        """
        _, policy_probs, _ = self.predict_move(board_state, current_player, move_history)
        
        # Get top K indices
        top_indices = np.argsort(policy_probs)[-top_k:][::-1]
        
        top_moves = []
        for idx in top_indices:
            y = idx // self.board_size
            x = idx % self.board_size
            prob = policy_probs[idx]
            top_moves.append((x, y, prob))
        
        return top_moves

# Sử dụng trong game
model = GoAIModel('checkpoints/best_model.pt', device='cpu')

# Trong game loop
board_state = get_current_board_state()  # Hàm của bạn
current_player = 'B'
move_history = get_recent_moves()  # Hàm của bạn

# Predict move
best_move, policy_probs, win_prob = model.predict_move(
    board_state, current_player, move_history
)

# Lấy top 5 moves
top_moves = model.get_top_moves(board_state, current_player, move_history, top_k=5)
for x, y, prob in top_moves:
    print(f"Move ({x}, {y}): {prob:.4f}")
```

### 6.2. Tích hợp với game engine

```python
# Ví dụ tích hợp với game engine
class GameEngine:
    def __init__(self):
        self.model = GoAIModel('checkpoints/best_model.pt', device='cpu')
        self.board = initialize_board()
        self.current_player = 'B'
        self.move_history = []
    
    def ai_move(self):
        """AI thực hiện nước đi dựa trên model."""
        # Lấy board state
        board_state = self.get_board_state()
        
        # Predict
        best_move, _, win_prob = self.model.predict_move(
            board_state, self.current_player, self.move_history
        )
        
        # Thực hiện move
        x, y = best_move
        if self.is_valid_move(x, y):
            self.make_move(x, y)
            self.move_history.append((x, y))
            self.current_player = 'W' if self.current_player == 'B' else 'B'
            return True
        
        return False
    
    def get_ai_suggestion(self):
        """Lấy gợi ý nước đi từ AI (không thực hiện move)."""
        board_state = self.get_board_state()
        best_move, policy_probs, win_prob = self.model.predict_move(
            board_state, self.current_player, self.move_history
        )
        
        return {
            'best_move': best_move,
            'win_probability': win_prob,
            'top_moves': self.model.get_top_moves(
                board_state, self.current_player, self.move_history, top_k=5
            )
        }
```

---

## 7. Ví Dụ Hoàn Chỉnh

### 7.1. Script đơn giản để test model

```python
import torch
import numpy as np
from pathlib import Path

# Import các hàm đã định nghĩa ở trên
from load_model import load_trained_model
from features import board_to_features_17_planes
from predict import predict

def test_model():
    """Test model với board state mẫu."""
    
    # Load model
    print("Loading model...")
    checkpoint_path = 'checkpoints/best_model.pt'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    policy_net, value_net, board_size = load_trained_model(
        checkpoint_path, device=device
    )
    print(f"✅ Model loaded! Board size: {board_size}")
    
    # Tạo board state mẫu
    board_state = np.zeros((board_size, board_size), dtype=np.int32)
    # Thêm một số quân cờ mẫu
    board_state[3, 3] = 1  # Black
    board_state[3, 4] = 2  # White
    board_state[4, 3] = 2  # White
    board_state[4, 4] = 1  # Black
    
    current_player = 'B'
    move_history = [(3, 3), (3, 4)]
    
    # Tạo features
    features = board_to_features_17_planes(
        board_state, current_player, move_history, board_size
    )
    features = features.unsqueeze(0).to(device)
    
    # Predict
    print("\nPredicting...")
    policy_probs, best_move, win_prob = predict(
        policy_net, value_net, features, board_size
    )
    
    print(f"\n📊 Results:")
    print(f"  Best move: {best_move}")
    print(f"  Win probability: {win_prob:.4f} ({win_prob * 100:.2f}%)")
    print(f"  Top 5 moves:")
    
    # Top 5 moves
    top_indices = torch.argsort(policy_probs, descending=True)[:5]
    for i, idx in enumerate(top_indices):
        y = idx.item() // board_size
        x = idx.item() % board_size
        prob = policy_probs[idx].item()
        print(f"    {i+1}. ({x}, {y}): {prob:.4f}")

if __name__ == '__main__':
    test_model()
```

### 7.2. Batch prediction (nhiều positions cùng lúc)

```python
def batch_predict(policy_net, value_net, features_batch, board_size):
    """
    Predict cho nhiều positions cùng lúc (batch).
    
    Args:
        features_batch: Tensor [batch_size, 17, board_size, board_size]
    
    Returns:
        policy_probs_batch: Tensor [batch_size, board_size * board_size]
        value_batch: Tensor [batch_size, 1]
    """
    policy_net.eval()
    value_net.eval()
    
    with torch.no_grad():
        policy_logits = policy_net(features_batch)
        policy_probs = torch.exp(policy_logits)
        
        value_pred = value_net(features_batch)
    
    return policy_probs, value_pred

# Sử dụng
batch_size = 32
features_batch = torch.randn(batch_size, 17, board_size, board_size)

policy_batch, value_batch = batch_predict(
    policy_net, value_net, features_batch, board_size
)

print(f"Policy shape: {policy_batch.shape}")  # [32, board_size * board_size]
print(f"Value shape: {value_batch.shape}")  # [32, 1]
```

---

## 8. Troubleshooting

### 8.1. Lỗi: "File not found"

```python
# Kiểm tra đường dẫn
checkpoint_path = Path('checkpoints/best_model.pt')
if not checkpoint_path.exists():
    print(f"❌ Checkpoint not found: {checkpoint_path}")
    print(f"   Current directory: {Path.cwd()}")
    print(f"   Available files: {list(Path('checkpoints').glob('*.pt'))}")
```

### 8.2. Lỗi: "KeyError: 'policy_config'"

Checkpoint format có thể khác nhau. Kiểm tra keys trong checkpoint:

```python
checkpoint = torch.load('checkpoints/best_model.pt', map_location='cpu')
print("Checkpoint keys:", checkpoint.keys())

# Nếu không có 'policy_config', có thể cần load thủ công:
if 'policy_config' not in checkpoint:
    # Thử load với format cũ
    policy_config = PolicyConfig(
        board_size=checkpoint.get('board_size', 9),
        input_planes=17,
        channels=128  # hoặc từ checkpoint nếu có
    )
```

### 8.3. Lỗi: "Shape mismatch"

Đảm bảo features có đúng shape:

```python
# Kiểm tra shape
print(f"Features shape: {features.shape}")
print(f"Expected: [1, 17, {board_size}, {board_size}]")

# Nếu thiếu batch dimension
if features.dim() == 3:
    features = features.unsqueeze(0)
```

### 8.4. Model chạy chậm

Tối ưu tốc độ:

```python
# 1. Dùng GPU nếu có
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 2. Dùng torch.jit.script hoặc torch.compile (PyTorch 2.0+)
if hasattr(torch, 'compile'):
    policy_net = torch.compile(policy_net)
    value_net = torch.compile(value_net)

# 3. Batch predictions thay vì từng cái một
# Thay vì predict 100 lần riêng lẻ, gom lại thành 1 batch

# 4. Dùng half precision (FP16) nếu GPU hỗ trợ
if device.type == 'cuda':
    policy_net = policy_net.half()
    value_net = value_net.half()
    features = features.half()
```

### 8.5. Memory issues

```python
# 1. Clear cache sau mỗi prediction
torch.cuda.empty_cache()

# 2. Dùng CPU nếu GPU hết memory
device = 'cpu'

# 3. Giảm batch size
batch_size = 1  # Thay vì 32
```

---

## 📚 Tài Liệu Tham Khảo

- **Training guide**: `scripts/README_COLAB_TRAINING.md`
- **Model architecture**: `src/ml/policy_network.py`, `src/ml/value_network.py`
- **Feature generation**: `scripts/generate_features_colab.py`
- **Comprehensive ML guide**: `docs/ML_COMPREHENSIVE_GUIDE.md`

---

## 💡 Tips và Best Practices

1. **Luôn dùng `best_model.pt`**: Model này có validation loss thấp nhất
2. **Set `eval()` mode**: Quan trọng để tắt dropout và batch norm
3. **Dùng `torch.no_grad()`**: Tắt gradient để tiết kiệm memory
4. **Batch predictions**: Gom nhiều predictions lại để tăng tốc
5. **Cache features**: Nếu cùng board state, cache features để tránh tính lại
6. **Validate moves**: Luôn kiểm tra move có hợp lệ trước khi thực hiện
7. **Monitor performance**: Đo thời gian inference để tối ưu

---

## ✅ Checklist Trước Khi Sử Dụng

- [ ] Model đã được train và có checkpoint file
- [ ] Đã cài đặt đầy đủ dependencies (torch, numpy)
- [ ] Đã import đúng các class (PolicyNetwork, ValueNetwork)
- [ ] Features có đúng shape [1, 17, board_size, board_size]
- [ ] Model đã được set sang `eval()` mode
- [ ] Đã test với board state mẫu trước khi tích hợp vào game

---
<!-- test load mode -->
python scripts/test_model_in_game.py

**Chúc bạn sử dụng model thành công! 🎉**

