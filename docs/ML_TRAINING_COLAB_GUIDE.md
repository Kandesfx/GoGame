# 🚀 HƯỚNG DẪN TRAINING ML TRÊN COLAB/KAGGLE

## 📑 MỤC LỤC

1. [Tổng quan](#1-tổng-quan)
2. [Setup Colab/Kaggle](#2-setup-colabkaggle)
3. [Thu thập dữ liệu chuyên nghiệp](#3-thu-thập-dữ-liệu-chuyên-nghiệp)
4. [Xử lý dữ liệu](#4-xử-lý-dữ-liệu)
5. [Quy trình Training](#5-quy-trình-training)
6. [Deployment Model](#6-deployment-model)

---

## 1. TỔNG QUAN

### 1.1. Tại sao dùng Colab/Kaggle?

| Platform | GPU | Storage | Thời gian | Giới hạn |
|----------|-----|---------|-----------|----------|
| **Google Colab** | ✅ Free T4 (16GB) | 15GB | 12h/session | Cần reconnect |
| **Kaggle** | ✅ Free P100 (16GB) | 30GB | 9h/session | Stable hơn |
| **Local** | ❌ Cần GPU riêng | Unlimited | Unlimited | Tốn tiền |

**Khuyến nghị**: Dùng **Kaggle** vì ổn định hơn, hoặc **Colab Pro** ($10/tháng) nếu cần thời gian dài hơn.

### 1.2. Workflow tổng quan

```
┌─────────────────────────────────────────────────┐
│  STEP 1: Download Professional Games           │
│  • KGS Archive (70K games)                      │
│  • OGS API (recent games)                       │
│  • GoGoD (optional, paid)                        │
└─────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│  STEP 2: Parse SGF → Positions                  │
│  • Extract board states                         │
│  • Filter quality games                         │
│  • Generate features (17 planes)                │
└─────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│  STEP 3: Generate Labels                        │
│  • Threat maps (rule-based)                      │
│  • Attack maps (rule-based)                      │
│  • Intent labels (pattern-based)                │
└─────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│  STEP 4: Upload to Colab/Kaggle                 │
│  • Compress dataset                              │
│  • Upload to Google Drive / Kaggle Dataset       │
└─────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│  STEP 5: Train Model                            │
│  • Load dataset                                  │
│  • Train multi-task model                       │
│  • Monitor với TensorBoard                      │
└─────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│  STEP 6: Download Model                         │
│  • Save checkpoint                               │
│  • Download về local                            │
│  • Deploy vào backend                           │
└─────────────────────────────────────────────────┘
```

### 1.3. Yêu cầu dữ liệu

| Board Size | Min Games | Min Positions | Recommended |
|------------|-----------|---------------|-------------|
| 9×9 | 1,000 | 80,000 | 5,000 games |
| 13×13 | 500 | 60,000 | 2,000 games |
| 19×19 | 2,000 | 400,000 | 10,000 games |

**Tổng cần**: ~17,000 games chuyên nghiệp (từ rank 5d trở lên)

---

## 2. SETUP COLAB/KAGGLE

### 2.1. Google Colab Setup

#### Bước 1: Tạo Notebook mới

1. Vào https://colab.research.google.com
2. File → New Notebook
3. Đặt tên: `GoGame_ML_Training.ipynb`

#### Bước 2: Enable GPU

```python
# Cell 1: Check GPU
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

**Runtime → Change runtime type → GPU (T4)**

#### Bước 3: Mount Google Drive

```python
# Cell 2: Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Tạo thư mục làm việc
import os
WORK_DIR = '/content/drive/MyDrive/GoGame_ML'
os.makedirs(WORK_DIR, exist_ok=True)
os.chdir(WORK_DIR)
print(f"Working directory: {WORK_DIR}")
```

#### Bước 4: Clone Repository (hoặc upload code)

**Option A: Clone từ GitHub** (nếu có repo)
```python
# Cell 3: Clone repo
!git clone https://github.com/yourusername/GoGame.git
%cd GoGame
```

**Option B: Upload code thủ công**
```python
# Cell 3: Upload files
from google.colab import files
# Upload: src/ml/models/*.py, src/ml/training/*.py, src/ml/features.py
```

#### Bước 5: Install Dependencies

```python
# Cell 4: Install packages
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install numpy pandas tqdm tensorboard
!pip install sgf  # For parsing SGF files
!pip install go  # If needed for Go utilities
```

### 2.2. Kaggle Setup

#### Bước 1: Tạo Notebook mới

1. Vào https://www.kaggle.com/code
2. New Notebook
3. Đặt tên: `gogame-ml-training`

#### Bước 2: Enable GPU

**Settings → Accelerator → GPU (P100)**

#### Bước 3: Upload Dataset

1. **Data → Add data → New dataset**
2. Upload dataset files (sẽ hướng dẫn ở phần sau)
3. Dataset sẽ có path: `/kaggle/input/your-dataset-name/`

#### Bước 4: Install Dependencies

```python
# Cell 1: Install packages
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install numpy pandas tqdm tensorboard sgf
```

#### Bước 5: Setup Working Directory

```python
# Cell 2: Setup
import os
WORK_DIR = '/kaggle/working'
os.chdir(WORK_DIR)

# Copy code files (hoặc clone repo)
# !git clone https://github.com/yourusername/GoGame.git
```

---

## 3. THU THẬP DỮ LIỆU CHUYÊN NGHIỆP

### 3.1. Nguồn dữ liệu

#### 3.1.1. KGS Game Archive (⭐ RECOMMENDED - FREE)

**URL**: https://u-go.net/gamerecords/

**Thông tin**:
- ~70,000 games chuyên nghiệp
- Format: SGF
- Ranks: 1d - 9d professional
- **FREE và không giới hạn**

**Script download** (chạy trên local trước khi upload lên Colab):

```python
# scripts/download_kgs_games.py

import requests
import os
from pathlib import Path
from tqdm import tqdm
import time

KGS_BASE_URL = "https://u-go.net/gamerecords/"
OUTPUT_DIR = Path("data/raw/kgs")

def download_kgs_games(min_rank=5, max_games=10000, output_dir=OUTPUT_DIR):
    """
    Download games từ KGS Archive
    
    Args:
        min_rank: Minimum rank (dan) - chỉ lấy từ 5d trở lên
        max_games: Số lượng games tối đa
        output_dir: Thư mục lưu
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded = 0
    failed = 0
    
    # KGS có nhiều năm, mỗi năm có nhiều tháng
    years = range(2000, 2024)  # 2000-2023
    
    print(f"Bắt đầu download từ KGS Archive...")
    print(f"Target: {max_games} games, min rank: {min_rank}d")
    
    for year in years:
        if downloaded >= max_games:
            break
            
        for month in range(1, 13):
            if downloaded >= max_games:
                break
            
            # URL format: https://u-go.net/gamerecords/YYYY/MM/
            url = f"{KGS_BASE_URL}{year}/{month:02d}/"
            
            try:
                # Get list of SGF files
                response = requests.get(url, timeout=10)
                if response.status_code != 200:
                    continue
                
                # Parse HTML để tìm links .sgf
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.text, 'html.parser')
                
                sgf_links = []
                for link in soup.find_all('a'):
                    href = link.get('href', '')
                    if href.endswith('.sgf'):
                        sgf_links.append(href)
                
                # Download từng file
                for sgf_file in sgf_links:
                    if downloaded >= max_games:
                        break
                    
                    sgf_url = f"{url}{sgf_file}"
                    output_path = output_dir / f"{year}_{month:02d}_{sgf_file}"
                    
                    # Skip nếu đã có
                    if output_path.exists():
                        continue
                    
                    try:
                        sgf_response = requests.get(sgf_url, timeout=10)
                        if sgf_response.status_code == 200:
                            # Check rank trong SGF metadata
                            sgf_content = sgf_response.text
                            if f"{min_rank}d" in sgf_content or f"{min_rank+1}d" in sgf_content:
                                output_path.write_text(sgf_content, encoding='utf-8')
                                downloaded += 1
                                
                                if downloaded % 100 == 0:
                                    print(f"Downloaded: {downloaded}/{max_games}")
                                
                                time.sleep(0.1)  # Rate limiting
                    except Exception as e:
                        failed += 1
                        if failed % 100 == 0:
                            print(f"Failed downloads: {failed}")
                        continue
                
            except Exception as e:
                print(f"Error processing {year}/{month}: {e}")
                continue
    
    print(f"\n✅ Hoàn thành!")
    print(f"Downloaded: {downloaded} games")
    print(f"Failed: {failed}")
    print(f"Saved to: {output_dir}")

if __name__ == '__main__':
    download_kgs_games(min_rank=5, max_games=10000)
```

**Cách chạy**:
```bash
# Trên local machine
python scripts/download_kgs_games.py --min-rank 5 --max-games 10000
```

#### 3.1.2. OGS API (FREE)

**URL**: https://online-go.com/api/v1/

**Script download**:

```python
# scripts/download_ogs_games.py

import requests
import json
from pathlib import Path
from tqdm import tqdm

OGS_API_BASE = "https://online-go.com/api/v1/"

def download_ogs_games(min_rank=5, max_games=5000, output_dir=Path("data/raw/ogs")):
    """
    Download games từ OGS API
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded = 0
    page = 1
    
    print(f"Downloading from OGS API...")
    
    while downloaded < max_games:
        # Get games list
        url = f"{OGS_API_BASE}games/"
        params = {
            'ordering': '-ended',
            'page': page,
            'page_size': 100,
            'ranked': True
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code != 200:
                break
            
            data = response.json()
            games = data.get('results', [])
            
            if not games:
                break
            
            for game in games:
                if downloaded >= max_games:
                    break
                
                # Check rank
                black_rank = game.get('black', {}).get('ranking', 0)
                white_rank = game.get('white', {}).get('ranking', 0)
                
                if black_rank < min_rank * 100 or white_rank < min_rank * 100:
                    continue  # OGS uses numeric ranks (500 = 5d)
                
                # Download SGF
                game_id = game['id']
                sgf_url = f"{OGS_API_BASE}games/{game_id}/sgf"
                
                try:
                    sgf_response = requests.get(sgf_url, timeout=10)
                    if sgf_response.status_code == 200:
                        output_path = output_dir / f"ogs_{game_id}.sgf"
                        output_path.write_text(sgf_response.text, encoding='utf-8')
                        downloaded += 1
                        
                        if downloaded % 100 == 0:
                            print(f"Downloaded: {downloaded}/{max_games}")
                except:
                    continue
            
            page += 1
            
        except Exception as e:
            print(f"Error: {e}")
            break
    
    print(f"✅ Downloaded {downloaded} games from OGS")

if __name__ == '__main__':
    download_ogs_games(min_rank=5, max_games=5000)
```

#### 3.1.3. GoGoD Database (PAID - Optional)

**URL**: https://www.gogodonline.co.uk/

**Thông tin**:
- ~100,000 historical games
- Very high quality
- Cost: ~$40 one-time
- Format: SGF

**Nếu mua**: Download và extract vào `data/raw/gogod/`

---

## 4. XỬ LÝ DỮ LIỆU

### 4.1. Parse SGF → Positions

**File**: `scripts/parse_sgf_to_positions.py`

```python
# scripts/parse_sgf_to_positions.py

import sgf
from pathlib import Path
import torch
from tqdm import tqdm
import numpy as np

def parse_sgf_file(sgf_path):
    """
    Parse 1 SGF file và extract tất cả positions
    
    Returns:
        List of (board_state, move, outcome) tuples
    """
    try:
        with open(sgf_path, 'r', encoding='utf-8') as f:
            sgf_content = f.read()
        
        # Parse SGF
        game = sgf.parse(sgf_content)
        
        # Extract metadata
        root = game.root
        board_size = int(root.properties.get('SZ', ['19'])[0])
        result = root.properties.get('RE', [''])[0]  # "B+12.5" or "W+R"
        
        # Determine winner
        if result.startswith('B'):
            winner = 'B'
        elif result.startswith('W'):
            winner = 'W'
        else:
            winner = None  # Unknown
        
        # Extract moves
        positions = []
        board = create_empty_board(board_size)
        current_player = 'B'  # Black starts
        
        for node in game.rest:
            # Get move
            if 'B' in node.properties:
                move = node.properties['B'][0]
                color = 'B'
            elif 'W' in node.properties:
                move = node.properties['W'][0]
                color = 'W'
            else:
                continue  # Pass or other
            
            # Parse move coordinate
            if move and move != '' and move != 'tt':  # 'tt' = pass
                x, y = parse_sgf_coord(move, board_size)
                
                # Save position BEFORE move
                positions.append({
                    'board_state': board.copy(),
                    'move': (x, y),
                    'current_player': current_player,
                    'move_number': len(positions),
                    'board_size': board_size,
                    'game_result': result,
                    'winner': winner
                })
                
                # Apply move
                board[y, x] = 1 if color == 'B' else 2
                # TODO: Apply Go rules (captures, ko, etc.)
            
            current_player = 'W' if current_player == 'B' else 'B'
        
        return positions
        
    except Exception as e:
        print(f"Error parsing {sgf_path}: {e}")
        return []

def parse_sgf_coord(sgf_coord, board_size):
    """
    Convert SGF coordinate to (x, y)
    SGF: 'aa' = (0, 0), 'sa' = (18, 0) for 19x19
    """
    if len(sgf_coord) < 2:
        return None, None
    
    x = ord(sgf_coord[0]) - ord('a')
    y = ord(sgf_coord[1]) - ord('a')
    
    # Skip 'i' (no I in Go coordinates)
    if x >= 8:
        x -= 1
    if y >= 8:
        y -= 1
    
    return x, y

def create_empty_board(size):
    """Create empty Go board"""
    return np.zeros((size, size), dtype=np.int8)

def process_all_sgf_files(sgf_dir, output_path, board_sizes=[9, 13, 19]):
    """
    Process tất cả SGF files và tạo dataset
    
    Args:
        sgf_dir: Thư mục chứa SGF files
        output_path: Path để lưu PyTorch dataset
        board_sizes: Các board sizes cần xử lý
    """
    sgf_dir = Path(sgf_dir)
    all_positions = {size: [] for size in board_sizes}
    
    sgf_files = list(sgf_dir.glob("*.sgf"))
    print(f"Found {len(sgf_files)} SGF files")
    
    for sgf_file in tqdm(sgf_files, desc="Parsing SGF"):
        positions = parse_sgf_file(sgf_file)
        
        for pos in positions:
            board_size = pos['board_size']
            if board_size in board_sizes:
                all_positions[board_size].append(pos)
    
    # Save datasets
    for board_size in board_sizes:
        if all_positions[board_size]:
            output_file = output_path / f"positions_{board_size}x{board_size}.pt"
            torch.save({
                'positions': all_positions[board_size],
                'board_size': board_size,
                'total': len(all_positions[board_size])
            }, output_file)
            print(f"✅ Saved {len(all_positions[board_size])} positions for {board_size}x{board_size}")
    
    return all_positions

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='SGF directory')
    parser.add_argument('--output', type=str, default='data/processed', help='Output directory')
    parser.add_argument('--board-sizes', type=int, nargs='+', default=[9, 13, 19])
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    process_all_sgf_files(args.input, output_dir, args.board_sizes)
```

**Cách chạy**:
```bash
python scripts/parse_sgf_to_positions.py \
  --input data/raw/kgs \
  --output data/processed \
  --board-sizes 9 13 19
```

### 4.2. Generate Features (17 Planes)

**File**: `src/ml/features.py` (đã có, cần update)

```python
# src/ml/features.py (update)

import torch
import numpy as np
from typing import Tuple

def board_to_tensor(board_state: np.ndarray, current_player: str, board_size: int) -> torch.Tensor:
    """
    Convert board state to 17-plane tensor
    
    Args:
        board_state: (board_size, board_size) array, 0=empty, 1=black, 2=white
        current_player: 'B' or 'W'
        board_size: Board size
    
    Returns:
        Tensor of shape (17, board_size, board_size)
    """
    features = torch.zeros((17, board_size, board_size), dtype=torch.float32)
    
    # Plane 0: Black stones
    features[0] = (board_state == 1).float()
    
    # Plane 1: White stones
    features[1] = (board_state == 2).float()
    
    # Plane 2-7: Liberty counts (simplified - cần implement đầy đủ)
    # TODO: Calculate actual liberties for each stone
    # For now, use simple heuristics
    
    # Plane 8-15: History (last 4 moves, 2 planes each)
    # TODO: Track move history
    
    # Plane 16: Turn indicator
    features[16] = 1.0 if current_player == 'B' else 0.0
    
    return features

def process_positions_to_features(positions_data, board_size):
    """
    Convert positions to feature tensors
    
    Args:
        positions_data: List of position dicts from parse_sgf
        board_size: Board size
    
    Returns:
        List of feature tensors
    """
    features_list = []
    
    for pos in positions_data:
        board_state = pos['board_state']
        current_player = pos['current_player']
        
        features = board_to_tensor(board_state, current_player, board_size)
        features_list.append(features)
    
    return features_list
```

### 4.3. Generate Labels

**File**: `scripts/generate_labels.py`

```python
# scripts/generate_labels.py

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

def generate_threat_map(board_state, current_player):
    """
    Generate threat map using rule-based heuristics
    
    Returns:
        (board_size, board_size) tensor, values 0-1
    """
    board_size = board_state.shape[0]
    threat_map = np.zeros((board_size, board_size), dtype=np.float32)
    
    # TODO: Implement threat detection
    # - Groups with 1 liberty → 1.0
    # - Groups with 2 liberties → 0.7
    # - False eyes → 0.6
    # - Cutting points → 0.5
    
    return torch.from_numpy(threat_map)

def generate_attack_map(board_state, current_player):
    """
    Generate attack opportunity map
    
    Returns:
        (board_size, board_size) tensor, values 0-1
    """
    board_size = board_state.shape[0]
    attack_map = np.zeros((board_size, board_size), dtype=np.float32)
    
    # TODO: Implement attack detection
    # - Opponent in atari → 1.0
    # - Can cut → 0.8
    # - Invasion points → 0.6
    # - Ladder works → 0.7
    
    return torch.from_numpy(attack_map)

def generate_intent_label(board_state, move, prev_moves):
    """
    Generate intent label from move pattern
    
    Returns:
        intent_type: str ('territory', 'attack', 'defense', 'connection', 'cut')
        confidence: float
    """
    # TODO: Implement intent recognition
    # - Pattern matching
    # - Heuristic analysis
    
    return 'attack', 0.5  # Placeholder

def process_dataset_with_labels(input_path, output_path):
    """
    Process dataset và generate labels
    """
    print(f"Loading positions from {input_path}...")
    data = torch.load(input_path)
    positions = data['positions']
    board_size = data['board_size']
    
    print(f"Processing {len(positions)} positions...")
    
    labeled_data = []
    
    for pos in tqdm(positions, desc="Generating labels"):
        board_state = pos['board_state']
        current_player = pos['current_player']
        move = pos['move']
        
        # Generate features
        features = board_to_tensor(board_state, current_player, board_size)
        
        # Generate labels
        threat_map = generate_threat_map(board_state, current_player)
        attack_map = generate_attack_map(board_state, current_player)
        intent_type, intent_conf = generate_intent_label(board_state, move, [])
        
        labeled_data.append({
            'features': features,
            'threat_map': threat_map,
            'attack_map': attack_map,
            'intent': {
                'type': intent_type,
                'confidence': intent_conf
            },
            'metadata': {
                'move_number': pos['move_number'],
                'game_result': pos['game_result'],
                'winner': pos['winner']
            }
        })
    
    # Save
    torch.save({
        'labeled_data': labeled_data,
        'board_size': board_size,
        'total': len(labeled_data)
    }, output_path)
    
    print(f"✅ Saved labeled dataset to {output_path}")
    print(f"Total samples: {len(labeled_data)}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    
    args = parser.parse_args()
    process_dataset_with_labels(args.input, args.output)
```

### 4.4. Prepare Dataset cho Colab/Kaggle

**Sau khi xử lý xong, compress và upload**:

```bash
# Compress dataset
cd data/processed
tar -czf gogame_dataset.tar.gz *.pt
# Hoặc zip
zip -r gogame_dataset.zip *.pt

# Upload lên Google Drive (cho Colab)
# Hoặc upload lên Kaggle Dataset (cho Kaggle)
```

---

## 5. QUY TRÌNH TRAINING

### 5.1. Setup trên Colab

**Notebook structure**:

```python
# ============================================
# CELL 1: Setup & Install
# ============================================

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os

# Check GPU
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Setup paths
WORK_DIR = Path('/content/drive/MyDrive/GoGame_ML')
WORK_DIR.mkdir(exist_ok=True)
os.chdir(WORK_DIR)

# Install packages
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install numpy pandas tqdm tensorboard sgf

print("✅ Setup complete!")
```

```python
# ============================================
# CELL 2: Load Dataset
# ============================================

# Upload dataset file hoặc load từ Drive
dataset_path = WORK_DIR / 'gogame_dataset_9x9.pt'

# Load dataset
print("Loading dataset...")
dataset = torch.load(dataset_path)
labeled_data = dataset['labeled_data']
board_size = dataset['board_size']

print(f"✅ Loaded {len(labeled_data)} samples")
print(f"Board size: {board_size}x{board_size}")

# Split train/val/test
from sklearn.model_selection import train_test_split

train_data, temp_data = train_test_split(
    labeled_data, 
    test_size=0.2, 
    random_state=42
)
val_data, test_data = train_test_split(
    temp_data,
    test_size=0.5,
    random_state=42
)

print(f"Train: {len(train_data)}")
print(f"Val: {len(val_data)}")
print(f"Test: {len(test_data)}")
```

```python
# ============================================
# CELL 3: Create Dataset Class
# ============================================

class GoPositionDataset(Dataset):
    def __init__(self, data, augment=False):
        self.data = data
        self.augment = augment
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        features = sample['features']
        threat_map = sample['threat_map']
        attack_map = sample['attack_map']
        intent = sample['intent']
        
        # TODO: Add augmentation if self.augment
        
        return {
            'features': features,
            'threat_map': threat_map,
            'attack_map': attack_map,
            'intent_type': intent['type'],
            'intent_conf': intent['confidence']
        }

# Create datasets
train_dataset = GoPositionDataset(train_data, augment=True)
val_dataset = GoPositionDataset(val_data, augment=False)
test_dataset = GoPositionDataset(test_data, augment=False)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

print("✅ Datasets created")
```

```python
# ============================================
# CELL 4: Load Model Architecture
# ============================================

# Import model code (upload files trước)
import sys
sys.path.append(str(WORK_DIR))

# Copy model files vào Colab
# Hoặc import từ uploaded files
from src.ml.models.multi_task_model import MultiTaskModel
from src.ml.models.shared_backbone import SharedBackbone
from src.ml.models.threat_head import ThreatHead
from src.ml.models.attack_head import AttackHead
from src.ml.models.intent_head import IntentHead

# Create model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultiTaskModel(board_size=9).to(device)

print(f"✅ Model created on {device}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

```python
# ============================================
# CELL 5: Training Loop
# ============================================

# Config
config = {
    'num_epochs': 50,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'patience': 10,
    'checkpoint_dir': WORK_DIR / 'checkpoints',
    'log_dir': WORK_DIR / 'logs'
}

config['checkpoint_dir'].mkdir(exist_ok=True)
config['log_dir'].mkdir(exist_ok=True)

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config['learning_rate'],
    weight_decay=config['weight_decay']
)

# Scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=config['num_epochs']
)

# Loss functions
threat_loss_fn = nn.MSELoss()
attack_loss_fn = nn.MSELoss()
intent_loss_fn = nn.CrossEntropyLoss()

# TensorBoard
writer = SummaryWriter(config['log_dir'])

# Training
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(config['num_epochs']):
    print(f"\n=== Epoch {epoch+1}/{config['num_epochs']} ===")
    
    # Train
    model.train()
    train_loss = 0
    train_threat_loss = 0
    train_attack_loss = 0
    train_intent_loss = 0
    
    for batch in tqdm(train_loader, desc='Training'):
        features = batch['features'].to(device)
        threat_map = batch['threat_map'].to(device)
        attack_map = batch['attack_map'].to(device)
        intent_type = batch['intent_type']  # TODO: Convert to class index
        
        optimizer.zero_grad()
        
        # Forward
        outputs = model(features)
        
        # Losses
        threat_loss = threat_loss_fn(outputs['threat_map'], threat_map)
        attack_loss = attack_loss_fn(outputs['attack_map'], attack_map)
        # intent_loss = intent_loss_fn(outputs['intent_logits'], intent_type)
        
        total_loss = threat_loss + attack_loss  # + intent_loss
        
        # Backward
        total_loss.backward()
        optimizer.step()
        
        # Track
        train_loss += total_loss.item()
        train_threat_loss += threat_loss.item()
        train_attack_loss += attack_loss.item()
    
    # Average
    train_loss /= len(train_loader)
    train_threat_loss /= len(train_loader)
    train_attack_loss /= len(train_loader)
    
    # Validate
    model.eval()
    val_loss = 0
    val_threat_loss = 0
    val_attack_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating'):
            features = batch['features'].to(device)
            threat_map = batch['threat_map'].to(device)
            attack_map = batch['attack_map'].to(device)
            
            outputs = model(features)
            
            threat_loss = threat_loss_fn(outputs['threat_map'], threat_map)
            attack_loss = attack_loss_fn(outputs['attack_map'], attack_map)
            total_loss = threat_loss + attack_loss
            
            val_loss += total_loss.item()
            val_threat_loss += threat_loss.item()
            val_attack_loss += attack_loss.item()
    
    val_loss /= len(val_loader)
    val_threat_loss /= len(val_loader)
    val_attack_loss /= len(val_loader)
    
    # Log
    writer.add_scalar('Loss/Train', train_loss, epoch)
    writer.add_scalar('Loss/Val', val_loss, epoch)
    writer.add_scalar('Loss/Train_Threat', train_threat_loss, epoch)
    writer.add_scalar('Loss/Val_Threat', val_threat_loss, epoch)
    writer.add_scalar('Loss/Train_Attack', train_attack_loss, epoch)
    writer.add_scalar('Loss/Val_Attack', val_attack_loss, epoch)
    writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)
    
    print(f"Train Loss: {train_loss:.4f} (Threat: {train_threat_loss:.4f}, Attack: {train_attack_loss:.4f})")
    print(f"Val Loss: {val_loss:.4f} (Threat: {val_threat_loss:.4f}, Attack: {val_attack_loss:.4f})")
    
    # Save best model
    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        patience_counter = 0
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'config': config
        }
        torch.save(checkpoint, config['checkpoint_dir'] / f'best_model_epoch_{epoch}.pt')
        print(f"✅ Saved best model (val_loss: {val_loss:.4f})")
    else:
        patience_counter += 1
    
    # Early stopping
    if patience_counter >= config['patience']:
        print(f"Early stopping at epoch {epoch}")
        break
    
    # Scheduler step
    scheduler.step()
    
    # Periodic checkpoint
    if (epoch + 1) % 10 == 0:
        torch.save(checkpoint, config['checkpoint_dir'] / f'checkpoint_epoch_{epoch}.pt')

writer.close()
print("\n✅ Training complete!")
```

```python
# ============================================
# CELL 6: Evaluate Model
# ============================================

# Load best model
best_checkpoint = torch.load(config['checkpoint_dir'] / 'best_model_epoch_X.pt')
model.load_state_dict(best_checkpoint['model_state_dict'])

# Evaluate on test set
model.eval()
test_loss = 0

with torch.no_grad():
    for batch in tqdm(test_loader, desc='Testing'):
        features = batch['features'].to(device)
        threat_map = batch['threat_map'].to(device)
        attack_map = batch['attack_map'].to(device)
        
        outputs = model(features)
        
        threat_loss = threat_loss_fn(outputs['threat_map'], threat_map)
        attack_loss = attack_loss_fn(outputs['attack_map'], attack_map)
        total_loss = threat_loss + attack_loss
        
        test_loss += total_loss.item()

test_loss /= len(test_loader)
print(f"Test Loss: {test_loss:.4f}")
```

```python
# ============================================
# CELL 7: Download Model
# ============================================

# Model sẽ tự động lưu vào Google Drive
# Hoặc download về local:
from google.colab import files

# Download checkpoint
files.download(str(config['checkpoint_dir'] / 'best_model_epoch_X.pt'))
```

---

## 6. DEPLOYMENT MODEL

### 6.1. Download Model về Local

```bash
# Từ Google Drive
# Hoặc từ Colab: files.download()

# Save vào project
mkdir -p models/ml
cp best_model_epoch_X.pt models/ml/multi_task_9x9.pt
```

### 6.2. Load Model trong Backend

```python
# backend/app/services/ml_analysis_service.py

import torch
from pathlib import Path
from src.ml.models.multi_task_model import MultiTaskModel

class MLAnalysisService:
    def __init__(self, model_path: Path):
        self.device = torch.device('cpu')  # Hoặc 'cuda' nếu có GPU
        self.model = MultiTaskModel(board_size=9)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✅ Loaded ML model from {model_path}")
    
    def analyze(self, board_state):
        """Run inference"""
        with torch.no_grad():
            outputs = self.model(board_state)
        return outputs
```

---

## 7. CHECKLIST HOÀN CHỈNH

### Phase 1: Data Collection (Local)
- [ ] Download KGS games (10,000+ games)
- [ ] Download OGS games (5,000+ games)
- [ ] Parse SGF → Positions
- [ ] Filter quality games (5d+)
- [ ] Generate features (17 planes)
- [ ] Generate labels (threat, attack, intent)
- [ ] Split train/val/test
- [ ] Compress dataset

### Phase 2: Upload to Colab/Kaggle
- [ ] Upload dataset to Google Drive / Kaggle Dataset
- [ ] Upload model code files
- [ ] Setup Colab/Kaggle notebook

### Phase 3: Training
- [ ] Load dataset
- [ ] Create DataLoader
- [ ] Initialize model
- [ ] Train model
- [ ] Monitor với TensorBoard
- [ ] Save checkpoints
- [ ] Evaluate on test set

### Phase 4: Deployment
- [ ] Download model
- [ ] Test model locally
- [ ] Integrate vào backend
- [ ] Deploy to production

---

**END OF PART 1**

*Tiếp tục với phần chi tiết hơn ở file tiếp theo...*

