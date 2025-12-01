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

### 1.0. 🚀 QUICK START (Cho người đã setup Colab)

Nếu bạn đã:
- ✅ Tạo notebook mới
- ✅ Enable GPU
- ✅ Mount Google Drive

**Bước tiếp theo:**

1. **Tạo cấu trúc thư mục** (chạy Cell 1 trong template)
2. **Upload SGF Files** vào `GoGame_ML/raw_sgf/` (hoặc đã có sẵn)
3. **Upload Code Scripts** vào `GoGame_ML/code/`:
   - `policy_network.py`
   - `value_network.py`
   - `generate_features_colab.py`
   - `generate_labels_colab.py`
   - `train_colab.py`
   - `parse_sgf_colab.py`
4. **Chạy theo thứ tự các cells** trong template:
   - Cell 1-2: Setup
   - Cell 3-6: Load code
   - Cell 7-8: Parse SGF → Positions
   - Cell 9: Generate Labels
   - Cell 10: Verify Dataset
   - Cell 11-12: Training
   - Cell 13: Download Model

**Hoặc sử dụng template:** Copy từng cell từ `scripts/colab_notebook_template.py`

**Vị trí Scripts:** `scripts/` trong repository

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

### 2.1. Cấu trúc Thư mục Google Drive (QUAN TRỌNG)

Trước khi bắt đầu, hãy tạo cấu trúc thư mục trên Google Drive như sau:

```
Google Drive/MyDrive/GoGame_ML/
├── raw_sgf/              # ⭐ UPLOAD SGF FILES VÀO ĐÂY (nếu có)
│   ├── game1.sgf
│   ├── game2.sgf
│   └── ...
├── processed/            # (Tự động tạo khi parse SGF)
│   └── positions_*.pt
├── datasets/             # ⭐ DATASET ĐÃ XỬ LÝ (để training)
│   ├── positions_9x9.pt
│   ├── positions_13x13.pt
│   └── positions_19x19.pt
├── code/                 # ⭐ UPLOAD CODE MODEL VÀO ĐÂY
│   ├── models/
│   │   ├── __init__.py
│   │   ├── multi_task_model.py
│   │   ├── shared_backbone.py
│   │   ├── threat_head.py
│   │   ├── attack_head.py
│   │   └── intent_head.py
│   └── features.py
├── checkpoints/          # (Tự động tạo khi training)
│   └── best_model_epoch_X.pt
├── logs/                 # (Tự động tạo khi training)
│   └── TensorBoard logs
└── outputs/              # (Tự động tạo khi training)
    └── Evaluation results
```

**Lưu ý:**
- **SGF Files**: Upload vào `raw_sgf/` nếu bạn có dataset dạng `.sgf`
- **Dataset .pt**: Upload file `.pt` đã xử lý vào `datasets/` (hoặc sẽ tự động tạo từ SGF)
- **Code**: Upload các file model vào `code/models/` và `code/features.py`
- Các thư mục `processed/`, `checkpoints/`, `logs/`, `outputs/` sẽ tự động được tạo

### 2.2. Google Colab Setup

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

#### Bước 3: Mount Google Drive và Setup Cấu trúc Thư mục

```python
# Cell 2: Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Tạo cấu trúc thư mục chuẩn
import os
from pathlib import Path

# Thư mục gốc trên Google Drive
WORK_DIR = Path('/content/drive/MyDrive/GoGame_ML')
WORK_DIR.mkdir(exist_ok=True)

# Cấu trúc thư mục:
# GoGame_ML/
#   ├── datasets/          # Dataset đã xử lý (upload vào đây)
#   │   ├── positions_9x9.pt
#   │   ├── positions_13x13.pt
#   │   └── positions_19x19.pt
#   ├── code/              # Code model (upload vào đây)
#   │   ├── models/
#   │   ├── features.py
#   │   └── ...
#   ├── checkpoints/       # Model checkpoints (tự động tạo)
#   ├── logs/              # TensorBoard logs (tự động tạo)
#   └── outputs/           # Kết quả training (tự động tạo)

# Tạo các thư mục cần thiết
(WORK_DIR / 'raw_sgf').mkdir(exist_ok=True)      # Cho SGF files
(WORK_DIR / 'processed').mkdir(exist_ok=True)   # Cho positions sau khi parse
(WORK_DIR / 'datasets').mkdir(exist_ok=True)    # Cho dataset đã xử lý
(WORK_DIR / 'code').mkdir(exist_ok=True)
(WORK_DIR / 'checkpoints').mkdir(exist_ok=True)
(WORK_DIR / 'logs').mkdir(exist_ok=True)
(WORK_DIR / 'outputs').mkdir(exist_ok=True)

os.chdir(WORK_DIR)
print(f"✅ Working directory: {WORK_DIR}")
print(f"📁 Dataset folder: {WORK_DIR / 'datasets'}")
print(f"📁 Code folder: {WORK_DIR / 'code'}")
```

#### Bước 4: Upload Code Model (KHÔNG cần clone git)

**Cách 1: Upload trực tiếp từ Colab** (Khuyến nghị cho lần đầu)

```python
# Cell 3: Upload code files
from google.colab import files
import zipfile
from pathlib import Path

print("📤 Bước 1: Upload file ZIP chứa code model")
print("   (Tạo ZIP từ local: zip -r gogame_ml_code.zip src/ml/models/ src/ml/features.py)")
print("   Hoặc upload từng file riêng lẻ")

# Option A: Upload ZIP file
uploaded = files.upload()  # Chọn file ZIP

# Extract ZIP
for filename in uploaded.keys():
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(WORK_DIR / 'code')
        print(f"✅ Extracted {filename} to {WORK_DIR / 'code'}")
```

**Cách 2: Copy code trực tiếp vào Colab** (Nhanh nhất)

```python
# Cell 3: Tạo code files trực tiếp trong Colab
# Copy nội dung từ các file trong src/ml/models/ và paste vào đây

# Ví dụ: Tạo file multi_task_model.py
code_dir = WORK_DIR / 'code'
code_dir.mkdir(exist_ok=True)

# Tạo __init__.py
(code_dir / '__init__.py').write_text('')

# Tạo thư mục models
(code_dir / 'models').mkdir(exist_ok=True)
(code_dir / 'models' / '__init__.py').write_text('')

print("📝 Bây giờ hãy copy nội dung từ các file sau vào Colab:")
print("   1. src/ml/models/multi_task_model.py")
print("   2. src/ml/models/shared_backbone.py")
print("   3. src/ml/models/threat_head.py")
print("   4. src/ml/models/attack_head.py")
print("   5. src/ml/models/intent_head.py")
print("   6. src/ml/features.py")
print("\nSau đó chạy lệnh để lưu vào file:")
print("   %%writefile code/models/multi_task_model.py")
print("   [paste code here]")
```

**Cách 3: Clone từ GitHub** (Nếu đã push code lên GitHub)

```python
# Cell 3: Clone repo (nếu có)
!git clone https://github.com/yourusername/GoGame.git temp_repo
!cp -r temp_repo/src/ml/models/* {WORK_DIR}/code/models/
!cp temp_repo/src/ml/features.py {WORK_DIR}/code/
!rm -rf temp_repo
print("✅ Code đã được copy vào code/")
```

#### Bước 5: Upload Dataset (SGF hoặc .pt)

**Nếu bạn có dataset dạng `.sgf` (Smart Game Format):**

```python
# Cell 4: Upload SGF Files
from google.colab import files
from pathlib import Path
import zipfile
import shutil

print("📤 Upload SGF files")
print("   Option 1: Upload ZIP file chứa nhiều .sgf files")
print("   Option 2: Upload từng file .sgf riêng lẻ")
print("   Option 3: Nếu đã có trên Google Drive, copy vào raw_sgf/")

# Tạo thư mục cho SGF files
(WORK_DIR / 'raw_sgf').mkdir(exist_ok=True)

# Option A: Upload ZIP file
uploaded = files.upload()  # Chọn file ZIP chứa .sgf files

for filename in uploaded.keys():
    if filename.endswith('.zip'):
        # Extract ZIP vào raw_sgf/
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(WORK_DIR / 'raw_sgf')
        print(f"✅ Extracted {filename} to raw_sgf/")
    elif filename.endswith('.sgf'):
        # Di chuyển file .sgf vào raw_sgf/
        shutil.move(filename, WORK_DIR / 'raw_sgf' / filename)
        print(f"✅ Moved {filename} to raw_sgf/")

# Option B: Copy từ Google Drive (nếu đã upload trước)
# !cp -r /content/drive/MyDrive/your_sgf_folder/* {WORK_DIR}/raw_sgf/

# Đếm số file SGF
sgf_files = list((WORK_DIR / 'raw_sgf').glob('*.sgf'))
print(f"\n✅ Total SGF files: {len(sgf_files)}")
print(f"   Location: {WORK_DIR / 'raw_sgf'}")
```

**Nếu bạn đã có dataset dạng `.pt` (đã xử lý sẵn):**

```python
# Cell 4: Upload Dataset .pt
from google.colab import files
import torch
import shutil

print("📤 Upload dataset file (.pt)")
print("   Dataset nên được đặt tại: datasets/positions_9x9.pt")

uploaded = files.upload()  # Chọn file .pt

for filename in uploaded.keys():
    if filename.endswith('.pt'):
        shutil.move(filename, WORK_DIR / 'datasets' / filename)
        print(f"✅ Moved {filename} to datasets/")
        
        # Kiểm tra dataset
        data = torch.load(WORK_DIR / 'datasets' / filename, map_location='cpu')
        print(f"   Dataset info: {len(data.get('positions', data.get('labeled_data', [])))} samples")

# Nếu dataset đã có trên Google Drive
# !cp /content/drive/MyDrive/your_dataset.pt {WORK_DIR}/datasets/
```

**Lưu ý:**
- **SGF files**: Cần parse thành positions trước khi training (xem Cell 5-7)
- **.pt files**: Đã xử lý sẵn, có thể training ngay (skip Cell 5-7)
- Nếu dataset lớn (>1GB), nên upload lên Google Drive trước, rồi copy vào Colab

#### Bước 6: Install Dependencies và Setup Python Path

```python
# Cell 5: Install packages
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install numpy pandas tqdm tensorboard scikit-learn
!pip install sgf  # For parsing SGF files

# Setup Python path để import code
import sys
sys.path.insert(0, str(WORK_DIR / 'code'))
sys.path.insert(0, str(WORK_DIR / 'code' / 'models'))

print("✅ Dependencies installed")
print(f"✅ Python path updated: {sys.path[:3]}")
```

#### Bước 7: Parse SGF → Positions (CHỈ CẦN NẾU CÓ SGF FILES)

**Nếu bạn đã có dataset .pt, SKIP bước này và chuyển sang Bước 8.**

```python
# Cell 6: Parse SGF Files thành Positions
import sgf
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch

def parse_sgf_coord(sgf_coord, board_size):
    """Convert SGF coordinate to (x, y)"""
    if not sgf_coord or len(sgf_coord) < 2 or sgf_coord == 'tt':
        return None, None  # Pass move
    
    x = ord(sgf_coord[0]) - ord('a')
    y = ord(sgf_coord[1]) - ord('a')
    
    # Skip 'i' (no I in Go coordinates)
    if x >= 8:
        x -= 1
    if y >= 8:
        y -= 1
    
    if x < 0 or x >= board_size or y < 0 or y >= board_size:
        return None, None
    
    return x, y

def parse_sgf_file(sgf_path):
    """Parse 1 SGF file và extract tất cả positions"""
    try:
        with open(sgf_path, 'r', encoding='utf-8', errors='ignore') as f:
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
            winner = None
        
        # Extract moves
        positions = []
        board = np.zeros((board_size, board_size), dtype=np.int8)
        current_player = 'B'  # Black starts
        
        for node in game.rest:
            # Get move
            move = None
            color = None
            
            if 'B' in node.properties:
                move = node.properties['B'][0]
                color = 'B'
            elif 'W' in node.properties:
                move = node.properties['W'][0]
                color = 'W'
            else:
                continue  # Pass or other
            
            # Parse move coordinate
            x, y = parse_sgf_coord(move, board_size)
            
            if x is not None and y is not None:
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
                
                # Apply move (simplified - không xử lý captures, ko, etc.)
                board[y, x] = 1 if color == 'B' else 2
            
            current_player = 'W' if current_player == 'B' else 'B'
        
        return positions
        
    except Exception as e:
        print(f"Error parsing {sgf_path}: {e}")
        return []

# Parse tất cả SGF files
sgf_dir = WORK_DIR / 'raw_sgf'
sgf_files = list(sgf_dir.glob('*.sgf'))

print(f"📊 Parsing {len(sgf_files)} SGF files...")

all_positions = {9: [], 13: [], 19: []}

for sgf_file in tqdm(sgf_files, desc="Parsing SGF"):
    positions = parse_sgf_file(sgf_file)
    
    for pos in positions:
        board_size = pos['board_size']
        if board_size in all_positions:
            all_positions[board_size].append(pos)

# Save positions theo board size
(WORK_DIR / 'processed').mkdir(exist_ok=True)

for board_size in [9, 13, 19]:
    if all_positions[board_size]:
        output_file = WORK_DIR / 'processed' / f'positions_{board_size}x{board_size}.pt'
        torch.save({
            'positions': all_positions[board_size],
            'board_size': board_size,
            'total': len(all_positions[board_size])
        }, output_file)
        print(f"✅ Saved {len(all_positions[board_size]):,} positions for {board_size}x{board_size}")

print("\n✅ Parsing complete!")
```

#### Bước 8: Generate Features và Labels (CHỈ CẦN NẾU CÓ SGF FILES)

```python
# Cell 7: Generate Features và Labels từ Positions
import torch
import numpy as np
from tqdm import tqdm

def board_to_tensor_simple(board_state, current_player, board_size):
    """Convert board state to 17-plane tensor (simplified version)"""
    features = torch.zeros((17, board_size, board_size), dtype=torch.float32)
    
    # Plane 0: Black stones
    features[0] = (board_state == 1).float()
    
    # Plane 1: White stones
    features[1] = (board_state == 2).float()
    
    # Plane 2-7: Liberty counts (simplified - có thể cải thiện sau)
    # TODO: Calculate actual liberties
    
    # Plane 8-15: History (last 4 moves, 2 planes each)
    # TODO: Track move history
    
    # Plane 16: Turn indicator
    features[16] = 1.0 if current_player == 'B' else 0.0
    
    return features

def generate_threat_map_simple(board_state, current_player):
    """Generate threat map (simplified rule-based)"""
    board_size = board_state.shape[0]
    threat_map = np.zeros((board_size, board_size), dtype=np.float32)
    
    # TODO: Implement actual threat detection
    # For now, return zeros
    return torch.from_numpy(threat_map)

def generate_attack_map_simple(board_state, current_player):
    """Generate attack map (simplified rule-based)"""
    board_size = board_state.shape[0]
    attack_map = np.zeros((board_size, board_size), dtype=np.float32)
    
    # TODO: Implement actual attack detection
    # For now, return zeros
    return torch.from_numpy(attack_map)

# Process positions và generate features/labels
for board_size in [9, 13, 19]:
    input_file = WORK_DIR / 'processed' / f'positions_{board_size}x{board_size}.pt'
    
    if not input_file.exists():
        continue
    
    print(f"\n📊 Processing {board_size}x{board_size}...")
    data = torch.load(input_file, map_location='cpu')
    positions = data['positions']
    
    labeled_data = []
    
    for pos in tqdm(positions, desc=f"Generating features {board_size}x{board_size}"):
        board_state = pos['board_state']
        current_player = pos['current_player']
        move = pos['move']
        
        # Generate features
        features = board_to_tensor_simple(
            torch.from_numpy(board_state),
            current_player,
            board_size
        )
        
        # Generate labels
        threat_map = generate_threat_map_simple(board_state, current_player)
        attack_map = generate_attack_map_simple(board_state, current_player)
        
        labeled_data.append({
            'features': features,
            'threat_map': threat_map,
            'attack_map': attack_map,
            'intent': {
                'type': 'unknown',  # TODO: Implement intent recognition
                'confidence': 0.5
            },
            'metadata': {
                'move_number': pos['move_number'],
                'game_result': pos['game_result'],
                'winner': pos['winner']
            }
        })
    
    # Save labeled dataset
    output_file = WORK_DIR / 'datasets' / f'positions_{board_size}x{board_size}.pt'
    torch.save({
        'labeled_data': labeled_data,
        'board_size': board_size,
        'total': len(labeled_data)
    }, output_file)
    
    print(f"✅ Saved {len(labeled_data):,} labeled samples to {output_file}")

print("\n✅ Feature generation complete!")
print("📁 Dataset ready tại: datasets/positions_*.pt")
```

#### Bước 9: Verify Setup

```python
# Cell 8: Kiểm tra setup
import torch
from pathlib import Path

print("=" * 50)
print("🔍 VERIFY SETUP")
print("=" * 50)

# Check GPU
print(f"\n1. GPU Check:")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Check code files
print(f"\n2. Code Files Check:")
code_dir = WORK_DIR / 'code'
models_dir = code_dir / 'models'
required_files = [
    'models/multi_task_model.py',
    'models/shared_backbone.py',
    'models/threat_head.py',
    'models/attack_head.py',
    'models/intent_head.py',
    'features.py'
]

all_ok = True
for file_path in required_files:
    full_path = code_dir / file_path
    exists = full_path.exists()
    status = "✅" if exists else "❌"
    print(f"   {status} {file_path}")
    if not exists:
        all_ok = False

# Check datasets
print(f"\n3. Dataset Files Check:")
dataset_dir = WORK_DIR / 'datasets'
if dataset_dir.exists():
    dataset_files = list(dataset_dir.glob("*.pt"))
    if dataset_files:
        for ds_file in dataset_files:
            try:
                data = torch.load(ds_file, map_location='cpu')
                size = data.get('board_size', 'unknown')
                total = data.get('total', len(data.get('positions', data.get('labeled_data', []))))
                print(f"   ✅ {ds_file.name} - Board: {size}x{size}, Samples: {total:,}")
            except Exception as e:
                print(f"   ❌ {ds_file.name} - Error: {e}")
    else:
        print(f"   ⚠️  No dataset files found in {dataset_dir}")
        print(f"   💡 Upload dataset .pt vào đây, hoặc upload SGF files vào raw_sgf/ để parse")

# Check SGF files (nếu có)
sgf_dir = WORK_DIR / 'raw_sgf'
if sgf_dir.exists():
    sgf_files = list(sgf_dir.glob("*.sgf"))
    if sgf_files:
        print(f"\n4. SGF Files Check:")
        print(f"   ✅ Found {len(sgf_files)} SGF files in raw_sgf/")
        print(f"   💡 Run Cell 6-7 to parse SGF → positions → features")
else:
    print(f"\n4. SGF Files: Not found (OK if you have .pt dataset)")

# Check directories
print(f"\n5. Directories Check:")
dirs = ['raw_sgf', 'processed', 'datasets', 'checkpoints', 'logs', 'outputs']
for dir_name in dirs:
    dir_path = WORK_DIR / dir_name
    exists = dir_path.exists()
    status = "✅" if exists else "❌"
    print(f"   {status} {dir_name}/")

print("\n" + "=" * 50)
if all_ok and dataset_files:
    print("✅ Setup hoàn tất! Sẵn sàng để training!")
else:
    print("⚠️  Còn thiếu một số files. Hãy kiểm tra lại.")
print("=" * 50)
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

# Load dataset từ thư mục datasets/
dataset_path = WORK_DIR / 'datasets' / 'positions_9x9.pt'  # Thay đổi theo board size bạn có

# Load dataset
print(f"Loading dataset from {dataset_path}...")
dataset = torch.load(dataset_path, map_location='cpu')

# Dataset có thể có format khác nhau
if 'labeled_data' in dataset:
labeled_data = dataset['labeled_data']
elif 'positions' in dataset:
    # Nếu chưa có labels, sẽ cần generate sau
    positions = dataset['positions']
    print("⚠️  Dataset chưa có labels. Cần generate labels trước khi training.")
    labeled_data = None
else:
    raise ValueError("Dataset format không hợp lệ!")

board_size = dataset['board_size']

print(f"✅ Loaded dataset")
print(f"   Board size: {board_size}x{board_size}")
if labeled_data:
    print(f"   Samples: {len(labeled_data):,}")

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

# Import model code từ thư mục code/
import sys
sys.path.insert(0, str(WORK_DIR / 'code'))
sys.path.insert(0, str(WORK_DIR / 'code' / 'models'))

# Import models
from models.multi_task_model import MultiTaskModel, MultiTaskConfig
# Hoặc nếu đã copy trực tiếp:
# from multi_task_model import MultiTaskModel, MultiTaskConfig

# Create model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Config model (thay đổi board_size theo dataset của bạn)
config = MultiTaskConfig(
    input_planes=17,
    board_size=board_size,  # Sử dụng board_size từ dataset
    base_channels=64,
    num_res_blocks=4
)

model = MultiTaskModel(config=config).to(device)

print(f"✅ Model created on {device}")
print(f"   Board size: {board_size}x{board_size}")
print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"   Model size: {sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024:.2f} MB")
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
    'log_dir': WORK_DIR / 'logs',
    'output_dir': WORK_DIR / 'outputs'
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

### Phase 1: Data Collection (Local hoặc Colab)
- [ ] Download KGS games (10,000+ games) - hoặc đã có SGF files
- [ ] Download OGS games (5,000+ games) - hoặc đã có SGF files
- [ ] Upload SGF files lên Colab vào `raw_sgf/` (Cell 4)
- [ ] Parse SGF → Positions (Cell 6 trên Colab)
- [ ] Filter quality games (5d+) - có thể làm trong parse
- [ ] Generate features (17 planes) (Cell 7 trên Colab)
- [ ] Generate labels (threat, attack, intent) (Cell 7 trên Colab)
- [ ] Dataset `.pt` đã có trong `datasets/`

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

## 8. TÓM TẮT VỊ TRÍ FILES (QUAN TRỌNG)

### 8.1. Vị trí Dataset trên Google Drive

**Nếu bạn có SGF files:**

```
/content/drive/MyDrive/GoGame_ML/
├── raw_sgf/              ← Upload SGF files vào đây
│   ├── game1.sgf
│   └── game2.sgf
├── processed/            ← Tự động tạo (positions sau khi parse)
│   └── positions_*.pt
└── datasets/             ← Tự động tạo (labeled data để training)
    └── positions_*.pt
```

**Workflow với SGF:**
1. Upload SGF files vào `raw_sgf/` (Cell 4)
2. Parse SGF → positions (Cell 6)
3. Generate features & labels (Cell 7)
4. Dataset sẵn sàng tại `datasets/` để training

**Nếu bạn đã có dataset .pt:**

```
/content/drive/MyDrive/GoGame_ML/datasets/
├── positions_9x9.pt      ← Upload vào đây
├── positions_13x13.pt   ← Upload vào đây
└── positions_19x19.pt   ← Upload vào đây
```

**Cách upload .pt:**
1. Upload file `.pt` trực tiếp từ Colab: `files.upload()`
2. Hoặc copy từ Google Drive khác: `!cp /path/to/dataset.pt {WORK_DIR}/datasets/`
3. Hoặc upload qua Google Drive web interface, rồi copy vào Colab

### 8.2. Vị trí Code Model trên Google Drive

```
/content/drive/MyDrive/GoGame_ML/code/
├── models/
│   ├── __init__.py
│   ├── multi_task_model.py
│   ├── shared_backbone.py
│   ├── threat_head.py
│   ├── attack_head.py
│   └── intent_head.py
└── features.py
```

**Cách upload:**
1. **Option A (Khuyến nghị)**: Chạy `scripts/setup_colab_helper.py` trên local để tạo ZIP, rồi upload ZIP lên Colab
2. **Option B**: Copy code trực tiếp vào Colab cells và lưu vào file
3. **Option C**: Clone từ GitHub (nếu đã push code)

### 8.3. Checklist Setup

**Nếu bạn có SGF files:**

- [ ] ✅ GPU enabled (T4 hoặc P100)
- [ ] ✅ Google Drive mounted
- [ ] ✅ Thư mục `GoGame_ML/` đã tạo với cấu trúc đúng
- [ ] ✅ SGF files đã upload vào `raw_sgf/`
- [ ] ✅ Code model đã upload vào `code/models/`
- [ ] ✅ Dependencies đã install (bao gồm `sgf` package)
- [ ] ✅ Python path đã setup đúng
- [ ] ✅ Parse SGF → positions (Cell 6)
- [ ] ✅ Generate features & labels (Cell 7)
- [ ] ✅ Dataset `.pt` đã có trong `datasets/`
- [ ] ✅ Verify setup passed (Cell 8)

**Nếu bạn đã có dataset .pt:**

- [ ] ✅ GPU enabled (T4 hoặc P100)
- [ ] ✅ Google Drive mounted
- [ ] ✅ Thư mục `GoGame_ML/` đã tạo với cấu trúc đúng
- [ ] ✅ Dataset file `.pt` đã upload vào `datasets/`
- [ ] ✅ Code model đã upload vào `code/models/`
- [ ] ✅ Dependencies đã install
- [ ] ✅ Python path đã setup đúng
- [ ] ✅ Verify setup passed (Cell 8)

### 8.4. Script Helper

Chạy script helper trên local để chuẩn bị files:

```bash
# Trên local machine
python scripts/setup_colab_helper.py
```

Script này sẽ:
- ✅ Tạo ZIP file chứa code model (`gogame_ml_code.zip`)
- ✅ In hướng dẫn setup chi tiết
- ✅ Tạo notebook template (optional)

---

**END OF PART 1**

*Tiếp tục với phần chi tiết hơn ở file tiếp theo...*

