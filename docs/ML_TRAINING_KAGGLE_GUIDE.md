# 🎯 HƯỚNG DẪN TRAINING ML TRÊN KAGGLE - DỄ HIỂU

## 📑 MỤC LỤC

1. [Kaggle là gì? Tại sao dùng Kaggle?](#1-kaggle-là-gì-tại-sao-dùng-kaggle)
2. [Chuẩn bị dữ liệu](#2-chuẩn-bị-dữ-liệu)
3. [Setup Kaggle Notebook](#3-setup-kaggle-notebook)
4. [Upload dữ liệu lên Kaggle](#4-upload-dữ-liệu-lên-kaggle)
5. [Training Model - Từng bước chi tiết](#5-training-model---từng-bước-chi-tiết)
6. [Download Model về máy](#6-download-model-về-máy)
7. [Troubleshooting - Xử lý lỗi](#7-troubleshooting---xử-lý-lỗi)

---

## 1. KAGGLE LÀ GÌ? TẠI SAO DÙNG KAGGLE?

### 1.1. Kaggle là gì?

**Kaggle** là một platform miễn phí của Google cho phép bạn:
- ✅ Chạy code Python với GPU mạnh (P100, 16GB VRAM)
- ✅ Lưu trữ dataset lớn (30GB free)
- ✅ Chạy notebook Jupyter trực tiếp trên trình duyệt
- ✅ Không cần cài đặt gì trên máy tính của bạn

**So sánh với các platform khác:**

| Platform | GPU | Storage | Thời gian | Ưu điểm |
|----------|-----|---------|-----------|---------|
| **Kaggle** | ✅ P100 (16GB) | 30GB | 9h/session | Ổn định, dễ dùng |
| **Google Colab** | ✅ T4 (16GB) | 15GB | 12h/session | Tích hợp Google Drive |
| **Local** | ❌ Cần mua | Unlimited | Unlimited | Tốn tiền GPU |

**👉 Khuyến nghị: Dùng Kaggle vì ổn định và dễ sử dụng hơn Colab.**

### 1.2. Tại sao cần GPU?

**Training ML model** cần tính toán rất nhiều:
- Một model Go có thể có hàng triệu tham số
- Training trên CPU: **10-20 giờ** cho 1 epoch
- Training trên GPU: **10-20 phút** cho 1 epoch

**👉 GPU nhanh hơn CPU khoảng 50-100 lần cho deep learning!**

---

## 2. CHUẨN BỊ DỮ LIỆU

### 2.1. Dữ liệu cần có gì?

Để train model, bạn cần:

1. **Board States** (Trạng thái bàn cờ)
   - Format: Tensor `[17, board_size, board_size]`
   - 17 planes = 8 lịch sử + 1 hiện tại + 8 features khác

2. **Labels** (Nhãn để train)
   - **Policy labels**: Nước đi đúng (từ professional games)
   - **Value labels**: Xác suất thắng (0.0 - 1.0)

### 2.2. Cách tạo dữ liệu

#### Option A: Từ Professional Games (Khuyến nghị)

```bash
# Bước 1: Download games từ KGS
python scripts/download_kgs_games.py \
  --output data/raw/kgs/ \
  --min-rank 5d \
  --max-games 5000

# Bước 2: Parse SGF files thành positions
python scripts/parse_sgf_colab.py \
  --input data/raw/kgs/ \
  --output data/processed/positions_9x9.pt \
  --board-size 9

# Output: data/processed/positions_9x9.pt
```

**Giải thích:**
- `--min-rank 5d`: Chỉ lấy games từ rank 5 dan trở lên (chất lượng cao)
- `--max-games 5000`: Tối đa 5000 games
- `--board-size 9`: Bàn cờ 9x9

#### Option B: Từ Self-Play (Nếu không có professional games)

```bash
# Tạo games bằng AI tự đánh với nhau
python src/ml/training/data_collector.py \
  --board-size 9 \
  --num-games 1000 \
  --output data/training/self_play_9x9.pt
```

### 2.3. Kiểm tra dữ liệu

Trước khi upload, kiểm tra file có đúng format không:

```python
import torch

# Load file
data = torch.load('data/processed/positions_9x9.pt')

# Kiểm tra
print(f"Số lượng positions: {len(data)}")
print(f"Ví dụ một position:")
print(f"  - Board state shape: {data[0]['board_state'].shape}")
print(f"  - Policy shape: {data[0]['policy'].shape}")
print(f"  - Value: {data[0]['value']}")
```

**Output mong đợi:**
```
Số lượng positions: 80000
Ví dụ một position:
  - Board state shape: torch.Size([17, 9, 9])
  - Policy shape: torch.Size([81])
  - Value: 0.65
```

---

## 3. SETUP KAGGLE NOTEBOOK

### 3.1. Tạo tài khoản Kaggle

1. Vào https://www.kaggle.com/
2. Click **"Sign Up"** hoặc **"Sign In"** (nếu đã có tài khoản)
3. Đăng nhập bằng Google account (dễ nhất)

### 3.2. Tạo Notebook mới

1. Vào https://www.kaggle.com/code
2. Click **"New Notebook"** (góc trên bên phải)
3. Chọn:
   - **Language**: Python
   - **Accelerator**: **GPU P100** (quan trọng!)
   - **Internet**: **On** (để download packages)

### 3.3. Cấu trúc thư mục Kaggle

Kaggle có cấu trúc thư mục như sau:

```
/kaggle/
├── input/          # Nơi chứa datasets (chỉ đọc)
├── working/        # Nơi bạn code và lưu output (có thể ghi)
└── output/         # Nơi lưu files để download (có thể ghi)
```

**Giải thích:**
- `/kaggle/input/`: Dataset bạn upload (read-only)
- `/kaggle/working/`: Nơi bạn code, train model (read-write)
- `/kaggle/output/`: Nơi lưu model để download (read-write)

---

## 4. UPLOAD DỮ LIỆU LÊN KAGGLE

### 4.1. Tạo Kaggle Dataset

1. Vào https://www.kaggle.com/datasets
2. Click **"New Dataset"**
3. Upload file `.pt` của bạn (ví dụ: `positions_9x9.pt`)
4. Đặt tên dataset: `gogame-training-data-9x9`
5. Click **"Create"**

**Lưu ý:**
- File phải nhỏ hơn 20GB (Kaggle giới hạn)
- Nếu file lớn, nén bằng `.zip` hoặc `.tar.gz` trước

### 4.2. Add Dataset vào Notebook

1. Trong notebook của bạn, click **"Add data"** (góc trên bên phải)
2. Tìm dataset vừa tạo: `gogame-training-data-9x9`
3. Click **"Add"**

**Sau khi add, dataset sẽ ở:** `/kaggle/input/gogame-training-data-9x9/`

### 4.3. Upload Code Model

Bạn có 2 cách:

#### Cách 1: Copy-paste code trực tiếp (Đơn giản)

Copy toàn bộ code từ project vào các cells trong notebook.

#### Cách 2: Upload file code (Khuyến nghị)

1. Nén folder `src/ml/` thành `ml_code.zip`
2. Upload vào Kaggle Dataset (giống như upload data)
3. Add dataset vào notebook
4. Giải nén trong notebook:

```python
import zipfile
import os

# Giải nén code
with zipfile.ZipFile('/kaggle/input/gogame-ml-code/ml_code.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/')

# Thêm vào Python path
import sys
sys.path.append('/kaggle/working/src')
```

---

## 5. TRAINING MODEL - TỪNG BƯỚC CHI TIẾT

### 5.1. Cell 1: Import và Setup

```python
# ============================================
# CELL 1: Import Libraries
# ============================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

# Kiểm tra GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Thêm code vào path (nếu upload code)
sys.path.append('/kaggle/working/src')
```

**Giải thích:**
- `torch`: PyTorch library cho deep learning
- `device`: Chọn GPU nếu có, không thì dùng CPU
- `sys.path.append`: Thêm thư mục code vào Python path

### 5.2. Cell 2: Load Model Code

```python
# ============================================
# CELL 2: Import Model Classes
# ============================================

# Nếu đã upload code, import từ đó
from ml.policy_network import PolicyNetwork, PolicyConfig
from ml.value_network import ValueNetwork, ValueConfig
from ml.features import board_to_tensor

# Hoặc định nghĩa lại model (nếu không upload code)
# (Copy code từ src/ml/policy_network.py và value_network.py)
```

**Giải thích:**
- Import các class model từ code đã upload
- Nếu không upload, bạn cần copy-paste code model vào cell này

### 5.3. Cell 3: Tạo Dataset Class

```python
# ============================================
# CELL 3: Dataset Class
# ============================================

class GoDataset(Dataset):
    """
    Dataset class để load training data.
    
    Mỗi sample gồm:
    - board_state: Tensor [17, 9, 9] - Trạng thái bàn cờ
    - policy: Tensor [81] - Xác suất nước đi (ground truth)
    - value: float - Xác suất thắng (0.0 - 1.0)
    """
    
    def __init__(self, data_path, board_size=9):
        """
        Args:
            data_path: Đường dẫn đến file .pt chứa data
            board_size: Kích thước bàn cờ (9, 13, hoặc 19)
        """
        self.data = torch.load(data_path)
        self.board_size = board_size
        
    def __len__(self):
        """Trả về số lượng samples"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Lấy một sample tại vị trí idx.
        
        Returns:
            board_state: Tensor [17, board_size, board_size]
            policy: Tensor [board_size * board_size]
            value: Tensor [1]
        """
        sample = self.data[idx]
        
        # Đảm bảo đúng format
        board_state = sample['board_state'].float()
        policy = sample['policy'].float()
        value = torch.tensor([sample['value']], dtype=torch.float32)
        
        return board_state, policy, value

# Test dataset
dataset_path = '/kaggle/input/gogame-training-data-9x9/positions_9x9.pt'
dataset = GoDataset(dataset_path, board_size=9)
print(f"Dataset size: {len(dataset)} samples")

# Xem một sample
board, policy, value = dataset[0]
print(f"Board shape: {board.shape}")
print(f"Policy shape: {policy.shape}")
print(f"Value: {value.item()}")
```

**Giải thích:**
- `GoDataset`: Class kế thừa `Dataset` của PyTorch
- `__len__()`: Trả về số lượng samples
- `__getitem__()`: Lấy một sample (PyTorch tự động gọi khi training)
- `float()`: Chuyển sang float32 (cần cho training)

### 5.4. Cell 4: Tạo DataLoader

```python
# ============================================
# CELL 4: DataLoader
# ============================================

# Chia train/validation
train_size = int(0.9 * len(dataset))  # 90% train
val_size = len(dataset) - train_size  # 10% validation

train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")

# Tạo DataLoader
batch_size = 64  # Số samples mỗi batch

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,  # Xáo trộn data mỗi epoch
    num_workers=2,  # Số threads để load data
    pin_memory=True  # Tăng tốc transfer lên GPU
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,  # Không cần shuffle validation
    num_workers=2,
    pin_memory=True
)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
```

**Giải thích:**
- `random_split`: Chia dataset thành train (90%) và validation (10%)
- `batch_size`: Số samples xử lý cùng lúc (64 = tốc độ và bộ nhớ cân bằng)
- `shuffle=True`: Xáo trộn data để model học tốt hơn
- `num_workers`: Số threads load data song song (2-4 là tốt)
- `pin_memory=True`: Tăng tốc transfer data lên GPU

### 5.5. Cell 5: Khởi tạo Model

```python
# ============================================
# CELL 5: Initialize Model
# ============================================

board_size = 9
input_planes = 17  # Số feature planes

# Tạo Policy Network
policy_config = PolicyConfig(
    board_size=board_size,
    input_planes=input_planes,
    channels=128
)
policy_net = PolicyNetwork(policy_config).to(device)

# Tạo Value Network
value_config = ValueConfig(
    board_size=board_size,
    input_planes=input_planes,
    channels=128
)
value_net = ValueNetwork(value_config).to(device)

# Đếm số tham số
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

policy_params = count_parameters(policy_net)
value_params = count_parameters(value_net)

print(f"Policy Network parameters: {policy_params:,}")
print(f"Value Network parameters: {value_params:,}")
print(f"Total parameters: {policy_params + value_params:,}")

# Test forward pass
test_input = torch.randn(1, input_planes, board_size, board_size).to(device)
with torch.no_grad():
    policy_out = policy_net(test_input)
    value_out = value_net(test_input)
    
print(f"Policy output shape: {policy_out.shape}")
print(f"Value output shape: {value_out.shape}")
```

**Giải thích:**
- `PolicyConfig` / `ValueConfig`: Cấu hình model (kích thước, số channels)
- `.to(device)`: Chuyển model lên GPU
- `count_parameters`: Đếm số tham số (để biết model lớn nhỏ thế nào)
- Test forward pass: Kiểm tra model chạy đúng không

### 5.6. Cell 6: Setup Training

```python
# ============================================
# CELL 6: Training Setup
# ============================================

# Loss functions
policy_loss_fn = nn.CrossEntropyLoss()  # Cho policy (classification)
value_loss_fn = nn.MSELoss()  # Cho value (regression)

# Optimizers
learning_rate = 1e-3  # 0.001
policy_optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
value_optimizer = optim.Adam(value_net.parameters(), lr=learning_rate)

# Learning rate scheduler (giảm LR khi loss không giảm)
policy_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    policy_optimizer, mode='min', factor=0.5, patience=3, verbose=True
)
value_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    value_optimizer, mode='min', factor=0.5, patience=3, verbose=True
)

# Training parameters
num_epochs = 10  # Số lần train toàn bộ dataset
save_every = 2  # Lưu checkpoint mỗi 2 epochs

print("Training setup complete!")
print(f"Learning rate: {learning_rate}")
print(f"Epochs: {num_epochs}")
```

**Giải thích:**
- `CrossEntropyLoss`: Loss cho policy (phân loại nước đi)
- `MSELoss`: Loss cho value (dự đoán xác suất thắng)
- `Adam`: Optimizer (tốt hơn SGD cho deep learning)
- `ReduceLROnPlateau`: Tự động giảm learning rate khi loss không giảm
- `num_epochs`: Số lần train toàn bộ dataset (10 = train 10 lần)

### 5.7. Cell 7: Training Loop

```python
# ============================================
# CELL 7: Training Loop
# ============================================

def train_one_epoch(policy_net, value_net, train_loader, 
                    policy_optimizer, value_optimizer,
                    policy_loss_fn, value_loss_fn, device):
    """
    Train một epoch.
    
    Returns:
        avg_policy_loss: Loss trung bình của policy
        avg_value_loss: Loss trung bình của value
    """
    policy_net.train()  # Chế độ training
    value_net.train()
    
    policy_losses = []
    value_losses = []
    
    # Tạo progress bar
    pbar = tqdm(train_loader, desc="Training")
    
    for batch_idx, (boards, policies, values) in enumerate(pbar):
        # Chuyển lên GPU
        boards = boards.to(device)
        policies = policies.to(device)
        values = values.to(device)
        
        # ===== TRAIN POLICY NETWORK =====
        policy_optimizer.zero_grad()  # Reset gradients
        
        policy_pred = policy_net(boards)  # Forward pass
        policy_loss = policy_loss_fn(policy_pred, policies.argmax(dim=1))  # Tính loss
        
        policy_loss.backward()  # Backward pass (tính gradients)
        policy_optimizer.step()  # Update weights
        
        policy_losses.append(policy_loss.item())
        
        # ===== TRAIN VALUE NETWORK =====
        value_optimizer.zero_grad()
        
        value_pred = value_net(boards)
        value_loss = value_loss_fn(value_pred.squeeze(), values.squeeze())
        
        value_loss.backward()
        value_optimizer.step()
        
        value_losses.append(value_loss.item())
        
        # Update progress bar
        pbar.set_postfix({
            'policy_loss': f'{policy_loss.item():.4f}',
            'value_loss': f'{value_loss.item():.4f}'
        })
    
    return np.mean(policy_losses), np.mean(value_losses)


def validate(policy_net, value_net, val_loader,
             policy_loss_fn, value_loss_fn, device):
    """
    Validate model trên validation set.
    
    Returns:
        avg_policy_loss: Loss trung bình của policy
        avg_value_loss: Loss trung bình của value
    """
    policy_net.eval()  # Chế độ evaluation
    value_net.eval()
    
    policy_losses = []
    value_losses = []
    
    with torch.no_grad():  # Không tính gradients (tiết kiệm bộ nhớ)
        for boards, policies, values in tqdm(val_loader, desc="Validating"):
            boards = boards.to(device)
            policies = policies.to(device)
            values = values.to(device)
            
            # Policy
            policy_pred = policy_net(boards)
            policy_loss = policy_loss_fn(policy_pred, policies.argmax(dim=1))
            policy_losses.append(policy_loss.item())
            
            # Value
            value_pred = value_net(boards)
            value_loss = value_loss_fn(value_pred.squeeze(), values.squeeze())
            value_losses.append(value_loss.item())
    
    return np.mean(policy_losses), np.mean(value_losses)


# Bắt đầu training
print("Starting training...")
print("=" * 50)

best_val_loss = float('inf')

for epoch in range(1, num_epochs + 1):
    print(f"\nEpoch {epoch}/{num_epochs}")
    print("-" * 50)
    
    # Train
    train_policy_loss, train_value_loss = train_one_epoch(
        policy_net, value_net, train_loader,
        policy_optimizer, value_optimizer,
        policy_loss_fn, value_loss_fn, device
    )
    
    # Validate
    val_policy_loss, val_value_loss = validate(
        policy_net, value_net, val_loader,
        policy_loss_fn, value_loss_fn, device
    )
    
    # Update learning rate
    policy_scheduler.step(val_policy_loss)
    value_scheduler.step(val_value_loss)
    
    # Print results
    print(f"Train - Policy Loss: {train_policy_loss:.4f}, Value Loss: {train_value_loss:.4f}")
    print(f"Val   - Policy Loss: {val_policy_loss:.4f}, Value Loss: {val_value_loss:.4f}")
    
    # Save checkpoint
    if epoch % save_every == 0:
        checkpoint = {
            'epoch': epoch,
            'policy_net_state_dict': policy_net.state_dict(),
            'value_net_state_dict': value_net.state_dict(),
            'policy_optimizer_state_dict': policy_optimizer.state_dict(),
            'value_optimizer_state_dict': value_optimizer.state_dict(),
            'train_policy_loss': train_policy_loss,
            'train_value_loss': train_value_loss,
            'val_policy_loss': val_policy_loss,
            'val_value_loss': val_value_loss,
        }
        
        checkpoint_path = f'/kaggle/working/checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
    
    # Save best model
    current_val_loss = val_policy_loss + val_value_loss
    if current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
        best_checkpoint_path = '/kaggle/working/best_model.pt'
        torch.save({
            'epoch': epoch,
            'policy_net_state_dict': policy_net.state_dict(),
            'value_net_state_dict': value_net.state_dict(),
            'val_policy_loss': val_policy_loss,
            'val_value_loss': val_value_loss,
        }, best_checkpoint_path)
        print(f"Saved best model: {best_checkpoint_path}")

print("\n" + "=" * 50)
print("Training complete!")
```

**Giải thích chi tiết:**

1. **`train_one_epoch()`**:
   - `model.train()`: Bật chế độ training (bật dropout, batch norm update)
   - `zero_grad()`: Reset gradients về 0 (quan trọng!)
   - `forward()`: Tính output từ input
   - `loss.backward()`: Tính gradients (đạo hàm)
   - `optimizer.step()`: Update weights dựa trên gradients

2. **`validate()`**:
   - `model.eval()`: Bật chế độ evaluation (tắt dropout, freeze batch norm)
   - `torch.no_grad()`: Không tính gradients (tiết kiệm bộ nhớ và nhanh hơn)

3. **Training loop**:
   - Train trên training set
   - Validate trên validation set
   - Lưu checkpoint mỗi `save_every` epochs
   - Lưu best model (model có validation loss thấp nhất)

### 5.8. Cell 8: Lưu Model cuối cùng

```python
# ============================================
# CELL 8: Save Final Models
# ============================================

# Lưu model cuối cùng
final_checkpoint = {
    'policy_net_state_dict': policy_net.state_dict(),
    'value_net_state_dict': value_net.state_dict(),
    'policy_config': policy_config.__dict__,
    'value_config': value_config.__dict__,
    'board_size': board_size,
}

# Lưu vào /kaggle/output để download
torch.save(final_checkpoint, '/kaggle/output/final_model.pt')
print("Saved final model to /kaggle/output/final_model.pt")

# Copy best model
import shutil
shutil.copy('/kaggle/working/best_model.pt', '/kaggle/output/best_model.pt')
print("Saved best model to /kaggle/output/best_model.pt")
```

**Giải thích:**
- `state_dict()`: Chỉ lưu weights, không lưu toàn bộ model (nhẹ hơn)
- `/kaggle/output/`: Thư mục để download files
- `/kaggle/working/`: Thư mục làm việc (không download được)

---

## 6. DOWNLOAD MODEL VỀ MÁY

### 6.1. Cách 1: Download từ Kaggle UI

1. Sau khi training xong, vào tab **"Output"** trong notebook
2. Click vào file `final_model.pt` hoặc `best_model.pt`
3. Click **"Download"**

### 6.2. Cách 2: Dùng Kaggle API

```bash
# Cài Kaggle API
pip install kaggle

# Setup API token (lấy từ Kaggle Account Settings)
# Copy kaggle.json vào ~/.kaggle/

# Download file
kaggle kernels output <username>/<kernel-slug> -p ./models/
```

### 6.3. Sử dụng Model

Sau khi download, load model trong code:

```python
import torch
from src.ml.policy_network import PolicyNetwork, PolicyConfig
from src.ml.value_network import ValueNetwork, ValueConfig

# Load checkpoint
checkpoint = torch.load('models/final_model.pt', map_location='cpu')

# Khởi tạo model
policy_config = PolicyConfig(**checkpoint['policy_config'])
policy_net = PolicyNetwork(policy_config)
policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
policy_net.eval()

value_config = ValueConfig(**checkpoint['value_config'])
value_net = ValueNetwork(value_config)
value_net.load_state_dict(checkpoint['value_net_state_dict'])
value_net.eval()

# Sử dụng model
# ... (code inference)
```

---

## 7. TROUBLESHOOTING - XỬ LÝ LỖI

### 7.1. Lỗi: "CUDA out of memory"

**Nguyên nhân:** Batch size quá lớn, không đủ VRAM

**Giải pháp:**
```python
# Giảm batch size
batch_size = 32  # Thay vì 64

# Hoặc dùng gradient accumulation
# (Train với batch nhỏ nhưng update weights như batch lớn)
```

### 7.2. Lỗi: "File not found"

**Nguyên nhân:** Đường dẫn dataset sai

**Giải pháp:**
```python
# Kiểm tra đường dẫn
import os
print(os.listdir('/kaggle/input/'))

# Tìm đúng tên dataset
# Dataset name thường có format: username/dataset-name
```

### 7.3. Lỗi: "Module not found"

**Nguyên nhân:** Chưa import code hoặc path sai

**Giải pháp:**
```python
# Kiểm tra path
import sys
print(sys.path)

# Thêm path đúng
sys.path.append('/kaggle/working/src')

# Hoặc copy-paste code trực tiếp vào notebook
```

### 7.4. Training quá chậm

**Nguyên nhân:**
- Chưa bật GPU
- Batch size quá nhỏ
- DataLoader không tối ưu

**Giải pháp:**
```python
# Kiểm tra GPU
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# Tăng batch size (nếu đủ VRAM)
batch_size = 128

# Tối ưu DataLoader
train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=4,  # Tăng số workers
    pin_memory=True,
    persistent_workers=True  # Giữ workers giữa các epochs
)
```

### 7.5. Loss không giảm

**Nguyên nhân:**
- Learning rate quá cao hoặc quá thấp
- Data chất lượng kém
- Model quá nhỏ hoặc quá lớn

**Giải pháp:**
```python
# Thử learning rate khác
learning_rate = 1e-4  # Thay vì 1e-3

# Hoặc dùng learning rate scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs
)

# Kiểm tra data
print(f"Sample value range: {values.min()} - {values.max()}")
print(f"Sample policy sum: {policies.sum(dim=1)}")  # Phải = 1
```

### 7.6. Session timeout

**Nguyên nhân:** Kaggle giới hạn 9 giờ/session

**Giải pháp:**
- Lưu checkpoint thường xuyên (mỗi epoch)
- Resume training từ checkpoint:

```python
# Load checkpoint
checkpoint = torch.load('/kaggle/working/checkpoint_epoch_5.pt')

# Resume
policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
value_net.load_state_dict(checkpoint['value_net_state_dict'])
policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])

start_epoch = checkpoint['epoch'] + 1

# Tiếp tục training từ epoch start_epoch
for epoch in range(start_epoch, num_epochs + 1):
    # ... training code
```

---

## 8. TIPS & BEST PRACTICES

### 8.1. Tối ưu Training

1. **Mixed Precision Training** (Nhanh hơn 2x):
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# Trong training loop
with autocast():
    policy_pred = policy_net(boards)
    policy_loss = policy_loss_fn(policy_pred, policies.argmax(dim=1))

scaler.scale(policy_loss).backward()
scaler.step(policy_optimizer)
scaler.update()
```

2. **Early Stopping** (Dừng sớm nếu không cải thiện):
```python
patience = 5
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(num_epochs):
    # ... training ...
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping!")
            break
```

3. **TensorBoard Logging** (Theo dõi training):
```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('/kaggle/working/logs')

# Trong training loop
writer.add_scalar('Loss/Train_Policy', train_policy_loss, epoch)
writer.add_scalar('Loss/Val_Policy', val_policy_loss, epoch)
```

### 8.2. Tiết kiệm thời gian

1. **Chỉ train trên subset nhỏ** để test code trước:
```python
# Test với 1000 samples
train_dataset = torch.utils.data.Subset(train_dataset, range(1000))
```

2. **Dùng pre-trained model** (nếu có):
```python
# Load weights từ model cũ
checkpoint = torch.load('old_model.pt')
policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
```

3. **Cache data** (load nhanh hơn):
```python
# Lưu processed data
torch.save(processed_data, '/kaggle/working/cached_data.pt')
```

---

## 9. TÓM TẮT QUY TRÌNH

```
1. Chuẩn bị data (.pt file)
   ↓
2. Tạo Kaggle account
   ↓
3. Upload data lên Kaggle Dataset
   ↓
4. Tạo Notebook mới (GPU P100)
   ↓
5. Add dataset vào notebook
   ↓
6. Copy code model vào notebook
   ↓
7. Chạy các cells training
   ↓
8. Download model từ Output tab
   ↓
9. Sử dụng model trong project
```

---

## 10. TÀI LIỆU THAM KHẢO

- **Kaggle Documentation**: https://www.kaggle.com/docs
- **PyTorch Tutorial**: https://pytorch.org/tutorials/
- **Kaggle Notebooks Examples**: https://www.kaggle.com/code

---

## ✅ CHECKLIST

Trước khi bắt đầu:
- [ ] Có tài khoản Kaggle
- [ ] Đã chuẩn bị data (.pt file)
- [ ] Đã upload data lên Kaggle Dataset
- [ ] Đã tạo Notebook với GPU

Trong khi training:
- [ ] Đã add dataset vào notebook
- [ ] Đã copy code model
- [ ] Đã chạy tất cả cells
- [ ] Đã lưu checkpoint thường xuyên

Sau khi training:
- [ ] Đã download model về máy
- [ ] Đã test model hoạt động đúng
- [ ] Đã lưu model vào project

---

**Chúc bạn training thành công! 🎉**

Nếu có vấn đề, xem phần [Troubleshooting](#7-troubleshooting---xử-lý-lỗi) hoặc đọc thêm:
- `docs/ML_TRAINING_COLAB_GUIDE.md` - Hướng dẫn Colab (tương tự)
- `docs/ML_COMPREHENSIVE_GUIDE.md` - Hướng dẫn toàn diện

