"""
Colab Notebook Template - Copy từng cell vào Colab

Hướng dẫn:
1. Tạo notebook mới trên Colab
2. Enable GPU: Runtime → Change runtime type → GPU
3. Mount Google Drive: Chạy Cell 1
4. Copy từng cell vào notebook và chạy theo thứ tự
"""

# ============================================
# CELL 1: Setup và Mount Drive
# ============================================

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Setup paths
from pathlib import Path
WORK_DIR = Path('/content/drive/MyDrive/GoGame_ML')
WORK_DIR.mkdir(parents=True, exist_ok=True)

# Create directories
(WORK_DIR / 'raw_sgf').mkdir(exist_ok=True)
(WORK_DIR / 'processed').mkdir(exist_ok=True)
(WORK_DIR / 'datasets').mkdir(exist_ok=True)
(WORK_DIR / 'checkpoints').mkdir(exist_ok=True)
(WORK_DIR / 'code').mkdir(exist_ok=True)

print(f"✅ Setup complete! Working directory: {WORK_DIR}")


# ============================================
# CELL 2: Install Dependencies
# ============================================

!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install numpy pandas tqdm sgf

print("✅ Dependencies installed!")


# ============================================
# CELL 3: Upload Code Models (hoặc copy từ local)
# ============================================

# Option A: Upload files từ local
# - policy_network.py
# - value_network.py
# - generate_features_colab.py
# - generate_labels_colab.py
# - train_colab.py

# Option B: Copy code trực tiếp vào cells (xem các cell sau)

print("📝 Code files ready!")


# ============================================
# CELL 4: Copy Policy Network Code
# ============================================

# Copy toàn bộ nội dung từ src/ml/policy_network.py vào đây
# Hoặc upload file và import:
# import sys
# sys.path.append(str(WORK_DIR / 'code'))
# from policy_network import PolicyNetwork, PolicyConfig

print("✅ Policy Network loaded!")


# ============================================
# CELL 5: Copy Value Network Code
# ============================================

# Copy toàn bộ nội dung từ src/ml/value_network.py vào đây
# Hoặc upload file và import:
# from value_network import ValueNetwork, ValueConfig

print("✅ Value Network loaded!")


# ============================================
# CELL 6: Copy Feature Generation Code
# ============================================

# Copy toàn bộ nội dung từ scripts/generate_features_colab.py vào đây
# Hoặc upload file và import:
# from generate_features_colab import (
#     board_to_features_17_planes,
#     generate_policy_label,
#     generate_value_label
# )

print("✅ Feature generation loaded!")


# ============================================
# CELL 7: Upload SGF Files (nếu chưa có)
# ============================================

# Nếu đã có SGF files trong Drive, skip cell này
# Nếu chưa có, upload vào WORK_DIR / 'raw_sgf'

from pathlib import Path
sgf_files = list((WORK_DIR / 'raw_sgf').glob('*.sgf'))
print(f"📊 Found {len(sgf_files)} SGF files")


# ============================================
# CELL 8: Parse SGF Files → Positions
# ============================================

# Copy code từ scripts/parse_sgf_colab.py
# Hoặc import:
# from parse_sgf_colab import process_sgf_directory

from parse_sgf_colab import process_sgf_directory

process_sgf_directory(
    sgf_dir=WORK_DIR / 'raw_sgf',
    output_dir=WORK_DIR / 'processed',
    board_sizes=[9, 13, 19]  # Chọn board sizes cần train
)

print("✅ SGF parsing complete!")


# ============================================
# CELL 9: Generate Labels từ Positions (với Incremental Save)
# ============================================

# Copy code từ scripts/generate_labels_colab.py
# Hoặc import:
# from generate_labels_colab import process_dataset_file

from generate_labels_colab import process_dataset_file

# Process cho từng board size hoặc từng năm
# Option 1: Process theo board size
for board_size in [9, 13, 19]:  # Chọn board sizes cần train
    input_file = WORK_DIR / 'processed' / f'positions_{board_size}x{board_size}.pt'
    output_file = WORK_DIR / 'datasets' / f'labeled_{board_size}x{board_size}.pt'
    
    if input_file.exists():
        print(f"\n🔄 Processing {board_size}x{board_size}...")
        process_dataset_file(
            input_path=str(input_file),
            output_path=str(output_file),
            filter_handicap=True,
            save_chunk_size=50000,  # Save mỗi 50K samples (~1.2GB) để tránh MemoryError
            auto_enable_incremental=True,  # Tự động enable nếu estimated memory > 4GB
            skip_merge=False  # Nếu True, bỏ qua merge (dùng khi RAM không đủ)
        )
    else:
        print(f"⚠️  Skipping {board_size}x{board_size} (file not found)")

# Option 2: Process theo năm (nếu có data theo năm)
# for year in [2019, 2020, 2021]:
#     input_file = WORK_DIR / 'processed' / f'positions_19x19_{year}.pt'
#     output_file = WORK_DIR / 'datasets' / f'labeled_19x19_{year}.pt'
#     
#     if input_file.exists():
#         print(f"\n🔄 Processing year {year}...")
#         process_dataset_file(
#             input_path=str(input_file),
#             output_path=str(output_file),
#             filter_handicap=True,
#             save_chunk_size=50000,
#             auto_enable_incremental=True
#         )
#     else:
#         print(f"⚠️  Skipping year {year} (file not found)")

print("\n✅ Label generation complete!")


# ============================================
# CELL 10: Verify Dataset
# ============================================

import torch

# Load và kiểm tra dataset
dataset_path = WORK_DIR / 'datasets' / 'labeled_9x9.pt'
data = torch.load(dataset_path, map_location='cpu')

print(f"📊 Dataset info:")
print(f"   Board size: {data['board_size']}x{data['board_size']}")
print(f"   Total samples: {len(data['labeled_data']):,}")

# Xem một sample
sample = data['labeled_data'][0]
print(f"\n📝 Sample structure:")
print(f"   Features shape: {sample['features'].shape}")
print(f"   Policy shape: {sample['policy'].shape}")
print(f"   Value: {sample['value']}")

print("✅ Dataset verified!")


# ============================================
# CELL 11: Training Setup
# ============================================

# Copy code từ scripts/train_colab.py
# Hoặc import:
# from train_colab import train_model

from train_colab import train_model

# Check GPU
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print("✅ Training setup complete!")


# ============================================
# CELL 12: Start Training
# ============================================

# Option 1: Training từ merged file (dataset nhỏ)
# train_model(
#     train_dataset_path=str(WORK_DIR / 'datasets' / 'labeled_9x9.pt'),
#     val_dataset_path=None,  # Auto-split từ training
#     board_size=None,  # Auto-detect từ dataset
#     batch_size=16,  # ⭐ Giảm nếu gặp RAM issues
#     num_epochs=10,
#     learning_rate=0.001,
#     checkpoint_dir=str(WORK_DIR / 'checkpoints'),
#     use_chunks=False
# )

# Option 2: Training từ chunks (dataset lớn, khuyến nghị)
train_model(
    train_dataset_path=str(WORK_DIR / 'datasets' / 'labeled_19x19_2019_chunks'),  # ⭐ Chunks directory
    val_dataset_path=None,  # Có thể dùng chunks riêng cho validation
    board_size=None,  # ⭐ Auto-detect từ dataset
    batch_size=16,  # ⭐ Giảm để tránh RAM overflow (8-16 là tốt)
    num_epochs=10,
    learning_rate=0.001,
    checkpoint_dir=str(WORK_DIR / 'checkpoints'),
    use_chunks=True  # ⭐ Enable chunks mode
)

print("✅ Training complete!")


# ============================================
# CELL 13: Download Model
# ============================================

# Download best model
from google.colab import files

best_model_path = WORK_DIR / 'checkpoints' / 'best_model.pt'
if best_model_path.exists():
    files.download(str(best_model_path))
    print("✅ Best model downloaded!")

final_model_path = WORK_DIR / 'checkpoints' / 'final_model.pt'
if final_model_path.exists():
    files.download(str(final_model_path))
    print("✅ Final model downloaded!")


# ============================================
# CELL 14: Test Model (Optional)
# ============================================

# Load và test model
import torch
from train_colab import GoDataset
from policy_network import PolicyNetwork, PolicyConfig
from value_network import ValueNetwork, ValueConfig

# Load model
checkpoint = torch.load(WORK_DIR / 'checkpoints' / 'best_model.pt', map_location='cpu')

policy_config = PolicyConfig(**checkpoint['policy_config'])
policy_net = PolicyNetwork(policy_config)
policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
policy_net.eval()

value_config = ValueConfig(**checkpoint['value_config'])
value_net = ValueNetwork(value_config)
value_net.load_state_dict(checkpoint['value_net_state_dict'])
value_net.eval()

# Test với một sample
dataset_path = WORK_DIR / 'datasets' / 'labeled_9x9.pt'
data = torch.load(dataset_path, map_location='cpu')
sample = data['labeled_data'][0]

features = sample['features'].unsqueeze(0)  # Add batch dimension

with torch.no_grad():
    policy_logits = policy_net(features)
    value_pred = value_net(features)

print(f"Policy output shape: {policy_logits.shape}")
print(f"Value output: {value_pred.item():.4f}")
print(f"Best move index: {policy_logits.argmax().item()}")

print("✅ Model test complete!")

