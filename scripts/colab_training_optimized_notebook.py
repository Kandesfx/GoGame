"""
📓 COLAB NOTEBOOK TEMPLATE - TRAINING TỐI ƯU VỚI CHUNKS

Copy từng cell vào Colab notebook và chạy theo thứ tự.
"""

# ============================================
# CELL 1: Mount Google Drive
# ============================================

from google.colab import drive
drive.mount('/content/drive')

print("✅ Google Drive mounted!")


# ============================================
# CELL 2: Setup Directories
# ============================================

from pathlib import Path

WORK_DIR = Path('/content/drive/MyDrive/GoGame_ML')

# Tạo cấu trúc thư mục
(WORK_DIR / 'code').mkdir(parents=True, exist_ok=True)
(WORK_DIR / 'datasets').mkdir(parents=True, exist_ok=True)
(WORK_DIR / 'checkpoints').mkdir(parents=True, exist_ok=True)

print(f"✅ Directories created at: {WORK_DIR}")


# ============================================
# CELL 3: Upload Code Files (Manual)
# ============================================

print("📤 Please upload the following files to GoGame_ML/code/:")
print("   1. train_colab_optimized.py")
print("   2. chunk_dataset_optimized.py")
print("   3. policy_network.py")
print("   4. value_network.py")
print("\nOr use:")
print("   from google.colab import files")
print("   files.upload()  # Select files to upload")


# ============================================
# CELL 4: Verify Uploaded Files
# ============================================

code_dir = WORK_DIR / 'code'
required_files = [
    'train_colab_optimized.py',
    'chunk_dataset_optimized.py',
    'policy_network.py',
    'value_network.py'
]

print("🔍 Checking uploaded files...")
for file in required_files:
    file_path = code_dir / file
    if file_path.exists():
        print(f"   ✅ {file}")
    else:
        print(f"   ❌ {file} - NOT FOUND!")

print("\n✅ All files verified!")


# ============================================
# CELL 5: Add Code Directory to Path
# ============================================

import sys
sys.path.insert(0, str(WORK_DIR / 'code'))

# Verify imports
try:
    from train_colab_optimized import train_model_optimized
    from chunk_dataset_optimized import create_chunk_dataset
    print("✅ Imports successful!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Please check that all files are uploaded correctly.")


# ============================================
# CELL 6: Check Chunks Directory
# ============================================

# Thay đổi path này theo dataset của bạn
chunks_dir = WORK_DIR / 'datasets' / 'labeled_19x19_2012_chunks'

if chunks_dir.exists():
    chunk_files = sorted(chunks_dir.glob("*.pt"))
    print(f"📦 Found {len(chunk_files)} chunk files:")
    for i, f in enumerate(chunk_files[:10], 1):
        size_mb = f.stat().st_size / (1024**2)
        print(f"   {i}. {f.name} ({size_mb:.1f} MB)")
    if len(chunk_files) > 10:
        print(f"   ... and {len(chunk_files) - 10} more files")
    
    total_size_gb = sum(f.stat().st_size for f in chunk_files) / (1024**3)
    print(f"\n📊 Total size: {total_size_gb:.2f} GB")
    print(f"📊 Total samples: ~{len(chunk_files) * 1000000:,} (estimated)")
else:
    print(f"❌ Chunks directory not found: {chunks_dir}")
    print("   Please update the path in this cell.")


# ============================================
# CELL 7: System Check
# ============================================

import torch
import psutil

print("🖥️  System Check:")
print(f"   Python: {sys.version.split()[0]}")
print(f"   PyTorch: {torch.__version__}")

# RAM
mem = psutil.virtual_memory()
print(f"   RAM: {mem.total / (1024**3):.1f} GB total, {mem.available / (1024**3):.1f} GB available")

# GPU
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"   CUDA: {torch.version.cuda}")
    if torch.cuda.is_bf16_supported():
        print("   ✅ Mixed precision supported")
    else:
        print("   ⚠️  Mixed precision not supported")
else:
    print("   ⚠️  No GPU detected!")

print("\n✅ System check complete!")


# ============================================
# CELL 8: Start Training
# ============================================

from train_colab_optimized import train_model_optimized
from pathlib import Path

WORK_DIR = Path('/content/drive/MyDrive/GoGame_ML')

# ⚙️ CONFIGURATION - Thay đổi theo nhu cầu
CONFIG = {
    'train_dataset_path': str(WORK_DIR / 'datasets' / 'labeled_19x19_2012_chunks'),
    'val_dataset_path': None,  # Có thể dùng chunks riêng cho validation
    'board_size': None,  # Auto-detect từ dataset
    'batch_size': None,  # ⭐ Auto-detect optimal batch size
    'num_epochs': 10,
    'learning_rate': 0.001,
    'checkpoint_dir': str(WORK_DIR / 'checkpoints'),
    'save_every': 2,  # Save checkpoint mỗi 2 epochs
    'use_chunks': True,  # ⭐ Enable chunks mode
    'use_mixed_precision': True,  # ⭐ Mixed precision (nhanh hơn)
    'chunk_pattern': None,  # ⭐ Auto-detect pattern (labeled_*_*.pt)
    'pin_memory': True,  # ⭐ Tối ưu GPU transfer
    'prefetch_factor': 2  # Prefetch batches
}

print("🚀 Starting training with configuration:")
for key, value in CONFIG.items():
    print(f"   {key}: {value}")

print("\n" + "=" * 80)
print("⚠️  This may take several hours. Make sure Colab Pro session won't timeout!")
print("=" * 80)

# Start training
train_model_optimized(**CONFIG)


# ============================================
# CELL 9: Download Results (Optional)
# ============================================

from google.colab import files
from pathlib import Path

WORK_DIR = Path('/content/drive/MyDrive/GoGame_ML')
checkpoint_dir = WORK_DIR / 'checkpoints'

print("📥 Downloading results...")

# Download best model
best_model = checkpoint_dir / 'best_model.pt'
if best_model.exists():
    print(f"   Downloading: {best_model.name}")
    files.download(str(best_model))
else:
    print("   ⚠️  best_model.pt not found")

# Download final model
final_model = checkpoint_dir / 'final_model.pt'
if final_model.exists():
    print(f"   Downloading: {final_model.name}")
    files.download(str(final_model))
else:
    print("   ⚠️  final_model.pt not found")

print("\n✅ Download complete!")


# ============================================
# CELL 10: Load and Test Model (Optional)
# ============================================

import torch
from pathlib import Path

# Load best model
checkpoint_path = WORK_DIR / 'checkpoints' / 'best_model.pt'

if checkpoint_path.exists():
    print(f"📂 Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"   Board size: {checkpoint.get('board_size', 'N/A')}")
    print(f"   Val Policy Loss: {checkpoint.get('val_policy_loss', 'N/A'):.4f}")
    print(f"   Val Value Loss: {checkpoint.get('val_value_loss', 'N/A'):.4f}")
    
    print("\n✅ Model loaded successfully!")
else:
    print(f"❌ Model not found: {checkpoint_path}")

