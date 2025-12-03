# ============================================
# FIXED VERSION: Split Dataset Script for Colab
# ============================================
# Issue: Files are in /processed/, not /processed/positions_19x19_2018/
# ============================================

from google.colab import drive
drive.mount('/content/drive')

import sys, os, torch
sys.path.append('/content/drive/MyDrive/GoGame_ML/code')

print("✅ Drive mounted!\n")

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ============================================
# 2) SPLIT DATASET to small files (~200MB)
# ============================================

from pathlib import Path

# FIXED: Point to the correct folder where files actually are
input_folder  = "/content/drive/MyDrive/GoGame_ML/processed"  # Files are directly here
output_folder = "/content/drive/MyDrive/GoGame_ML/processed/positions_19x19_2018_split"
target_chunk_size_mb = 200

os.makedirs(output_folder, exist_ok=True)

def split_chunk(file_path, output_folder, target_size_mb):
    print(f"\n📂 Đang xử lý: {file_path.name}")

    # Load 1 file lớn
    data = torch.load(file_path, map_location="cpu")
    samples = data["labeled_data"]
    total_samples = len(samples)

    print(f"➡ Tổng samples: {total_samples}")

    # Ước tính số samples mỗi file nhỏ
    est_size_per_sample = os.path.getsize(file_path) / total_samples
    target_samples = int((target_size_mb * 1024 * 1024) / est_size_per_sample)
    target_samples = max(1, target_samples)

    print(f"➡ Mỗi file nhỏ ~{target_samples} samples")

    # Chia nhỏ
    part = 0
    for start in range(0, total_samples, target_samples):
        end = min(start + target_samples, total_samples)
        part_data = samples[start:end]

        out_file = Path(output_folder) / f"{file_path.stem}_part{part:03d}.pt"
        torch.save({"labeled_data": part_data}, out_file)

        print(f"   ✔ Saved: {out_file.name} ({end-start} samples)")
        part += 1

    print(f"🎉 Đã chia thành {part} files nhỏ.\n")


# ===== RUN SPLITTING =====
input_path = Path(input_folder)

# FIXED: Filter for the specific file you want to split
# Option 1: Split only positions_19x19_2018.pt
files = list(input_path.glob("positions_19x19_2018.pt"))

# Option 2: If you want to split all .pt files in processed folder, use:
# files = sorted(list(input_path.glob("*.pt")))

print("\n🔍 Tìm thấy", len(files), "file lớn cần chia nhỏ.\n")

if len(files) == 0:
    print("❌ ERROR: Không tìm thấy file!")
    print(f"   Đang tìm trong: {input_folder}")
    print(f"   Files có sẵn:")
    all_files = list(input_path.glob("*.pt"))
    for f in all_files:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"     - {f.name} ({size_mb:.1f} MB)")
else:
    for f in files:
        split_chunk(f, output_folder, target_chunk_size_mb)

    print("\n🎯 DONE! Dataset đã được chia nhỏ hoàn toàn.\n")

# ============================================
# 3) Kiểm tra dataset split
# ============================================

print("📁 Kiểm tra 10 file đầu tiên:")
os.system('ls -lh "/content/drive/MyDrive/GoGame_ML/processed/positions_19x19_2018_split" | head -n 10')

# ============================================
# 4) Kiểm tra GPU
# ============================================
print("\nCUDA available:", torch.cuda.is_available())
print("Device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# ============================================
# 5) COPY SPLIT DATASET TỪ DRIVE → LOCAL
# ============================================

os.makedirs("/content/split18", exist_ok=True)
os.system('cp /content/drive/MyDrive/GoGame_ML/processed/positions_19x19_2018_split/*.pt /content/split18/')

# Verify copy
copied_files = list(Path("/content/split18").glob("*.pt"))
print(f"✅ Copied {len(copied_files)} split chunks to /content/split18")

# ============================================
# 6) TRAINING USING SPLIT DATASET
# ============================================

from train_colab import train_model

train_model(
    train_dataset_path="/content/split18",
    use_chunks=True,
    num_epochs=10,
    learning_rate=0.001,
    checkpoint_dir="/content/drive/MyDrive/GoGame_ML/checkpoints"
)

print("\n🚀 Training started with SPLIT dataset!")

