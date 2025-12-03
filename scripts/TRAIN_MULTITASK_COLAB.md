# Hướng dẫn Training MultiTaskModel trên Colab

## 1. Setup và Mount Drive

```python
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
```

## 2. Copy models/ vào Colab

Đảm bảo thư mục `models/` (từ `src/ml/models/`) đã được copy vào Drive:
- `/content/drive/MyDrive/GoGame_ML/code/models/`

Hoặc copy trực tiếp:
```python
# Copy models từ local
!cp -r /content/drive/MyDrive/GoGame_ML/code/models /content/
sys.path.insert(0, '/content')
```

## 3. Copy training scripts

```python
# Copy train_colab.py và chunk_dataset.py vào /content
!cp /content/drive/MyDrive/GoGame_ML/code/train_colab.py /content/
!cp /content/drive/MyDrive/GoGame_ML/code/chunk_dataset.py /content/
```

## 4. Split Dataset (nếu cần)

```python
from pathlib import Path

input_folder  = "/content/drive/MyDrive/GoGame_ML/datasets/labeled_19x19_2018_chunks"
output_folder = "/content/drive/MyDrive/GoGame_ML/datasets/labeled_19x19_2018_chunks_split"
target_chunk_size_mb = 200

os.makedirs(output_folder, exist_ok=True)

def split_chunk(file_path, output_folder, target_size_mb):
    print(f"\n📂 Đang xử lý: {file_path.name}")
    data = torch.load(file_path, map_location="cpu")
    samples = data["labeled_data"]
    total_samples = len(samples)
    print(f"➡ Tổng samples: {total_samples}")

    est_size_per_sample = os.path.getsize(file_path) / total_samples
    target_samples = int((target_size_mb * 1024 * 1024) / est_size_per_sample)
    target_samples = max(1, target_samples)
    print(f"➡ Mỗi file nhỏ ~{target_samples} samples")

    part = 0
    for start in range(0, total_samples, target_samples):
        end = min(start + target_samples, total_samples)
        part_data = samples[start:end]
        out_file = Path(output_folder) / f"{file_path.stem}_part{part:03d}.pt"
        torch.save({"labeled_data": part_data}, out_file)
        print(f"   ✔ Saved: {out_file.name} ({end-start} samples)")
        part += 1
    print(f"🎉 Đã chia thành {part} files nhỏ.\n")

input_path = Path(input_folder)
files = sorted(list(input_path.glob("*.pt")))
print("\n🔍 Tìm thấy", len(files), "file lớn cần chia nhỏ.\n")

for f in files:
    split_chunk(f, output_folder, target_chunk_size_mb)

print("\n🎯 DONE! Dataset đã được chia nhỏ hoàn toàn.\n")
```

## 5. Copy Split Dataset từ Drive → Local

```python
!mkdir -p /content/split19
!cp /content/drive/MyDrive/GoGame_ML/datasets/labeled_19x19_2018_chunks_split/*.pt /content/split19/
print("✅ Copied all split chunks to /content/split19")
```

## 6. Training MultiTaskModel

```python
from train_colab import train_multi_task_model

train_multi_task_model(
    train_dataset_path="/content/split19",
    use_chunks=True,
    num_epochs=10,
    learning_rate=0.001,
    checkpoint_dir="/content/drive/MyDrive/GoGame_ML/checkpoints",
    batch_size=1024,  # Có thể điều chỉnh theo GPU RAM
    base_channels=64,
    num_res_blocks=4,
    use_detection_labels=True,  # Bật nếu dataset có labels cho detection heads
    loss_weights={
        'threat': 1.0,
        'attack': 1.0,
        'intent': 1.0
    }
)

print("\n🚀 Training MultiTaskModel started!")
```

## Lưu ý:

1. **Detection Labels**: Nếu dataset chưa có labels cho detection heads (threat_map, attack_map, intent_label), model sẽ train với dummy labels (zeros). Để train thực sự, cần tạo labels từ game data.

2. **Model Path**: Đảm bảo `models/` được import đúng. Có thể cần điều chỉnh `sys.path` tùy theo cấu trúc thư mục.

3. **GPU Memory**: Điều chỉnh `batch_size` và `base_channels` theo GPU RAM của bạn.

4. **Checkpoints**: Model sẽ được lưu với prefix `_multitask_` để phân biệt với Policy/Value models.

