"""
Helper script để tạo cấu trúc thư mục và code cho Colab.

Chạy script này trên local để:
1. Tạo ZIP file chứa code model
2. Hướng dẫn upload dataset
3. Tạo notebook template
"""

import zipfile
from pathlib import Path
import shutil

def create_code_zip(output_path: Path = Path("gogame_ml_code.zip")):
    """Tạo ZIP file chứa code model để upload lên Colab."""
    print("📦 Creating code ZIP file...")
    
    # Paths
    src_ml = Path("src/ml")
    code_files = [
        src_ml / "models" / "multi_task_model.py",
        src_ml / "models" / "shared_backbone.py",
        src_ml / "models" / "threat_head.py",
        src_ml / "models" / "attack_head.py",
        src_ml / "models" / "intent_head.py",
        src_ml / "models" / "__init__.py",
        src_ml / "features.py",
    ]
    
    # Create ZIP
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add models directory structure
        for file_path in code_files:
            if file_path.exists():
                # Preserve directory structure
                arcname = file_path.relative_to(src_ml)
                zipf.write(file_path, arcname)
                print(f"  ✅ Added: {arcname}")
            else:
                print(f"  ⚠️  Not found: {file_path}")
    
    print(f"\n✅ Created: {output_path}")
    print(f"   Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"\n📤 Upload file này lên Colab vào thư mục GoGame_ML/code/")
    return output_path


def print_setup_instructions():
    """In hướng dẫn setup."""
    print("\n" + "=" * 60)
    print("📋 HƯỚNG DẪN SETUP COLAB")
    print("=" * 60)
    
    print("\n1️⃣  CẤU TRÚC THƯ MỤC GOOGLE DRIVE:")
    print("   GoGame_ML/")
    print("   ├── datasets/          ← Upload dataset .pt vào đây")
    print("   ├── code/              ← Upload code ZIP vào đây")
    print("   ├── checkpoints/       (tự động tạo)")
    print("   ├── logs/              (tự động tạo)")
    print("   └── outputs/           (tự động tạo)")
    
    print("\n2️⃣  UPLOAD DATASET:")
    print("   - File format: .pt (PyTorch)")
    print("   - Vị trí: GoGame_ML/datasets/positions_9x9.pt")
    print("   - Dataset phải có keys: 'positions' hoặc 'labeled_data', 'board_size'")
    
    print("\n3️⃣  UPLOAD CODE:")
    print("   - Option A: Upload ZIP file (gogame_ml_code.zip)")
    print("   - Option B: Copy code trực tiếp vào Colab cells")
    print("   - Option C: Clone từ GitHub (nếu có)")
    
    print("\n4️⃣  CHẠY TRÊN COLAB:")
    print("   - Cell 1: Check GPU")
    print("   - Cell 2: Mount Drive + Setup folders")
    print("   - Cell 3: Upload/Extract code")
    print("   - Cell 4: Upload dataset")
    print("   - Cell 5: Install dependencies")
    print("   - Cell 6: Verify setup")
    print("   - Cell 7+: Training loop")
    
    print("\n" + "=" * 60)


def create_colab_notebook_template(output_path: Path = Path("GoGame_ML_Training_Template.ipynb")):
    """Tạo notebook template cho Colab."""
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# GoGame ML Training trên Colab\n",
                    "\n",
                    "## Setup và Training Pipeline"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Cell 1: Check GPU\n",
                    "import torch\n",
                    "print(f\"PyTorch version: {torch.__version__}\")\n",
                    "print(f\"CUDA available: {torch.cuda.is_available()}\")\n",
                    "if torch.cuda.is_available():\n",
                    "    print(f\"GPU: {torch.cuda.get_device_name(0)}\")\n",
                    "    print(f\"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Cell 2: Mount Drive và Setup\n",
                    "from google.colab import drive\n",
                    "from pathlib import Path\n",
                    "import os\n",
                    "\n",
                    "drive.mount('/content/drive')\n",
                    "\n",
                    "WORK_DIR = Path('/content/drive/MyDrive/GoGame_ML')\n",
                    "WORK_DIR.mkdir(exist_ok=True)\n",
                    "\n",
                    "# Tạo thư mục\n",
                    "(WORK_DIR / 'datasets').mkdir(exist_ok=True)\n",
                    "(WORK_DIR / 'code').mkdir(exist_ok=True)\n",
                    "(WORK_DIR / 'checkpoints').mkdir(exist_ok=True)\n",
                    "(WORK_DIR / 'logs').mkdir(exist_ok=True)\n",
                    "(WORK_DIR / 'outputs').mkdir(exist_ok=True)\n",
                    "\n",
                    "os.chdir(WORK_DIR)\n",
                    "print(f\"✅ Working directory: {WORK_DIR}\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Cell 3: Upload Code (chọn 1 trong 3 cách)\n",
                    "# Cách 1: Upload ZIP\n",
                    "from google.colab import files\n",
                    "import zipfile\n",
                    "\n",
                    "uploaded = files.upload()\n",
                    "for filename in uploaded.keys():\n",
                    "    if filename.endswith('.zip'):\n",
                    "        with zipfile.ZipFile(filename, 'r') as zip_ref:\n",
                    "            zip_ref.extractall(WORK_DIR / 'code')\n",
                    "        print(f\"✅ Extracted {filename}\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Cell 4: Upload Dataset\n",
                    "from google.colab import files\n",
                    "import shutil\n",
                    "\n",
                    "uploaded = files.upload()\n",
                    "for filename in uploaded.keys():\n",
                    "    if filename.endswith('.pt'):\n",
                    "        shutil.move(filename, WORK_DIR / 'datasets' / filename)\n",
                    "        print(f\"✅ Moved {filename} to datasets/\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Cell 5: Install Dependencies\n",
                    "!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118\n",
                    "!pip install numpy pandas tqdm tensorboard scikit-learn\n",
                    "!pip install sgfmill\n",
                    "\n",
                    "import sys\n",
                    "sys.path.insert(0, str(WORK_DIR / 'code'))\n",
                    "sys.path.insert(0, str(WORK_DIR / 'code' / 'models'))\n",
                    "\n",
                    "print(\"✅ Dependencies installed\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Cell 6: Verify Setup\n",
                    "# (Xem trong guide để có code đầy đủ)"
                ]
            }
        ],
        "metadata": {
            "colab": {
                "provenance": []
            },
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }
    
    import json
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook_content, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Created notebook template: {output_path}")
    print(f"   Upload file này lên Colab để bắt đầu!")


if __name__ == "__main__":
    print("🚀 GoGame ML - Colab Setup Helper\n")
    
    # Create code ZIP
    create_code_zip()
    
    # Print instructions
    print_setup_instructions()
    
    # Create notebook template (optional)
    create_colab_notebook_template()
    
    print("\n✅ Setup helper completed!")
    print("\n📝 Next steps:")
    print("   1. Upload gogame_ml_code.zip lên Colab")
    print("   2. Upload dataset .pt file lên Colab")
    print("   3. Follow guide trong ML_TRAINING_COLAB_GUIDE.md")

