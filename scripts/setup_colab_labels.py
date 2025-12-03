"""
Setup script để upload và import các modules cần thiết trên Colab.

Cách sử dụng:
1. Upload file này lên Colab
2. Chạy: exec(open('setup_colab_labels.py').read())
3. Hoặc copy-paste nội dung vào một cell và chạy
"""

import sys
from pathlib import Path

# Thêm scripts directory vào Python path
SCRIPTS_DIR = Path('/content/drive/MyDrive/GoGame_ML/scripts')
if SCRIPTS_DIR.exists():
    sys.path.insert(0, str(SCRIPTS_DIR))
    print(f"✅ Added {SCRIPTS_DIR} to Python path")
else:
    print(f"⚠️  Scripts directory not found: {SCRIPTS_DIR}")
    print("   Please upload scripts to this location or update SCRIPTS_DIR")

# Kiểm tra các modules cần thiết
required_modules = [
    'generate_labels_colab',
    'label_generators',
    'generate_features_colab'
]

missing_modules = []
for module in required_modules:
    try:
        __import__(module)
        print(f"✅ {module} imported successfully")
    except ImportError as e:
        missing_modules.append(module)
        print(f"❌ {module} not found: {e}")

if missing_modules:
    print(f"\n⚠️  Missing modules: {missing_modules}")
    print("\nPlease upload these files to:")
    print(f"  {SCRIPTS_DIR}")
    print("\nRequired files:")
    for module in missing_modules:
        print(f"  - {module}.py")
else:
    print("\n✅ All modules ready!")
    print("\nYou can now use:")
    print("  from generate_labels_colab import process_dataset_file")

