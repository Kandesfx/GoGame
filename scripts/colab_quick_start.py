"""
🚀 QUICK START CODE CHO COLAB
Copy-paste code này vào các cells trên Colab
"""

# ============================================================================
# CELL 1: Mount Google Drive và Setup Path
# ============================================================================
"""
from google.colab import drive
drive.mount('/content/drive')

import sys
from pathlib import Path

# Thêm scripts directory vào Python path
SCRIPTS_DIR = Path('/content/drive/MyDrive/GoGame_ML/scripts')
if SCRIPTS_DIR.exists():
    sys.path.insert(0, str(SCRIPTS_DIR))
    print(f"✅ Added {SCRIPTS_DIR} to Python path")
else:
    print(f"⚠️  Scripts directory not found: {SCRIPTS_DIR}")
    print("   Please upload scripts to this location first!")
    print("   Required files:")
    print("     - generate_labels_colab.py")
    print("     - label_generators.py")
    print("     - generate_features_colab.py")
"""

# ============================================================================
# CELL 2: Verify Files Exist
# ============================================================================
"""
from pathlib import Path

SCRIPTS_DIR = Path('/content/drive/MyDrive/GoGame_ML/scripts')
required_files = [
    'generate_labels_colab.py',
    'label_generators.py',
    'generate_features_colab.py'
]

print("Checking required files...")
all_exist = True
for file in required_files:
    file_path = SCRIPTS_DIR / file
    exists = file_path.exists()
    status = "✅" if exists else "❌"
    print(f"{status} {file}: {exists}")
    if not exists:
        all_exist = False

if all_exist:
    print("\n✅ All files found!")
else:
    print("\n⚠️  Some files are missing. Please upload them to:")
    print(f"   {SCRIPTS_DIR}")
"""

# ============================================================================
# CELL 3: Test Import
# ============================================================================
"""
try:
    from generate_labels_colab import process_dataset_file
    print("✅ generate_labels_colab imported successfully")
except ImportError as e:
    print(f"❌ Error importing generate_labels_colab: {e}")
    print("\nTroubleshooting:")
    print("1. Check if file exists in scripts directory")
    print("2. Check if Python path is set correctly")
    print("3. Try: import sys; print(sys.path)")

try:
    from label_generators import ThreatLabelGenerator
    print("✅ label_generators imported successfully")
except ImportError as e:
    print(f"❌ Error importing label_generators: {e}")

try:
    from generate_features_colab import board_to_features_17_planes
    print("✅ generate_features_colab imported successfully")
except ImportError as e:
    print(f"❌ Error importing generate_features_colab: {e}")
"""

# ============================================================================
# CELL 4: Use (Sau khi import thành công)
# ============================================================================
"""
from pathlib import Path
from generate_labels_colab import process_dataset_file

WORK_DIR = Path('/content/drive/MyDrive/GoGame_ML')

# Process một file
process_dataset_file(
    input_path=WORK_DIR / 'processed' / 'positions_19x19_2019.pt',
    output_path=WORK_DIR / 'datasets' / 'labeled_19x19_2019.pt',
    filter_handicap=True,
    save_chunk_size=50000  # Quan trọng cho Colab RAM limit
)
"""

# ============================================================================
# ALTERNATIVE: Nếu không muốn dùng Google Drive, copy files vào /content
# ============================================================================
"""
# Cell 1: Upload files qua Colab UI, sau đó:
import sys
sys.path.insert(0, '/content')

# Hoặc copy từ Drive:
from pathlib import Path
import shutil

drive_scripts = Path('/content/drive/MyDrive/GoGame_ML/scripts')
local_scripts = Path('/content/scripts')
local_scripts.mkdir(exist_ok=True)

for file in ['generate_labels_colab.py', 'label_generators.py', 'generate_features_colab.py']:
    src = drive_scripts / file
    dst = local_scripts / file
    if src.exists():
        shutil.copy(src, dst)
        print(f"✅ Copied {file}")

sys.path.insert(0, str(local_scripts))
"""

