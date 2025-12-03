"""
Quick test để kiểm tra ML model có load được không.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "backend" / "app" / "services"))
sys.path.insert(0, str(project_root / "src" / "ml"))

print("Testing ML model service...")

try:
    from ml_model_service import get_ml_model_service
    
    service = get_ml_model_service()
    if service:
        if service.is_loaded():
            print(f"✅ ML Model is loaded!")
            print(f"   Board size: {service.board_size}")
            print(f"   Device: {service.device}")
        else:
            print("❌ ML Model service exists but model not loaded")
    else:
        print("❌ ML Model service not available")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()









