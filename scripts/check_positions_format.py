"""
Script kiểm tra format của các file positions để xác định có cần parse lại không.

Chạy: python scripts/check_positions_format.py [path_to_positions_file.pt]
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

# Fix encoding for Windows
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Try importing torch
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[WARNING] torch not installed. Will use pickle to load files (slower).")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("[WARNING] numpy not installed. Some checks may be limited.")

def check_position_file(file_path: Path) -> Dict:
    """
    Kiểm tra format của một file positions.
    
    Returns:
        Dict với thông tin về format và compatibility
    """
    print(f"\n{'='*70}")
    print(f"📂 Checking: {file_path.name}")
    print(f"{'='*70}")
    
    try:
        # Load file
        if HAS_TORCH:
            data = torch.load(file_path, map_location='cpu', weights_only=False)
        else:
            # Fallback: use pickle
            import pickle
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
    except Exception as e:
        return {
            'valid': False,
            'error': str(e),
            'needs_reparse': True
        }
    
    # Check structure
    if 'positions' not in data:
        print("❌ File không có 'positions' key")
        return {
            'valid': False,
            'error': "Missing 'positions' key",
            'needs_reparse': True
        }
    
    positions = data['positions']
    board_size = data.get('board_size', 19)
    total = len(positions)
    
    print(f"📊 Total positions: {total:,}")
    print(f"📐 Board size: {board_size}x{board_size}")
    
    if total == 0:
        print("⚠️  File rỗng!")
        return {
            'valid': False,
            'error': "Empty file",
            'needs_reparse': True
        }
    
    # Check sample position
    sample = positions[0]
    print(f"\n📋 Sample position fields:")
    for key in sample.keys():
        value = sample[key]
        if HAS_NUMPY and isinstance(value, np.ndarray):
            print(f"   ✅ {key}: {type(value).__name__} {value.shape}")
        elif isinstance(value, (tuple, list)):
            print(f"   ✅ {key}: {type(value).__name__} (len={len(value)})")
        else:
            print(f"   ✅ {key}: {type(value).__name__} = {value}")
    
    # Required fields
    required_fields = [
        'board_state',
        'move',
        'current_player',
        'move_number',
        'board_size'
    ]
    
    optional_fields = [
        'winner',
        'game_result',
        'handicap'
    ]
    
    print(f"\n🔍 Checking required fields:")
    missing_required = []
    for field in required_fields:
        if field in sample:
            print(f"   ✅ {field}: Present")
        else:
            print(f"   ❌ {field}: MISSING")
            missing_required.append(field)
    
    print(f"\n🔍 Checking optional fields:")
    missing_optional = []
    for field in optional_fields:
        if field in sample:
            print(f"   ✅ {field}: Present")
        else:
            print(f"   ⚠️  {field}: Missing (optional but recommended)")
            missing_optional.append(field)
    
    # Check pass moves support
    print(f"\n🔍 Checking pass moves support:")
    pass_moves = [p for p in positions if p.get('move') is None]
    pass_count = len(pass_moves)
    pass_percentage = (pass_count / total * 100) if total > 0 else 0
    
    print(f"   Pass moves: {pass_count:,} / {total:,} ({pass_percentage:.2f}%)")
    
    if pass_count > 0:
        print(f"   ✅ Pass moves được hỗ trợ (move = None)")
        has_pass_support = True
    else:
        print(f"   ⚠️  Không có pass moves trong file")
        print(f"      (Có thể do: không có pass moves trong games, hoặc bị bỏ qua khi parse)")
        has_pass_support = None  # Unknown
    
    # Check move format
    print(f"\n🔍 Checking move format:")
    move_types = {}
    for pos in positions[:1000]:  # Check first 1000
        move = pos.get('move')
        if move is None:
            move_type = 'None (pass)'
        elif isinstance(move, tuple):
            if len(move) == 2:
                move_type = 'tuple (x, y)'
            else:
                move_type = f'tuple (invalid len={len(move)})'
        elif isinstance(move, list):
            move_type = 'list'
        else:
            move_type = f'other ({type(move).__name__})'
        
        move_types[move_type] = move_types.get(move_type, 0) + 1
    
    for move_type, count in move_types.items():
        print(f"   {move_type}: {count:,}")
    
    # Check board_state format
    print(f"\n🔍 Checking board_state format:")
    board_state = sample.get('board_state')
    if board_state is not None:
        if HAS_NUMPY and isinstance(board_state, np.ndarray):
            print(f"   ✅ Type: numpy.ndarray")
            print(f"   ✅ Shape: {board_state.shape}")
            print(f"   ✅ Dtype: {board_state.dtype}")
            if board_state.shape == (board_size, board_size):
                print(f"   ✅ Shape matches board_size")
            else:
                print(f"   ❌ Shape mismatch: expected ({board_size}, {board_size})")
        else:
            print(f"   ⚠️  Type: {type(board_state).__name__} (expected numpy.ndarray)")
    else:
        print(f"   ❌ board_state is None")
    
    # Determine if reparse needed
    needs_reparse = False
    reasons = []
    
    if missing_required:
        needs_reparse = True
        reasons.append(f"Missing required fields: {', '.join(missing_required)}")
    
    if 'winner' not in sample and 'game_result' not in sample:
        needs_reparse = False  # Optional, but will affect value labels
        reasons.append("Missing winner/game_result (will use default values)")
    
    # Check if old format (no pass moves support)
    # This is hard to determine without knowing when file was created
    # But we can check if there are any obvious issues
    
    print(f"\n{'='*70}")
    print(f"📊 SUMMARY")
    print(f"{'='*70}")
    
    if needs_reparse:
        print(f"❌ CẦN PARSE LẠI")
        for reason in reasons:
            print(f"   - {reason}")
    else:
        print(f"✅ KHÔNG CẦN PARSE LẠI")
        print(f"   - Tất cả required fields đều có")
        if pass_count > 0:
            print(f"   - Pass moves được hỗ trợ")
        if not missing_optional:
            print(f"   - Tất cả optional fields đều có")
        else:
            print(f"   - Optional fields missing: {', '.join(missing_optional)} (không bắt buộc)")
    
    return {
        'valid': True,
        'total': total,
        'board_size': board_size,
        'has_pass_support': has_pass_support,
        'pass_count': pass_count,
        'missing_required': missing_required,
        'missing_optional': missing_optional,
        'needs_reparse': needs_reparse,
        'reasons': reasons
    }


def check_directory(directory: Path) -> None:
    """Kiểm tra tất cả .pt files trong directory"""
    pt_files = list(directory.glob('positions_*.pt'))
    
    if not pt_files:
        print(f"⚠️  Không tìm thấy file positions_*.pt trong {directory}")
        return
    
    print(f"📁 Found {len(pt_files)} position files")
    
    results = []
    for pt_file in sorted(pt_files):
        result = check_position_file(pt_file)
        results.append((pt_file, result))
    
    # Summary
    print(f"\n{'='*70}")
    print(f"📊 OVERALL SUMMARY")
    print(f"{'='*70}")
    
    needs_reparse_count = sum(1 for _, r in results if r.get('needs_reparse', False))
    valid_count = sum(1 for _, r in results if r.get('valid', False))
    
    print(f"Total files: {len(pt_files)}")
    print(f"Valid files: {valid_count}")
    print(f"Files cần parse lại: {needs_reparse_count}")
    
    if needs_reparse_count > 0:
        print(f"\n❌ Files cần parse lại:")
        for pt_file, result in results:
            if result.get('needs_reparse', False):
                print(f"   - {pt_file.name}")
                for reason in result.get('reasons', []):
                    print(f"     → {reason}")
    else:
        print(f"\n✅ Tất cả files đều OK - KHÔNG CẦN PARSE LẠI")
        print(f"   Bạn có thể chạy labeling script ngay!")


def main():
    """Main function"""
    if len(sys.argv) > 1:
        # Check specific file
        file_path = Path(sys.argv[1])
        if not file_path.exists():
            print(f"❌ File không tồn tại: {file_path}")
            sys.exit(1)
        
        result = check_position_file(file_path)
        sys.exit(0 if result.get('valid', False) and not result.get('needs_reparse', False) else 1)
    else:
        # Check current directory
        current_dir = Path('.')
        print(f"🔍 Checking position files in: {current_dir.absolute()}")
        check_directory(current_dir)


if __name__ == "__main__":
    main()

