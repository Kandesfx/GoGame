"""
Script kiểm tra đơn giản format position files.
Cần torch và numpy để chạy.

Chạy: python scripts/check_positions_simple.py [file_path]
"""

import sys
from pathlib import Path

try:
    import torch
    import numpy as np
except ImportError as e:
    print(f"ERROR: Cần cài torch và numpy:")
    print(f"  pip install torch numpy")
    sys.exit(1)


def check_file(file_path: Path):
    """Kiểm tra một file positions"""
    print(f"\n{'='*70}")
    print(f"Checking: {file_path.name}")
    print(f"{'='*70}\n")
    
    if not file_path.exists():
        print(f"ERROR: File not found")
        return False
    
    try:
        # Load file
        print("Loading file...")
        data = torch.load(file_path, map_location='cpu', weights_only=False)
        print("SUCCESS: File loaded\n")
    except Exception as e:
        print(f"ERROR: Cannot load file: {e}")
        return False
    
    # Check structure
    if 'positions' not in data:
        print("ERROR: File không có 'positions' key")
        return False
    
    positions = data['positions']
    total = len(positions)
    board_size = data.get('board_size', 19)
    
    print(f"Total positions: {total:,}")
    print(f"Board size: {board_size}x{board_size}\n")
    
    if total == 0:
        print("ERROR: File rỗng!")
        return False
    
    # Check sample
    sample = positions[0]
    print("Sample position fields:")
    for key in sorted(sample.keys()):
        value = sample[key]
        if isinstance(value, np.ndarray):
            print(f"  {key}: numpy.ndarray {value.shape}")
        elif value is None:
            print(f"  {key}: None")
        elif isinstance(value, (tuple, list)):
            print(f"  {key}: {type(value).__name__} (len={len(value)})")
        else:
            print(f"  {key}: {type(value).__name__} = {value}")
    
    # Check required fields
    print(f"\nRequired fields check:")
    required = {
        'board_state': 'numpy.ndarray',
        'move': 'tuple or None',
        'current_player': 'str (B/W)',
        'move_number': 'int',
        'board_size': 'int'
    }
    
    missing_required = []
    for field, expected in required.items():
        if field in sample:
            value = sample[field]
            if field == 'move':
                if value is None:
                    print(f"  {field}: None (PASS MOVE - OK)")
                elif isinstance(value, (tuple, list)) and len(value) == 2:
                    print(f"  {field}: {value} (normal move - OK)")
                else:
                    print(f"  {field}: {value} (unexpected format)")
            else:
                print(f"  {field}: Present - OK")
        else:
            print(f"  {field}: MISSING - ERROR")
            missing_required.append(field)
    
    # Check optional but important fields
    print(f"\nOptional fields check:")
    optional = ['winner', 'game_result', 'handicap']
    missing_optional = []
    for field in optional:
        if field in sample:
            print(f"  {field}: Present - OK")
        else:
            print(f"  {field}: Missing (optional)")
            missing_optional.append(field)
    
    # Check pass moves
    print(f"\nPass moves check:")
    pass_moves = [p for p in positions[:min(10000, total)] if p.get('move') is None]
    pass_count = len(pass_moves)
    pass_percentage = (pass_count / min(10000, total) * 100) if total > 0 else 0
    
    print(f"  Pass moves found: {pass_count:,} / {min(10000, total):,} ({pass_percentage:.2f}%)")
    
    if pass_count > 0:
        print(f"  -> PASS MOVES SUPPORTED (move = None)")
        has_pass_support = True
    else:
        print(f"  -> No pass moves found")
        print(f"     (Note: Games might not have pass moves, or they were skipped)")
        has_pass_support = None
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    needs_reparse = False
    reasons = []
    
    if missing_required:
        needs_reparse = True
        reasons.append(f"Missing required fields: {', '.join(missing_required)}")
    
    if not missing_required and not missing_optional:
        print("RESULT: File format is COMPLETE")
        print("  -> All required and optional fields present")
        if has_pass_support:
            print("  -> Pass moves supported")
        print("  -> NO NEED TO REPARSE")
        print("  -> Ready for labeling!")
    elif not missing_required:
        print("RESULT: File format is VALID (missing some optional fields)")
        print(f"  -> Missing optional: {', '.join(missing_optional)}")
        if has_pass_support:
            print("  -> Pass moves supported")
        print("  -> NO NEED TO REPARSE (but optional fields recommended)")
        print("  -> Ready for labeling!")
    else:
        print("RESULT: File format is INCOMPLETE")
        for reason in reasons:
            print(f"  -> {reason}")
        print("  -> NEED TO REPARSE")
    
    print(f"{'='*70}\n")
    
    return not needs_reparse


def main():
    if len(sys.argv) > 1:
        file_path = Path(sys.argv[1])
        if not file_path.exists():
            print(f"ERROR: File not found: {file_path}")
            sys.exit(1)
        check_file(file_path)
    else:
        # Check all files in data/processed
        data_dir = Path('data/processed')
        if not data_dir.exists():
            print(f"ERROR: Directory not found: {data_dir}")
            print("Usage: python scripts/check_positions_simple.py [file_path]")
            sys.exit(1)
        
        pt_files = sorted(data_dir.glob('positions_*.pt'))
        
        if not pt_files:
            print(f"No position files found in {data_dir}")
            return
        
        print(f"Found {len(pt_files)} position files\n")
        
        all_ok = True
        for pt_file in pt_files:
            if not check_file(pt_file):
                all_ok = False
        
        if all_ok:
            print("\n✅ All files are valid - NO NEED TO REPARSE")
        else:
            print("\n❌ Some files need to be reparsed")


if __name__ == "__main__":
    main()

