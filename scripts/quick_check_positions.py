"""
Script kiểm tra nhanh format của position files (chỉ check metadata, không load toàn bộ).

Chạy: python scripts/quick_check_positions.py [file_or_directory]
"""

import sys
from pathlib import Path
import struct

def check_file_header(file_path: Path):
    """Kiểm tra header của file để xác định format"""
    print(f"\n{'='*70}")
    print(f"Checking: {file_path.name}")
    print(f"{'='*70}")
    
    if not file_path.exists():
        print(f"ERROR: File not found")
        return None
    
    file_size = file_path.stat().st_size
    print(f"File size: {file_size / (1024**2):.2f} MB")
    
    # Try to detect if it's a PyTorch file
    with open(file_path, 'rb') as f:
        # PyTorch files start with specific magic numbers
        header = f.read(8)
        
        # PyTorch format detection
        if header[:2] == b'PK':  # ZIP format (PyTorch 1.6+)
            print("Format: PyTorch (ZIP-based)")
            is_pytorch = True
        elif header[:4] == b'PK\x03\x04':  # ZIP
            print("Format: PyTorch (ZIP-based)")
            is_pytorch = True
        else:
            print("Format: Unknown (might be pickle)")
            is_pytorch = False
    
    return {
        'file_path': file_path,
        'file_size_mb': file_size / (1024**2),
        'is_pytorch': is_pytorch
    }


def try_load_sample(file_path: Path, max_positions=5):
    """Thử load một vài positions để kiểm tra format"""
    print(f"\nTrying to load sample positions...")
    
    # Try with pickle first (no dependencies)
    try:
        import pickle
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print("SUCCESS: Loaded with pickle")
        
        if 'positions' in data:
            positions = data['positions']
            total = len(positions)
            print(f"Total positions: {total:,}")
            
            if total > 0:
                sample = positions[0]
                print(f"\nSample position fields:")
                for key in sorted(sample.keys()):
                    value = sample[key]
                    if value is None:
                        print(f"  {key}: None")
                    elif isinstance(value, (int, float, str)):
                        print(f"  {key}: {type(value).__name__} = {value}")
                    elif isinstance(value, (tuple, list)):
                        print(f"  {key}: {type(value).__name__} (len={len(value)})")
                    else:
                        print(f"  {key}: {type(value).__name__}")
                
                # Check critical fields
                print(f"\nCritical fields check:")
                required = ['board_state', 'move', 'current_player']
                for field in required:
                    if field in sample:
                        value = sample[field]
                        if field == 'move':
                            if value is None:
                                print(f"  {field}: None (PASS MOVE - OK)")
                            else:
                                print(f"  {field}: {value} (normal move - OK)")
                        else:
                            print(f"  {field}: Present - OK")
                    else:
                        print(f"  {field}: MISSING - ERROR")
                
                # Check pass moves
                pass_count = sum(1 for p in positions[:1000] if p.get('move') is None)
                print(f"\nPass moves in first 1000: {pass_count}")
                
                if pass_count > 0:
                    print("  -> PASS MOVES SUPPORTED (move = None)")
                else:
                    print("  -> No pass moves found (might need reparse if games had passes)")
                
                return {
                    'valid': True,
                    'total': total,
                    'has_pass_support': pass_count > 0,
                    'sample_fields': list(sample.keys())
                }
        
        return {'valid': False, 'error': "No 'positions' key"}
        
    except Exception as e:
        print(f"ERROR loading with pickle: {e}")
        return {'valid': False, 'error': str(e)}


def main():
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
    else:
        # Check data/processed directory
        target = Path('data/processed')
    
    if target.is_file():
        # Check single file
        check_file_header(target)
        result = try_load_sample(target)
        
        if result and result.get('valid'):
            print(f"\n{'='*70}")
            print("RESULT: File format is VALID")
            if result.get('has_pass_support'):
                print("  -> Pass moves supported")
                print("  -> NO NEED TO REPARSE")
            else:
                print("  -> No pass moves found")
                print("  -> May need to reparse if you want pass moves")
            print(f"{'='*70}")
        else:
            print(f"\n{'='*70}")
            print("RESULT: Could not validate file")
            print(f"{'='*70}")
    
    elif target.is_dir():
        # Check all .pt files in directory
        pt_files = sorted(target.glob('positions_*.pt'))
        
        if not pt_files:
            print(f"No position files found in {target}")
            return
        
        print(f"Found {len(pt_files)} position files")
        
        for pt_file in pt_files:
            check_file_header(pt_file)
            result = try_load_sample(pt_file)
            print()
    
    else:
        print(f"ERROR: {target} is not a file or directory")
        sys.exit(1)


if __name__ == "__main__":
    main()

