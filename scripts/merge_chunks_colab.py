"""
Script helper để merge chunks sau khi đã có đủ RAM.

Sử dụng khi:
- Gán nhãn với skip_merge=True (để tránh MemoryError)
- Muốn merge chunks sau khi có đủ RAM
- Hoặc merge chunks từ session trước
"""

from pathlib import Path
from generate_labels_colab import merge_chunks
import torch

def merge_chunks_from_directory(chunks_dir: str, output_path: str):
    """
    Merge tất cả chunks từ một directory.
    
    Args:
        chunks_dir: Directory chứa chunk files (ví dụ: 'labeled_19x19_2019_chunks')
        output_path: Path để save merged file
    """
    chunks_dir = Path(chunks_dir)
    output_path = Path(output_path)
    
    # Tìm tất cả chunk files
    chunk_files = sorted(chunks_dir.glob('chunk_*.pt'))
    
    if not chunk_files:
        print(f"❌ No chunk files found in {chunks_dir}")
        return
    
    print(f"📦 Found {len(chunk_files)} chunks in {chunks_dir}")
    
    # Estimate total size
    total_samples = 0
    for chunk_file in chunk_files:
        chunk_data = torch.load(chunk_file, map_location='cpu', weights_only=False)
        total_samples += chunk_data.get('total_samples', len(chunk_data['labeled_data']))
    
    estimated_mb = total_samples * 50 / 1024
    print(f"   Estimated total: {total_samples:,} samples (~{estimated_mb:.0f}MB)")
    
    if estimated_mb > 10000:  # > 10GB
        print(f"⚠️  Warning: Large dataset (~{estimated_mb/1024:.1f}GB). Make sure you have enough RAM.")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return
    
    # Merge
    try:
        total_samples = merge_chunks(chunk_files, output_path)
        print(f"✅ Successfully merged {total_samples:,} samples to {output_path}")
    except MemoryError as e:
        print(f"❌ MemoryError: {e}")
        print(f"💡 Try:")
        print(f"   1. Restart runtime to free RAM")
        print(f"   2. Reduce number of chunks (merge in batches)")
        print(f"   3. Use a machine with more RAM")


if __name__ == "__main__":
    # Example usage trên Colab:
    from pathlib import Path
    
    WORK_DIR = Path('/content/drive/MyDrive/GoGame_ML')
    
    # Merge chunks từ directory
    chunks_dir = WORK_DIR / 'datasets' / 'labeled_19x19_2019_chunks'
    output_path = WORK_DIR / 'datasets' / 'labeled_19x19_2019.pt'
    
    merge_chunks_from_directory(chunks_dir, output_path)

