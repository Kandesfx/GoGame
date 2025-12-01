"""
Script tối ưu để generate labels trên Google Colab.

Tính năng:
- Incremental save để tránh MemoryError (quan trọng với Colab RAM limit)
- Batch processing để tối ưu memory
- Error handling và logging
- Tích hợp với Google Drive
- Progress tracking với tqdm

Sử dụng:
    from generate_labels_colab import process_dataset_file
    
    process_dataset_file(
        input_path='/content/drive/MyDrive/GoGame_ML/processed/positions_19x19_2019.pt',
        output_path='/content/drive/MyDrive/GoGame_ML/datasets/labeled_19x19_2019.pt',
        filter_handicap=True,
        save_chunk_size=50000  # Save mỗi 50K samples (~1.2GB)
    )
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
from datetime import datetime
import traceback
import time
import gc
from typing import Optional, List, Dict, Tuple

# Import features
try:
    from generate_features_colab import (
        board_to_features_17_planes,
        generate_policy_label,
        generate_value_label
    )
except ImportError:
    # Nếu chưa có, thử import từ thư mục hiện tại
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from generate_features_colab import (
        board_to_features_17_planes,
        generate_policy_label,
        generate_value_label
    )

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('generate_labels_colab.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def process_single_position(pos: Dict, board_size: int, move_history: List = None) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Process một position thành labeled sample.
    
    Args:
        pos: Position dict với board_state, current_player, move, etc.
        board_size: Board size
        move_history: List of (x, y) tuples cho last 4 moves
    
    Returns:
        (labeled_sample, error_info) tuple
    """
    if move_history is None:
        move_history = []
    
    try:
        board_state = pos['board_state']
        current_player = pos['current_player']
        move = pos['move']
        winner = pos.get('winner')
        game_result = pos.get('game_result')
        move_number = pos.get('move_number', 0)
        
        # Convert numpy board to tensor
        if isinstance(board_state, np.ndarray):
            board_np = board_state
        else:
            board_np = np.array(board_state)
        
        # Validate board size
        if board_np.shape[0] != board_size or board_np.shape[1] != board_size:
            return None, {
                'error': f'Board size mismatch: {board_np.shape} vs {board_size}',
                'type': 'size_mismatch',
                'move_number': move_number
            }
        
        # Generate 17-plane features
        features = board_to_features_17_planes(
            board_np,
            current_player,
            move_history=move_history,
            board_size=board_size
        )
        
        # Generate policy label
        policy = generate_policy_label(move, board_size)
        
        # Generate value label
        value = generate_value_label(winner, current_player, game_result)
        
        # Create labeled sample
        labeled_sample = {
            'features': features,
            'policy': policy,
            'value': value,
            'metadata': {
                'move_number': move_number,
                'game_result': game_result,
                'winner': winner,
                'handicap': pos.get('handicap', 0)
            }
        }
        
        return labeled_sample, None
        
    except Exception as e:
        error_info = {
            'error': str(e),
            'type': 'exception',
            'traceback': traceback.format_exc(),
            'position': {
                'move_number': pos.get('move_number', -1),
                'current_player': pos.get('current_player', '?')
            }
        }
        return None, error_info


def process_positions_to_labels(
    positions: List[Dict],
    board_size: int,
    save_chunk_size: Optional[int] = None,
    output_dir: Optional[Path] = None,
    chunk_prefix: str = 'chunk'
) -> Tuple[List[Dict], List[Dict], List[Path]]:
    """
    Convert positions thành labeled data với incremental save.
    
    Args:
        positions: List of position dicts
        board_size: Board size
        save_chunk_size: Nếu set, save định kỳ mỗi N samples để giảm memory
        output_dir: Directory để save chunks (nếu dùng incremental save)
        chunk_prefix: Prefix cho chunk files
    
    Returns:
        (labeled_data, errors, saved_chunks) tuple
        - labeled_data: List of labeled samples (chỉ có nếu không dùng incremental save)
        - errors: List of error dicts
        - saved_chunks: List of chunk file paths (nếu dùng incremental save)
    """
    labeled_data = []
    errors = []
    saved_chunks = []
    
    # Track move history for each game
    move_history = []
    last_move_num = -1
    chunk_counter = 0
    
    use_incremental_save = save_chunk_size is not None and save_chunk_size > 0
    
    if use_incremental_save and output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Incremental save enabled: chunks will be saved to {output_dir}")
        logger.info(f"   Chunk size: {save_chunk_size:,} samples (~{save_chunk_size * 50 / 1024:.0f}MB per chunk)")
    
    start_time = time.time()
    last_speed_check_time = start_time
    last_speed_check_positions = 0
    
    for idx, pos in enumerate(tqdm(positions, desc="Generating labels", unit="pos")):
        move_num = pos.get('move_number', 0)
        
        # Reset history nếu là game mới
        if move_num < last_move_num or move_num == 0:
            move_history = []
        
        # Process position
        labeled_sample, error_info = process_single_position(pos, board_size, move_history)
        
        if error_info is not None:
            errors.append(error_info)
        elif labeled_sample is not None:
            labeled_data.append(labeled_sample)
        
        # Update move history
        move = pos.get('move')
        if move:
            move_history.append(move)
            if len(move_history) > 4:
                move_history = move_history[-4:]  # Keep last 4 only
        
        last_move_num = move_num
        
        # Incremental save nếu cần
        if use_incremental_save and output_dir and len(labeled_data) >= save_chunk_size:
            chunk_counter += 1
            chunk_file = output_dir / f'{chunk_prefix}_{chunk_counter:04d}.pt'
            
            logger.info(f"💾 Saving chunk {chunk_counter} ({len(labeled_data):,} samples) to {chunk_file.name}")
            
            # Save chunk
            torch.save({
                'labeled_data': labeled_data,
                'board_size': board_size,
                'chunk_num': chunk_counter,
                'total_samples': len(labeled_data)
            }, chunk_file)
            
            saved_chunks.append(chunk_file)
            
            # Clear memory và force GC
            labeled_data = []
            gc.collect()
            
            logger.info(f"✅ Chunk {chunk_counter} saved. Memory cleared.")
        
        # Periodic GC và speed check mỗi 10K samples
        elif len(labeled_data) % 10000 == 0 and len(labeled_data) > 0:
            gc.collect()
            
            # Speed check
            current_time = time.time()
            time_since_last_check = current_time - last_speed_check_time
            if time_since_last_check >= 30.0:  # Mỗi 30 giây
                positions_since_last_check = idx + 1 - last_speed_check_positions
                real_time_speed = positions_since_last_check / time_since_last_check if time_since_last_check > 0 else 0
                total_elapsed = current_time - start_time
                avg_speed = (idx + 1) / total_elapsed if total_elapsed > 0 else 0
                
                # Estimate memory usage
                estimated_memory_mb = len(labeled_data) * 50 / 1024
                
                logger.info(
                    f"Speed check - Real-time: {real_time_speed:.0f} pos/s | "
                    f"Average: {avg_speed:.0f} pos/s | "
                    f"Samples in memory: {len(labeled_data):,} (~{estimated_memory_mb:.0f}MB) | "
                    f"Progress: {idx+1:,}/{len(positions):,} ({(idx+1)/len(positions)*100:.1f}%)"
                )
                
                # Memory warning
                if estimated_memory_mb > 3000:  # > 3GB
                    logger.warning(
                        f"⚠️  High memory usage: ~{estimated_memory_mb:.0f}MB. "
                        f"Consider enabling incremental save (save_chunk_size=50000)."
                    )
                
                last_speed_check_time = current_time
                last_speed_check_positions = idx + 1
    
    # Final GC
    gc.collect()
    
    return labeled_data, errors, saved_chunks


def merge_chunks(chunk_files: List[Path], output_path: Path) -> int:
    """
    Merge các chunk files thành một file duy nhất.
    Tối ưu memory: Load từng chunk, clear ngay sau khi extend.
    
    Args:
        chunk_files: List of chunk file paths
        output_path: Path để save merged file
    
    Returns:
        Total number of samples merged
    """
    logger.info(f"📦 Merging {len(chunk_files)} chunks (memory-optimized)...")
    
    all_labeled_data = []
    board_size = None
    total_samples = 0
    
    # Estimate memory để cảnh báo
    estimated_total_mb = len(chunk_files) * 50 * 50000 / 1024  # Rough: 50KB per sample * 50K per chunk
    if estimated_total_mb > 10000:  # > 10GB
        logger.warning(
            f"⚠️  Large dataset (~{estimated_total_mb/1024:.1f}GB). "
            f"Merge may take time and use significant RAM."
        )
    
    for idx, chunk_file in enumerate(tqdm(chunk_files, desc="Loading chunks")):
        # Load chunk
        chunk_data = torch.load(chunk_file, map_location='cpu', weights_only=False)
        chunk_samples = chunk_data['labeled_data']
        
        # Extend vào list
        all_labeled_data.extend(chunk_samples)
        total_samples += len(chunk_samples)
        
        # Get board_size từ chunk đầu tiên
        if board_size is None:
            board_size = chunk_data['board_size']
        
        # Clear chunk data ngay để giải phóng memory
        del chunk_data
        del chunk_samples
        
        # GC sau mỗi 3 chunks để tránh memory buildup
        if (idx + 1) % 3 == 0:
            gc.collect()
            
            # Log memory usage
            estimated_mb = len(all_labeled_data) * 50 / 1024
            logger.info(
                f"   Processed {idx + 1}/{len(chunk_files)} chunks | "
                f"Samples: {total_samples:,} | "
                f"Memory: ~{estimated_mb:.0f}MB"
            )
    
    logger.info(f"✅ Loaded {total_samples:,} samples from {len(chunk_files)} chunks")
    
    # Final GC trước khi save
    gc.collect()
    
    # Save merged file
    logger.info(f"💾 Saving merged dataset to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save với metadata
    merged_data = {
        'labeled_data': all_labeled_data,
        'board_size': board_size,
        'total': total_samples,
        'metadata': {
            'merged_from_chunks': len(chunk_files),
            'chunk_files': [str(f) for f in chunk_files]
        }
    }
    
    torch.save(merged_data, output_path)
    
    # Clear memory ngay sau khi save
    del merged_data
    del all_labeled_data
    gc.collect()
    
    logger.info(f"✅ Saved merged dataset to {output_path}")
    
    return total_samples


def process_dataset_file(
    input_path: str,
    output_path: str,
    filter_handicap: bool = True,
    save_chunk_size: Optional[int] = None,
    auto_enable_incremental: bool = True,
    skip_merge: bool = False  # Nếu True, giữ chunks riêng, không merge
):
    """
    Process một file positions và generate labels.
    
    Args:
        input_path: Path to positions file (.pt)
        output_path: Path to save labeled dataset (.pt)
        filter_handicap: Nếu True, bỏ qua positions có handicap
        save_chunk_size: Nếu set, save định kỳ mỗi N samples (None = auto-detect)
        auto_enable_incremental: Tự động enable incremental save nếu estimated memory > 4GB
    """
    logger.info(f"📂 Loading positions from: {input_path}")
    
    try:
        # PyTorch 2.6+ requires weights_only=False for files with numpy arrays
        data = torch.load(input_path, map_location='cpu', weights_only=False)
    except Exception as e:
        logger.error(f"Failed to load {input_path}: {e}")
        return None
    
    positions = data['positions']
    board_size = data['board_size']
    year = data.get('year')
    
    logger.info(f"   Board size: {board_size}x{board_size}")
    logger.info(f"   Total positions: {len(positions):,}")
    if year:
        logger.info(f"   Year: {year}")
    
    # Filter handicap nếu cần
    if filter_handicap:
        original_count = len(positions)
        positions = [p for p in positions if p.get('handicap', 0) == 0]
        filtered_count = len(positions)
        if filtered_count < original_count:
            logger.info(
                f"   Filtered out {original_count - filtered_count:,} "
                f"handicap positions"
            )
    
    # Auto-enable incremental save nếu estimated memory > 4GB
    estimated_memory_mb = len(positions) * 50 / 1024
    if auto_enable_incremental and save_chunk_size is None and estimated_memory_mb > 4000:
        save_chunk_size = 50000  # Save mỗi 50K samples (~1.2GB)
        logger.info(
            f"💡 Auto-enabling incremental save (chunk size: {save_chunk_size:,}) "
            f"to prevent MemoryError (estimated: ~{estimated_memory_mb:.0f}MB)"
        )
    elif estimated_memory_mb > 2000:
        logger.warning(
            f"⚠️  WARNING: Estimated memory usage: ~{estimated_memory_mb:.0f}MB. "
            f"Consider enabling incremental save (save_chunk_size=50000) to avoid RAM issues."
        )
    
    # Setup output directory cho chunks
    output_path_obj = Path(output_path)
    output_dir = output_path_obj.parent
    chunks_dir = output_dir / f'{output_path_obj.stem}_chunks'
    
    # Generate labels
    labeled_data, errors, saved_chunks = process_positions_to_labels(
        positions,
        board_size,
        save_chunk_size=save_chunk_size,
        output_dir=chunks_dir if save_chunk_size else None,
        chunk_prefix=output_path_obj.stem
    )
    
    # Nếu dùng incremental save, merge chunks
    if saved_chunks:
        if skip_merge:
            logger.info(
                f"⏭️  Skipping merge (skip_merge=True). "
                f"Chunks saved in: {chunks_dir}\n"
                f"   Total chunks: {len(saved_chunks)}\n"
                f"   To merge later, use: merge_chunks({saved_chunks}, {output_path_obj})"
            )
        else:
            try:
                total_samples = merge_chunks(saved_chunks, output_path_obj)
                logger.info(f"✅ Merged {total_samples:,} samples from {len(saved_chunks)} chunks")
                
                # Optional: Cleanup chunks (uncomment nếu muốn)
                # import shutil
                # logger.info(f"🗑️  Cleaning up chunks directory...")
                # shutil.rmtree(chunks_dir)
                # logger.info(f"✅ Chunks cleaned up")
            except MemoryError as e:
                logger.error(
                    f"❌ MemoryError during merge: {e}\n"
                    f"💡 Solution: Re-run with skip_merge=True to keep chunks separate, "
                    f"or reduce save_chunk_size to create smaller chunks."
                )
                raise
    else:
        # Save labeled dataset trực tiếp
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'labeled_data': labeled_data,
            'board_size': board_size,
            'total': len(labeled_data),
            'year': year,
            'metadata': {
                'filtered_handicap': filter_handicap,
                'input_file': str(input_path)
            }
        }, output_path_obj)
        
        logger.info(f"✅ Saved {len(labeled_data):,} labeled samples to {output_path_obj}")
    
    # Log errors
    if errors:
        error_log_file = output_dir / f'label_errors_{year or "all"}.log'
        with open(error_log_file, 'w', encoding='utf-8') as f:
            f.write(f"Label Generation Errors Summary\n")
            f.write(f"Total errors: {len(errors)}\n")
            f.write(f"Date: {datetime.now()}\n\n")
            
            # Group errors by type
            errors_by_type = {}
            for err in errors:
                err_type = err.get('type', 'unknown')
                if err_type not in errors_by_type:
                    errors_by_type[err_type] = []
                errors_by_type[err_type].append(err)
            
            for err_type, err_list in errors_by_type.items():
                f.write(f"\n=== {err_type.upper()} ({len(err_list)} errors) ===\n")
                for err in err_list[:10]:  # Show first 10 of each type
                    f.write(f"  {err.get('error', 'Unknown error')}\n")
                    if 'position' in err:
                        f.write(f"    Position: {err['position']}\n")
                if len(err_list) > 10:
                    f.write(f"  ... and {len(err_list) - 10} more\n")
        
        logger.warning(f"⚠️  {len(errors):,} errors occurred. See {error_log_file}")
    
    return labeled_data if not saved_chunks else None


if __name__ == "__main__":
    # Example usage trên Colab:
    from pathlib import Path
    
    # Mount Google Drive (chạy trong notebook)
    # from google.colab import drive
    # drive.mount('/content/drive')
    
    WORK_DIR = Path('/content/drive/MyDrive/GoGame_ML')
    
    # Process một năm
    process_dataset_file(
        input_path=WORK_DIR / 'processed' / 'positions_19x19_2019.pt',
        output_path=WORK_DIR / 'datasets' / 'labeled_19x19_2019.pt',
        filter_handicap=True,
        save_chunk_size=50000  # Save mỗi 50K samples
    )
