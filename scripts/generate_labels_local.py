"""
Script tối ưu để generate labels trên local máy tính.

Tính năng:
- Multiprocessing để tăng tốc
- Error handling và logging
- Xử lý theo batch để tiết kiệm memory
- Progress tracking
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
from datetime import datetime
import traceback
from multiprocessing import Pool, cpu_count
import sys
import signal
import time
import gc

# Import features
try:
    from generate_features_colab import (
        board_to_features_17_planes,
        generate_policy_label,
        generate_value_label
    )
except ImportError:
    # Nếu chưa có, thử import từ thư mục hiện tại
    sys.path.insert(0, str(Path(__file__).parent))
    from generate_features_colab import (
        board_to_features_17_planes,
        generate_policy_label,
        generate_value_label
    )

# Setup logging with UTF-8 encoding
import sys

# Create file handler with UTF-8
file_handler = logging.FileHandler('generate_labels_local.log', encoding='utf-8')
file_handler.setLevel(logging.INFO)

# Create console handler with UTF-8 (if possible)
try:
    console_handler = logging.StreamHandler(sys.stdout)
    # Try to set UTF-8 encoding for console
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
except:
    console_handler = logging.StreamHandler()

console_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)


def process_single_position(pos, board_size):
    """
    Process một position thành labeled sample.
    
    Returns:
        (labeled_sample, error_info) tuple
    """
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
                'type': 'size_mismatch'
            }
        
        # Get move history (simplified - từ move_number)
        move_history = []  # Will be handled in batch processing
        
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


def process_positions_batch(positions_batch, board_size):
    """
    Process một batch positions với move history tracking.
    
    Args:
        positions_batch: List of positions từ cùng một game
        board_size: Board size
    
    Returns:
        (labeled_samples, errors) tuple
    """
    labeled_samples = []
    errors = []
    move_history = []
    
    for pos in positions_batch:
        # Update move history
        if pos.get('move_number', 0) == 0:
            move_history = []
        
        # Process position
        labeled_sample, error_info = process_single_position(pos, board_size)
        
        if error_info is not None:
            errors.append(error_info)
            continue
        
        if labeled_sample is None:
            continue
        
        # Update features với move history (nếu cần)
        # Note: Move history đã được tính trong batch processing
        labeled_samples.append(labeled_sample)
        
        # Update move history
        move = pos.get('move')
        if move:
            move_history.append(move)
            if len(move_history) > 4:
                move_history = move_history[-4:]
    
    return labeled_samples, errors


def _process_batch_wrapper(args):
    """Wrapper function for multiprocessing (không thể dùng lambda)."""
    batch, board_size = args
    return process_positions_batch(batch, board_size)


def process_positions_to_labels_parallel(
    positions,
    board_size,
    num_workers=None,
    batch_size=5000,  # Batch size mặc định (tối ưu cho performance)
    save_chunk_size=None,  # Nếu set, sẽ save định kỳ thay vì giữ tất cả trong memory
    output_dir=None  # Directory để save chunks nếu dùng incremental save
):
    """
    Process positions với multiprocessing (tối ưu).
    
    Args:
        positions: List of position dicts
        board_size: Board size
        num_workers: Number of worker processes
        batch_size: Batch size for processing (giảm để giảm memory)
        save_chunk_size: Nếu set, save định kỳ mỗi N samples để giảm memory
    
    Returns:
        (labeled_data, errors) tuple
    """
    if num_workers is None:
        num_workers = min(cpu_count(), 8)
    
    # Memory warning nếu quá nhiều workers
    if num_workers > 12:
        logger.warning(
            f"⚠️  WARNING: {num_workers} workers có thể gây RAM overflow. "
            f"Khuyến nghị: giảm xuống 8 hoặc ít hơn."
        )
    
    logger.info(
        f"Processing {len(positions):,} positions with {num_workers} workers "
        f"(batch size: {batch_size:,})"
    )
    
    # Group positions by game (simplified - dựa vào move_number)
    batches = []
    current_batch = []
    last_move_num = -1
    
    for pos in positions:
        move_num = pos.get('move_number', 0)
        
        # Start new batch if move_number resets
        if move_num < last_move_num:
            if current_batch:
                batches.append(current_batch)
            current_batch = [pos]
        else:
            current_batch.append(pos)
        
        # Flush batch if too large
        if len(current_batch) >= batch_size:
            batches.append(current_batch)
            current_batch = []
        
        last_move_num = move_num
    
    if current_batch:
        batches.append(current_batch)
    
    logger.info(f"Created {len(batches):,} batches")
    
    # Initialize lists
    total_positions = len(positions)
    all_labeled_data = []
    all_errors = []
    
    processed_positions = 0
    total_errors = 0
    last_log_progress = 0
    
    # Tracking thời gian để tính tốc độ thực tế
    start_time = time.time()
    last_speed_check_time = start_time
    last_speed_check_positions = 0
    
    # Memory management: nếu save_chunk_size được set, sẽ save định kỳ
    # và clear memory để tránh RAM overflow
    use_incremental_save = save_chunk_size is not None and save_chunk_size > 0
    saved_chunks = []  # List để lưu path của các chunk files nếu dùng incremental save
    chunk_counter = 0
    
    # Tạo chunks directory nếu cần
    if use_incremental_save and output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Chunks will be saved to: {output_dir}")
    
    # Process in parallel với imap_unordered (nhanh hơn vì không cần giữ thứ tự)
    # Dùng pool thủ công thay vì context manager để có thể terminate khi interrupt
    pool = None
    try:
        pool = Pool(processes=num_workers)
        # Progress bar với thông tin chi tiết
        with tqdm(
            total=total_positions,
            desc="Generating labels",
            unit="pos",
            unit_scale=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] | Errors: {postfix}",
            miniters=1000,  # Update ít hơn để giảm overhead
            smoothing=0.05  # Smoothing nhẹ hơn để phản ánh tốc độ thực tế
        ) as pbar:
            # Dùng imap_unordered để tăng tốc độ (không cần giữ thứ tự batches)
            for labeled_samples, errors in pool.imap_unordered(
                _process_batch_wrapper,
                [(batch, board_size) for batch in batches],
                chunksize=max(1, len(batches) // (num_workers * 4))  # Chunk size tối ưu
            ):
                # Update progress
                batch_size_processed = len(labeled_samples) + len(errors)
                processed_positions += batch_size_processed
                total_errors += len(errors)
                
                # Tính tốc độ thực tế (không phụ thuộc vào tqdm smoothing)
                current_time = time.time()
                time_since_last_check = current_time - last_speed_check_time
                
                # Log tốc độ thực tế mỗi 15 giây để phát hiện slowdown
                if time_since_last_check >= 15.0:
                    positions_since_last_check = processed_positions - last_speed_check_positions
                    real_time_speed = positions_since_last_check / time_since_last_check if time_since_last_check > 0 else 0
                    total_elapsed = current_time - start_time
                    avg_speed = processed_positions / total_elapsed if total_elapsed > 0 else 0
                    
                    # Estimate memory usage
                    # Mỗi labeled sample: features (17 planes x 19x19 x float32) + policy + value
                    # ~17 * 19 * 19 * 4 bytes + 361 * 4 bytes + 4 bytes ≈ 25KB per sample
                    estimated_memory_mb = len(all_labeled_data) * 25 / 1024
                    
                    # Log vào file để phân tích
                    logger.info(
                        f"Speed check - Real-time: {real_time_speed:.0f} pos/s | "
                        f"Average: {avg_speed:.0f} pos/s | "
                        f"Samples in memory: {len(all_labeled_data):,} (~{estimated_memory_mb:.0f}MB) | "
                        f"Progress: {processed_positions:,}/{total_positions:,} ({processed_positions/total_positions*100:.1f}%)"
                    )
                    
                    # Memory warning nếu quá cao
                    if estimated_memory_mb > 3000:  # > 3GB
                        logger.warning(
                            f"⚠️  High memory usage: ~{estimated_memory_mb:.0f}MB. "
                            f"Consider reducing batch_size or num_workers."
                        )
                    
                    last_speed_check_time = current_time
                    last_speed_check_positions = processed_positions
                
                # Collect results (dùng extend thay vì append từng item)
                if labeled_samples:
                    all_labeled_data.extend(labeled_samples)
                if errors:
                    all_errors.extend(errors)
                
                # Memory management: Save định kỳ nếu cần để tránh MemoryError
                if use_incremental_save and output_dir is not None and len(all_labeled_data) >= save_chunk_size:
                    chunk_counter += 1
                    chunk_file = Path(output_dir) / f'chunk_{chunk_counter:04d}.pt'
                    
                    logger.info(f"💾 Saving chunk {chunk_counter} ({len(all_labeled_data):,} samples) to {chunk_file.name}")
                    
                    # Save chunk
                    torch.save({
                        'labeled_data': all_labeled_data,
                        'board_size': board_size,
                        'chunk_num': chunk_counter
                    }, chunk_file)
                    
                    saved_chunks.append(chunk_file)
                    
                    # Clear memory và force GC
                    all_labeled_data = []
                    gc.collect()
                    
                    logger.info(f"✅ Chunk {chunk_counter} saved. Memory cleared.")
                
                # Periodic GC mỗi 20K samples (ngay cả khi không dùng incremental save)
                elif len(all_labeled_data) % 20000 == 0 and len(all_labeled_data) > 0:
                    gc.collect()
                
                # Update progress bar (ít thường xuyên hơn)
                pbar.update(batch_size_processed)
                pbar.set_postfix_str(f"{total_errors:,}")
                
                # Log progress every 10% (không làm hỏng progress bar)
                progress_pct = (processed_positions / total_positions) * 100
                if progress_pct - last_log_progress >= 10.0:
                    success_rate = ((processed_positions - total_errors) / processed_positions * 100) if processed_positions > 0 else 0
                    elapsed = current_time - start_time
                    avg_speed = processed_positions / elapsed if elapsed > 0 else 0
                    tqdm.write(
                        f"Progress: {progress_pct:.1f}% | "
                        f"Processed: {processed_positions:,}/{total_positions:,} | "
                        f"Success: {success_rate:.1f}% | "
                        f"Errors: {total_errors:,} | "
                        f"Avg Speed: {avg_speed:.0f} pos/s"
                    )
                    last_log_progress = progress_pct
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Interrupted by user (Ctrl+C). Cleaning up worker processes...")
        if pool is not None:
            logger.info("Terminating worker processes...")
            pool.terminate()  # Force terminate all workers immediately
            # Python 3.12 không hỗ trợ timeout trong pool.join()
            # Dùng cách khác để timeout
            import threading
            def join_with_timeout():
                pool.join()
            join_thread = threading.Thread(target=join_with_timeout)
            join_thread.daemon = True
            join_thread.start()
            join_thread.join(timeout=5)  # Wait up to 5 seconds
            if join_thread.is_alive():
                logger.warning("Some worker processes may still be running")
            else:
                logger.info("Worker processes terminated.")
        raise  # Re-raise để main có thể handle
    finally:
        # Đảm bảo cleanup pool
        if pool is not None:
            pool.close()  # Prevent new tasks
            # Python 3.12 không hỗ trợ timeout trong pool.join()
            pool.join()  # Wait for cleanup (không có timeout)
    
    # Nếu dùng incremental save, return saved_chunks thay vì all_labeled_data
    if use_incremental_save:
        return saved_chunks, all_errors
    else:
        return all_labeled_data, all_errors


def process_dataset_file(
    input_path,
    output_path,
    filter_handicap=True,
    num_workers=None,
    batch_size=5000  # Batch size mặc định (tối ưu cho performance)
):
    """
    Process một file positions và generate labels.
    
    Args:
        input_path: Path to positions file (.pt)
        output_path: Path to save labeled dataset (.pt)
        filter_handicap: Nếu True, bỏ qua positions có handicap
        num_workers: Number of worker processes
        batch_size: Batch size for processing
    """
    logger.info(f"Loading positions from: {input_path}")
    
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
    
    # Memory warning nếu quá nhiều positions
    # Estimate: mỗi position ~1-2KB raw, sau khi label ~50KB
    estimated_memory_mb = len(positions) * 50 / 1024  # Rough estimate
    if estimated_memory_mb > 2000:  # > 2GB
        logger.warning(
            f"⚠️  WARNING: Estimated memory usage: ~{estimated_memory_mb:.0f}MB. "
            f"Consider reducing num_workers or batch_size to avoid RAM issues."
        )
        if num_workers is None or num_workers > 8:
            suggested_workers = min(8, cpu_count())
            logger.info(f"💡 Suggested: Use --workers {suggested_workers} or less")
    
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
    auto_save_chunk_size = None
    if estimated_memory_mb > 4000:  # > 4GB
        # Auto-enable: save mỗi 50K samples (~1.2GB)
        auto_save_chunk_size = 50000
        logger.info(
            f"💡 Auto-enabling incremental save (chunk size: {auto_save_chunk_size:,}) "
            f"to prevent MemoryError (estimated: ~{estimated_memory_mb:.0f}MB)"
        )
    
    # Setup output directory cho chunks
    output_path_obj = Path(output_path)
    output_dir = output_path_obj.parent
    chunks_dir = output_dir / f'{output_path_obj.stem}_chunks'
    
    # Generate labels với multiprocessing
    result, errors = process_positions_to_labels_parallel(
        positions,
        board_size,
        num_workers=num_workers,
        batch_size=batch_size,
        save_chunk_size=auto_save_chunk_size,
        output_dir=chunks_dir if auto_save_chunk_size else None
    )
    
    # Nếu dùng incremental save, result là list of chunk files
    if auto_save_chunk_size and isinstance(result, list):
        saved_chunks = result
        logger.info(f"📦 Merging {len(saved_chunks)} chunks...")
        
        # Merge chunks
        all_labeled_data = []
        for chunk_file in tqdm(saved_chunks, desc="Loading chunks"):
            chunk_data = torch.load(chunk_file, map_location='cpu', weights_only=False)
            all_labeled_data.extend(chunk_data['labeled_data'])
            # Cleanup chunk file sau khi load (optional - có thể giữ để backup)
            # chunk_file.unlink()
        
        logger.info(f"✅ Merged {len(all_labeled_data):,} samples from {len(saved_chunks)} chunks")
        
        # Cleanup chunks directory (optional)
        # import shutil
        # shutil.rmtree(chunks_dir)
    else:
        all_labeled_data = result
    
    labeled_data = all_labeled_data
    
    # Log errors
    if errors:
        error_log_file = Path(output_path).parent / f'label_errors_{year or "all"}.log'
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
            
            f.write("Errors by type:\n")
            for err_type, err_list in errors_by_type.items():
                f.write(f"  {err_type}: {len(err_list)}\n")
            
            f.write("\nSample errors (first 100):\n")
            for err in errors[:100]:
                f.write(f"\nType: {err.get('type', 'unknown')}\n")
                f.write(f"Error: {err.get('error', 'N/A')}\n")
                if 'position' in err:
                    f.write(f"Position: {err['position']}\n")
        
        logger.warning(
            f"WARNING: {len(errors)} positions had errors. "
            f"See {error_log_file}"
        )
    
    # Save labeled dataset
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'labeled_data': labeled_data,
        'board_size': board_size,
        'total': len(labeled_data),
        'year': year,
        'metadata': {
            'filtered_handicap': filter_handicap,
            'input_file': str(input_path),
            'errors': len(errors),
            'date_processed': datetime.now().isoformat()
        }
    }, output_path)
    
    logger.info(
        f"SUCCESS: Saved {len(labeled_data):,} labeled samples to {output_path}"
    )
    
    # Summary
    total_processed = len(labeled_data) + len(errors)
    success_rate = (len(labeled_data) / total_processed * 100) if total_processed > 0 else 0
    
    logger.info("\n" + "="*50)
    logger.info("Label Generation Summary:")
    logger.info(f"  Input positions: {len(positions):,}")
    logger.info(f"  Processed positions: {total_processed:,}")
    logger.info(f"  Labeled samples: {len(labeled_data):,}")
    logger.info(f"  Errors: {len(errors):,}")
    logger.info(f"  Success rate: {success_rate:.2f}%")
    if len(errors) > 0:
        logger.info(f"  Error rate: {len(errors) / total_processed * 100:.2f}%")
    logger.info("="*50)
    
    return labeled_data


if __name__ == "__main__":
    import argparse
    
    # Signal handler để cleanup khi Ctrl+C
    def signal_handler(sig, frame):
        logger.warning("\n⚠️  Received interrupt signal. Cleaning up...")
        sys.exit(1)
    
    # Register signal handler cho SIGINT (Ctrl+C) và SIGTERM
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)
    
    parser = argparse.ArgumentParser(description='Generate labels locally')
    parser.add_argument('--input', type=str, required=True,
                        help='Input positions file (.pt)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output labeled dataset file (.pt)')
    parser.add_argument('--filter-handicap', action='store_true', default=True,
                        help='Filter out handicap positions')
    parser.add_argument('--no-filter-handicap', dest='filter_handicap',
                        action='store_false',
                        help='Keep handicap positions')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of worker processes (default: auto, max 8). '
                             'Giảm nếu RAM bị chiếm nhiều (ví dụ: --workers 8)')
    parser.add_argument('--batch-size', type=int, default=5000,
                        help='Batch size for processing (default: 5000, tối ưu cho performance). '
                             'Giảm nếu RAM bị chiếm nhiều (ví dụ: --batch-size 2000)')
    
    args = parser.parse_args()
    
    # Auto-adjust workers nếu không được chỉ định
    if args.workers is None:
        # Load file để estimate số positions
        try:
            data = torch.load(args.input, map_location='cpu', weights_only=False)
            num_positions = len(data.get('positions', []))
            
            # Tự động điều chỉnh: nhiều positions -> ít workers hơn để tránh RAM overflow
            if num_positions > 1_000_000:  # > 1M positions
                suggested_workers = min(6, cpu_count())
            elif num_positions > 500_000:  # > 500K positions
                suggested_workers = min(8, cpu_count())
            else:
                suggested_workers = min(8, cpu_count())
            
            logger.info(
                f"Auto-detected {num_positions:,} positions. "
                f"Using {suggested_workers} workers (có thể điều chỉnh bằng --workers)"
            )
            args.workers = suggested_workers
        except Exception as e:
            logger.warning(f"Could not auto-detect positions count: {e}. Using default workers.")
            args.workers = min(8, cpu_count())
    
    # Validate workers
    if args.workers > 16:
        logger.warning(
            f"⚠️  WARNING: {args.workers} workers có thể gây RAM overflow. "
            f"Khuyến nghị: --workers 8 hoặc ít hơn."
        )
    
    try:
        process_dataset_file(
            input_path=args.input,
            output_path=args.output,
            filter_handicap=args.filter_handicap,
            num_workers=args.workers,
            batch_size=args.batch_size
        )
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Script interrupted by user. Exiting...")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)

