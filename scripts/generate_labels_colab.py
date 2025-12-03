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

# Multiprocessing for Colab Pro
try:
    from multiprocessing import Pool, cpu_count
    MULTIPROCESSING_AVAILABLE = True
except ImportError:
    MULTIPROCESSING_AVAILABLE = False
    cpu_count = lambda: 1

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

# Import label generators cho Multi-task Model
try:
    from label_generators import (
        ThreatLabelGenerator,
        AttackLabelGenerator,
        IntentLabelGenerator,
        EvaluationLabelGenerator
    )
except ImportError:
    # Nếu chưa có, thử import từ thư mục hiện tại
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from label_generators import (
        ThreatLabelGenerator,
        AttackLabelGenerator,
        IntentLabelGenerator,
        EvaluationLabelGenerator
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


def process_single_position_optimized(
    pos: Dict, 
    board_size: int, 
    move_history: List = None,
    threat_gen: Optional[ThreatLabelGenerator] = None,
    attack_gen: Optional[AttackLabelGenerator] = None,
    intent_gen: Optional[IntentLabelGenerator] = None,
    eval_gen: Optional[EvaluationLabelGenerator] = None
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Process một position với reused generators (tối ưu memory và speed).
    """
    # Tạo generators nếu chưa có (backward compatibility)
    if threat_gen is None:
        threat_gen = ThreatLabelGenerator(board_size)
    if attack_gen is None:
        attack_gen = AttackLabelGenerator(board_size)
    if intent_gen is None:
        intent_gen = IntentLabelGenerator(board_size)
    if eval_gen is None:
        eval_gen = EvaluationLabelGenerator(board_size)
    
    return _process_single_position_core(pos, board_size, move_history, threat_gen, attack_gen, intent_gen, eval_gen)


def process_single_position(pos: Dict, board_size: int, move_history: List = None) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Backward compatibility wrapper."""
    return process_single_position_optimized(pos, board_size, move_history)


def _process_single_position_core(
    pos: Dict, 
    board_size: int, 
    move_history: List,
    threat_gen: ThreatLabelGenerator,
    attack_gen: AttackLabelGenerator,
    intent_gen: IntentLabelGenerator,
    eval_gen: EvaluationLabelGenerator
) -> Tuple[Optional[Dict], Optional[Dict]]:
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
        
        # Convert numpy board to tensor (tối ưu: tránh copy và type conversion)
        if isinstance(board_state, np.ndarray):
            board_np = board_state
            # Chỉ convert nếu cần (tránh overhead)
            if board_np.dtype != np.int8:
                board_np = board_np.astype(np.int8, copy=False)
        else:
            board_np = np.array(board_state, dtype=np.int8)
        
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
        
        # Validate move format
        if move is None:
            # Pass move - OK
            pass
        elif isinstance(move, (tuple, list)) and len(move) == 2:
            # Normal move - validate coordinates
            mx, my = move
            if not (0 <= mx < board_size and 0 <= my < board_size):
                return None, {
                    'error': f'Move coordinates ({mx}, {my}) out of bounds for board size {board_size}',
                    'type': 'invalid_move',
                    'move_number': move_number
                }
        else:
            return None, {
                'error': f'Invalid move format: {move}. Expected (x, y) tuple or None for pass.',
                'type': 'invalid_move_format',
                'move_number': move_number
            }
        
        # Validate current_player
        if current_player not in ('B', 'W', 'b', 'w'):
            return None, {
                'error': f'Invalid current_player: {current_player}. Must be B or W.',
                'type': 'invalid_player',
                'move_number': move_number
            }
        
        # Normalize current_player
        current_player = current_player.upper()
        
        # Generate Multi-task Model labels (theo tài liệu)
        # Tối ưu: Tính groups một lần, reuse cho cả threat và attack
        try:
            # Tính groups một lần (tốn thời gian nhất)
            groups = threat_gen.find_groups(board_np)
            
            # Validate groups
            if groups is None:
                groups = []
            
            # Reuse groups cho threat_map
            threat_map = threat_gen.generate_threat_map(board_np, current_player, groups=groups)
        except Exception as e:
            import traceback
            return None, {
                'error': f'Threat map generation failed: {str(e)}\n{traceback.format_exc()}',
                'type': 'threat_map_error',
                'move_number': move_number
            }
        
        try:
            # Reuse groups cho attack_map (tối ưu quan trọng - tránh tính lại groups!)
            attack_map = attack_gen.generate_attack_map(board_np, current_player, groups=groups)
        except Exception as e:
            import traceback
            return None, {
                'error': f'Attack map generation failed: {str(e)}\n{traceback.format_exc()}',
                'type': 'attack_map_error',
                'move_number': move_number
            }
        
        try:
            intent_label = intent_gen.generate_intent_label(
                board_np, move, move_history or [], current_player
            )
        except Exception as e:
            return None, {
                'error': f'Intent label generation failed: {str(e)}',
                'type': 'intent_label_error',
                'move_number': move_number
            }
        
        try:
            evaluation_label = eval_gen.generate_evaluation(
                board_np, current_player, winner, game_result
            )
        except Exception as e:
            return None, {
                'error': f'Evaluation label generation failed: {str(e)}',
                'type': 'evaluation_label_error',
                'move_number': move_number
            }
        
        # Generate policy/value labels (cho Policy/Value Network - backward compatibility)
        try:
            policy = generate_policy_label(move, board_size)
        except ValueError as e:
            return None, {
                'error': f'Policy label generation failed: {str(e)}',
                'type': 'policy_label_error',
                'move_number': move_number
            }
        
        try:
            value = generate_value_label(winner, current_player, game_result)
        except ValueError as e:
            return None, {
                'error': f'Value label generation failed: {str(e)}',
                'type': 'value_label_error',
                'move_number': move_number
            }
        
        # Validate value is in valid range
        if not (0.0 <= value <= 1.0):
            return None, {
                'error': f'Invalid value label: {value}. Must be between 0.0 and 1.0.',
                'type': 'invalid_value',
                'move_number': move_number
            }
        
        # Create labeled sample theo format tài liệu ML_COMPREHENSIVE_GUIDE.md
        labeled_sample = {
            # Core data
            'features': features,  # Tensor[17, board_size, board_size]
            
            # Labels cho Multi-task Model (theo tài liệu)
            'labels': {
                'threat_map': threat_map,  # Tensor[board_size, board_size]
                'attack_map': attack_map,  # Tensor[board_size, board_size]
                'intent': intent_label,    # Dict với type, confidence, region
                'evaluation': evaluation_label  # Dict với win_probability, territory_map, influence_map
            },
            
            # Policy/Value labels (backward compatibility)
            'policy': policy,  # Tensor[board_size * board_size + 1]
            'value': value,   # float
            
            # Metadata
            'metadata': {
                'move_number': move_number,
                'game_result': game_result,
                'winner': winner,
                'handicap': pos.get('handicap', 0),
                'board_size': board_size,
                'current_player': current_player
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


def process_positions_batch(batch: List[Dict], board_size: int) -> Tuple[List[Dict], List[Dict]]:
    """
    Process một batch positions (dùng trong multiprocessing).
    Tối ưu: Reuse label generators cho toàn bộ batch.
    
    Args:
        batch: List of position dicts từ cùng một game
        board_size: Board size
    
    Returns:
        (labeled_samples, errors) tuple
    """
    labeled_samples = []
    errors = []
    move_history = []
    
    # REUSE generators cho toàn bộ batch (tối ưu quan trọng!)
    threat_gen = ThreatLabelGenerator(board_size)
    attack_gen = AttackLabelGenerator(board_size)
    intent_gen = IntentLabelGenerator(board_size)
    eval_gen = EvaluationLabelGenerator(board_size)
    
    for pos in batch:
        # Update move history
        move_num = pos.get('move_number', 0)
        if move_num == 0:
            move_history = []
        
        # Process position với reused generators
        labeled_sample, error_info = process_single_position_optimized(
            pos, board_size, move_history,
            threat_gen, attack_gen, intent_gen, eval_gen
        )
        
        if error_info is not None:
            errors.append(error_info)
        elif labeled_sample is not None:
            labeled_samples.append(labeled_sample)
        
        # Update move history
        move = pos.get('move')
        if move:
            move_history.append(move)
            if len(move_history) > 4:
                move_history = move_history[-4:]
    
    return labeled_samples, errors


def _process_batch_wrapper(args):
    """Wrapper function for multiprocessing."""
    batch, board_size = args
    return process_positions_batch(batch, board_size)


def process_positions_to_labels(
    positions: List[Dict],
    board_size: int,
    save_chunk_size: Optional[int] = None,
    output_dir: Optional[Path] = None,
    chunk_prefix: str = 'chunk',
    num_workers: Optional[int] = None,
    use_multiprocessing: bool = True
) -> Tuple[List[Dict], List[Dict], List[Path]]:
    """
    Convert positions thành labeled data với incremental save.
    Tối ưu cho Colab Pro với multiprocessing.
    
    Args:
        positions: List of position dicts
        board_size: Board size
        save_chunk_size: Nếu set, save định kỳ mỗi N samples để giảm memory
        output_dir: Directory để save chunks (nếu dùng incremental save)
        chunk_prefix: Prefix cho chunk files
        num_workers: Số worker processes (None = auto-detect, Colab Pro thường 4-8)
        use_multiprocessing: Có dùng multiprocessing không (True = nhanh hơn)
    
    Returns:
        (labeled_data, errors, saved_chunks) tuple
        - labeled_data: List of labeled samples (chỉ có nếu không dùng incremental save)
        - errors: List of error dicts
        - saved_chunks: List of chunk file paths (nếu dùng incremental save)
    """
    use_incremental_save = save_chunk_size is not None and save_chunk_size > 0
    
    if use_incremental_save and output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Incremental save enabled: chunks will be saved to {output_dir}")
        logger.info(f"   Chunk size: {save_chunk_size:,} samples (~{save_chunk_size * 50 / 1024:.0f}MB per chunk)")
    
    # Default: Single-threaded (1 worker) - tận dụng 50GB RAM với chunk size lớn
    if num_workers is None:
        num_workers = 1
    
    # Chỉ dùng multiprocessing nếu explicitly enabled và num_workers > 1
    if use_multiprocessing and MULTIPROCESSING_AVAILABLE and num_workers > 1 and len(positions) > 1000:
        logger.info(f"🚀 Using multiprocessing with {num_workers} workers")
        return _process_positions_parallel(
            positions, board_size, num_workers, save_chunk_size, output_dir, chunk_prefix
        )
    
    # Single-threaded (default) - tối ưu với reused generators và chunk size lớn
    logger.info("📝 Using single-threaded processing (1 worker, optimized for 50GB RAM)")
    return _process_positions_single_threaded(
        positions, board_size, save_chunk_size, output_dir, chunk_prefix
    )


def _process_positions_parallel(
    positions: List[Dict],
    board_size: int,
    num_workers: int,
    save_chunk_size: Optional[int],
    output_dir: Optional[Path],
    chunk_prefix: str
) -> Tuple[List[Dict], List[Dict], List[Path]]:
    """Parallel processing với multiprocessing (tối ưu cho Colab Pro)."""
    labeled_data = []
    errors = []
    saved_chunks = []
    chunk_counter = 0
    
    # Group positions by game để maintain move history
    batches = []
    current_batch = []
    last_move_num = -1
    
    # Optimal batch size: đủ lớn để giảm overhead, đủ nhỏ để fit memory
    batch_size = max(500, min(5000, len(positions) // (num_workers * 4)))
    
    for pos in positions:
        move_num = pos.get('move_number', 0)
        
        # Start new batch if move_number resets (new game)
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
    
    logger.info(f"   Created {len(batches):,} batches (avg size: {len(positions)//len(batches) if batches else 0})")
    
    start_time = time.time()
    
    # Process in parallel
    pool = None
    try:
        pool = Pool(processes=num_workers)
        
        # Use imap_unordered for better performance
        results = list(tqdm(
            pool.imap_unordered(
                _process_batch_wrapper,
                [(batch, board_size) for batch in batches],
                chunksize=max(1, len(batches) // (num_workers * 4))
            ),
            total=len(batches),
            desc="Processing batches",
            unit="batch"
        ))
        
        # Collect results
        for batch_labeled, batch_errors in results:
            labeled_data.extend(batch_labeled)
            errors.extend(batch_errors)
            
            # Incremental save nếu cần
            if save_chunk_size and output_dir and len(labeled_data) >= save_chunk_size:
                chunk_counter += 1
                chunk_file = output_dir / f'{chunk_prefix}_{chunk_counter:04d}.pt'
                
                logger.info(f"💾 Saving chunk {chunk_counter} ({len(labeled_data):,} samples)")
                
                torch.save({
                    'labeled_data': labeled_data,
                    'board_size': board_size,
                    'chunk_num': chunk_counter,
                    'total_samples': len(labeled_data)
                }, chunk_file)
                
                saved_chunks.append(chunk_file)
                labeled_data = []
                gc.collect()
    
    except KeyboardInterrupt:
        logger.warning("⚠️  Interrupted by user")
        if pool:
            pool.terminate()
        raise
    finally:
        if pool:
            pool.close()
            pool.join()
    
    elapsed = time.time() - start_time
    speed = len(positions) / elapsed if elapsed > 0 else 0
    logger.info(f"✅ Processed {len(positions):,} positions in {elapsed:.1f}s ({speed:.0f} pos/s)")
    
    return labeled_data, errors, saved_chunks


def _process_positions_single_threaded(
    positions: List[Dict],
    board_size: int,
    save_chunk_size: Optional[int],
    output_dir: Optional[Path],
    chunk_prefix: str
) -> Tuple[List[Dict], List[Dict], List[Path]]:
    """Single-threaded processing (optimized for memory and speed)."""
    labeled_data = []
    errors = []
    saved_chunks = []
    
    # REUSE label generators (không tạo mới mỗi lần) - Tối ưu quan trọng!
    threat_gen = ThreatLabelGenerator(board_size)
    attack_gen = AttackLabelGenerator(board_size)
    intent_gen = IntentLabelGenerator(board_size)
    eval_gen = EvaluationLabelGenerator(board_size)
    
    # Track move history for each game
    move_history = []
    last_move_num = -1
    chunk_counter = 0
    
    # Auto-enable incremental save với chunk size 50K
    if save_chunk_size is None or save_chunk_size <= 0:
        save_chunk_size = 50000  # Default: 50K samples (~2.5GB)
        logger.info(f"💡 Auto-enabling incremental save (chunk_size={save_chunk_size:,})")
    
    if output_dir is None:
        output_dir = Path('/tmp/label_chunks')
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.warning(f"⚠️  Using temporary directory: {output_dir}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    last_speed_check_time = start_time
    last_speed_check_positions = 0
    
    for idx, pos in enumerate(tqdm(positions, desc="Generating labels", unit="pos")):
        move_num = pos.get('move_number', 0)
        
        # Reset history nếu là game mới
        if move_num < last_move_num or move_num == 0:
            move_history = []
        
        # Process position với reused generators
        labeled_sample, error_info = process_single_position_optimized(
            pos, board_size, move_history, 
            threat_gen, attack_gen, intent_gen, eval_gen
        )
        
        if error_info is not None:
            errors.append(error_info)
        elif labeled_sample is not None:
            labeled_data.append(labeled_sample)
        
        # Update move history
        move = pos.get('move')
        if move:
            move_history.append(move)
            if len(move_history) > 4:
                move_history = move_history[-4:]
        
        last_move_num = move_num
        
        # Incremental save (bắt buộc)
        if len(labeled_data) >= save_chunk_size:
            chunk_counter += 1
            chunk_file = output_dir / f'{chunk_prefix}_{chunk_counter:04d}.pt'
            
            logger.info(f"💾 Saving chunk {chunk_counter} ({len(labeled_data):,} samples) to {chunk_file.name}")
            
            # Save với compression để giảm I/O time
            torch.save({
                'labeled_data': labeled_data,
                'board_size': board_size,
                'chunk_num': chunk_counter,
                'total_samples': len(labeled_data)
            }, chunk_file, _use_new_zipfile_serialization=True)
            
            saved_chunks.append(chunk_file)
            
            # Clear memory ngay lập tức
            del labeled_data
            labeled_data = []
            gc.collect()
        
        # Periodic GC và speed check (mỗi 20K samples để giảm overhead - với 50GB RAM có thể ít hơn)
        if len(labeled_data) % 20000 == 0 and len(labeled_data) > 0:
            gc.collect()
            
            current_time = time.time()
            time_since_last_check = current_time - last_speed_check_time
            if time_since_last_check >= 30.0:
                positions_since_last_check = idx + 1 - last_speed_check_positions
                real_time_speed = positions_since_last_check / time_since_last_check if time_since_last_check > 0 else 0
                total_elapsed = current_time - start_time
                avg_speed = (idx + 1) / total_elapsed if total_elapsed > 0 else 0
                
                estimated_memory_mb = len(labeled_data) * 50 / 1024
                
                logger.info(
                    f"Speed: {real_time_speed:.0f} pos/s (avg: {avg_speed:.0f}) | "
                    f"Memory: ~{estimated_memory_mb:.0f}MB | "
                    f"Progress: {idx+1:,}/{len(positions):,} ({(idx+1)/len(positions)*100:.1f}%)"
                )
                
                last_speed_check_time = current_time
                last_speed_check_positions = idx + 1
    
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
            'chunk_files': [str(f) for f in chunk_files],
            'date_processed': datetime.now().isoformat()
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
    skip_merge: bool = False,  # Nếu True, giữ chunks riêng, không merge
    num_workers: Optional[int] = None,  # Số workers (None = 1, single-threaded)
    use_multiprocessing: bool = False  # Default False - single-threaded với 1 worker
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
    
    # Auto-calculate chunk size dựa trên memory
    # Default: 50K samples (~2.5GB)
    estimated_memory_mb = len(positions) * 50 / 1024
    if save_chunk_size is None or save_chunk_size <= 0:
        # Default chunk size: 50K samples
        save_chunk_size = 50000  # 50K samples (~2.5GB)
        
        logger.info(
            f"💡 Auto-enabling incremental save (chunk size: {save_chunk_size:,}) "
            f"(estimated: ~{estimated_memory_mb:.0f}MB)"
        )
    elif estimated_memory_mb > 2000:
        logger.info(
            f"📊 Estimated memory usage: ~{estimated_memory_mb:.0f}MB. "
            f"Incremental save enabled with chunk_size={save_chunk_size:,}"
        )
    
    # Setup output directory cho chunks
    output_path_obj = Path(output_path)
    output_dir = output_path_obj.parent
    chunks_dir = output_dir / f'{output_path_obj.stem}_chunks'
    
    # Generate labels với multiprocessing (Colab Pro optimized)
    labeled_data, errors, saved_chunks = process_positions_to_labels(
        positions,
        board_size,
        save_chunk_size=save_chunk_size,
        output_dir=chunks_dir if save_chunk_size else None,
        chunk_prefix=output_path_obj.stem,
        num_workers=num_workers,
        use_multiprocessing=use_multiprocessing
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
                'input_file': str(input_path),
                'errors': len(errors),
                'date_processed': datetime.now().isoformat()
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
    
    # Process một năm (Colab Pro optimized)
    process_dataset_file(
        input_path=WORK_DIR / 'processed' / 'positions_19x19_2019.pt',
        output_path=WORK_DIR / 'datasets' / 'labeled_19x19_2019.pt',
        filter_handicap=True,
        save_chunk_size=50000,  # Save mỗi 50K samples
        num_workers=None,  # Auto-detect (Colab Pro: 4-8 workers)
        use_multiprocessing=True  # Enable multiprocessing
    )
