"""
Script tối ưu để parse SGF files trên local máy tính.

Tính năng:
- Xử lý theo năm (từ tên file YYYY-MM-DD-XX.sgf)
- Multiprocessing để tăng tốc
- Error handling và logging
- Bỏ qua file lỗi và tiếp tục
- Output theo năm
"""

import sgf
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import logging
from datetime import datetime
import traceback
from multiprocessing import Pool, cpu_count
import re
import json


# Setup logging with UTF-8 encoding
import sys

# Create file handler with UTF-8
file_handler = logging.FileHandler('parse_sgf_local.log', encoding='utf-8')
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


def parse_sgf_coord(sgf_coord, board_size):
    """
    Convert SGF coordinate to (x, y).
    
    Handles:
    - Normal moves: 'cd', 'dd', etc.
    - Pass moves: '', 'tt', None, or any invalid coordinate
    - 9x9 board: coordinates from 'a' to 'i' (skip 'i' in Go)
    """
    # Handle pass moves and invalid coordinates
    if not sgf_coord:
        return None, None  # Pass move (empty string)
    
    # Convert to string if needed
    sgf_coord = str(sgf_coord).strip()
    
    # Pass move indicators
    if not sgf_coord or len(sgf_coord) < 2 or sgf_coord.lower() == 'tt':
        return None, None  # Pass move
    
    # Parse coordinates
    try:
        x = ord(sgf_coord[0].lower()) - ord('a')
        y = ord(sgf_coord[1].lower()) - ord('a')
        
        # Skip 'i' (no I in Go coordinates)
        if x >= 8:
            x -= 1
        if y >= 8:
            y -= 1
        
        # Validate bounds
        if x < 0 or x >= board_size or y < 0 or y >= board_size:
            return None, None  # Invalid coordinate (treat as pass)
        
        return x, y
    except (ValueError, IndexError, TypeError):
        # Invalid coordinate format (treat as pass)
        return None, None


def parse_single_sgf_file(sgf_path):
    """
    Parse một SGF file và return positions.
    
    Returns:
        (positions, error_info) tuple
        - positions: list of position dicts hoặc None nếu lỗi
        - error_info: dict với thông tin lỗi hoặc None nếu thành công
    """
    sgf_path = Path(sgf_path)
    
    try:
        with open(sgf_path, 'r', encoding='utf-8', errors='ignore') as f:
            sgf_content = f.read()
        
        # Validate SGF content
        if not sgf_content.strip():
            return None, {
                'file': str(sgf_path),
                'error': 'Empty file',
                'type': 'empty'
            }
        
        # Parse SGF
        try:
            parsed = sgf.parse(sgf_content)
        except Exception as e:
            return None, {
                'file': str(sgf_path),
                'error': f'Parse error: {str(e)}',
                'type': 'parse_error'
            }
        
        # sgf.parse() always returns a Collection (even for single game)
        # Get first game from collection
        try:
            if hasattr(parsed, '__iter__'):
                # It's a Collection, get first game
                games = list(parsed)
                if not games:
                    return None, {
                        'file': str(sgf_path),
                        'error': 'No games found in SGF file',
                        'type': 'no_games'
                    }
                game = games[0]
            else:
                # Should not happen, but handle just in case
                return None, {
                    'file': str(sgf_path),
                    'error': f'Unexpected parse result type: {type(parsed)}',
                    'type': 'parse_error'
                }
        except Exception as e:
            return None, {
                'file': str(sgf_path),
                'error': f'Error extracting game from collection: {str(e)}',
                'type': 'parse_error'
            }
        
        # Extract metadata
        root = game.root
        try:
            board_size = int(root.properties.get('SZ', ['19'])[0])
        except (ValueError, TypeError):
            board_size = 19  # Default
        
        # Validate board size
        if board_size not in [9, 13, 19]:
            return None, {
                'file': str(sgf_path),
                'error': f'Invalid board size: {board_size}',
                'type': 'invalid_board_size'
            }
        
        # Get result property (RE = Result) - cải tiến parsing
        result = ''
        winner = None
        try:
            result_prop = root.properties.get('RE', [])
            if result_prop and len(result_prop) > 0:
                result_value = result_prop[0]
                # Handle both string and bytes
                if isinstance(result_value, bytes):
                    result = result_value.decode('utf-8', errors='ignore')
                else:
                    result = str(result_value)
                
                # Parse winner from result
                # Format examples: "B+12.5", "W+R", "B+", "W+0.5", "0" (draw)
                result_upper = result.upper().strip()
                if result_upper.startswith('B+') or result_upper == 'B':
                    winner = 'B'
                elif result_upper.startswith('W+') or result_upper == 'W':
                    winner = 'W'
                elif result_upper == '0' or result_upper == 'DRAW':
                    winner = 'DRAW'  # Hòa
                # If result doesn't match expected format, winner stays None
        except Exception as e:
            # If RE property doesn't exist or can't be parsed, result stays empty
            logger.debug(f"Could not parse result in {sgf_path}: {e}")
        
        # Extract handicap info
        try:
            handicap = int(root.properties.get('HA', ['0'])[0])
        except (ValueError, TypeError):
            handicap = 0
        
        handicap_stones_black = root.properties.get('AB', [])
        handicap_stones_white = root.properties.get('AW', [])
        
        # Initialize board
        board = np.zeros((board_size, board_size), dtype=np.int8)
        
        # Place handicap stones
        if handicap_stones_black:
            for stone_coord in handicap_stones_black:
                x, y = parse_sgf_coord(stone_coord, board_size)
                if x is not None and y is not None:
                    board[y, x] = 1  # Black = 1
        
        if handicap_stones_white:
            for stone_coord in handicap_stones_white:
                x, y = parse_sgf_coord(stone_coord, board_size)
                if x is not None and y is not None:
                    board[y, x] = 2  # White = 2
        
        # Determine starting player
        current_player = 'W' if handicap > 0 else 'B'
        
        # Extract moves
        positions = []
        move_count = 0
        
        try:
            # game.rest might be None for empty games
            if game.rest is None:
                return None, {
                    'file': str(sgf_path),
                    'error': 'No moves found in game',
                    'type': 'no_moves'
                }
            
            # Convert to list to avoid issues with generators and limit moves
            # Some SGF files might have corrupted or extremely long move sequences
            try:
                rest_nodes = list(game.rest)
            except Exception as e:
                return None, {
                    'file': str(sgf_path),
                    'error': f'Error converting moves to list: {str(e)}',
                    'type': 'move_conversion_error'
                }
            
            # Limit maximum moves to prevent infinite loops from corrupted files
            max_moves = 10000  # Reasonable limit for a Go game
            if len(rest_nodes) > max_moves:
                logger.warning(f"File {sgf_path} has {len(rest_nodes)} moves, limiting to {max_moves}")
                rest_nodes = rest_nodes[:max_moves]
            
            for node in rest_nodes:
                move = None
                color = None
                
                # Check for move properties (B or W)
                # Skip comment-only nodes (C property without B/W)
                if 'B' in node.properties:
                    move = node.properties['B'][0]
                    color = 'B'
                elif 'W' in node.properties:
                    move = node.properties['W'][0]
                    color = 'W'
                else:
                    # Node không có move (chỉ có comment, CC, etc.) → skip
                    # Nhưng vẫn cần switch player nếu có comment về resignation
                    if 'C' in node.properties:
                        # Check if it's a resignation comment
                        comment = node.properties['C'][0]
                        if isinstance(comment, bytes):
                            comment = comment.decode('utf-8', errors='ignore')
                        comment_str = str(comment).lower()
                        if 'resign' in comment_str:
                            # Game ended, stop parsing
                            break
                    continue
                
                # Parse move coordinate (handles both normal moves and pass moves)
                x, y = parse_sgf_coord(move, board_size)
                
                # Handle both normal moves and pass moves
                if x is not None and y is not None:
                    # Normal move - validate (không được đặt vào ô đã có quân)
                    if board[y, x] != 0:
                        # Skip invalid move (occupied square)
                        logger.warning(
                            f"Invalid move in {sgf_path} at move {move_count}: "
                            f"Position ({x}, {y}) already occupied"
                        )
                        continue
                    
                    # Save position BEFORE move
                    positions.append({
                        'board_state': board.copy(),
                        'move': (x, y),  # Normal move as tuple
                        'current_player': current_player,
                        'move_number': move_count,
                        'board_size': board_size,
                        'game_result': result,
                        'winner': winner,
                        'handicap': handicap
                    })
                    
                    # Apply move (simplified - không xử lý captures, ko, etc.)
                    board[y, x] = 1 if color == 'B' else 2
                    move_count += 1
                else:
                    # Pass move - lưu với move = None
                    positions.append({
                        'board_state': board.copy(),
                        'move': None,  # Pass move marked as None
                        'current_player': current_player,
                        'move_number': move_count,
                        'board_size': board_size,
                        'game_result': result,
                        'winner': winner,
                        'handicap': handicap
                    })
                    move_count += 1
                    # Không cần apply move cho pass
                
                current_player = 'W' if current_player == 'B' else 'B'
        except Exception as e:
            # Nếu lỗi khi parse moves, vẫn return positions đã parse được
            logger.warning(f"Error parsing moves in {sgf_path}: {e}")
        
        if len(positions) == 0:
            return None, {
                'file': str(sgf_path),
                'error': 'No valid moves found',
                'type': 'no_moves'
            }
        
        return positions, None
        
    except Exception as e:
        error_info = {
            'file': str(sgf_path),
            'error': str(e),
            'type': 'exception',
            'traceback': traceback.format_exc()
        }
        return None, error_info


def extract_year_from_filename(filename):
    """Extract year from filename format: YYYY-M-D-X.sgf or YYYY-MM-DD-XX.sgf"""
    # Match both formats: YYYY-M-D-X.sgf and YYYY-MM-DD-XX.sgf
    match = re.match(r'(\d{4})-\d{1,2}-\d{1,2}-\d+\.sgf', filename)
    if match:
        return int(match.group(1))
    return None


def process_sgf_files_parallel(sgf_files, num_workers=None):
    """
    Process SGF files với multiprocessing.
    
    Returns:
        (all_positions, errors) tuple
    """
    if num_workers is None:
        num_workers = min(cpu_count(), 8)  # Max 8 workers
    
    logger.info(f"Processing {len(sgf_files)} files with {num_workers} workers")
    
    all_positions = defaultdict(list)
    errors = []
    
    # Limit number of files processed at once to avoid memory issues
    # Process in chunks if there are too many files
    chunk_size = 1000  # Process 1000 files at a time
    total_files = len(sgf_files)
    
    for chunk_start in range(0, total_files, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_files)
        chunk_files = sgf_files[chunk_start:chunk_end]
        
        logger.info(f"Processing chunk {chunk_start//chunk_size + 1}/{(total_files + chunk_size - 1)//chunk_size}: "
                   f"files {chunk_start+1}-{chunk_end} of {total_files}")
        
        try:
            with Pool(processes=num_workers) as pool:
                results = list(tqdm(
                    pool.imap(parse_single_sgf_file, chunk_files),
                    total=len(chunk_files),
                    desc=f"Parsing chunk {chunk_start//chunk_size + 1}",
                    miniters=10  # Update progress bar less frequently
                ))
            
            # Collect results from this chunk
            for positions, error_info in results:
                if error_info is not None:
                    errors.append(error_info)
                    continue
                
                if positions is None:
                    continue
                
                for pos in positions:
                    board_size = pos['board_size']
                    all_positions[board_size].append(pos)
        
        except KeyboardInterrupt:
            logger.warning("Interrupted by user. Stopping processing.")
            raise
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_start//chunk_size + 1}: {e}")
            import traceback
            traceback.print_exc()
            # Continue with next chunk instead of stopping completely
            continue
    
    return all_positions, errors


def process_sgf_directory_by_year(
    sgf_dir,
    output_dir,
    year=None,
    board_sizes=[9, 13, 19],
    num_workers=None,
    min_positions_per_game=10
):
    """
    Process SGF files, có thể filter theo năm.
    
    Args:
        sgf_dir: Path to directory chứa SGF files
        output_dir: Path to directory để lưu positions
        year: Năm cần xử lý (None = tất cả)
        board_sizes: List các board sizes cần xử lý
        num_workers: Số worker processes (None = auto)
        min_positions_per_game: Số positions tối thiểu mỗi game
    """
    sgf_dir = Path(sgf_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find SGF files
    sgf_files = list(sgf_dir.glob('*.sgf'))
    
    if not sgf_files:
        logger.warning(f"⚠️  No SGF files found in {sgf_dir}")
        return
    
    logger.info(f"📊 Found {len(sgf_files)} SGF files")
    
    # Filter by year if specified
    if year is not None:
        sgf_files = [
            f for f in sgf_files
            if extract_year_from_filename(f.name) == year
        ]
        logger.info(f"📅 Filtered to {len(sgf_files)} files for year {year}")
    
    if not sgf_files:
        logger.warning(f"⚠️  No files found for year {year}")
        return
    
    # Group by year for statistics
    files_by_year = defaultdict(list)
    for f in sgf_files:
        file_year = extract_year_from_filename(f.name)
        if file_year:
            files_by_year[file_year].append(f)
    
    logger.info(f"📅 Files by year: {dict(files_by_year)}")
    
    # Process files
    all_positions, errors = process_sgf_files_parallel(sgf_files, num_workers)
    
    # Log errors
    if errors:
        error_log_file = output_dir / f'parse_errors_{year or "all"}.log'
        with open(error_log_file, 'w', encoding='utf-8') as f:
            f.write(f"Parse Errors Summary\n")
            f.write(f"Total errors: {len(errors)}\n")
            f.write(f"Date: {datetime.now()}\n\n")
            
            # Group errors by type
            errors_by_type = defaultdict(list)
            for err in errors:
                errors_by_type[err['type']].append(err)
            
            f.write("Errors by type:\n")
            for err_type, err_list in errors_by_type.items():
                f.write(f"  {err_type}: {len(err_list)}\n")
            
            f.write("\nDetailed errors:\n")
            for err in errors:
                f.write(f"\nFile: {err['file']}\n")
                f.write(f"Type: {err['type']}\n")
                f.write(f"Error: {err['error']}\n")
                if 'traceback' in err:
                    f.write(f"Traceback:\n{err['traceback']}\n")
        
        if errors:
            logger.warning(f"{len(errors)} files had errors. See {error_log_file}")
    
    # Filter positions by minimum count
    filtered_positions = {}
    for board_size in board_sizes:
        positions = all_positions.get(board_size, [])
        
        # Group by game and filter
        games = defaultdict(list)
        for pos in positions:
            # Use move_number to group (simplified)
            game_key = pos.get('move_number', 0)  # Simplified grouping
            games[game_key].append(pos)
        
        # Filter games with too few positions
        valid_positions = []
        for game_positions in games.values():
            if len(game_positions) >= min_positions_per_game:
                valid_positions.extend(game_positions)
        
        if valid_positions:
            filtered_positions[board_size] = valid_positions
            logger.info(
                f"✅ {board_size}x{board_size}: "
                f"{len(valid_positions):,} positions "
                f"({len(positions):,} before filtering)"
            )
    
    # Save positions theo board size và year
    year_suffix = f"_{year}" if year else ""
    
    for board_size in board_sizes:
        if board_size not in filtered_positions:
            continue
        
        positions = filtered_positions[board_size]
        output_file = output_dir / f'positions_{board_size}x{board_size}{year_suffix}.pt'
        
        torch.save({
            'positions': positions,
            'board_size': board_size,
            'total': len(positions),
            'year': year,
            'metadata': {
                'source_files': len(sgf_files),
                'errors': len(errors),
                'date_processed': datetime.now().isoformat()
            }
        }, output_file)
        
        logger.info(
            f"💾 Saved {len(positions):,} positions for {board_size}x{board_size} "
            f"to {output_file}"
        )
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("Parsing Summary:")
    logger.info(f"  Total files processed: {len(sgf_files)}")
    logger.info(f"  Files with errors: {len(errors)}")
    for board_size in board_sizes:
        if board_size in filtered_positions:
            logger.info(f"  {board_size}x{board_size}: {len(filtered_positions[board_size]):,} positions")


def process_sgf_directory_with_chunking(
    sgf_dir,
    output_dir,
    board_sizes=[9, 13, 19],
    num_workers=None,
    min_positions_per_game=10,
    positions_per_chunk=50000,
    chunk_prefix="chunk"
):
    """
    Process SGF files và tự động chia thành các file output với kích thước hợp lý.
    Phù hợp cho các file SGF không có năm trong tên file.
    
    Args:
        sgf_dir: Path to directory chứa SGF files
        output_dir: Path to directory để lưu positions
        board_sizes: List các board sizes cần xử lý
        num_workers: Số worker processes (None = auto)
        min_positions_per_game: Số positions tối thiểu mỗi game
        positions_per_chunk: Số positions mỗi chunk file (default: 50K)
        chunk_prefix: Prefix cho tên file chunk (default: "chunk")
    """
    sgf_dir = Path(sgf_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find SGF files (không filter theo năm)
    sgf_files = sorted(list(sgf_dir.glob('*.sgf')))
    
    if not sgf_files:
        logger.warning(f"⚠️  No SGF files found in {sgf_dir}")
        return
    
    logger.info(f"📊 Found {len(sgf_files)} SGF files")
    logger.info(f"📁 Files range: {sgf_files[0].name} to {sgf_files[-1].name}")
    
    # Process files
    all_positions, errors = process_sgf_files_parallel(sgf_files, num_workers)
    
    # Log errors
    if errors:
        error_log_file = output_dir / 'parse_errors.log'
        with open(error_log_file, 'w', encoding='utf-8') as f:
            f.write(f"Parse Errors Summary\n")
            f.write(f"Total errors: {len(errors)}\n")
            f.write(f"Date: {datetime.now()}\n\n")
            
            # Group errors by type
            errors_by_type = defaultdict(list)
            for err in errors:
                errors_by_type[err['type']].append(err)
            
            f.write("Errors by type:\n")
            for err_type, err_list in errors_by_type.items():
                f.write(f"  {err_type}: {len(err_list)}\n")
            
            f.write("\nDetailed errors:\n")
            for err in errors:
                f.write(f"\nFile: {err['file']}\n")
                f.write(f"Type: {err['type']}\n")
                f.write(f"Error: {err['error']}\n")
                if 'traceback' in err:
                    f.write(f"Traceback:\n{err['traceback']}\n")
        
        logger.warning(f"⚠️  {len(errors)} errors occurred. See {error_log_file}")
    
    # Filter positions by minimum count
    filtered_positions = {}
    for board_size in board_sizes:
        positions = all_positions.get(board_size, [])
        
        # Group by game and filter
        games = defaultdict(list)
        for pos in positions:
            # Use move_number to group (simplified)
            game_key = pos.get('move_number', 0)  # Simplified grouping
            games[game_key].append(pos)
        
        # Filter games with too few positions
        valid_positions = []
        for game_positions in games.values():
            if len(game_positions) >= min_positions_per_game:
                valid_positions.extend(game_positions)
        
        if valid_positions:
            filtered_positions[board_size] = valid_positions
            logger.info(
                f"✅ {board_size}x{board_size}: "
                f"{len(valid_positions):,} positions "
                f"({len(positions):,} before filtering)"
            )
    
    # Save positions theo board size với chunking
    for board_size in board_sizes:
        if board_size not in filtered_positions:
            continue
        
        positions = filtered_positions[board_size]
        total_positions = len(positions)
        
        # Chia thành các chunks
        num_chunks = (total_positions + positions_per_chunk - 1) // positions_per_chunk
        
        logger.info(
            f"💾 Saving {total_positions:,} positions for {board_size}x{board_size} "
            f"into {num_chunks} chunk(s) ({positions_per_chunk:,} positions/chunk)"
        )
        
        saved_files = []
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * positions_per_chunk
            end_idx = min(start_idx + positions_per_chunk, total_positions)
            chunk_positions = positions[start_idx:end_idx]
            
            # Tên file: {chunk_prefix}_{board_size}x{board_size}_{chunk_num:04d}.pt
            chunk_file = output_dir / f'{chunk_prefix}_{board_size}x{board_size}_{chunk_idx+1:04d}.pt'
            
            torch.save({
                'positions': chunk_positions,
                'board_size': board_size,
                'chunk_num': chunk_idx + 1,
                'total_chunks': num_chunks,
                'positions_in_chunk': len(chunk_positions),
                'start_index': start_idx,
                'end_index': end_idx - 1,
                'metadata': {
                    'source_files': len(sgf_files),
                    'source_file_range': f"{sgf_files[0].name} to {sgf_files[-1].name}",
                    'errors': len(errors),
                    'date_processed': datetime.now().isoformat(),
                    'positions_per_chunk': positions_per_chunk
                }
            }, chunk_file)
            
            saved_files.append(chunk_file)
            logger.info(
                f"  ✅ Chunk {chunk_idx+1}/{num_chunks}: "
                f"{len(chunk_positions):,} positions → {chunk_file.name}"
            )
        
        # Tạo file index để dễ quản lý
        index_file = output_dir / f'{chunk_prefix}_{board_size}x{board_size}_index.json'
        index_data = {
            'board_size': board_size,
            'total_positions': total_positions,
            'total_chunks': num_chunks,
            'positions_per_chunk': positions_per_chunk,
            'chunks': [
                {
                    'chunk_num': i + 1,
                    'filename': f.name,
                    'positions': len(positions[i*positions_per_chunk:(i+1)*positions_per_chunk])
                }
                for i, f in enumerate(saved_files)
            ],
            'source_files': len(sgf_files),
            'date_created': datetime.now().isoformat()
        }
        
        import json
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📋 Index file saved: {index_file.name}")
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("Parsing Summary:")
    logger.info(f"  Total files processed: {len(sgf_files)}")
    logger.info(f"  Files with errors: {len(errors)}")
    for board_size in board_sizes:
        if board_size in filtered_positions:
            num_chunks = (len(filtered_positions[board_size]) + positions_per_chunk - 1) // positions_per_chunk
            logger.info(
                f"  {board_size}x{board_size}: "
                f"{len(filtered_positions[board_size]):,} positions "
                f"→ {num_chunks} chunk file(s)"
            )
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Success rate: {(len(sgf_files) - len(errors)) / len(sgf_files) * 100:.1f}%")
    for board_size in board_sizes:
        if board_size in filtered_positions:
            logger.info(
                f"  {board_size}x{board_size}: "
                f"{len(filtered_positions[board_size]):,} positions"
            )
    logger.info("="*50)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Parse SGF files locally')
    parser.add_argument('--input', type=str, required=True,
                        help='Input directory containing SGF files')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for processed positions')
    parser.add_argument('--year', type=int, default=None,
                        help='Process only files from this year (e.g., 2019)')
    parser.add_argument('--board-sizes', type=int, nargs='+', default=[9, 13, 19],
                        help='Board sizes to process (default: 9 13 19)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of worker processes (default: auto)')
    parser.add_argument('--min-positions', type=int, default=10,
                        help='Minimum positions per game (default: 10)')
    
    args = parser.parse_args()
    
    process_sgf_directory_by_year(
        sgf_dir=args.input,
        output_dir=args.output,
        year=args.year,
        board_sizes=args.board_sizes,
        num_workers=args.workers,
        min_positions_per_game=args.min_positions
    )

