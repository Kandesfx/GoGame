"""
Script để parse SGF files thành positions trên Colab.

Copy script này vào Colab Cell để parse SGF files.
"""

from sgfmill import sgf
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm


def parse_sgf_coord(sgf_coord, board_size):
    """Convert SGF coordinate to (x, y)"""
    if not sgf_coord or len(sgf_coord) < 2 or sgf_coord == 'tt':
        return None, None  # Pass move
    
    x = ord(sgf_coord[0]) - ord('a')
    y = ord(sgf_coord[1]) - ord('a')
    
    # Skip 'i' (no I in Go coordinates)
    if x >= 8:
        x -= 1
    if y >= 8:
        y -= 1
    
    if x < 0 or x >= board_size or y < 0 or y >= board_size:
        return None, None
    
    return x, y


def parse_sgf_file(sgf_path):
    """Parse 1 SGF file và extract tất cả positions
    
    Hỗ trợ:
    - Handicap stones (;AB[...])
    - White stones đặt sẵn (;AW[...])
    - Starting player thay đổi khi có handicap
    - Result parsing với nhiều format
    """
    try:
        with open(sgf_path, 'rb') as f:
            sgf_data = f.read()
        
        # Parse SGF using sgfmill
        game = sgf.Sgf_game.from_bytes(sgf_data)
        
        # Extract metadata
        root = game.get_root()
        board_size = game.get_size()
        
        # Get result property (RE = Result) - từ ThuanBranch
        result = ''
        winner = None
        try:
            result_prop = root.get_raw('RE')
            if result_prop and len(result_prop) > 0:
                # result_prop is a list, get first element
                result_value = result_prop[0]
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
            pass
        
        # Extract handicap info - từ HEAD (master)
        try:
            handicap = int(root.properties.get('HA', ['0'])[0])  # Handicap number
        except (ValueError, TypeError):
            handicap = 0
        
        handicap_stones_black = root.properties.get('AB', [])  # Black handicap stones
        handicap_stones_white = root.properties.get('AW', [])  # White handicap stones (rare)
        
        # Initialize board
        board = np.zeros((board_size, board_size), dtype=np.int8)
        
        # Place handicap stones (Black stones đặt sẵn)
        if handicap_stones_black:
            for stone_coord in handicap_stones_black:
                x, y = parse_sgf_coord(stone_coord, board_size)
                if x is not None and y is not None:
                    board[y, x] = 1  # Black = 1
        
        # Place white handicap stones (nếu có, rất hiếm)
        if handicap_stones_white:
            for stone_coord in handicap_stones_white:
                x, y = parse_sgf_coord(stone_coord, board_size)
                if x is not None and y is not None:
                    board[y, x] = 2  # White = 2
        
        # Determine starting player
        # Nếu có handicap, White đi trước (không phải Black)
        current_player = 'W' if handicap > 0 else 'B'
        
        # Extract moves by traversing the main line - từ ThuanBranch
        positions = []
        
        # Traverse main line (first child of each node)
        node = root
        move_number = 0
        
        while True:
            # Get children
            children = list(node)
            if not children:
                break
            
            # Follow main line (first child)
            node = children[0]
            
            # Get move from this node - từ ThuanBranch
            move = node.get_move()
            if move:
                color, move_coord = move
                if move_coord is not None:
                    # Normal move - move_coord is (x, y) tuple
                    x, y = move_coord
                    
                    # Validate move (không được đặt vào ô đã có quân)
                    if board[y, x] != 0:
                        # Skip invalid move
                        continue
                    
                    # Save position BEFORE move
                    positions.append({
                        'board_state': board.copy(),
                        'move': (x, y),  # Normal move as tuple
                        'current_player': current_player,
                        'move_number': move_number,
                        'board_size': board_size,
                        'game_result': result,
                        'winner': winner,
                        'handicap': handicap  # Lưu thông tin handicap để filter sau
                    })
                    
                    # Apply move (simplified - không xử lý captures, ko, etc.)
                    board[y, x] = 1 if color == 'b' else 2
                    move_number += 1
                else:
                    # Pass move - move_coord is None
                    positions.append({
                        'board_state': board.copy(),
                        'move': None,  # Pass move marked as None
                        'current_player': current_player,
                        'move_number': move_number,
                        'board_size': board_size,
                        'game_result': result,
                        'winner': winner,
                        'handicap': handicap
                    })
                    move_number += 1
                    # Không cần apply move cho pass
                
                current_player = 'W' if current_player == 'B' else 'B'
        
        return positions
        
    except Exception as e:
        print(f"Error parsing {sgf_path}: {e}")
        return []


def process_sgf_directory(sgf_dir, output_dir, board_sizes=[9, 13, 19]):
    """
    Process tất cả SGF files trong thư mục
    
    Args:
        sgf_dir: Path to directory chứa SGF files
        output_dir: Path to directory để lưu positions
        board_sizes: List các board sizes cần xử lý
    """
    sgf_dir = Path(sgf_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sgf_files = list(sgf_dir.glob('*.sgf'))
    
    if not sgf_files:
        print(f"⚠️  No SGF files found in {sgf_dir}")
        return
    
    print(f"📊 Found {len(sgf_files)} SGF files")
    print(f"📁 Processing from: {sgf_dir}")
    print(f"💾 Output to: {output_dir}")
    
    all_positions = {size: [] for size in board_sizes}
    
    for sgf_file in tqdm(sgf_files, desc="Parsing SGF"):
        positions = parse_sgf_file(sgf_file)
        
        for pos in positions:
            board_size = pos['board_size']
            if board_size in all_positions:
                all_positions[board_size].append(pos)
    
    # Save positions theo board size
    for board_size in board_sizes:
        if all_positions[board_size]:
            output_file = output_dir / f'positions_{board_size}x{board_size}.pt'
            torch.save({
                'positions': all_positions[board_size],
                'board_size': board_size,
                'total': len(all_positions[board_size])
            }, output_file)
            print(f"✅ Saved {len(all_positions[board_size]):,} positions for {board_size}x{board_size}")
        else:
            print(f"⚠️  No positions found for {board_size}x{board_size}")
    
    print("\n✅ Parsing complete!")
    return all_positions


if __name__ == "__main__":
    # Example usage trên Colab:
    # WORK_DIR = Path('/content/drive/MyDrive/GoGame_ML')
    # process_sgf_directory(
    #     sgf_dir=WORK_DIR / 'raw_sgf',
    #     output_dir=WORK_DIR / 'processed',
    #     board_sizes=[9, 13, 19]
    # )
    pass
