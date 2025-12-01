"""SGF (Smart Game Format) export utility."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def export_sgf(
    moves: List[Dict[str, Any]],
    board_size: int,
    black_player: Optional[str] = None,
    white_player: Optional[str] = None,
    result: Optional[str] = None,
    date: Optional[datetime] = None,
    event: str = "GoGame",
    handicap: int = 0,
    handicap_stones_black: Optional[List[tuple]] = None,
    handicap_stones_white: Optional[List[tuple]] = None,
    komi: Optional[float] = None,
) -> str:
    """Export game moves sang SGF format.
    
    Args:
        moves: List of moves với format {"number": int, "color": "B"|"W", "position": [x, y]|None}
        board_size: Kích thước bàn cờ (9, 13, 19)
        black_player: Tên người chơi đen
        white_player: Tên người chơi trắng
        result: Kết quả ván đấu (e.g., "B+R", "W+R", "B+2.5")
        date: Ngày chơi
        event: Tên sự kiện/tournament
        handicap: Số handicap stones (0 = no handicap)
        handicap_stones_black: List of (x, y) tuples cho Black handicap stones
        handicap_stones_white: List of (x, y) tuples cho White handicap stones (rare)
        komi: Komi value (default: 7.5 for no handicap, 0.5 for handicap)
    
    Returns:
        SGF string
    """
    sgf_lines = ["("]  # Root node
    
    # Game info
    sgf_lines.append(";FF[4]")  # SGF format version
    sgf_lines.append(f";SZ[{board_size}]")  # Board size
    sgf_lines.append(f";EV[{event}]")  # Event
    
    if date:
        sgf_lines.append(f";DT[{date.strftime('%Y-%m-%d')}]")
    else:
        sgf_lines.append(f";DT[{datetime.now().strftime('%Y-%m-%d')}]")
    
    if black_player:
        sgf_lines.append(f";PB[{black_player}]")
    if white_player:
        sgf_lines.append(f";PW[{white_player}]")
    if result:
        sgf_lines.append(f";RE[{result}]")
    
    # Handicap info
    if handicap > 0:
        sgf_lines.append(f";HA[{handicap}]")
        
        # Add handicap stones (Black)
        if handicap_stones_black:
            ab_line = ";AB"
            for x, y in handicap_stones_black:
                # Convert to SGF coordinates (skip 'i')
                sgf_x = chr(ord("a") + x + (1 if x >= 8 else 0))
                sgf_y = chr(ord("a") + y + (1 if y >= 8 else 0))
                ab_line += f"[{sgf_x}{sgf_y}]"
            sgf_lines.append(ab_line)
        
        # Add handicap stones (White) - rare
        if handicap_stones_white:
            aw_line = ";AW"
            for x, y in handicap_stones_white:
                sgf_x = chr(ord("a") + x + (1 if x >= 8 else 0))
                sgf_y = chr(ord("a") + y + (1 if y >= 8 else 0))
                aw_line += f"[{sgf_x}{sgf_y}]"
            sgf_lines.append(aw_line)
    
    # Komi
    if komi is not None:
        sgf_lines.append(f";KM[{komi:.2f}]")
    elif handicap > 0:
        sgf_lines.append(";KM[0.50]")  # Default komi for handicap games
    else:
        sgf_lines.append(";KM[7.50]")  # Default komi for normal games
    
    # Moves
    for move in moves:
        color = move.get("color", "")
        position = move.get("position")
        
        if position and len(position) == 2:
            x, y = position
            # SGF uses letters: a-z for 1-26, but skips 'i'
            # Convert 0-indexed to SGF format (a-z, skip 'i')
            sgf_x = chr(ord("a") + x + (1 if x >= 8 else 0))
            sgf_y = chr(ord("a") + y + (1 if y >= 8 else 0))
            
            if color == "B":
                sgf_lines.append(f";B[{sgf_x}{sgf_y}]")
            elif color == "W":
                sgf_lines.append(f";W[{sgf_x}{sgf_y}]")
        else:
            # Pass move
            if color == "B":
                sgf_lines.append(";B[]")
            elif color == "W":
                sgf_lines.append(";W[]")
    
    sgf_lines.append(")")
    
    return "".join(sgf_lines)


def parse_sgf(sgf_string: str) -> Dict[str, Any]:
    """Parse SGF string thành game data.
    
    Hỗ trợ:
    - Handicap stones (;AB[...], ;AW[...])
    - Handicap number (;HA[n])
    - Starting player thay đổi khi có handicap
    
    Args:
        sgf_string: SGF format string
    
    Returns:
        Dict với game info và moves:
        {
            "board_size": int,
            "moves": [{"number": int, "color": "B"|"W", "position": [x, y]|None}],
            "black_player": str|None,
            "white_player": str|None,
            "result": str|None,
            "date": datetime|None,
            "handicap": int,  # Number of handicap stones
            "handicap_stones_black": List[str],  # SGF coordinates
            "handicap_stones_white": List[str],  # SGF coordinates (rare)
        }
    """
    import re
    from datetime import datetime
    
    result = {
        "board_size": 9,
        "moves": [],
        "black_player": None,
        "white_player": None,
        "result": None,
        "date": None,
        "handicap": 0,
        "handicap_stones_black": [],
        "handicap_stones_white": [],
    }
    
    # Extract properties using regex
    # SGF format: ;PB[PlayerName] or ;B[ab] for moves
    sgf_clean = sgf_string.replace("\n", "").replace(" ", "")
    
    # Extract board size
    sz_match = re.search(r"SZ\[(\d+)\]", sgf_clean)
    if sz_match:
        result["board_size"] = int(sz_match.group(1))
    
    # Extract players
    pb_match = re.search(r"PB\[([^\]]+)\]", sgf_clean)
    if pb_match:
        result["black_player"] = pb_match.group(1)
    
    pw_match = re.search(r"PW\[([^\]]+)\]", sgf_clean)
    if pw_match:
        result["white_player"] = pw_match.group(1)
    
    # Extract result
    re_match = re.search(r"RE\[([^\]]+)\]", sgf_clean)
    if re_match:
        result["result"] = re_match.group(1)
    
    # Extract date
    dt_match = re.search(r"DT\[([^\]]+)\]", sgf_clean)
    if dt_match:
        try:
            date_str = dt_match.group(1)
            # Try parsing date (format: YYYY-MM-DD)
            if len(date_str) >= 10:
                result["date"] = datetime.strptime(date_str[:10], "%Y-%m-%d")
        except Exception:
            pass
    
    # Extract handicap number
    ha_match = re.search(r"HA\[(\d+)\]", sgf_clean)
    if ha_match:
        result["handicap"] = int(ha_match.group(1))
    
    # Extract handicap stones (Black) - format: ;AB[pd][dp]
    # Note: SGF allows multiple AB properties or multiple coordinates in one
    ab_matches = re.findall(r"AB\[([a-z]{2})\]", sgf_clean)
    if ab_matches:
        result["handicap_stones_black"] = ab_matches
    
    # Extract handicap stones (White) - format: ;AW[pd][dp] (rare)
    aw_matches = re.findall(r"AW\[([a-z]{2})\]", sgf_clean)
    if aw_matches:
        result["handicap_stones_white"] = aw_matches
    
    # Extract moves: ;B[ab] or ;W[cd] or ;B[] for pass
    # Skip handicap stones (AB, AW) - they're not moves
    move_pattern = r";([BW])\[([a-z]{0,2})\]"
    moves = re.findall(move_pattern, sgf_clean)
    
    move_number = 1
    for color_char, pos_str in moves:
        color = "B" if color_char == "B" else "W"
        
        if not pos_str:  # Pass move
            result["moves"].append({
                "number": move_number,
                "color": color,
                "position": None,
            })
        else:
            # Convert SGF coordinates to 0-indexed
            # SGF: a=0, b=1, ..., z=25
            # Skip 'i' (no I in Go coordinates)
            if len(pos_str) == 2:
                x = ord(pos_str[0]) - ord("a")
                y = ord(pos_str[1]) - ord("a")
                
                # Adjust for missing 'i' in Go coordinates
                if x >= 8:
                    x -= 1
                if y >= 8:
                    y -= 1
                
                # Validate coordinates
                if 0 <= x < result["board_size"] and 0 <= y < result["board_size"]:
                    result["moves"].append({
                        "number": move_number,
                        "color": color,
                        "position": [x, y],
                    })
        
        move_number += 1
    
    logger.info(
        f"Parsed SGF: {len(result['moves'])} moves, board size: {result['board_size']}, "
        f"handicap: {result['handicap']}"
    )
    return result

