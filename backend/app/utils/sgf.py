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
    
    # Moves
    for move in moves:
        color = move.get("color", "")
        position = move.get("position")
        
        if position and len(position) == 2:
            x, y = position
            # SGF uses letters: a-z for 1-26
            # x, y are 0-indexed in our system, SGF uses 1-indexed (a=0, b=1, ...)
            # Convert 0-indexed to SGF format (a-z)
            sgf_x = chr(ord("a") + x) if x < 26 else chr(ord("A") + x - 26)
            sgf_y = chr(ord("a") + y) if y < 26 else chr(ord("A") + y - 26)
            
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
    
    # Extract moves: ;B[ab] or ;W[cd] or ;B[] for pass
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
            if len(pos_str) == 2:
                x = ord(pos_str[0]) - ord("a")
                y = ord(pos_str[1]) - ord("a")
                result["moves"].append({
                    "number": move_number,
                    "color": color,
                    "position": [x, y],
                })
        
        move_number += 1
    
    logger.info(f"Parsed SGF: {len(result['moves'])} moves, board size: {result['board_size']}")
    return result

