"""
Label Generators cho Multi-task Model theo tài liệu ML_COMPREHENSIVE_GUIDE.md

Các generators này tạo labels cho:
- Threat Detection Head
- Attack Opportunity Head
- Intent Recognition Head
- Position Evaluation Head
"""

import numpy as np
import torch
from typing import Tuple, List, Dict, Optional, Set
from collections import defaultdict


class ThreatLabelGenerator:
    """
    Generate threat labels using rule-based heuristics.
    
    Theo tài liệu ML_COMPREHENSIVE_GUIDE.md (dòng 600-633):
    - Groups with 1 liberty → 1.0 (atari)
    - Groups with 2 liberties → 0.7
    - False eyes → 0.6
    - Cutting points → 0.5
    """
    
    def __init__(self, board_size: int = 19):
        self.board_size = board_size
    
    def find_groups(self, board_state: np.ndarray) -> List[Dict]:
        """
        Tìm tất cả các nhóm quân trên bàn cờ.
        Tối ưu: Dùng scipy.ndimage.label nếu có, fallback về DFS.
        
        Returns:
            List of groups, mỗi group có:
            - 'color': 1 (black) hoặc 2 (white)
            - 'positions': List of (x, y) tuples
            - 'liberties': Số liberties
        """
        # Thử dùng scipy.ndimage.label để vectorize (nhanh hơn)
        # Nhưng chỉ dùng nếu scipy có sẵn và board không quá lớn
        try:
            from scipy import ndimage
            # Chỉ dùng scipy cho board 19x19 (tối ưu nhất)
            if self.board_size == 19:
                return self._find_groups_vectorized(board_state, ndimage)
            else:
                # Board nhỏ hơn, DFS đủ nhanh
                return self._find_groups_dfs(board_state)
        except ImportError:
            # Fallback về DFS (original code - đơn giản và nhanh)
            return self._find_groups_dfs(board_state)
    
    def _find_groups_vectorized(self, board_state: np.ndarray, ndimage) -> List[Dict]:
        """Vectorized version using scipy.ndimage.label (optimized)"""
        groups = []
        
        # Process black và white riêng
        for color in [1, 2]:
            mask = (board_state == color)
            if not np.any(mask):
                continue
            
            # Label connected components - dùng structure đơn giản hơn
            labeled, num_groups = ndimage.label(mask)
            
            # Process each group - tối ưu: dùng np.where trực tiếp
            for group_id in range(1, num_groups + 1):
                # Tối ưu: dùng np.where trực tiếp (nhanh hơn zip)
                y_coords, x_coords = np.where(labeled == group_id)
                # Convert to list of tuples (nhanh hơn)
                group_positions = [(int(x), int(y)) for x, y in zip(x_coords, y_coords)]
                
                # Tính liberties
                liberties = self._count_group_liberties(group_positions, board_state)
                
                groups.append({
                    'color': color,
                    'positions': group_positions,
                    'liberties': liberties
                })
        
        return groups
    
    def _find_groups_dfs(self, board_state: np.ndarray) -> List[Dict]:
        """DFS version (optimized)"""
        visited = set()
        groups = []
        
        # Tối ưu: chỉ scan các vị trí có quân (không phải empty)
        # Tạo list các vị trí có quân trước
        stone_positions = []
        for y in range(self.board_size):
            for x in range(self.board_size):
                if board_state[y, x] != 0:
                    stone_positions.append((x, y))
        
        # DFS chỉ cho các vị trí có quân
        for start_x, start_y in stone_positions:
            if (start_x, start_y) in visited:
                continue
            
            color = board_state[start_y, start_x]
            group_positions = []
            stack = [(start_x, start_y)]
            
            # DFS để tìm tất cả quân trong nhóm
            while stack:
                cx, cy = stack.pop()
                if (cx, cy) in visited:
                    continue
                
                visited.add((cx, cy))
                group_positions.append((cx, cy))
                
                # Check neighbors (tối ưu: chỉ check 4 hướng)
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = cx + dx, cy + dy
                    if (0 <= nx < self.board_size and 
                        0 <= ny < self.board_size and
                        board_state[ny, nx] == color and
                        (nx, ny) not in visited):
                        stack.append((nx, ny))
            
            # Tính liberties cho nhóm
            liberties = self._count_group_liberties(group_positions, board_state)
            
            groups.append({
                'color': color,
                'positions': group_positions,
                'liberties': liberties
            })
        
        return groups
    
    def _count_group_liberties(self, group_positions: List[Tuple[int, int]], 
                               board_state: np.ndarray) -> int:
        """Đếm số liberties của một nhóm quân (optimized)"""
        if not group_positions:
            return 0
        
        # Simple và nhanh hơn: dùng set để track unique liberties
        liberty_set = set()
        
        # Neighbors offsets
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for x, y in group_positions:
            for dx, dy in neighbors:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.board_size and 
                    0 <= ny < self.board_size and
                    board_state[ny, nx] == 0):
                    liberty_set.add((nx, ny))
        
        return len(liberty_set)
    
    def detect_false_eyes(self, board_state: np.ndarray) -> List[Tuple[int, int]]:
        """
        Phát hiện mắt giả (false eyes) - Vectorized version.
        
        Simplified: Mắt giả là empty point được bao quanh bởi cùng một màu
        nhưng có thể bị đối thủ chiếm.
        Tối ưu: Vectorized với NumPy để tăng tốc.
        """
        false_eyes = []
        
        try:
            # Đảm bảo board_state là numpy array
            if not isinstance(board_state, np.ndarray):
                board_state = np.array(board_state, dtype=np.int8)
            
            # Tối ưu: Chỉ scan empty cells (thường ít hơn 361 cells)
            empty_mask = (board_state == 0)
            empty_y, empty_x = np.where(empty_mask)
            
            if len(empty_y) == 0:
                return false_eyes
            
            # Vectorized neighbor checking
            for idx in range(len(empty_y)):
                x, y = int(empty_x[idx]), int(empty_y[idx])
                
                # Get neighbors (vectorized)
                neighbors = []
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                        neighbors.append(int(board_state[ny, nx]))
                
                # Nếu tất cả neighbors là cùng một màu (không phải empty)
                non_empty = [n for n in neighbors if n != 0]
                if len(non_empty) >= 3 and len(set(non_empty)) == 1:
                    # Có thể là false eye (simplified heuristic)
                    false_eyes.append((x, y))
        except Exception as e:
            # Fallback: return empty list nếu có lỗi
            return []
        
        return false_eyes
    
    def detect_cutting_points(self, board_state: np.ndarray) -> List[Tuple[int, int]]:
        """
        Phát hiện các điểm cắt (cutting points) - Vectorized version.
        
        Simplified: Điểm cắt là empty point nằm giữa 2 nhóm quân đối thủ
        có thể cắt đứt kết nối.
        Tối ưu: Vectorized với NumPy để tăng tốc.
        """
        cutting_points = []
        
        try:
            # Đảm bảo board_state là numpy array
            if not isinstance(board_state, np.ndarray):
                board_state = np.array(board_state, dtype=np.int8)
            
            # Tối ưu: Chỉ scan empty cells (thường ít hơn 361 cells)
            empty_mask = (board_state == 0)
            empty_y, empty_x = np.where(empty_mask)
            
            if len(empty_y) == 0:
                return cutting_points
            
            # Vectorized neighbor checking
            for idx in range(len(empty_y)):
                x, y = int(empty_x[idx]), int(empty_y[idx])
                
                # Get neighbors (vectorized)
                neighbors = []
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                        neighbors.append(int(board_state[ny, nx]))
                
                # Nếu có cả black (1) và white (2) neighbors
                has_black = 1 in neighbors
                has_white = 2 in neighbors
                
                if has_black and has_white:
                    cutting_points.append((x, y))
        except Exception as e:
            # Fallback: return empty list nếu có lỗi
            return []
        
        return cutting_points
    
    def generate_threat_map(self, board_state: np.ndarray, 
                           current_player: str, groups: Optional[List[Dict]] = None) -> torch.Tensor:
        """
        Generate threat map.
        
        Args:
            board_state: numpy array [board_size, board_size]
            current_player: 'B' hoặc 'W'
        
        Returns:
            Tensor [board_size, board_size] với values 0.0-1.0
        """
        threat_map = np.zeros((self.board_size, self.board_size), dtype=np.float32)
        
        # Determine opponent color
        opponent_color = 2 if current_player == 'B' else 1
        
        # Find all groups (reuse nếu đã có)
        if groups is None:
            groups = self.find_groups(board_state)
        
        # Tối ưu: chỉ tính false_eyes và cutting_points nếu cần
        # Rule 1: Groups with 1 liberty → 1.0 (atari) - ưu tiên cao nhất
        for group in groups:
            if group['color'] == opponent_color and group['liberties'] == 1:
                for x, y in group['positions']:
                    threat_map[y, x] = 1.0
        
        # Rule 2: Groups with 2 liberties → 0.7
        for group in groups:
            if group['color'] == opponent_color and group['liberties'] == 2:
                for x, y in group['positions']:
                    if threat_map[y, x] < 0.7:
                        threat_map[y, x] = 0.7
        
        # Rule 3: False eyes → 0.6 (theo tài liệu chính thức)
        false_eyes = self.detect_false_eyes(board_state)
        for x, y in false_eyes:
            if threat_map[y, x] < 0.6:
                threat_map[y, x] = 0.6
        
        # Rule 4: Cutting points → 0.5 (theo tài liệu chính thức)
        cutting_points = self.detect_cutting_points(board_state)
        for x, y in cutting_points:
            if threat_map[y, x] < 0.5:
                threat_map[y, x] = 0.5
        
        return torch.from_numpy(threat_map)


class AttackLabelGenerator:
    """
    Generate attack opportunity labels.
    
    Theo tài liệu ML_COMPREHENSIVE_GUIDE.md (dòng 635-663):
    - Opponent in atari → 1.0
    - Can cut → 0.8
    - Invasion points → 0.6
    - Ladder works → 0.7
    """
    
    def __init__(self, board_size: int = 19):
        self.board_size = board_size
        self.threat_gen = ThreatLabelGenerator(board_size)
    
    def find_opponent_groups(self, board_state: np.ndarray, 
                            current_player: str, groups: Optional[List[Dict]] = None) -> List[Dict]:
        """Tìm tất cả nhóm quân đối thủ (có thể reuse groups nếu đã tính)"""
        opponent_color = 2 if current_player == 'B' else 1
        if groups is None:
            all_groups = self.threat_gen.find_groups(board_state)
        else:
            all_groups = groups
        return [g for g in all_groups if g['color'] == opponent_color]
    
    def find_cut_opportunities(self, board_state: np.ndarray,
                               current_player: str) -> List[Tuple[int, int]]:
        """Tìm các điểm có thể cắt đứt đối thủ"""
        cutting_points = self.threat_gen.detect_cutting_points(board_state)
        # Filter: chỉ những điểm có thể cắt đối thủ
        opponent_color = 2 if current_player == 'B' else 1
        valid_cuts = []
        
        for x, y in cutting_points:
            # Check if placing stone here would cut opponent
            # Simplified: nếu neighbors có opponent groups
            neighbors = []
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                    neighbors.append(board_state[ny, nx])
            
            if opponent_color in neighbors:
                valid_cuts.append((x, y))
        
        return valid_cuts
    
    def find_invasion_points(self, board_state: np.ndarray,
                           current_player: str) -> List[Tuple[int, int]]:
        """
        Tìm các điểm xâm nhập vào lãnh thổ đối thủ - Vectorized version.
        
        Simplified: Empty points gần nhóm quân đối thủ lớn.
        Tối ưu: Vectorized với NumPy để tăng tốc.
        """
        invasion_points = []
        opponent_color = 2 if current_player == 'B' else 1
        
        try:
            # Đảm bảo board_state là numpy array
            if not isinstance(board_state, np.ndarray):
                board_state = np.array(board_state, dtype=np.int8)
            
            # Tối ưu: Chỉ scan empty cells (thường ít hơn 361 cells)
            empty_mask = (board_state == 0)
            empty_y, empty_x = np.where(empty_mask)
            
            if len(empty_y) == 0:
                return invasion_points
            
            # Vectorized neighbor checking
            for idx in range(len(empty_y)):
                x, y = int(empty_x[idx]), int(empty_y[idx])
                
                # Check if near opponent territory
                opponent_neighbors = 0
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < self.board_size and 
                        0 <= ny < self.board_size and
                        int(board_state[ny, nx]) == opponent_color):
                        opponent_neighbors += 1
                
                # Nếu có 2+ opponent neighbors → invasion point
                if opponent_neighbors >= 2:
                    invasion_points.append((x, y))
        except Exception as e:
            # Fallback: return empty list nếu có lỗi
            return []
        
        return invasion_points
    
    def find_working_ladders(self, board_state: np.ndarray,
                           current_player: str, groups: Optional[List[Dict]] = None) -> List[Tuple[int, int]]:
        """
        Tìm các ladder moves (nước đi có thể bắt quân bằng ladder) - Optimized version.
        
        Simplified: Nếu đối thủ có nhóm 1 liberty và có thể bắt bằng ladder.
        Tối ưu: Reuse groups nếu có, dùng set để tránh duplicates.
        """
        ladder_moves_set = set()
        opponent_groups = self.find_opponent_groups(board_state, current_player, groups)
        
        for group in opponent_groups:
            if group['liberties'] == 1:
                # Tìm liberty của nhóm này
                for x, y in group['positions']:
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < self.board_size and 
                            0 <= ny < self.board_size and
                            board_state[ny, nx] == 0):
                            # Simplified: coi như ladder move
                            ladder_moves_set.add((nx, ny))
        
        return list(ladder_moves_set)
    
    def generate_attack_map(self, board_state: np.ndarray,
                           current_player: str, groups: Optional[List[Dict]] = None) -> torch.Tensor:
        """
        Generate attack opportunity map.
        
        Args:
            board_state: numpy array [board_size, board_size]
            current_player: 'B' hoặc 'W'
            groups: Optional - reuse groups nếu đã tính (tối ưu)
        
        Returns:
            Tensor [board_size, board_size] với values 0.0-1.0
        """
        attack_map = np.zeros((self.board_size, self.board_size), dtype=np.float32)
        
        # Reuse groups nếu có (tối ưu quan trọng!)
        opponent_groups = self.find_opponent_groups(board_state, current_player, groups)
        
        # Rule 1: Opponent in atari → 1.0
        for group in opponent_groups:
            if group['liberties'] == 1:
                # Mark the capturing move (liberty point)
                for x, y in group['positions']:
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < self.board_size and 
                            0 <= ny < self.board_size and
                            board_state[ny, nx] == 0):
                            attack_map[ny, nx] = 1.0
        
        # Rule 2: Can cut → 0.8 (theo tài liệu chính thức)
        cut_points = self.find_cut_opportunities(board_state, current_player)
        for x, y in cut_points:
            if attack_map[y, x] < 0.8:
                attack_map[y, x] = 0.8
        
        # Rule 3: Invasion points → 0.6 (theo tài liệu chính thức)
        invasion_points = self.find_invasion_points(board_state, current_player)
        for x, y in invasion_points:
            if attack_map[y, x] < 0.6:
                attack_map[y, x] = 0.6
        
        # Rule 4: Ladder works → 0.7 (theo tài liệu chính thức)
        ladder_moves = self.find_working_ladders(board_state, current_player, groups=groups)
        for x, y in ladder_moves:
            if attack_map[y, x] < 0.7:
                attack_map[y, x] = 0.7
        
        return torch.from_numpy(attack_map)


class IntentLabelGenerator:
    """
    Generate intent labels from move sequences.
    
    Theo tài liệu ML_COMPREHENSIVE_GUIDE.md (dòng 666-712):
    Intent classes: territory, attack, defense, connection, cut
    """
    
    INTENT_PATTERNS = {
        'territory': [
            'enclosure_3_3',
            'side_extension',
            'shimari'
        ],
        'attack': [
            'attach',
            'cut',
            'peep',
            'atari'
        ],
        'defense': [
            'connect',
            'add_eye_space',
            'escape'
        ],
        'connection': [
            'kosumi',
            'keima',
            'one_point_jump'
        ],
        'cut': [
            'wedge',
            'diagonal_cut',
            'contact_cut'
        ]
    }
    
    def __init__(self, board_size: int = 19):
        self.board_size = board_size
    
    def matches_pattern(self, move: Optional[Tuple[int, int]], 
                       prev_moves: List[Tuple[int, int]],
                       pattern: str) -> bool:
        """
        Kiểm tra xem move có match với pattern không.
        
        Simplified heuristic-based matching.
        """
        if move is None:
            return False
        
        x, y = move
        
        # Simplified pattern matching
        if pattern == 'enclosure_3_3':
            # 3-3 point (corner enclosure)
            return (x == 2 and y == 2) or (x == self.board_size - 3 and y == 2) or \
                   (x == 2 and y == self.board_size - 3) or \
                   (x == self.board_size - 3 and y == self.board_size - 3)
        
        elif pattern == 'side_extension':
            # Extension along side
            return x == 0 or x == self.board_size - 1 or y == 0 or y == self.board_size - 1
        
        elif pattern == 'attach':
            # Attach to opponent stone
            return True  # Simplified
        
        elif pattern == 'cut':
            # Cutting move
            return True  # Simplified
        
        # Default: heuristic analysis
        return False
    
    def heuristic_intent_analysis(self, board_state: np.ndarray,
                                 move: Optional[Tuple[int, int]],
                                 current_player: str) -> Tuple[str, float]:
        """
        Phân tích heuristic để xác định intent.
        
        Returns:
            (intent_type, confidence) tuple
        """
        if move is None:
            return 'defense', 0.5  # Pass move thường là defense
        
        x, y = move
        
        # Simplified heuristic
        # Check if near opponent stones → attack
        opponent_color = 2 if current_player == 'B' else 1
        opponent_neighbors = 0
        
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if (0 <= nx < self.board_size and 
                0 <= ny < self.board_size and
                board_state[ny, nx] == opponent_color):
                opponent_neighbors += 1
        
        if opponent_neighbors >= 2:
            return 'attack', 0.7
        elif opponent_neighbors == 1:
            return 'cut', 0.6
        else:
            return 'territory', 0.5
    
    def generate_intent_label(self, board_state: np.ndarray,
                             move: Optional[Tuple[int, int]],
                             prev_moves: List[Tuple[int, int]],
                             current_player: str) -> Dict:
        """
        Generate intent label.
        
        Args:
            board_state: numpy array [board_size, board_size]
            move: (x, y) tuple hoặc None
            prev_moves: List of previous moves
            current_player: 'B' hoặc 'W'
        
        Returns:
            Dict với:
            - 'type': str (one of 5 classes)
            - 'confidence': float (0.0-1.0)
            - 'region': List of positions
        """
        # Check against known patterns
        for intent_type, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if self.matches_pattern(move, prev_moves, pattern):
                    # Determine region (simplified: just the move position)
                    region = [move] if move else []
                    return {
                        'type': intent_type,
                        'confidence': 0.9,
                        'region': region
                    }
        
        # Fallback: heuristic analysis
        intent_type, confidence = self.heuristic_intent_analysis(
            board_state, move, current_player
        )
        
        region = [move] if move else []
        return {
            'type': intent_type,
            'confidence': confidence,
            'region': region
        }


class EvaluationLabelGenerator:
    """
    Generate evaluation labels (win probability, territory, influence).
    
    Theo tài liệu ML_COMPREHENSIVE_GUIDE.md (dòng 271-290):
    - win_probability: 0.0-1.0
    - territory_map: Tensor[board_size, board_size]
    - influence_map: Tensor[board_size, board_size]
    """
    
    def __init__(self, board_size: int = 19):
        self.board_size = board_size
    
    def generate_territory_map(self, board_state: np.ndarray,
                              current_player: str) -> torch.Tensor:
        """
        Generate territory map (simplified).
        
        Simplified: Territory = empty points gần stones của current player.
        """
        territory_map = np.zeros((self.board_size, self.board_size), dtype=np.float32)
        player_color = 1 if current_player == 'B' else 2
        
        # Tối ưu: Tìm tất cả stones của player trước
        player_stones = []
        for y in range(self.board_size):
            for x in range(self.board_size):
                if board_state[y, x] == player_color:
                    territory_map[y, x] = 1.0
                    player_stones.append((x, y))
        
        # Chỉ tính territory cho empty points (nếu có player stones)
        if player_stones:
            for y in range(self.board_size):
                for x in range(self.board_size):
                    if board_state[y, x] == 0:
                        # Check distance to player stones (chỉ check trong player_stones)
                        min_dist = float('inf')
                        for sx, sy in player_stones:
                            dist = abs(x - sx) + abs(y - sy)
                            min_dist = min(min_dist, dist)
                        
                        # Inverse distance (closer = higher territory value)
                        if min_dist < float('inf'):
                            territory_map[y, x] = max(0.0, 1.0 - min_dist / 5.0)
        
        return torch.from_numpy(territory_map)
    
    def generate_influence_map(self, board_state: np.ndarray,
                              current_player: str) -> torch.Tensor:
        """
        Generate influence map (simplified).
        
        Simplified: Influence = spread từ stones của current player.
        """
        influence_map = np.zeros((self.board_size, self.board_size), dtype=np.float32)
        player_color = 1 if current_player == 'B' else 2
        
        # Tối ưu: Tìm tất cả stones của player trước
        player_stones = []
        for y in range(self.board_size):
            for x in range(self.board_size):
                if board_state[y, x] == player_color:
                    influence_map[y, x] = 1.0
                    player_stones.append((x, y))
        
        # Chỉ tính influence cho empty points (nếu có player stones)
        if player_stones:
            for y in range(self.board_size):
                for x in range(self.board_size):
                    if board_state[y, x] == 0:
                        # Calculate influence from all player stones (chỉ check trong player_stones)
                        total_influence = 0.0
                        for sx, sy in player_stones:
                            dist = ((x - sx) ** 2 + (y - sy) ** 2) ** 0.5
                            if dist > 0:
                                total_influence += 1.0 / (1.0 + dist)
                        
                        influence_map[y, x] = min(1.0, total_influence)
        
        return torch.from_numpy(influence_map)
    
    def generate_evaluation(self, board_state: np.ndarray,
                           current_player: str,
                           winner: Optional[str],
                           game_result: Optional[str] = None) -> Dict:
        """
        Generate evaluation labels.
        
        Args:
            board_state: numpy array [board_size, board_size]
            current_player: 'B' hoặc 'W'
            winner: 'B', 'W', 'DRAW', hoặc None
            game_result: String như "B+12.5"
        
        Returns:
            Dict với:
            - 'win_probability': float (0.0-1.0)
            - 'territory_map': Tensor[board_size, board_size]
            - 'influence_map': Tensor[board_size, board_size]
        """
        # Calculate win probability
        if winner is None:
            win_probability = 0.5
        elif winner == 'DRAW':
            win_probability = 0.5
        else:
            win_probability = 1.0 if winner == current_player else 0.0
        
        # Generate territory and influence maps
        territory_map = self.generate_territory_map(board_state, current_player)
        influence_map = self.generate_influence_map(board_state, current_player)
        
        return {
            'win_probability': win_probability,
            'territory_map': territory_map,
            'influence_map': influence_map
        }

