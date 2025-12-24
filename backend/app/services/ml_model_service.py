"""
Service để load và sử dụng ML model đã train cho game AI.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Try to import ML dependencies (optional)
# Không báo warning - vì có thể chưa cần ML features
try:
    import numpy as np
    import torch
    _NUMPY_AVAILABLE = True
    _TORCH_AVAILABLE = True
except ImportError:
    # Dependencies chưa được cài - không báo warning, chỉ set flag
    np = None
    torch = None
    _NUMPY_AVAILABLE = False
    _TORCH_AVAILABLE = False

# Thêm src/ml vào path để import models
_project_root = Path(__file__).parent.parent.parent.parent
_src_ml_path = _project_root / "src" / "ml"
if str(_src_ml_path) not in sys.path:
    sys.path.insert(0, str(_src_ml_path))

# Try to import ML model classes (optional - chỉ khi có dependencies)
_ML_MODELS_AVAILABLE = False
PolicyNetwork = None
ValueNetwork = None
PolicyConfig = None
ValueConfig = None

if _NUMPY_AVAILABLE and _TORCH_AVAILABLE:
    try:
        from policy_network import PolicyNetwork, PolicyConfig
        from value_network import ValueNetwork, ValueConfig
        _ML_MODELS_AVAILABLE = True
    except ImportError:
        # Model files chưa có - không báo warning, chỉ set flag
        PolicyNetwork = None
        ValueNetwork = None
        PolicyConfig = None
        ValueConfig = None
        _ML_MODELS_AVAILABLE = False


def get_liberties_simple(board_state, x: int, y: int, board_size: int) -> int:
    """Tính số liberties của một quân cờ tại (x, y)."""
    if not _NUMPY_AVAILABLE:
        raise ImportError("numpy is required for ML model features")
    
    if board_state[y, x] == 0:
        return 0
    
    liberties = 0
    # Check 4 neighbors
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < board_size and 0 <= ny < board_size:
            if board_state[ny, nx] == 0:
                liberties += 1
    
    return liberties


def board_to_features_17_planes(
    board_state: np.ndarray,
    current_player: str,
    move_history: Optional[list] = None,
    board_size: int = 9
) -> torch.Tensor:
    """
    Convert board state thành 17-plane tensor.
    
    Args:
        board_state: numpy array [board_size, board_size]
                    0 = empty, 1 = black, 2 = white
        current_player: 'B' hoặc 'W'
        move_history: List of (x, y) tuples cho last 4 moves
        board_size: Kích thước bàn cờ
    
    Returns:
        Tensor [17, board_size, board_size]
    """
    features = torch.zeros((17, board_size, board_size), dtype=torch.float32)
    
    # Plane 0: Current player stones
    # Plane 1: Opponent stones
    if current_player == 'B':
        features[0] = torch.from_numpy((board_state == 1).astype(np.float32))
        features[1] = torch.from_numpy((board_state == 2).astype(np.float32))
    else:  # White
        features[0] = torch.from_numpy((board_state == 2).astype(np.float32))
        features[1] = torch.from_numpy((board_state == 1).astype(np.float32))
    
    # Plane 2-7: Liberty counts
    for y in range(board_size):
        for x in range(board_size):
            if board_state[y, x] == 0:
                continue
            
            # Determine if this is current player's stone
            is_current = (
                (current_player == 'B' and board_state[y, x] == 1) or
                (current_player == 'W' and board_state[y, x] == 2)
            )
            
            if is_current:
                liberties = get_liberties_simple(board_state, x, y, board_size)
                if liberties == 1:
                    features[2, y, x] = 1.0
                elif liberties == 2:
                    features[4, y, x] = 1.0
                elif liberties >= 3:
                    features[6, y, x] = 1.0
            else:
                # Opponent stones
                liberties = get_liberties_simple(board_state, x, y, board_size)
                if liberties == 1:
                    features[3, y, x] = 1.0
                elif liberties == 2:
                    features[5, y, x] = 1.0
                elif liberties >= 3:
                    features[7, y, x] = 1.0
    
    # Plane 8-15: Move history (last 4 moves)
    if move_history:
        for i, (mx, my) in enumerate(move_history[-4:]):  # Last 4 moves only
            if 0 <= mx < board_size and 0 <= my < board_size:
                features[8 + i * 2, my, mx] = 1.0  # X coordinate
                features[8 + i * 2 + 1, my, mx] = 1.0  # Y coordinate
    
    # Plane 16: Turn indicator (1 = Black, 0 = White)
    features[16].fill_(1.0 if current_player == 'B' else 0.0)
    
    return features


def board_position_to_numpy(
    board_position: dict,
    board_size: int
) -> np.ndarray:
    """
    Convert board_position dict (từ MongoDB) thành numpy array.
    
    Args:
        board_position: Dict với keys như "x,y" và values "B" hoặc "W"
        board_size: Kích thước bàn cờ
    
    Returns:
        numpy array [board_size, board_size]
        0 = empty, 1 = black, 2 = white
    """
    board_state = np.zeros((board_size, board_size), dtype=np.int32)
    
    for key, color in board_position.items():
        try:
            x, y = map(int, key.split(','))
            if 0 <= x < board_size and 0 <= y < board_size:
                if color == 'B':
                    board_state[y, x] = 1
                elif color == 'W':
                    board_state[y, x] = 2
        except (ValueError, IndexError):
            continue
    
    return board_state


class MLModelService:
    """Service để load và sử dụng ML model."""
    
    def __init__(self, checkpoint_path: Optional[str] = None, device: str = 'cpu'):
        """
        Args:
            checkpoint_path: Đường dẫn đến checkpoint file. Nếu None, sẽ tìm trong backend/models/ hoặc checkpoints/
            device: 'cpu' hoặc 'cuda'
        """
        if not _NUMPY_AVAILABLE or not _TORCH_AVAILABLE:
            raise ImportError("ML dependencies not available. Please install numpy and torch: pip install numpy torch")
        
        if not _ML_MODELS_AVAILABLE:
            raise ImportError("ML models not available. Please ensure model files are installed.")
        
        self.device = torch.device(device)
        self.policy_net: Optional[PolicyNetwork] = None
        self.value_net: Optional[ValueNetwork] = None
        self.board_size: Optional[int] = None
        self._loaded = False
        
        # Tìm checkpoint path
        if checkpoint_path is None:
            checkpoint_path = self._find_checkpoint()
        
        if checkpoint_path:
            self.load_model(checkpoint_path)
    
    def _find_checkpoint(self) -> Optional[str]:
        """
        Tìm checkpoint file trong thư mục models/.
        
        Tìm theo thứ tự:
        1. backend/models/ (cho deploy)
        2. checkpoints/ (backward compatibility)
        """
        # Ưu tiên: backend/models/ (cho deploy)
        models_dir = Path(__file__).parent.parent.parent / "models"
        checkpoints_dir = _project_root / "checkpoints"
        
        # Danh sách thư mục để tìm (ưu tiên backend/models trước)
        search_dirs = [
            ("backend/models", models_dir),
            ("checkpoints", checkpoints_dir),  # Backward compatibility
        ]
        
        # Ưu tiên: final_model.pt > best_model.pt > các file khác
        for filename in ["final_model.pt", "best_model.pt"]:
            for dir_name, dir_path in search_dirs:
                checkpoint_path = dir_path / filename
                if checkpoint_path.exists():
                    logger.info(f"Found checkpoint in {dir_name}: {checkpoint_path}")
                    return str(checkpoint_path)
        
        # Tìm bất kỳ file .pt nào
        for dir_name, dir_path in search_dirs:
            pt_files = list(dir_path.glob("*.pt"))
            if pt_files:
                checkpoint_path = pt_files[0]
                logger.info(f"Found checkpoint in {dir_name}: {checkpoint_path}")
                return str(checkpoint_path)
        
        logger.warning(f"No checkpoint found in {models_dir} or {checkpoints_dir}")
        return None
    
    def load_model(self, checkpoint_path: str) -> bool:
        """
        Load model từ checkpoint.
        
        Returns:
            True nếu load thành công, False nếu có lỗi
        """
        try:
            checkpoint_path_obj = Path(checkpoint_path)
            if not checkpoint_path_obj.exists():
                logger.error(f"Checkpoint not found: {checkpoint_path}")
                return False
            
            logger.info(f"Loading model from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # Lấy config từ checkpoint
            policy_config = PolicyConfig(**checkpoint['policy_config'])
            value_config = ValueConfig(**checkpoint['value_config'])
            self.board_size = checkpoint.get('board_size', policy_config.board_size)
            
            # Khởi tạo models
            self.policy_net = PolicyNetwork(policy_config)
            self.value_net = ValueNetwork(value_config)
            
            # Load weights (xử lý cả trường hợp model đã được compile)
            policy_state_dict = checkpoint['policy_net_state_dict']
            value_state_dict = checkpoint['value_net_state_dict']
            
            # Nếu state_dict có prefix "_orig_mod.", loại bỏ nó
            if any(k.startswith('_orig_mod.') for k in policy_state_dict.keys()):
                logger.info("Removing '_orig_mod.' prefix from state_dict (model was compiled)")
                policy_state_dict = {k.replace('_orig_mod.', ''): v for k, v in policy_state_dict.items()}
                value_state_dict = {k.replace('_orig_mod.', ''): v for k, v in value_state_dict.items()}
            
            self.policy_net.load_state_dict(policy_state_dict)
            self.value_net.load_state_dict(value_state_dict)
            
            # Chuyển sang device và set eval mode
            self.policy_net = self.policy_net.to(self.device)
            self.value_net = self.value_net.to(self.device)
            self.policy_net.eval()
            self.value_net.eval()
            
            self._loaded = True
            logger.info(f"✅ Model loaded successfully! Board size: {self.board_size}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            self._loaded = False
            return False
    
    def is_loaded(self) -> bool:
        """Kiểm tra model đã được load chưa."""
        return self._loaded and self.policy_net is not None and self.value_net is not None
    
    def predict_move(
        self,
        board_position: dict,
        current_player: str,
        move_history: Optional[list] = None,
        valid_moves: Optional[list] = None
    ) -> Tuple[Optional[Tuple[int, int]], float, float]:
        """
        Dự đoán nước đi tốt nhất từ board state.
        
        Args:
            board_position: Dict với keys như "x,y" và values "B" hoặc "W"
            current_player: 'B' hoặc 'W'
            move_history: List of (x, y) tuples cho last 4 moves
            valid_moves: List of (x, y) tuples của các nước đi hợp lệ (optional)
        
        Returns:
            Tuple (best_move, policy_prob, win_prob)
            - best_move: (x, y) hoặc None nếu không có move hợp lệ
            - policy_prob: Xác suất của move tốt nhất
            - win_prob: Xác suất thắng của current_player
        """
        if not self.is_loaded():
            logger.error("Model not loaded!")
            return None, 0.0, 0.5
        
        if self.board_size is None:
            logger.error("Board size not set!")
            return None, 0.0, 0.5
        
        try:
            # Convert board_position sang numpy
            board_state = board_position_to_numpy(board_position, self.board_size)
            
            # Tạo features
            features = board_to_features_17_planes(
                board_state, current_player, move_history, self.board_size
            )
            features = features.unsqueeze(0).to(self.device)  # [1, 17, board_size, board_size]
            
            # Predict
            with torch.no_grad():
                # Policy prediction
                policy_logits = self.policy_net(features)  # [1, board_size * board_size]
                policy_probs = torch.exp(policy_logits[0])  # [board_size * board_size]
                
                # Value prediction
                value_pred = self.value_net(features)  # [1, 1]
                win_prob = value_pred[0, 0].item()
            
            # Tìm move tốt nhất
            if valid_moves:
                # Chỉ xem xét valid moves
                best_move = None
                best_prob = 0.0
                for x, y in valid_moves:
                    if 0 <= x < self.board_size and 0 <= y < self.board_size:
                        idx = y * self.board_size + x
                        prob = policy_probs[idx].item()
                        if prob > best_prob:
                            best_prob = prob
                            best_move = (x, y)
            else:
                # Lấy move có xác suất cao nhất
                best_move_idx = torch.argmax(policy_probs).item()
                best_move_y = best_move_idx // self.board_size
                best_move_x = best_move_idx % self.board_size
                best_move = (best_move_x, best_move_y)
                best_prob = policy_probs[best_move_idx].item()
            
            return best_move, best_prob, win_prob
            
        except Exception as e:
            logger.error(f"Error in predict_move: {e}", exc_info=True)
            return None, 0.0, 0.5


# Global instance (singleton)
_ml_model_service: Optional[MLModelService] = None


def get_ml_model_service(checkpoint_path: Optional[str] = None, device: str = 'cpu') -> Optional[MLModelService]:
    """
    Get hoặc tạo ML model service instance (singleton).
    
    Args:
        checkpoint_path: Đường dẫn đến checkpoint (chỉ dùng lần đầu)
        device: 'cpu' hoặc 'cuda'
    
    Returns:
        MLModelService instance hoặc None nếu không load được (không báo lỗi)
    """
    global _ml_model_service
    
    # Nếu dependencies không có, trả về None ngay (không báo lỗi)
    if not _NUMPY_AVAILABLE or not _TORCH_AVAILABLE:
        return None
    
    if _ml_model_service is None:
        try:
            _ml_model_service = MLModelService(checkpoint_path, device)
            if not _ml_model_service.is_loaded():
                # Chỉ log debug, không warning - vì có thể chưa train model
                logger.debug("ML model service created but model not loaded (model files may not exist yet)")
        except ImportError as e:
            # ImportError = thiếu dependencies hoặc model files - chỉ log debug, không error
            logger.debug(f"ML model service not available: {e} (this is OK if you haven't trained the model)")
            _ml_model_service = None
        except Exception as e:
            # Các lỗi khác - log debug, không warning
            logger.debug(f"ML model service initialization failed: {e}")
            _ml_model_service = None
    
    return _ml_model_service

