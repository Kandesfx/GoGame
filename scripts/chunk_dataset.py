"""
ChunkDataset: Memory-efficient dataset để load từ nhiều chunk files.
Chỉ cache 1 chunk tại một thời điểm để tránh MemoryError.
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List
import gc


class ChunkDataset(Dataset):
    """
    Dataset load từ chunks với memory-efficient caching.
    Chỉ cache 1 chunk tại một thời điểm.
    """
    
    def __init__(self, chunk_files: List[Path], augment: bool = False):
        """
        Args:
            chunk_files: List of chunk file paths
            augment: Nếu True, apply data augmentation
        """
        self.chunk_files = sorted(chunk_files)
        self.augment = augment
        
        if not self.chunk_files:
            raise ValueError("No chunk files provided")
        
        # Load metadata từ chunk đầu tiên
        print(f"📊 Loading metadata from {len(self.chunk_files)} chunks...")
        first_chunk = torch.load(self.chunk_files[0], map_location='cpu', weights_only=False)
        self.board_size = first_chunk['board_size']
        del first_chunk
        gc.collect()
        
        # Tính tổng samples và offsets (chỉ load metadata, không load data)
        self._chunk_sizes = []
        self._chunk_offsets = [0]
        total = 0
        
        for chunk_file in self.chunk_files:
            chunk_data = torch.load(chunk_file, map_location='cpu', weights_only=False)
            chunk_size = len(chunk_data['labeled_data'])
            self._chunk_sizes.append(chunk_size)
            total += chunk_size
            self._chunk_offsets.append(total)
            del chunk_data
            gc.collect()
        
        self._total_samples = total
        
        # Cache cho chunk hiện tại (chỉ cache 1 chunk)
        self._cached_chunk_idx = None
        self._cached_chunk_data = None
        
        print(f"✅ Total samples: {self._total_samples:,} from {len(self.chunk_files)} chunks")
        print(f"   Board size: {self.board_size}x{self.board_size}")
    
    def __len__(self):
        return self._total_samples
    
    def _load_chunk(self, chunk_idx: int):
        """Load chunk và cache (chỉ cache 1 chunk, clear chunk cũ)"""
        if self._cached_chunk_idx != chunk_idx:
            # Clear cache cũ
            if self._cached_chunk_data is not None:
                del self._cached_chunk_data
                gc.collect()
            
            # Load chunk mới
            chunk_data = torch.load(
                self.chunk_files[chunk_idx],
                map_location='cpu',
                weights_only=False
            )
            self._cached_chunk_data = chunk_data['labeled_data']
            self._cached_chunk_idx = chunk_idx
    
    def __getitem__(self, idx):
        """Get sample tại idx, tự động load chunk cần thiết"""
        # Tìm chunk chứa sample này
        chunk_idx = 0
        for i in range(len(self._chunk_offsets) - 1):
            if idx < self._chunk_offsets[i + 1]:
                chunk_idx = i
                local_idx = idx - self._chunk_offsets[i]
                break
        
        # Load chunk (cache nếu cần)
        self._load_chunk(chunk_idx)
        sample = self._cached_chunk_data[local_idx]
        
        # Process sample
        features = sample['features'].clone()
        policy = sample['policy'].clone()
        value = torch.tensor([sample['value']], dtype=torch.float32)
        
        # Data augmentation
        if self.augment and torch.rand(1).item() > 0.5:
            k = torch.randint(0, 4, (1,)).item()
            features = torch.rot90(features, k, dims=[1, 2])
            policy = self._rotate_policy(policy, k, features.shape[1])
            
            if torch.rand(1).item() > 0.5:
                features = torch.flip(features, dims=[2])
                policy = self._flip_policy(policy, features.shape[1])
        
        return {
            'features': features,
            'policy': policy,
            'value': value
        }
    
    def _rotate_policy(self, policy: torch.Tensor, k: int, board_size: int) -> torch.Tensor:
        """Rotate policy tensor"""
        policy_2d = policy.view(board_size, board_size)
        policy_2d = torch.rot90(policy_2d, k, dims=[0, 1])
        return policy_2d.view(-1)
    
    def _flip_policy(self, policy: torch.Tensor, board_size: int) -> torch.Tensor:
        """Flip policy tensor horizontally"""
        policy_2d = policy.view(board_size, board_size)
        policy_2d = torch.flip(policy_2d, dims=[1])
        return policy_2d.view(-1)
    
    def clear_cache(self):
        """Clear cache để giải phóng memory"""
        if self._cached_chunk_data is not None:
            del self._cached_chunk_data
            self._cached_chunk_data = None
            self._cached_chunk_idx = None
            gc.collect()


def create_chunk_dataset(chunks_dir: str, augment: bool = True):
    """
    Helper function để tạo ChunkDataset từ directory.
    
    Args:
        chunks_dir: Directory chứa chunk files
        augment: Nếu True, apply data augmentation
    
    Returns:
        ChunkDataset instance
    """
    chunks_dir = Path(chunks_dir)
    chunk_files = sorted(chunks_dir.glob('chunk_*.pt'))
    
    if not chunk_files:
        raise ValueError(f"No chunk files found in {chunks_dir}")
    
    return ChunkDataset(chunk_files, augment=augment)

