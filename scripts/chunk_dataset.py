import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List
import gc


class ChunkDataset(Dataset):
    """
    Dataset load từ nhiều chunk .pt – chỉ load 1 chunk tại 1 thời điểm.
    Bản tối ưu: không load full chunk trong __init__ (tránh KeyboardInterrupt).
    """

    def __init__(self, chunk_files: List[Path], augment: bool = False):
        self.chunk_files = sorted(chunk_files)
        self.augment = augment

        if not self.chunk_files:
            raise ValueError("No chunk files provided")

        print(f"📦 Found {len(self.chunk_files)} chunk files.")

        # --- Load metadata nhẹ từ 1 chunk ---
        tmp = torch.load(self.chunk_files[0], map_location='cpu', mmap=True)

        # detect board size if missing
        if "board_size" in tmp:
            self.board_size = tmp["board_size"]
        else:
            sample = tmp["labeled_data"][0]
            self.board_size = sample["features"].shape[-1]

        first_len = len(tmp["labeled_data"])
        print(f"📐 Board size = {self.board_size}, first chunk size = {first_len}")
        del tmp
        gc.collect()

        # --- Precompute sizes WITHOUT loading data ---
        self._chunk_sizes = []
        self._chunk_offsets = [0]

        print("🔍 Scanning chunk sizes (without loading data)...")
        total = 0
        for f in self.chunk_files:
            meta = torch.load(f, map_location='cpu', mmap=True)
            size = len(meta['labeled_data'])   # lightweight
            total += size
            self._chunk_sizes.append(size)
            self._chunk_offsets.append(total)
            del meta
            gc.collect()

        self._total_samples = total
        print(f"✅ Total samples: {self._total_samples:,}")

        # Cache
        self._cached_chunk_idx = None
        self._cached_chunk_data = None

    def __len__(self):
        return self._total_samples

    def _load_chunk(self, chunk_idx):
        """Load chunk nhẹ – chỉ dữ liệu, không metadata."""
        if self._cached_chunk_idx != chunk_idx:

            if self._cached_chunk_data is not None:
                del self._cached_chunk_data
                gc.collect()

            chunk = torch.load(
                self.chunk_files[chunk_idx],
                map_location='cpu',
                mmap=True
            )
            self._cached_chunk_data = chunk['labeled_data']
            self._cached_chunk_idx = chunk_idx
            del chunk
            gc.collect()

    def __getitem__(self, idx):
        # Locate chunk
        chunk_idx = 0
        for i in range(len(self._chunk_offsets) - 1):
            if idx < self._chunk_offsets[i + 1]:
                chunk_idx = i
                local_idx = idx - self._chunk_offsets[i]
                break

        # Load needed chunk
        self._load_chunk(chunk_idx)
        sample = self._cached_chunk_data[local_idx]

        features = sample['features'].clone()
        policy = sample['policy'].clone()
        value = torch.tensor([sample['value']], dtype=torch.float32)

        # Augmentation
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

    def _rotate_policy(self, policy, k, board_size):
        p = policy.view(board_size, board_size)
        p = torch.rot90(p, k, dims=[0, 1])
        return p.reshape(-1)

    def _flip_policy(self, policy, board_size):
        p = policy.view(board_size, board_size)
        p = torch.flip(p, dims=[1])
        return p.reshape(-1)

    def clear_cache(self):
        if self._cached_chunk_data is not None:
            del self._cached_chunk_data
            self._cached_chunk_data = None
            self._cached_chunk_idx = None
            gc.collect()


def create_chunk_dataset(chunks_dir: str, augment: bool = True):
    chunks_dir = Path(chunks_dir)
    chunk_files = sorted(chunks_dir.glob("*.pt"))

    if not chunk_files:
        raise ValueError(f"No chunk files found in {chunks_dir}")

    return ChunkDataset(chunk_files, augment=augment)
