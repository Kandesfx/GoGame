import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Optional
import gc
from tqdm import tqdm


class ChunkDataset(Dataset):
    """
    Dataset load từ nhiều chunk .pt – chỉ load 1 chunk tại 1 thời điểm.
    Bản tối ưu: không load full chunk trong __init__ (tránh KeyboardInterrupt).
    """

    def __init__(self, chunk_files: List[Path], augment: bool = False, use_detection_labels: bool = False):
        self.chunk_files = sorted(chunk_files)
        self.augment = augment
        self.use_detection_labels = use_detection_labels

        if not self.chunk_files:
            raise ValueError("No chunk files provided")

        print(f"📦 Found {len(self.chunk_files)} chunk files.")

        # --- Load metadata nhẹ từ 1 chunk ---
        # Bỏ mmap=True trên Colab để tránh treo khi load
        tmp = torch.load(self.chunk_files[0], map_location='cpu', weights_only=False)

        # detect board size if missing
        if "board_size" in tmp:
            self.board_size = tmp["board_size"]
        else:
            sample = tmp["labeled_data"][0]
            self.board_size = sample["features"].shape[-1]

        # Check detection labels availability
        if use_detection_labels and len(tmp["labeled_data"]) > 0:
            sample = tmp["labeled_data"][0]
            self.has_threat = 'threat_map' in sample
            self.has_attack = 'attack_map' in sample
            self.has_intent = 'intent_label' in sample
            print(f"📊 Detection labels: threat={self.has_threat}, attack={self.has_attack}, intent={self.has_intent}")
        else:
            self.has_threat = False
            self.has_attack = False
            self.has_intent = False

        first_len = len(tmp["labeled_data"])
        print(f"📐 Board size = {self.board_size}, first chunk size = {first_len}")
        del tmp
        gc.collect()

        # --- Load sizes từ chunks (không scan, chỉ load khi cần) ---
        self._chunk_sizes = []
        self._chunk_offsets = [0]

        print("📊 Loading chunk sizes...")
        total = 0
        for f in tqdm(self.chunk_files, desc="Loading chunk metadata", unit="file", leave=False):
            # Bỏ mmap=True trên Colab để tránh treo
            meta = torch.load(f, map_location='cpu', weights_only=False)
            size = len(meta['labeled_data'])
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
        
        # Preload chunk đầu tiên để tránh delay khi bắt đầu training
        print("🔄 Preloading first chunk...")
        self._load_chunk(0)
        print("✅ First chunk preloaded.")

    def __len__(self):
        return self._total_samples

    def _load_chunk(self, chunk_idx):
        """Load chunk nhẹ – chỉ dữ liệu, không metadata."""
        if self._cached_chunk_idx != chunk_idx:
            # Chỉ print khi load chunk mới (không phải reload)
            if not hasattr(self, '_loaded_chunks'):
                self._loaded_chunks = set()
            
            is_new_chunk = chunk_idx not in self._loaded_chunks
            if is_new_chunk:
                print(f"🔄 Loading chunk {chunk_idx} from {self.chunk_files[chunk_idx].name}...")
            
            if self._cached_chunk_data is not None:
                del self._cached_chunk_data
                gc.collect()

            # Bỏ mmap=True trên Colab để tránh treo khi load chunk
            chunk = torch.load(
                self.chunk_files[chunk_idx],
                map_location='cpu',
                weights_only=False
            )
            self._cached_chunk_data = chunk['labeled_data']
            self._cached_chunk_idx = chunk_idx
            del chunk
            gc.collect()
            
            # Chỉ print khi load chunk mới
            if is_new_chunk:
                print(f"✅ Chunk {chunk_idx} loaded ({len(self._cached_chunk_data)} samples)")
                self._loaded_chunks.add(chunk_idx)

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

        # Tối ưu: Dùng detach() thay vì clone() để nhanh hơn (nếu không cần gradient)
        features = sample['features'].detach().clone()  # detach() trước để nhanh hơn
        policy = sample['policy'].detach().clone()
        value = torch.tensor([sample['value']], dtype=torch.float32)

        result = {
            'features': features,
            'policy': policy,
            'value': value
        }
        
        # Load detection labels if available
        if self.use_detection_labels:
            if self.has_threat:
                result['threat_map'] = sample['threat_map'].detach().clone()
            else:
                result['threat_map'] = torch.zeros(self.board_size, self.board_size, dtype=torch.float32)
            
            if self.has_attack:
                result['attack_map'] = sample['attack_map'].detach().clone()
            else:
                result['attack_map'] = torch.zeros(self.board_size, self.board_size, dtype=torch.float32)
            
            if self.has_intent:
                result['intent_label'] = torch.tensor(sample['intent_label'], dtype=torch.long)
            else:
                result['intent_label'] = torch.tensor(0, dtype=torch.long)

        # Augmentation
        if self.augment and torch.rand(1).item() > 0.5:
            k = torch.randint(0, 4, (1,)).item()
            features = torch.rot90(features, k, dims=[1, 2])
            policy = self._rotate_policy(policy, k, features.shape[1])
            
            if self.use_detection_labels:
                if self.has_threat:
                    result['threat_map'] = torch.rot90(result['threat_map'], k, dims=[0, 1])
                if self.has_attack:
                    result['attack_map'] = torch.rot90(result['attack_map'], k, dims=[0, 1])

            if torch.rand(1).item() > 0.5:
                features = torch.flip(features, dims=[2])
                policy = self._flip_policy(policy, features.shape[1])
                
                if self.use_detection_labels:
                    if self.has_threat:
                        result['threat_map'] = torch.flip(result['threat_map'], dims=[1])
                    if self.has_attack:
                        result['attack_map'] = torch.flip(result['attack_map'], dims=[1])
            
            result['features'] = features
            result['policy'] = policy

        return result

    def _rotate_policy(self, policy, k, board_size):
    # Hỗ trợ cả 361 (không pass) và 362 (có pass ở cuối)
        if policy.numel() == board_size * board_size + 1:
            board_part = policy[:-1]              # 361 nước trên bàn
            pass_part = policy[-1:]               # 1 phần tử pass
            p = board_part.view(board_size, board_size)
            p = torch.rot90(p, k, dims=[0, 1])
            p = p.reshape(-1)
            return torch.cat([p, pass_part], dim=0)
        else:
            p = policy.view(board_size, board_size)
            p = torch.rot90(p, k, dims=[0, 1])
            return p.reshape(-1)

    def _flip_policy(self, policy, board_size):
        if policy.numel() == board_size * board_size + 1:
            board_part = policy[:-1]
            pass_part = policy[-1:]
            p = board_part.view(board_size, board_size)
            p = torch.flip(p, dims=[1])
            p = p.reshape(-1)
            return torch.cat([p, pass_part], dim=0)
        else:
            p = policy.view(board_size, board_size)
            p = torch.flip(p, dims=[1])
            return p.reshape(-1)

    def clear_cache(self):
        if self._cached_chunk_data is not None:
            del self._cached_chunk_data
            self._cached_chunk_data = None
            self._cached_chunk_idx = None
            gc.collect()


def create_chunk_dataset(chunks_dir: str, augment: bool = True, use_detection_labels: bool = False, pattern: str = None):
    """
    Tạo ChunkDataset từ directory chứa chunks.
    
    Args:
        chunks_dir: Directory chứa chunk files
        augment: Có apply data augmentation không
        use_detection_labels: Có sử dụng detection labels không
        pattern: Glob pattern để tìm files (mặc định: "*.pt")
                 Ví dụ: "labeled_19x19_*_*.pt" hoặc "chunk_*.pt"
    """
    chunks_dir = Path(chunks_dir)
    
    # Nếu không có pattern, tìm tất cả .pt files
    if pattern is None:
        # Tự động detect pattern: ưu tiên labeled_*_*.pt, sau đó chunk_*.pt, cuối cùng *.pt
        labeled_pattern = sorted(chunks_dir.glob("labeled_*_*.pt"))
        chunk_pattern = sorted(chunks_dir.glob("chunk_*.pt"))
        all_pt = sorted(chunks_dir.glob("*.pt"))
        
        if labeled_pattern:
            chunk_files = labeled_pattern
            print(f"📦 Detected pattern: labeled_*_*.pt ({len(chunk_files)} files)")
        elif chunk_pattern:
            chunk_files = chunk_pattern
            print(f"📦 Detected pattern: chunk_*.pt ({len(chunk_files)} files)")
        else:
            chunk_files = all_pt
            print(f"📦 Using all .pt files ({len(chunk_files)} files)")
    else:
        chunk_files = sorted(chunks_dir.glob(pattern))
    
    if not chunk_files:
        raise ValueError(f"No chunk files found in {chunks_dir} with pattern '{pattern or '*.pt'}'")

    return ChunkDataset(chunk_files, augment=augment, use_detection_labels=use_detection_labels)
