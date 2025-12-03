"""
Training script hoàn chỉnh cho Colab.

Sử dụng:
1. Load labeled dataset
2. Train Policy Network và Value Network
3. Save checkpoints
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np
from typing import Optional
import gc
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True

# Import models (cần copy vào Colab hoặc import từ drive)
try:
    from policy_network import PolicyNetwork, PolicyConfig
    from value_network import ValueNetwork, ValueConfig
except ImportError:
    # Nếu chưa có, sẽ define inline trong notebook
    PolicyNetwork = None
    ValueNetwork = None

# Import MultiTaskModel từ models/ (detection heads)
try:
    import sys
    from pathlib import Path
    # Thử nhiều paths khác nhau để tìm models
    possible_paths = [
        Path(__file__).parent.parent / "src" / "ml",  # Từ repo local
        Path("/content"),  # Colab default
        Path("/content/drive/MyDrive/GoGame_ML/code"),  # Colab Drive
    ]
    
    models_imported = False
    for base_path in possible_paths:
        models_path = base_path / "models"
        if models_path.exists() and str(base_path) not in sys.path:
            sys.path.insert(0, str(base_path))
            try:
                from models.multi_task_model import MultiTaskModel, MultiTaskConfig
                models_imported = True
                break
            except ImportError:
                continue
    
    if not models_imported:
        # Thử import trực tiếp nếu models đã ở trong path
        try:
            from models.multi_task_model import MultiTaskModel, MultiTaskConfig
            models_imported = True
        except ImportError:
            pass
    
    if not models_imported:
        raise ImportError("Could not find models.multi_task_model in any path")
    
    MULTI_TASK_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  MultiTaskModel not available: {e}")
    if 'possible_paths' in locals():
        print(f"   Tried paths: {possible_paths}")
    print(f"   Current sys.path (first 5): {sys.path[:5]}")
    MultiTaskModel = None
    MultiTaskConfig = None
    MULTI_TASK_AVAILABLE = False

# Import ChunkDataset để hỗ trợ load từ chunks
try:
    from chunk_dataset import ChunkDataset, create_chunk_dataset
except ImportError:
    ChunkDataset = None
    create_chunk_dataset = None


class GoDataset(Dataset):
    """
    Dataset class để load labeled data.
    
    Mỗi sample gồm:
    - features: Tensor [17, board_size, board_size]
    - policy: Tensor [board_size * board_size]
    - value: float
    - (optional) threat_map: Tensor [board_size, board_size]
    - (optional) attack_map: Tensor [board_size, board_size]
    - (optional) intent_label: int (0-4)
    """
    
    def __init__(self, labeled_data: list, augment: bool = False, use_detection_labels: bool = False):
        """
        Args:
            labeled_data: List of labeled samples
            augment: Nếu True, apply data augmentation (rotation, flip)
            use_detection_labels: Nếu True, load labels cho detection heads (threat, attack, intent)
        """
        self.data = labeled_data
        self.augment = augment
        self.use_detection_labels = use_detection_labels
        
        # Check if detection labels are available
        if use_detection_labels and len(labeled_data) > 0:
            sample = labeled_data[0]
            self.has_threat = 'threat_map' in sample
            self.has_attack = 'attack_map' in sample
            self.has_intent = 'intent_label' in sample
            print(f"📊 Detection labels available: threat={self.has_threat}, attack={self.has_attack}, intent={self.has_intent}")
        else:
            self.has_threat = False
            self.has_attack = False
            self.has_intent = False
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        board_size = sample['features'].shape[1] if sample['features'].dim() == 3 else sample['features'].shape[2]
        
        features = sample['features'].clone()
        policy = sample['policy'].clone()
        value = torch.tensor([sample['value']], dtype=torch.float32)
        
        # Load detection labels if available
        result = {
            'features': features,
            'policy': policy,
            'value': value
        }
        
        if self.use_detection_labels:
            # Threat map
            if self.has_threat:
                result['threat_map'] = sample['threat_map'].clone()
            else:
                # Dummy: zeros (will be ignored in loss)
                result['threat_map'] = torch.zeros(board_size, board_size, dtype=torch.float32)
            
            # Attack map
            if self.has_attack:
                result['attack_map'] = sample['attack_map'].clone()
            else:
                result['attack_map'] = torch.zeros(board_size, board_size, dtype=torch.float32)
            
            # Intent label
            if self.has_intent:
                result['intent_label'] = torch.tensor(sample['intent_label'], dtype=torch.long)
            else:
                result['intent_label'] = torch.tensor(0, dtype=torch.long)  # Default: territory
        
        # Data augmentation (rotation và flip)
        if self.augment and torch.rand(1).item() > 0.5:
            # Random rotation (0, 90, 180, 270 degrees)
            k = torch.randint(0, 4, (1,)).item()
            features = torch.rot90(features, k, dims=[1, 2])
            policy = self._rotate_policy(policy, k, board_size)
            
            if self.use_detection_labels:
                if self.has_threat:
                    result['threat_map'] = torch.rot90(result['threat_map'], k, dims=[0, 1])
                if self.has_attack:
                    result['attack_map'] = torch.rot90(result['attack_map'], k, dims=[0, 1])
            
            # Random flip
            if torch.rand(1).item() > 0.5:
                features = torch.flip(features, dims=[2])  # Flip horizontal
                policy = self._flip_policy(policy, board_size)
                
                if self.use_detection_labels:
                    if self.has_threat:
                        result['threat_map'] = torch.flip(result['threat_map'], dims=[1])
                    if self.has_attack:
                        result['attack_map'] = torch.flip(result['attack_map'], dims=[1])
            
            result['features'] = features
            result['policy'] = policy
        
        return result
    
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


def train_one_epoch(
    policy_net, value_net, train_loader,
    policy_optimizer, value_optimizer,
    policy_criterion, value_criterion,
    device, epoch, scaler,
    gradient_accumulation_steps: int = 1
):
    policy_net.train()
    value_net.train()

    # AMP tăng tốc 2–4 lần (sử dụng API mới)
    from torch.amp import autocast

    total_policy_loss = 0.0
    total_value_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", ncols=120)
    
    # Debug: Print khi bắt đầu epoch
    print(f"📊 Starting epoch {epoch}, total batches: {len(train_loader)}")

    for batch_idx, batch in enumerate(pbar):
        # Debug: Print batch đầu tiên
        if batch_idx == 0:
            print(f"🔄 Processing first batch (size: {batch['features'].shape[0]})...")

        # Transfer to GPU
        features = batch['features'].to(device, non_blocking=True)  # non_blocking=True với pin_memory
        policy_target = batch['policy'].to(device, non_blocking=True)
        value_target = batch['value'].to(device, non_blocking=True)
        
        # Debug: Print sau khi transfer batch đầu tiên
        if batch_idx == 0:
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"✅ First batch transferred to GPU")
            print(f"   GPU RAM - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

        # Scale loss by accumulation steps
        accumulation_scale = 1.0 / gradient_accumulation_steps

        # ---------------------------------------------------
        # POLICY NETWORK + AMP
        # ---------------------------------------------------
        # Only zero grad at the start of accumulation
        if batch_idx % gradient_accumulation_steps == 0:
            policy_optimizer.zero_grad(set_to_none=True)

        with autocast(device_type='cuda'):
            policy_logits = policy_net(features)
            policy_loss = policy_criterion(policy_logits, policy_target) * accumulation_scale

        scaler.scale(policy_loss).backward()
        
        # Only step optimizer after accumulation
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            scaler.step(policy_optimizer)

        # ---------------------------------------------------
        # VALUE NETWORK + AMP
        # ---------------------------------------------------
        # Only zero grad at the start of accumulation
        if batch_idx % gradient_accumulation_steps == 0:
            value_optimizer.zero_grad(set_to_none=True)

        with autocast(device_type='cuda'):
            value_pred = value_net(features)
            value_loss = value_criterion(value_pred, value_target) * accumulation_scale

        scaler.scale(value_loss).backward()
        
        # Only step optimizer after accumulation
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            scaler.step(value_optimizer)
            scaler.update()

        # Unscale loss for logging (multiply back by accumulation_steps)
        total_policy_loss += policy_loss.item() * gradient_accumulation_steps
        total_value_loss += value_loss.item() * gradient_accumulation_steps
        num_batches += 1

        pbar.set_postfix({
            "p_loss": f"{policy_loss.item():.4f}",
            "v_loss": f"{value_loss.item():.4f}"
        })

    return total_policy_loss / num_batches, total_value_loss / num_batches


def validate(
    policy_net: nn.Module,
    value_net: nn.Module,
    val_loader: DataLoader,
    policy_criterion: nn.Module,
    value_criterion: nn.Module,
    device: torch.device
) -> tuple:
    """Validate model"""
    policy_net.eval()
    value_net.eval()
    
    total_policy_loss = 0.0
    total_value_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            features = batch['features'].to(device,non_blocking=True)
            policy_target = batch['policy'].to(device,non_blocking=True)
            value_target = batch['value'].to(device,non_blocking=True)
            
            # Policy
            policy_logits = policy_net(features)
            policy_loss = policy_criterion(policy_logits, policy_target)
            
            # Value
            value_pred = value_net(features)
            value_loss = value_criterion(value_pred, value_target)
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1
    
    avg_policy_loss = total_policy_loss / num_batches
    avg_value_loss = total_value_loss / num_batches
    
    return avg_policy_loss, avg_value_loss


def train_model(
    train_dataset_path: str,
    val_dataset_path: Optional[str] = None,
    board_size: Optional[int] = None,  # Auto-detect nếu None
    batch_size: int = 512,  # Giảm xuống 2048 để DataLoader khởi động nhanh hơn trên Colab.
    num_epochs: int = 10,
    learning_rate: float = 0.001,
    device: Optional[torch.device] = None,
    checkpoint_dir: str = './checkpoints',
    save_every: int = 1,  # Lưu checkpoint mỗi epoch để không bỏ sót
    use_chunks: bool = False,  # Nếu True, load từ chunks thay vì merged file
    model_channels: int = 256,  # Tăng từ 128 để model lớn hơn, tận dụng GPU RAM
    max_train_samples: Optional[int] = None,  # None = dùng tất cả samples, hoặc set số cụ thể
    gradient_accumulation_steps: int = 1,  # Gradient accumulation để effective batch size lớn hơn
    enable_pin_memory: bool = True,  # Pin memory để tăng tốc data loading
    checkpoint_prefix: Optional[str] = None  # Prefix cho checkpoint files để tránh ghi đè (auto-detect từ dataset path nếu None)
):
    print(">>> DEBUG: Using train_dataset_path =", train_dataset_path)
    """
    Main training function với hỗ trợ chunks để tránh MemoryError.
    Tối ưu cho GPU RAM lớn (L4 24GB) với batch size lớn và model lớn hơn.
    
    Args:
        train_dataset_path: Path to labeled training dataset (file .pt) hoặc chunks directory
        val_dataset_path: Path to labeled validation dataset (optional)
        board_size: Board size (auto-detect từ dataset nếu None)
        batch_size: Batch size (mặc định 4096 để tận dụng GPU RAM, có thể tăng lên 8192)
        num_epochs: Number of epochs
        learning_rate: Learning rate
        device: torch.device (auto-detect nếu None)
        checkpoint_dir: Directory to save checkpoints
        save_every: Save checkpoint mỗi N epochs
        use_chunks: Nếu True, load từ chunks directory thay vì merged file
        model_channels: Số channels trong model (mặc định 256, tăng từ 128 để model lớn hơn)
        max_train_samples: Giới hạn số samples (None = dùng tất cả)
        gradient_accumulation_steps: Số steps để accumulate gradients (effective batch = batch_size * steps)
        enable_pin_memory: Bật pin_memory để tăng tốc data loading
        checkpoint_prefix: Prefix cho checkpoint files để tránh ghi đè (auto-detect từ dataset path nếu None)
    """
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Print GPU memory info
    if device.type == 'cuda':
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"💾 GPU Memory: {total_memory:.2f} GB")
        print(f"📊 Training config:")
        print(f"   - Batch size: {batch_size}")
        print(f"   - Model channels: {model_channels}")
        print(f"   - Gradient accumulation: {gradient_accumulation_steps}")
        print(f"   - Effective batch size: {batch_size * gradient_accumulation_steps}")
        if max_train_samples:
            print(f"   - Max training samples: {max_train_samples:,}")
        else:
            print(f"   - Using all available samples")
    
    # Load datasets
    train_dataset_path_obj = Path(train_dataset_path)
    
    if use_chunks:
        # Load từ chunks (memory-efficient)
        if ChunkDataset is None:
            raise ImportError("ChunkDataset not available! Please upload chunk_dataset.py")
        
        print(f"📦 Loading from chunks: {train_dataset_path}")
        train_dataset = create_chunk_dataset(train_dataset_path, augment=False)

        # Giới hạn số sample để train (subset từ chunks)
        # Với L4 24GB và batch_size lớn, có thể train nhiều hơn để tận dụng GPU RAM
        # max_train_samples=None sẽ dùng tất cả samples
        if max_train_samples is not None and len(train_dataset) > max_train_samples:
            from torch.utils.data import Subset
            train_dataset = Subset(train_dataset, range(max_train_samples))
            print(f"⚠️ Using subset of training data: {max_train_samples} samples")
        
        # Auto-detect board_size từ dataset
        if board_size is None:
            # Hỗ trợ cả trường hợp train_dataset là Subset(ChunkDataset)
            base_ds = getattr(train_dataset, "dataset", train_dataset)
            if hasattr(base_ds, "board_size"):
                board_size = base_ds.board_size
            else:
                raise AttributeError("Cannot infer board_size from training dataset")
            print(f"   Auto-detected board_size: {board_size}")
        
        # DataLoader với tối ưu cho GPU RAM lớn
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,  # Bật shuffle để training tốt hơn
            # Trên Colab, num_workers > 0 có thể gây deadlock với ChunkDataset.
            # Để num_workers=0 để tránh treo, dù sẽ chậm hơn một chút.
            num_workers=0,
            pin_memory=enable_pin_memory,  # Pin memory để tăng tốc transfer CPU->GPU
            persistent_workers=False,  # Tắt khi num_workers=0
            # prefetch_factor không được dùng khi num_workers=0
        )
    else:
        # Load từ merged file
        print(f"📂 Loading training dataset: {train_dataset_path}")
        train_data = torch.load(train_dataset_path, map_location='cpu', weights_only=False)
        train_labeled = train_data['labeled_data']
        
        # Auto-detect board_size từ dataset
        if board_size is None:
            if 'board_size' in train_data:
                board_size = train_data['board_size']
                print(f"   Auto-detected board_size: {board_size}")
            else:
                # Fallback: detect từ features shape
                if len(train_labeled) > 0:
                    sample_features = train_labeled[0]['features']
                    if sample_features.dim() == 3 and sample_features.shape[0] == 17:
                        board_size = sample_features.shape[1]
                        print(f"   Auto-detected board_size from features: {board_size}")
                    else:
                        board_size = 19  # Default
                        print(f"   Using default board_size: {board_size}")
                else:
                    raise ValueError("Empty dataset!")
        
        train_dataset = GoDataset(train_labeled, augment=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            # Trên Colab, để num_workers=0 để tránh deadlock (nhất quán với chunks mode)
            num_workers=0,
            pin_memory=enable_pin_memory,  # Pin memory để tăng tốc transfer CPU->GPU
            persistent_workers=False,
            # prefetch_factor không được dùng khi num_workers=0
        )
        
        # Clear loaded data để giải phóng memory
        del train_data, train_labeled
        gc.collect()
    
    print(f"   Training samples: {len(train_dataset):,}")
    
    # Validation dataset
    if val_dataset_path and Path(val_dataset_path).exists():
        val_path_obj = Path(val_dataset_path)
        
        # Check nếu là chunks directory
        if val_path_obj.is_dir() and any(val_path_obj.glob('chunk_*.pt')):
            print(f"📦 Loading validation from chunks: {val_dataset_path}")
            if ChunkDataset is None:
                raise ImportError("ChunkDataset not available!")
            val_dataset = create_chunk_dataset(val_dataset_path, augment=False)
        else:
            print(f"📂 Loading validation dataset: {val_dataset_path}")
            val_data = torch.load(val_dataset_path, map_location='cpu', weights_only=False)
            val_labeled = val_data['labeled_data']
            val_dataset = GoDataset(val_labeled, augment=False)
            del val_data, val_labeled
            gc.collect()
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            # Trên Colab, để num_workers=0 để tránh deadlock
            num_workers=0,
            pin_memory=enable_pin_memory  # Pin memory để tăng tốc transfer CPU->GPU
        )
        print(f"   Validation samples: {len(val_dataset):,}")
    else:
        # Split từ training data (chỉ với merged file, không support với chunks)
        if use_chunks:
            print("⚠️  Cannot split chunks dataset. Please provide separate validation chunks.")
            val_loader = None
        else:
            print("⚠️  No validation dataset, splitting from training data")
            train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset, [train_size, val_size]
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                # Trên Colab, để num_workers=0 để tránh deadlock
                num_workers=0,
                pin_memory=enable_pin_memory  # Pin memory để tăng tốc transfer CPU->GPU
            )
            print(f"   Validation samples: {len(val_dataset):,}")
    
    # Initialize models
    if PolicyNetwork is None or ValueNetwork is None:
        raise ImportError("PolicyNetwork và ValueNetwork chưa được import!")
    
    # Tăng channels để model lớn hơn, tận dụng GPU RAM
    policy_config = PolicyConfig(board_size=board_size, input_planes=17, channels=model_channels)
    value_config = ValueConfig(board_size=board_size, input_planes=17, channels=model_channels)
    print(f"🔧 Model channels: {model_channels} (tăng từ 128 để tận dụng GPU RAM)")
    
    policy_net = PolicyNetwork(policy_config).to(device)
    value_net = ValueNetwork(value_config).to(device)
    
    # Compile models để tăng tốc trên L4 (PyTorch 2.0+)
    try:
        if hasattr(torch, 'compile') and torch.__version__ >= '2.0.0':
            print("⚡ Compiling models for L4 GPU optimization...")
            policy_net = torch.compile(policy_net, mode='reduce-overhead')
            value_net = torch.compile(value_net, mode='reduce-overhead')
            print("✅ Models compiled successfully!")
    except Exception as e:
        print(f"⚠️  Model compilation skipped: {e}")
    
    print(f"📊 Policy Network parameters: {sum(p.numel() for p in policy_net.parameters()):,}")
    print(f"📊 Value Network parameters: {sum(p.numel() for p in value_net.parameters()):,}")
    
    # Optimizers
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    value_optimizer = optim.Adam(value_net.parameters(), lr=learning_rate)
    
    # Loss functions
    policy_criterion = nn.KLDivLoss(reduction='batchmean')  # Policy = distribution
    value_criterion = nn.MSELoss()  # Value = regression
    
    # Create checkpoint directory
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Tự động tạo checkpoint prefix từ dataset path nếu không có
    if checkpoint_prefix is None:
        # Lấy tên từ dataset path
        dataset_path = Path(train_dataset_path)
        if dataset_path.is_dir():
            # Nếu là directory, lấy tên thư mục
            checkpoint_prefix = dataset_path.name
        else:
            # Nếu là file, lấy tên file không có extension
            checkpoint_prefix = dataset_path.stem
        
        # Làm sạch prefix (loại bỏ ký tự đặc biệt)
        checkpoint_prefix = checkpoint_prefix.replace(' ', '_').replace('-', '_')
        # Giới hạn độ dài
        if len(checkpoint_prefix) > 30:
            checkpoint_prefix = checkpoint_prefix[:30]
    
    print(f"📝 Checkpoint prefix: '{checkpoint_prefix}' (để tránh ghi đè khi train nhiều dataset)")
    
    # Training loop
    best_val_loss = float('inf')
    from torch.amp import GradScaler
    scaler = GradScaler('cuda')

    print("\n🚀 Starting training...")
    
    # Test load 1 batch trước để đảm bảo DataLoader hoạt động
    print("🔍 Testing DataLoader with first batch...")
    try:
        test_batch = next(iter(train_loader))
        print(f"✅ DataLoader OK! Batch shape: features={test_batch['features'].shape}")
        del test_batch
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ DataLoader test failed: {e}")
        raise
    
    for epoch in range(1, num_epochs + 1):
        # Train
        train_policy_loss, train_value_loss = train_one_epoch(
            policy_net, value_net, train_loader,
            policy_optimizer, value_optimizer,
            policy_criterion, value_criterion,
            device, epoch, scaler,
            gradient_accumulation_steps=gradient_accumulation_steps
        )
        
        # Validate (nếu có)
        if val_loader is not None:
            val_policy_loss, val_value_loss = validate(
                policy_net, value_net, val_loader,
                policy_criterion, value_criterion, device
            )
            total_val_loss = val_policy_loss + val_value_loss
        else:
            val_policy_loss = 0.0
            val_value_loss = 0.0
            total_val_loss = float('inf')
        
        print(f"\nEpoch {epoch}/{num_epochs}:")
        print(f"  Train - Policy: {train_policy_loss:.4f}, Value: {train_value_loss:.4f}")
        if val_loader is not None:
            print(f"  Val   - Policy: {val_policy_loss:.4f}, Value: {val_value_loss:.4f}")
        
        # Clear cache sau mỗi epoch nếu dùng chunks
        if use_chunks:
            # Hỗ trợ cả trường hợp train_dataset là Subset(ChunkDataset)
            base_ds = getattr(train_dataset, "dataset", train_dataset)
            if hasattr(base_ds, 'clear_cache'):
                base_ds.clear_cache()
                gc.collect()
        
        # Save checkpoint - Lưu tất cả epochs (hoặc theo save_every)
        # Luôn lưu epoch cuối cùng và các epoch theo save_every
        should_save = (epoch % save_every == 0) or (epoch == num_epochs)
        if should_save:
            checkpoint_path = checkpoint_dir / f'{checkpoint_prefix}_checkpoint_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'policy_net_state_dict': policy_net.state_dict(),
                'value_net_state_dict': value_net.state_dict(),
                'policy_optimizer_state_dict': policy_optimizer.state_dict(),
                'value_optimizer_state_dict': value_optimizer.state_dict(),
                'train_policy_loss': train_policy_loss,
                'train_value_loss': train_value_loss,
                'val_policy_loss': val_policy_loss,
                'val_value_loss': val_value_loss,
                'policy_config': policy_config.__dict__,
                'value_config': value_config.__dict__,
                'board_size': board_size
            }, checkpoint_path)
            print(f"  💾 Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            best_path = checkpoint_dir / f'{checkpoint_prefix}_best_model.pt'
            torch.save({
                'epoch': epoch,
                'policy_net_state_dict': policy_net.state_dict(),
                'value_net_state_dict': value_net.state_dict(),
                'val_policy_loss': val_policy_loss,
                'val_value_loss': val_value_loss,
                'policy_config': policy_config.__dict__,
                'value_config': value_config.__dict__,
                'board_size': board_size
            }, best_path)
            print(f"  ⭐ Saved best model: {best_path}")
    
    # Save final model
    final_path = checkpoint_dir / f'{checkpoint_prefix}_final_model.pt'
    torch.save({
        'policy_net_state_dict': policy_net.state_dict(),
        'value_net_state_dict': value_net.state_dict(),
        'policy_config': policy_config.__dict__,
        'value_config': value_config.__dict__,
        'board_size': board_size
    }, final_path)
    print(f"\n✅ Training complete! Final model saved: {final_path}")
    print(">>> DEBUG: Using train_dataset_path =", train_dataset_path)



def train_multi_task_model(
    train_dataset_path: str,
    val_dataset_path: Optional[str] = None,
    board_size: Optional[int] = None,
    batch_size: int = 512,  # Giảm xuống 1024 để phù hợp với GPU RAM
    num_epochs: int = 10,
    learning_rate: float = 0.001,
    device: Optional[torch.device] = None,
    checkpoint_dir: str = './checkpoints',
    save_every: int = 1,
    use_chunks: bool = False,
    base_channels: int = 64,
    num_res_blocks: int = 4,
    max_train_samples: Optional[int] = None,
    gradient_accumulation_steps: int = 1,
    enable_pin_memory: bool = True,
    checkpoint_prefix: Optional[str] = None,
    use_detection_labels: bool = True,  # Bật detection labels
    loss_weights: Optional[dict] = None  # Weights cho các tasks: {'threat': 1.0, 'attack': 1.0, 'intent': 1.0}
):
    """
    Train MultiTaskModel với các detection heads (threat, attack, intent).
    
    Args:
        use_detection_labels: Nếu True, sẽ load labels cho detection heads từ dataset
        loss_weights: Dictionary với weights cho từng task loss
    """
    if not MULTI_TASK_AVAILABLE or MultiTaskModel is None:
        raise ImportError("MultiTaskModel không khả dụng! Vui lòng đảm bảo models/ được import đúng.")
    
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    if device.type == 'cuda':
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"💾 GPU Memory: {total_memory:.2f} GB")
        print(f"📊 Training config:")
        print(f"   - Batch size: {batch_size}")
        print(f"   - Base channels: {base_channels}")
        print(f"   - Use detection labels: {use_detection_labels}")
    
    # Default loss weights
    if loss_weights is None:
        loss_weights = {
            'threat': 1.0,
            'attack': 1.0,
            'intent': 1.0
        }
    
    # Load datasets
    train_dataset_path_obj = Path(train_dataset_path)
    
    if use_chunks:
        if ChunkDataset is None:
            raise ImportError("ChunkDataset not available!")
        
        print(f"📦 Loading from chunks: {train_dataset_path}")
        train_dataset = create_chunk_dataset(
            train_dataset_path, 
            augment=False, 
            use_detection_labels=use_detection_labels
        )
        
        if max_train_samples is not None and len(train_dataset) > max_train_samples:
            from torch.utils.data import Subset
            train_dataset = Subset(train_dataset, range(max_train_samples))
            print(f"⚠️ Using subset: {max_train_samples} samples")
        
        if board_size is None:
            base_ds = getattr(train_dataset, "dataset", train_dataset)
            if hasattr(base_ds, "board_size"):
                board_size = base_ds.board_size
            else:
                raise AttributeError("Cannot infer board_size")
            print(f"   Auto-detected board_size: {board_size}")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=enable_pin_memory,
            persistent_workers=False,
        )
    else:
        print(f"📂 Loading training dataset: {train_dataset_path}")
        train_data = torch.load(train_dataset_path, map_location='cpu', weights_only=False)
        train_labeled = train_data['labeled_data']
        
        if board_size is None:
            if 'board_size' in train_data:
                board_size = train_data['board_size']
            elif len(train_labeled) > 0:
                sample_features = train_labeled[0]['features']
                if sample_features.dim() == 3 and sample_features.shape[0] == 17:
                    board_size = sample_features.shape[1]
                else:
                    board_size = 19
            else:
                raise ValueError("Empty dataset!")
            print(f"   Auto-detected board_size: {board_size}")
        
        train_dataset = GoDataset(train_labeled, augment=True, use_detection_labels=use_detection_labels)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=enable_pin_memory,
            persistent_workers=False,
        )
        
        del train_data, train_labeled
        gc.collect()
    
    print(f"   Training samples: {len(train_dataset):,}")
    
    # Validation dataset
    if val_dataset_path and Path(val_dataset_path).exists():
        val_path_obj = Path(val_dataset_path)
        
        if val_path_obj.is_dir() and any(val_path_obj.glob('*.pt')):
            print(f"📦 Loading validation from chunks: {val_dataset_path}")
            if ChunkDataset is None:
                raise ImportError("ChunkDataset not available!")
            val_dataset = create_chunk_dataset(
                val_dataset_path, 
                augment=False, 
                use_detection_labels=use_detection_labels
            )
        else:
            print(f"📂 Loading validation dataset: {val_dataset_path}")
            val_data = torch.load(val_dataset_path, map_location='cpu', weights_only=False)
            val_labeled = val_data['labeled_data']
            val_dataset = GoDataset(val_labeled, augment=False, use_detection_labels=use_detection_labels)
            del val_data, val_labeled
            gc.collect()
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=enable_pin_memory
        )
        print(f"   Validation samples: {len(val_dataset):,}")
    else:
        if use_chunks:
            print("⚠️  Cannot split chunks dataset. Please provide separate validation chunks.")
            val_loader = None
        else:
            print("⚠️  No validation dataset, splitting from training data")
            train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset, [train_size, val_size]
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=enable_pin_memory
            )
            print(f"   Validation samples: {len(val_dataset):,}")
    
    # Initialize MultiTaskModel
    config = MultiTaskConfig(
        input_planes=17,
        board_size=board_size,
        base_channels=base_channels,
        num_res_blocks=num_res_blocks
    )
    model = MultiTaskModel(config).to(device)
    
    # Compile model
    try:
        if hasattr(torch, 'compile') and torch.__version__ >= '2.0.0':
            print("⚡ Compiling model for GPU optimization...")
            model = torch.compile(model, mode='reduce-overhead')
            print("✅ Model compiled successfully!")
    except Exception as e:
        print(f"⚠️  Model compilation skipped: {e}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 MultiTaskModel parameters: {total_params:,}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Loss functions
    threat_criterion = nn.BCELoss()  # Binary cross-entropy for heatmap
    attack_criterion = nn.BCELoss()
    intent_criterion = nn.CrossEntropyLoss()  # Classification
    
    # Create checkpoint directory
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    if checkpoint_prefix is None:
        dataset_path = Path(train_dataset_path)
        if dataset_path.is_dir():
            checkpoint_prefix = dataset_path.name
        else:
            checkpoint_prefix = dataset_path.stem
        checkpoint_prefix = checkpoint_prefix.replace(' ', '_').replace('-', '_')
        if len(checkpoint_prefix) > 30:
            checkpoint_prefix = checkpoint_prefix[:30]
    
    print(f"📝 Checkpoint prefix: '{checkpoint_prefix}'")
    
    # Training loop
    best_val_loss = float('inf')
    from torch.amp import GradScaler
    scaler = GradScaler('cuda')
    from torch.amp import autocast
    
    print("\n🚀 Starting MultiTaskModel training...")
    
    # Test DataLoader
    print("🔍 Testing DataLoader...")
    try:
        test_batch = next(iter(train_loader))
        print(f"✅ DataLoader OK! Batch shape: features={test_batch['features'].shape}")
        del test_batch
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ DataLoader test failed: {e}")
        raise
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_threat_loss = 0.0
        total_attack_loss = 0.0
        total_intent_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", ncols=120)
        
        for batch_idx, batch in enumerate(pbar):
            features = batch['features'].to(device, non_blocking=True)
            
            # Get targets
            threat_target = batch.get('threat_map', None)
            attack_target = batch.get('attack_map', None)
            intent_target = batch.get('intent_label', None)
            
            if threat_target is not None:
                threat_target = threat_target.to(device, non_blocking=True)
            if attack_target is not None:
                attack_target = attack_target.to(device, non_blocking=True)
            if intent_target is not None:
                intent_target = intent_target.to(device, non_blocking=True)
            
            accumulation_scale = 1.0 / gradient_accumulation_steps
            
            if batch_idx % gradient_accumulation_steps == 0:
                optimizer.zero_grad(set_to_none=True)
            
            with autocast(device_type='cuda'):
                outputs = model(features)
                
                total_loss = 0.0
                threat_loss_val = 0.0
                attack_loss_val = 0.0
                intent_loss_val = 0.0
                
                # Threat loss
                if threat_target is not None and use_detection_labels:
                    threat_loss = threat_criterion(outputs['threat_map'], threat_target)
                    total_loss += loss_weights['threat'] * threat_loss * accumulation_scale
                    threat_loss_val = threat_loss.item()
                
                # Attack loss
                if attack_target is not None and use_detection_labels:
                    attack_loss = attack_criterion(outputs['attack_map'], attack_target)
                    total_loss += loss_weights['attack'] * attack_loss * accumulation_scale
                    attack_loss_val = attack_loss.item()
                
                # Intent loss
                if intent_target is not None and use_detection_labels:
                    intent_loss = intent_criterion(outputs['intent_logits'], intent_target)
                    total_loss += loss_weights['intent'] * intent_loss * accumulation_scale
                    intent_loss_val = intent_loss.item()
            
            scaler.scale(total_loss).backward()
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
            
            total_threat_loss += threat_loss_val
            total_attack_loss += attack_loss_val
            total_intent_loss += intent_loss_val
            num_batches += 1
            
            pbar.set_postfix({
                "threat": f"{threat_loss_val:.4f}",
                "attack": f"{attack_loss_val:.4f}",
                "intent": f"{intent_loss_val:.4f}"
            })
        
        avg_threat_loss = total_threat_loss / num_batches
        avg_attack_loss = total_attack_loss / num_batches
        avg_intent_loss = total_intent_loss / num_batches
        
        # Validation
        if val_loader is not None:
            model.eval()
            val_threat_loss = 0.0
            val_attack_loss = 0.0
            val_intent_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validating"):
                    features = batch['features'].to(device, non_blocking=True)
                    outputs = model(features)
                    
                    if use_detection_labels:
                        threat_target = batch.get('threat_map', None)
                        attack_target = batch.get('attack_map', None)
                        intent_target = batch.get('intent_label', None)
                        
                        if threat_target is not None:
                            threat_target = threat_target.to(device, non_blocking=True)
                            val_threat_loss += threat_criterion(outputs['threat_map'], threat_target).item()
                        
                        if attack_target is not None:
                            attack_target = attack_target.to(device, non_blocking=True)
                            val_attack_loss += attack_criterion(outputs['attack_map'], attack_target).item()
                        
                        if intent_target is not None:
                            intent_target = intent_target.to(device, non_blocking=True)
                            val_intent_loss += intent_criterion(outputs['intent_logits'], intent_target).item()
                    
                    val_batches += 1
            
            val_threat_loss /= val_batches
            val_attack_loss /= val_batches
            val_intent_loss /= val_batches
            total_val_loss = val_threat_loss + val_attack_loss + val_intent_loss
        else:
            val_threat_loss = 0.0
            val_attack_loss = 0.0
            val_intent_loss = 0.0
            total_val_loss = float('inf')
        
        print(f"\nEpoch {epoch}/{num_epochs}:")
        print(f"  Train - Threat: {avg_threat_loss:.4f}, Attack: {avg_attack_loss:.4f}, Intent: {avg_intent_loss:.4f}")
        if val_loader is not None:
            print(f"  Val   - Threat: {val_threat_loss:.4f}, Attack: {val_attack_loss:.4f}, Intent: {val_intent_loss:.4f}")
        
        # Clear cache
        if use_chunks:
            base_ds = getattr(train_dataset, "dataset", train_dataset)
            if hasattr(base_ds, 'clear_cache'):
                base_ds.clear_cache()
                gc.collect()
        
        # Save checkpoint
        should_save = (epoch % save_every == 0) or (epoch == num_epochs)
        if should_save:
            checkpoint_path = checkpoint_dir / f'{checkpoint_prefix}_multitask_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_threat_loss': avg_threat_loss,
                'train_attack_loss': avg_attack_loss,
                'train_intent_loss': avg_intent_loss,
                'val_threat_loss': val_threat_loss,
                'val_attack_loss': val_attack_loss,
                'val_intent_loss': val_intent_loss,
                'config': config.__dict__,
                'board_size': board_size
            }, checkpoint_path)
            print(f"  💾 Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            best_path = checkpoint_dir / f'{checkpoint_prefix}_multitask_best.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_threat_loss': val_threat_loss,
                'val_attack_loss': val_attack_loss,
                'val_intent_loss': val_intent_loss,
                'config': config.__dict__,
                'board_size': board_size
            }, best_path)
            print(f"  ⭐ Saved best model: {best_path}")
    
    # Save final model
    final_path = checkpoint_dir / f'{checkpoint_prefix}_multitask_final.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.__dict__,
        'board_size': board_size
    }, final_path)
    print(f"\n✅ Training complete! Final model saved: {final_path}")


if __name__ == "__main__":
    # Example usage trên Colab:
    # from pathlib import Path
    # WORK_DIR = Path('/content/drive/MyDrive/GoGame_ML')
    # 
    # train_model(
    #     train_dataset_path=WORK_DIR / 'datasets' / 'labeled_9x9.pt',
    #     val_dataset_path=None,  # Auto-split
    #     board_size=9,
    #     batch_size=32,
    #     num_epochs=10,
    #     learning_rate=0.001,
    #     checkpoint_dir=str(WORK_DIR / 'checkpoints')
    # )
    pass

