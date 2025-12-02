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

# Import models (cần copy vào Colab hoặc import từ drive)
try:
    from policy_network import PolicyNetwork, PolicyConfig
    from value_network import ValueNetwork, ValueConfig
except ImportError:
    # Nếu chưa có, sẽ define inline trong notebook
    PolicyNetwork = None
    ValueNetwork = None

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
    """
    
    def __init__(self, labeled_data: list, augment: bool = False):
        """
        Args:
            labeled_data: List of labeled samples
            augment: Nếu True, apply data augmentation (rotation, flip)
        """
        self.data = labeled_data
        self.augment = augment
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        features = sample['features'].clone()
        policy = sample['policy'].clone()
        value = torch.tensor([sample['value']], dtype=torch.float32)
        
        # Data augmentation (rotation và flip)
        if self.augment and torch.rand(1).item() > 0.5:
            # Random rotation (0, 90, 180, 270 degrees)
            k = torch.randint(0, 4, (1,)).item()
            features = torch.rot90(features, k, dims=[1, 2])
            policy = self._rotate_policy(policy, k, features.shape[1])
            
            # Random flip
            if torch.rand(1).item() > 0.5:
                features = torch.flip(features, dims=[2])  # Flip horizontal
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


def train_one_epoch(
    policy_net, value_net, train_loader,
    policy_optimizer, value_optimizer,
    policy_criterion, value_criterion,
    device, epoch
):
    policy_net.train()
    value_net.train()
    
    total_policy_loss = 0.0
    total_value_loss = 0.0
    num_batches = 0

    from tqdm.auto import tqdm
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", ncols=120)

    for batch in pbar:
        features = batch['features'].to(device)
        policy_target = batch['policy'].to(device)
        value_target = batch['value'].to(device)

        # Policy network
        policy_optimizer.zero_grad()
        policy_logits = policy_net(features)
        policy_loss = policy_criterion(policy_logits, policy_target)
        policy_loss.backward()
        policy_optimizer.step()

        # Value network
        value_optimizer.zero_grad()
        value_pred = value_net(features)
        value_loss = value_criterion(value_pred, value_target)
        value_loss.backward()
        value_optimizer.step()

        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        num_batches += 1

        pbar.set_postfix({
            'p_loss': f'{policy_loss.item():.4f}',
            'v_loss': f'{value_loss.item():.4f}'
        })

    avg_policy_loss = total_policy_loss / num_batches
    avg_value_loss = total_value_loss / num_batches
    return avg_policy_loss, avg_value_loss


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
            features = batch['features'].to(device)
            policy_target = batch['policy'].to(device)
            value_target = batch['value'].to(device)
            
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
    batch_size: int = 16,  # Giảm mặc định để tránh RAM overflow
    num_epochs: int = 10,
    learning_rate: float = 0.001,
    device: Optional[torch.device] = None,
    checkpoint_dir: str = './checkpoints',
    save_every: int = 2,
    use_chunks: bool = False  # Nếu True, load từ chunks thay vì merged file
):
    """
    Main training function với hỗ trợ chunks để tránh MemoryError.
    
    Args:
        train_dataset_path: Path to labeled training dataset (file .pt) hoặc chunks directory
        val_dataset_path: Path to labeled validation dataset (optional)
        board_size: Board size (auto-detect từ dataset nếu None)
        batch_size: Batch size (giảm nếu gặp RAM issues)
        num_epochs: Number of epochs
        learning_rate: Learning rate
        device: torch.device (auto-detect nếu None)
        checkpoint_dir: Directory to save checkpoints
        save_every: Save checkpoint mỗi N epochs
        use_chunks: Nếu True, load từ chunks directory thay vì merged file
    """
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Load datasets
    train_dataset_path_obj = Path(train_dataset_path)
    
    if use_chunks:
        # Load từ chunks (memory-efficient)
        if ChunkDataset is None:
            raise ImportError("ChunkDataset not available! Please upload chunk_dataset.py")
        
        print(f"📦 Loading from chunks: {train_dataset_path}")
        train_dataset = create_chunk_dataset(train_dataset_path, augment=True)
        
        # Auto-detect board_size từ dataset
        if board_size is None:
            board_size = train_dataset.board_size
            print(f"   Auto-detected board_size: {board_size}")
        
        # Tối ưu DataLoader cho chunks (giảm memory)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            prefetch_factor=None,   # ⭐ fix lỗi
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
            num_workers=0,
            pin_memory=False  # Tắt pin_memory để giảm RAM
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
            num_workers=0,
            pin_memory=False
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
                num_workers=0,
                pin_memory=False
            )
            print(f"   Validation samples: {len(val_dataset):,}")
    
    # Initialize models
    if PolicyNetwork is None or ValueNetwork is None:
        raise ImportError("PolicyNetwork và ValueNetwork chưa được import!")
    
    policy_config = PolicyConfig(board_size=board_size, input_planes=17, channels=128)
    value_config = ValueConfig(board_size=board_size, input_planes=17, channels=128)
    
    policy_net = PolicyNetwork(policy_config).to(device)
    value_net = ValueNetwork(value_config).to(device)
    
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
    
    # Training loop
    best_val_loss = float('inf')
    
    print("\n🚀 Starting training...")
    for epoch in range(1, num_epochs + 1):
        # Train
        train_policy_loss, train_value_loss = train_one_epoch(
            policy_net, value_net, train_loader,
            policy_optimizer, value_optimizer,
            policy_criterion, value_criterion,
            device, epoch
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
        if use_chunks and hasattr(train_dataset, 'clear_cache'):
            train_dataset.clear_cache()
            gc.collect()
        
        # Save checkpoint
        if epoch % save_every == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
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
            best_path = checkpoint_dir / 'best_model.pt'
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
    final_path = checkpoint_dir / 'final_model.pt'
    torch.save({
        'policy_net_state_dict': policy_net.state_dict(),
        'value_net_state_dict': value_net.state_dict(),
        'policy_config': policy_config.__dict__,
        'value_config': value_config.__dict__,
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

