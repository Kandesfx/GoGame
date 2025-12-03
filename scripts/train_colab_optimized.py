"""
🚀 Training script tối ưu cho Colab Pro với chunks.
- Tối ưu GPU utilization
- Progress bars chi tiết
- Memory management
- Mixed precision training (nếu hỗ trợ)
- Support cấu trúc: labeled_19x19_YYYY_XXXX.pt
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

# GradScaler: Support cả PyTorch cũ và mới
try:
    from torch.amp import GradScaler  # PyTorch 2.0+
except ImportError:
    from torch.cuda.amp import GradScaler  # PyTorch < 2.0
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
from typing import Optional
import gc
import time
import psutil
import os
import shutil

# Import models
try:
    import sys
    # Thêm path nếu cần
    if '/content' in str(Path.cwd()) or 'colab' in str(Path.cwd()).lower():
        # Trên Colab, có thể cần thêm path
        sys.path.insert(0, str(Path('/content/drive/MyDrive/GoGame_ML/code')))
    
    from policy_network import PolicyNetwork, PolicyConfig
    from value_network import ValueNetwork, ValueConfig
except ImportError:
    try:
        from src.ml.policy_network import PolicyNetwork, PolicyConfig
        from src.ml.value_network import ValueNetwork, ValueConfig
    except ImportError:
        print("⚠️  Warning: PolicyNetwork/ValueNetwork not found. Please import manually.")
        PolicyNetwork = None
        ValueNetwork = None

# Import ChunkDataset
try:
    from chunk_dataset_optimized import ChunkDataset, create_chunk_dataset
except ImportError:
    print("⚠️  Warning: chunk_dataset_optimized not found. Please upload chunk_dataset_optimized.py")
    ChunkDataset = None
    create_chunk_dataset = None


def print_system_info():
    """In thông tin hệ thống và GPU"""
    print("=" * 80)
    print("🖥️  SYSTEM INFORMATION")
    print("=" * 80)
    
    # CPU & RAM
    mem = psutil.virtual_memory()
    print(f"💾 RAM: {mem.total / (1024**3):.1f} GB total, {mem.available / (1024**3):.1f} GB available")
    
    # GPU
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   cuDNN Version: {torch.backends.cudnn.version()}")
        
        # Check mixed precision
        if torch.cuda.is_bf16_supported():
            print("   ✅ bfloat16 supported (mixed precision)")
        else:
            print("   ⚠️  bfloat16 not supported")
    else:
        print("⚠️  No GPU available! Training will be very slow.")
    
    print("=" * 80)


def get_optimal_batch_size(device, board_size=19, start_batch=32):
    """
    Tự động tìm batch size tối ưu dựa trên GPU memory.
    """
    if not torch.cuda.is_available():
        return start_batch // 2
    
    # Test với batch size tăng dần
    test_batch = start_batch
    max_batch = 128
    
    # Tạo dummy data để test
    dummy_features = torch.randn(test_batch, 17, board_size, board_size).to(device)
    
    while test_batch <= max_batch:
        try:
            torch.cuda.empty_cache()
            # Test forward pass
            dummy_output = torch.randn(test_batch, board_size * board_size).to(device)
            _ = dummy_output * 2  # Simple operation
            torch.cuda.synchronize()
            
            # Check memory
            memory_used = torch.cuda.memory_allocated(device) / 1e9
            memory_total = torch.cuda.get_device_properties(device).total_memory / 1e9
            
            if memory_used / memory_total > 0.8:  # Nếu dùng >80% memory
                test_batch -= 4
                break
            
            test_batch += 4
        except RuntimeError as e:
            if "out of memory" in str(e):
                test_batch -= 4
                break
            raise
    
    torch.cuda.empty_cache()
    return max(8, test_batch - 4)  # Đảm bảo tối thiểu 8


def train_one_epoch(
    policy_net, value_net, train_loader,
    policy_optimizer, value_optimizer,
    policy_criterion, value_criterion,
    device, epoch, num_epochs,
    use_mixed_precision=False,
    scaler=None
):
    """Train một epoch với progress bar chi tiết"""
    policy_net.train()
    value_net.train()
    
    total_policy_loss = 0.0
    total_value_loss = 0.0
    num_batches = 0
    
    # Progress bar với thông tin chi tiết
    pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch}/{num_epochs}",
        ncols=120,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )
    
    # GPU memory tracking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        initial_memory = torch.cuda.memory_allocated(device) / 1e9
    
    start_time = time.time()
    
    for batch_idx, batch in enumerate(pbar):
        features = batch['features'].to(device, non_blocking=True)
        policy_target = batch['policy'].to(device, non_blocking=True)
        value_target = batch['value'].to(device, non_blocking=True)
        
        # Mixed precision training
        if use_mixed_precision and scaler is not None:
            # Policy network
            policy_optimizer.zero_grad()
            with autocast():
                policy_logits = policy_net(features)
                policy_loss = policy_criterion(policy_logits, policy_target)
            
            scaler.scale(policy_loss).backward()
            scaler.step(policy_optimizer)
            
            # Value network
            value_optimizer.zero_grad()
            with autocast():
                value_pred = value_net(features)
                value_loss = value_criterion(value_pred, value_target)
            
            scaler.scale(value_loss).backward()
            scaler.step(value_optimizer)
            
            scaler.update()
        else:
            # Standard training
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
        
        # Update progress bar với thông tin chi tiết
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated(device) / 1e9
            peak_memory = torch.cuda.max_memory_allocated(device) / 1e9
            memory_info = f"GPU: {current_memory:.1f}GB (peak: {peak_memory:.1f}GB)"
        else:
            memory_info = "CPU"
        
        pbar.set_postfix({
            'p_loss': f'{policy_loss.item():.4f}',
            'v_loss': f'{value_loss.item():.4f}',
            'mem': memory_info
        })
    
    elapsed_time = time.time() - start_time
    
    avg_policy_loss = total_policy_loss / num_batches
    avg_value_loss = total_value_loss / num_batches
    
    # Print summary
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"   ⏱️  Time: {elapsed_time:.1f}s | GPU Memory: {peak_memory:.2f} GB")
    
    return avg_policy_loss, avg_value_loss


def copy_chunks_to_local(
    source_dir: str,
    local_dir: Optional[str] = None,
    chunk_pattern: Optional[str] = None,
    force_copy: bool = False
) -> Path:
    """
    Copy chunks từ Google Drive vào local disk với progress bar chi tiết.
    
    Args:
        source_dir: Directory chứa chunks trên Google Drive
        local_dir: Directory local để copy (mặc định: /content/chunks_local)
        chunk_pattern: Pattern để tìm chunks (mặc định: auto-detect)
        force_copy: Nếu True, copy lại ngay cả khi đã có
    
    Returns:
        Path to local chunks directory
    """
    source_path = Path(source_dir)
    
    # Tạo local directory
    if local_dir is None:
        # Tạo tên dựa trên source directory
        local_dir_name = source_path.name
        local_path = Path('/content') / f'chunks_{local_dir_name}'
    else:
        local_path = Path(local_dir)
    
    local_path.mkdir(parents=True, exist_ok=True)
    
    # Tìm chunks
    if chunk_pattern is None:
        # Auto-detect pattern
        labeled_pattern = sorted(source_path.glob("labeled_*_*.pt"))
        chunk_pattern_files = sorted(source_path.glob("chunk_*.pt"))
        all_pt = sorted(source_path.glob("*.pt"))
        
        if labeled_pattern:
            chunk_files = labeled_pattern
            pattern_name = "labeled_*_*.pt"
        elif chunk_pattern_files:
            chunk_files = chunk_pattern_files
            pattern_name = "chunk_*.pt"
        else:
            chunk_files = all_pt
            pattern_name = "*.pt"
    else:
        chunk_files = sorted(source_path.glob(chunk_pattern))
        pattern_name = chunk_pattern
    
    if not chunk_files:
        raise ValueError(f"No chunk files found in {source_path} with pattern '{pattern_name}'")
    
    print(f"\n📦 Copying chunks to local disk...")
    print(f"   Source: {source_path}")
    print(f"   Pattern: {pattern_name} ({len(chunk_files)} files)")
    print(f"   Destination: {local_path}")
    
    # Kiểm tra xem đã copy chưa
    existing_files = list(local_path.glob("*.pt"))
    if not force_copy and len(existing_files) == len(chunk_files):
        # Check nếu tất cả files đã có
        all_exist = all((local_path / f.name).exists() for f in chunk_files)
        if all_exist:
            print(f"   ✅ Chunks already copied ({len(existing_files)} files)")
            return local_path
    
    # Copy với progress bar chi tiết
    total_size = sum(f.stat().st_size for f in chunk_files)
    total_size_gb = total_size / (1024**3)
    
    print(f"   Total size: {total_size_gb:.2f} GB")
    print(f"   ⏳ Copying... (this may take 15-25 minutes)")
    
    copied_size = 0
    with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024, desc="Copying chunks") as pbar:
        for chunk_file in chunk_files:
            dest_file = local_path / chunk_file.name
            file_size = chunk_file.stat().st_size
            
            # Update description với tên file
            pbar.set_description(f"Copying {chunk_file.name[:30]}...")
            
            # Copy file
            shutil.copy2(chunk_file, dest_file)
            
            copied_size += file_size
            pbar.update(file_size)
            
            # Update postfix với thông tin chi tiết
            copied_gb = copied_size / (1024**3)
            pbar.set_postfix({
                'copied': f'{copied_gb:.2f}GB',
                'files': f'{len([f for f in local_path.glob("*.pt")])}/{len(chunk_files)}'
            })
    
    print(f"   ✅ Copy complete! ({len(chunk_files)} files, {total_size_gb:.2f} GB)")
    return local_path


def validate(
    policy_net: nn.Module,
    value_net: nn.Module,
    val_loader: DataLoader,
    policy_criterion: nn.Module,
    value_criterion: nn.Module,
    device: torch.device,
    use_mixed_precision=False
) -> tuple:
    """Validate model với progress bar"""
    policy_net.eval()
    value_net.eval()
    
    total_policy_loss = 0.0
    total_value_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating", ncols=120)
        for batch in pbar:
            features = batch['features'].to(device, non_blocking=True)
            policy_target = batch['policy'].to(device, non_blocking=True)
            value_target = batch['value'].to(device, non_blocking=True)
            
            if use_mixed_precision:
                with autocast():
                    policy_logits = policy_net(features)
                    policy_loss = policy_criterion(policy_logits, policy_target)
                    
                    value_pred = value_net(features)
                    value_loss = value_criterion(value_pred, value_target)
            else:
                policy_logits = policy_net(features)
                policy_loss = policy_criterion(policy_logits, policy_target)
                
                value_pred = value_net(features)
                value_loss = value_criterion(value_pred, value_target)
            
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


def train_model_optimized(
    train_dataset_path: str,
    val_dataset_path: Optional[str] = None,
    board_size: Optional[int] = None,
    batch_size: Optional[int] = None,  # Auto-detect nếu None
    num_epochs: int = 10,
    learning_rate: float = 0.001,
    device: Optional[torch.device] = None,
    checkpoint_dir: str = './checkpoints',
    save_every: int = 2,
    use_chunks: bool = True,
    use_mixed_precision: bool = True,
    chunk_pattern: Optional[str] = None,  # Pattern để tìm chunks
    num_workers: int = 0,  # Colab không support multiprocessing tốt
    pin_memory: bool = True,  # Tối ưu GPU transfer
    prefetch_factor: int = 2  # Prefetch batches
):
    """
    Training function tối ưu cho Colab Pro với chunks.
    
    Args:
        train_dataset_path: Path to chunks directory hoặc merged file
        val_dataset_path: Path to validation chunks hoặc merged file
        board_size: Board size (auto-detect nếu None)
        batch_size: Batch size (auto-detect nếu None)
        num_epochs: Number of epochs
        learning_rate: Learning rate
        device: torch.device (auto-detect nếu None)
        checkpoint_dir: Directory to save checkpoints
        save_every: Save checkpoint mỗi N epochs
        use_chunks: Nếu True, load từ chunks
        use_mixed_precision: Sử dụng mixed precision training
        chunk_pattern: Glob pattern để tìm chunks (mặc định: auto-detect)
        num_workers: DataLoader workers (0 cho Colab)
        pin_memory: Pin memory cho GPU transfer nhanh hơn
        prefetch_factor: Số batches prefetch
    """
    print("\n" + "=" * 80)
    print("🚀 GO GAME TRAINING - OPTIMIZED FOR COLAB PRO")
    print("=" * 80)
    
    # System info
    print_system_info()
    
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔧 Using device: {device}")
    
    if not torch.cuda.is_available():
        print("⚠️  WARNING: No GPU detected! Training will be very slow.")
        use_mixed_precision = False
    
    # Mixed precision setup
    scaler = None
    if use_mixed_precision and torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            print("✅ Mixed precision training enabled (bfloat16)")
            # Support cả PyTorch cũ và mới
            try:
                scaler = GradScaler('cuda')  # PyTorch 2.0+
            except TypeError:
                scaler = GradScaler()  # PyTorch < 2.0
        else:
            print("⚠️  Mixed precision not supported, using FP32")
            use_mixed_precision = False
    
    # Load datasets
    train_dataset_path_obj = Path(train_dataset_path)
    
    print(f"\n📦 Loading training dataset...")
    print(f"   Path: {train_dataset_path}")
    
    if use_chunks:
        if ChunkDataset is None or create_chunk_dataset is None:
            raise ImportError("ChunkDataset not available! Please upload chunk_dataset_optimized.py")
        
        print(f"   Mode: Chunks")
        
        # Tự động copy chunks vào local nếu đang ở Google Drive
        if '/content/drive' in str(train_dataset_path):
            print(f"   🔄 Detected Google Drive path, copying to local disk...")
            local_chunks_path = copy_chunks_to_local(
                train_dataset_path,
                chunk_pattern=chunk_pattern,
                force_copy=False
            )
            actual_train_path = str(local_chunks_path)
            print(f"   ✅ Using local chunks: {actual_train_path}")
        else:
            actual_train_path = train_dataset_path
            print(f"   ✅ Using local path: {actual_train_path}")
        
        # Tạo dataset từ local chunks
        train_dataset = create_chunk_dataset(
            actual_train_path,
            augment=True,
            pattern=chunk_pattern
        )
        
        if board_size is None:
            board_size = train_dataset.board_size
            print(f"   ✅ Auto-detected board_size: {board_size}")
        
        print(f"   ✅ Total samples: {len(train_dataset):,}")
        
        # Tối ưu DataLoader cho chunks
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size or 32,  # Tạm thời, sẽ auto-detect sau
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=False
        )
    else:
        print(f"   Mode: Merged file")
        print(f"   ⏳ Loading merged file (this may take 1-2 minutes for large files)...")
        train_data = torch.load(train_dataset_path, map_location='cpu', weights_only=False)
        train_labeled = train_data['labeled_data']
        
        if board_size is None:
            if 'board_size' in train_data:
                board_size = train_data['board_size']
            else:
                sample_features = train_labeled[0]['features']
                board_size = sample_features.shape[1] if sample_features.dim() == 3 else 19
            print(f"   ✅ Auto-detected board_size: {board_size}")
        
        from train_colab import GoDataset
        train_dataset = GoDataset(train_labeled, augment=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size or 32,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            prefetch_factor=prefetch_factor if num_workers > 0 else None
        )
        
        del train_data, train_labeled
        gc.collect()
        
        print(f"   ✅ Total samples: {len(train_dataset):,}")
    
    # Auto-detect optimal batch size
    if batch_size is None and torch.cuda.is_available():
        print(f"\n🔍 Auto-detecting optimal batch size...")
        optimal_batch = get_optimal_batch_size(device, board_size)
        print(f"   ✅ Optimal batch size: {optimal_batch}")
        
        # Recreate DataLoader với batch size tối ưu
        train_loader = DataLoader(
            train_dataset,
            batch_size=optimal_batch,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=False
        )
        batch_size = optimal_batch
    elif batch_size is None:
        batch_size = 16
        print(f"   Using default batch size: {batch_size}")
    
    # Validation dataset
    val_loader = None
    if val_dataset_path and Path(val_dataset_path).exists():
        val_path_obj = Path(val_dataset_path)
        print(f"\n📦 Loading validation dataset...")
        
        if use_chunks and val_path_obj.is_dir():
            # Copy validation chunks vào local nếu cần
            if '/content/drive' in str(val_dataset_path):
                print(f"   🔄 Copying validation chunks to local disk...")
                local_val_path = copy_chunks_to_local(
                    val_dataset_path,
                    local_dir=str(Path('/content') / f'chunks_val_{Path(val_dataset_path).name}'),
                    chunk_pattern=chunk_pattern,
                    force_copy=False
                )
                actual_val_path = str(local_val_path)
                print(f"   ✅ Using local validation chunks: {actual_val_path}")
            else:
                actual_val_path = val_dataset_path
            
            val_dataset = create_chunk_dataset(
                actual_val_path,
                augment=False,
                pattern=chunk_pattern
            )
        else:
            print(f"   ⏳ Loading validation file...")
            val_data = torch.load(val_dataset_path, map_location='cpu', weights_only=False)
            val_labeled = val_data['labeled_data']
            from train_colab import GoDataset
            val_dataset = GoDataset(val_labeled, augment=False)
            del val_data, val_labeled
            gc.collect()
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            prefetch_factor=prefetch_factor if num_workers > 0 else None
        )
        print(f"   ✅ Validation samples: {len(val_dataset):,}")
    
    # Initialize models
    if PolicyNetwork is None or ValueNetwork is None:
        raise ImportError("PolicyNetwork và ValueNetwork chưa được import!")
    
    print(f"\n🧠 Initializing models...")
    policy_config = PolicyConfig(board_size=board_size, input_planes=17, channels=128)
    value_config = ValueConfig(board_size=board_size, input_planes=17, channels=128)
    
    policy_net = PolicyNetwork(policy_config).to(device)
    value_net = ValueNetwork(value_config).to(device)
    
    policy_params = sum(p.numel() for p in policy_net.parameters())
    value_params = sum(p.numel() for p in value_net.parameters())
    print(f"   📊 Policy Network: {policy_params:,} parameters")
    print(f"   📊 Value Network: {value_params:,} parameters")
    print(f"   📊 Total: {policy_params + value_params:,} parameters")
    
    # Optimizers
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    value_optimizer = optim.Adam(value_net.parameters(), lr=learning_rate)
    
    # Loss functions
    policy_criterion = nn.KLDivLoss(reduction='batchmean')
    value_criterion = nn.MSELoss()
    
    # Create checkpoint directory
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n💾 Checkpoints will be saved to: {checkpoint_dir}")
    
    # Training loop
    print(f"\n🚀 Starting training...")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Mixed precision: {'✅' if use_mixed_precision else '❌'}")
    print("=" * 80)
    
    best_val_loss = float('inf')
    training_start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()
        
        # Train
        train_policy_loss, train_value_loss = train_one_epoch(
            policy_net, value_net, train_loader,
            policy_optimizer, value_optimizer,
            policy_criterion, value_criterion,
            device, epoch, num_epochs,
            use_mixed_precision, scaler
        )
        
        # Validate
        if val_loader is not None:
            val_policy_loss, val_value_loss = validate(
                policy_net, value_net, val_loader,
                policy_criterion, value_criterion, device,
                use_mixed_precision
            )
            total_val_loss = val_policy_loss + val_value_loss
        else:
            val_policy_loss = 0.0
            val_value_loss = 0.0
            total_val_loss = float('inf')
        
        epoch_time = time.time() - epoch_start_time
        
        # Print epoch summary
        print(f"\n📊 Epoch {epoch}/{num_epochs} Summary:")
        print(f"   Train - Policy: {train_policy_loss:.4f}, Value: {train_value_loss:.4f}")
        if val_loader is not None:
            print(f"   Val   - Policy: {val_policy_loss:.4f}, Value: {val_value_loss:.4f}")
        print(f"   ⏱️  Time: {epoch_time:.1f}s ({epoch_time/60:.1f} min)")
        
        # Clear cache sau mỗi epoch nếu dùng chunks
        if use_chunks and hasattr(train_dataset, 'clear_cache'):
            train_dataset.clear_cache()
            gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
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
            print(f"   💾 Saved checkpoint: {checkpoint_path.name}")
        
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
            print(f"   ⭐ Saved best model: {best_path.name} (val_loss: {total_val_loss:.4f})")
    
    # Save final model
    total_time = time.time() - training_start_time
    final_path = checkpoint_dir / 'final_model.pt'
    torch.save({
        'policy_net_state_dict': policy_net.state_dict(),
        'value_net_state_dict': value_net.state_dict(),
        'policy_config': policy_config.__dict__,
        'value_config': value_config.__dict__,
        'board_size': board_size
    }, final_path)
    
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE!")
    print("=" * 80)
    print(f"⏱️  Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"💾 Final model saved: {final_path}")
    print(f"⭐ Best model saved: {checkpoint_dir / 'best_model.pt'}")
    print("=" * 80)


if __name__ == "__main__":
    # Example usage trên Colab:
    from pathlib import Path
    
    WORK_DIR = Path('/content/drive/MyDrive/GoGame_ML')
    
    # Training với chunks (cấu trúc: labeled_19x19_2012_0001.pt)
    train_model_optimized(
        train_dataset_path=str(WORK_DIR / 'datasets' / 'labeled_19x19_2012_chunks'),
        val_dataset_path=None,  # Có thể dùng chunks riêng
        board_size=None,  # Auto-detect
        batch_size=None,  # Auto-detect optimal
        num_epochs=10,
        learning_rate=0.001,
        checkpoint_dir=str(WORK_DIR / 'checkpoints'),
        save_every=2,
        use_chunks=True,
        use_mixed_precision=True,
        chunk_pattern=None,  # Auto-detect: labeled_*_*.pt
        pin_memory=True,
        prefetch_factor=2
    )

