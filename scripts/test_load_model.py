"""
Script đơn giản để test xem model có load được không.

Sử dụng:
    python scripts/test_load_model.py
"""

import sys
from pathlib import Path

# Thêm src/ml vào path để import models
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src' / 'ml'))

try:
    import torch
    from policy_network import PolicyNetwork, PolicyConfig
    from value_network import ValueNetwork, ValueConfig
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Hãy đảm bảo đã cài đặt torch và các dependencies")
    sys.exit(1)


def check_model_file(checkpoint_path: str):
    """Kiểm tra và load model."""
    checkpoint_path = Path(checkpoint_path)
    
    # Kiểm tra file có tồn tại không
    if not checkpoint_path.exists():
        print(f"❌ File không tồn tại: {checkpoint_path}")
        print(f"\n💡 Hướng dẫn:")
        print(f"   1. Đảm bảo file model đã được đặt trong thư mục 'checkpoints/'")
        print(f"   2. Kiểm tra tên file có đúng không")
        print(f"   3. Xem thêm: checkpoints/HUONG_DAN_DAT_FILE.md")
        return False
    
    print(f"✅ File tồn tại: {checkpoint_path}")
    file_size_mb = checkpoint_path.stat().st_size / 1024 / 1024
    print(f"   Kích thước: {file_size_mb:.2f} MB")
    
    # Thử load checkpoint
    try:
        print(f"\n🔄 Đang load checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print(f"✅ Load checkpoint thành công!")
        
        # Kiểm tra keys
        print(f"\n📋 Checkpoint keys:")
        for key in checkpoint.keys():
            print(f"   - {key}")
        
        # Kiểm tra board_size
        if 'board_size' in checkpoint:
            board_size = checkpoint['board_size']
            print(f"\n📐 Board size: {board_size}")
        else:
            print(f"\n⚠️  Không tìm thấy 'board_size' trong checkpoint")
        
        # Kiểm tra config
        if 'policy_config' in checkpoint:
            policy_config = checkpoint['policy_config']
            print(f"\n🔧 Policy config: {policy_config}")
        else:
            print(f"\n⚠️  Không tìm thấy 'policy_config' trong checkpoint")
        
        if 'value_config' in checkpoint:
            value_config = checkpoint['value_config']
            print(f"🔧 Value config: {value_config}")
        else:
            print(f"⚠️  Không tìm thấy 'value_config' trong checkpoint")
        
        # Thử khởi tạo models
        print(f"\n🔄 Đang khởi tạo models...")
        if 'policy_config' in checkpoint and 'value_config' in checkpoint:
            policy_config = PolicyConfig(**checkpoint['policy_config'])
            value_config = ValueConfig(**checkpoint['value_config'])
            
            policy_net = PolicyNetwork(policy_config)
            value_net = ValueNetwork(value_config)
            
            # Load weights
            policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            value_net.load_state_dict(checkpoint['value_net_state_dict'])
            
            policy_net.eval()
            value_net.eval()
            
            print(f"✅ Models khởi tạo thành công!")
            print(f"   Policy Network parameters: {sum(p.numel() for p in policy_net.parameters()):,}")
            print(f"   Value Network parameters: {sum(p.numel() for p in value_net.parameters()):,}")
            
            # Test forward pass
            print(f"\n🔄 Đang test forward pass...")
            board_size = checkpoint.get('board_size', 9)
            test_features = torch.randn(1, 17, board_size, board_size)
            
            with torch.no_grad():
                policy_logits = policy_net(test_features)
                value_pred = value_net(test_features)
            
            print(f"✅ Forward pass thành công!")
            print(f"   Policy output shape: {policy_logits.shape}")
            print(f"   Value output shape: {value_pred.shape}")
            print(f"   Value prediction: {value_pred[0, 0].item():.4f}")
            
            return True
        else:
            print(f"❌ Không thể khởi tạo models vì thiếu config")
            return False
        
    except Exception as e:
        print(f"❌ Lỗi khi load checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    print("=" * 60)
    print("🧪 TEST LOAD MODEL")
    print("=" * 60)
    
    # Tìm các file model có thể có
    checkpoints_dir = project_root / 'checkpoints'
    
    if not checkpoints_dir.exists():
        print(f"❌ Thư mục 'checkpoints/' không tồn tại!")
        print(f"   Đang tạo thư mục...")
        checkpoints_dir.mkdir(exist_ok=True)
        print(f"✅ Đã tạo thư mục 'checkpoints/'")
        print(f"\n💡 Hãy đặt file model (final_model.pt) vào thư mục này")
        print(f"   Xem thêm: checkpoints/HUONG_DAN_DAT_FILE.md")
        return
    
    # Tìm các file .pt trong checkpoints
    model_files = list(checkpoints_dir.glob('*.pt'))
    
    if not model_files:
        print(f"❌ Không tìm thấy file model nào trong 'checkpoints/'")
        print(f"\n💡 Hãy đặt file model (final_model.pt) vào thư mục:")
        print(f"   {checkpoints_dir}")
        print(f"   Xem thêm: checkpoints/HUONG_DAN_DAT_FILE.md")
        return
    
    print(f"\n📁 Tìm thấy {len(model_files)} file model:")
    for i, model_file in enumerate(model_files, 1):
        print(f"   {i}. {model_file.name}")
    
    # Test từng file
    print(f"\n" + "=" * 60)
    for model_file in model_files:
        print(f"\n🔍 Testing: {model_file.name}")
        print("-" * 60)
        success = check_model_file(model_file)
        
        if success:
            print(f"\n✅ {model_file.name} - OK! Model có thể sử dụng được.")
        else:
            print(f"\n❌ {model_file.name} - Có lỗi!")
        
        print()


if __name__ == '__main__':
    main()

