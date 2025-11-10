"""
快速开始脚本 - 测试数据处理和模型
"""

import torch
import numpy as np
from pathlib import Path
import sys
import io

# 设置标准输出编码为UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.append(str(Path(__file__).parent))

from dataset.data_loader import DatasetConfig, CrackDataset, get_training_augmentation
from models.convnext_upernet import ConvNeXtUPerNet
from training.losses import CombinedLoss
from inference.sliding_window import SlidingWindowInference, TTAInference


def test_data_loading():
    """测试数据加载"""
    print("=" * 60)
    print("测试 1: 数据加载与增强")
    print("=" * 60)
    
    try:
        # 配置
        config = DatasetConfig(
            root="../data/processed",
            crop_size=(512, 512),
            train_scales=[256, 384, 512]
        )
        
        # 创建数据集
        transform = get_training_augmentation(config, epoch_ratio=0.3)
        dataset = CrackDataset(config, transform=transform)
        
        print(f"✓ 数据集加载成功")
        print(f"  - 样本数: {len(dataset)}")
        
        # 测试获取样本
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"  - 图像形状: {sample['image'].shape}")
            print(f"  - 掩码形状: {sample['mask'].shape}")
            print(f"  - 样本ID: {sample['id']}")
        
        print("✓ 数据加载测试通过\n")
        return True
    
    except Exception as e:
        print(f"✗ 数据加载测试失败: {e}\n")
        return False


def test_model():
    """测试模型"""
    print("=" * 60)
    print("测试 2: 模型架构")
    print("=" * 60)
    
    try:
        # 创建模型
        model = ConvNeXtUPerNet(
            encoder_name='convnext_tiny',
            pretrained=False,  # 测试时不下载预训练权重
            num_classes=1,
            deep_supervision=True,
            edge_branch=True
        )
        
        print("✓ 模型创建成功")
        
        # 统计参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  - 总参数: {total_params / 1e6:.2f}M")
        print(f"  - 可训练参数: {trainable_params / 1e6:.2f}M")
        
        # 测试前向传播
        model.eval()
        with torch.no_grad():
            x = torch.randn(2, 3, 512, 512)
            outputs = model(x)
        
        print(f"  - 主输出形状: {outputs['out'].shape}")
        if 'edge' in outputs:
            print(f"  - 边界输出形状: {outputs['edge'].shape}")
        
        print("✓ 模型测试通过\n")
        return True
    
    except Exception as e:
        print(f"✗ 模型测试失败: {e}\n")
        return False


def test_loss():
    """测试损失函数"""
    print("=" * 60)
    print("测试 3: 损失函数")
    print("=" * 60)
    
    try:
        # 创建损失函数
        loss_fn = CombinedLoss(
            dice_weight=0.4,
            focal_weight=0.3,
            bce_weight=0.2,
            boundary_weight=0.1
        )
        
        print("✓ 损失函数创建成功")
        
        # 测试计算损失
        pred = torch.randn(2, 1, 256, 256)
        target = torch.randint(0, 2, (2, 1, 256, 256)).float()
        
        losses = loss_fn(pred, target)
        
        print("  - 损失分量:")
        for k, v in losses.items():
            if isinstance(v, torch.Tensor):
                print(f"    · {k}: {v.item():.4f}")
        
        print("✓ 损失函数测试通过\n")
        return True
    
    except Exception as e:
        print(f"✗ 损失函数测试失败: {e}\n")
        return False


def test_sliding_window():
    """测试滑窗推理"""
    print("=" * 60)
    print("测试 4: 滑窗推理")
    print("=" * 60)
    
    try:
        # 创建模型
        model = ConvNeXtUPerNet(
            encoder_name='convnext_tiny',
            pretrained=False,
            deep_supervision=False,
            edge_branch=False
        )
        model.eval()
        
        # 创建滑窗推理器
        sliding_window = SlidingWindowInference(
            window_size=(512, 512),
            overlap=0.25,
            batch_size=2
        )
        
        print("✓ 滑窗推理器创建成功")
        
        # 测试推理
        image = torch.randn(3, 1024, 1024)
        
        with torch.no_grad():
            pred = sliding_window(model, image, device='cpu')
        
        print(f"  - 输入形状: {image.shape}")
        print(f"  - 输出形状: {pred.shape}")
        print("✓ 滑窗推理测试通过\n")
        return True
    
    except Exception as e:
        print(f"✗ 滑窗推理测试失败: {e}\n")
        return False


def test_tta():
    """测试TTA"""
    print("=" * 60)
    print("测试 5: 测试时增强 (TTA)")
    print("=" * 60)
    
    try:
        # 创建模型
        model = ConvNeXtUPerNet(
            encoder_name='convnext_tiny',
            pretrained=False,
            deep_supervision=False,
            edge_branch=False
        )
        model.eval()
        
        # 创建TTA推理器
        tta = TTAInference(
            scales=[0.75, 1.0, 1.25],
            flip_h=True,
            flip_v=False
        )
        
        print("✓ TTA推理器创建成功")
        print("  - 尺度: [0.75, 1.0, 1.25]")
        print("  - 水平翻转: True")
        print("  - 总增强数: 6")
        
        # 测试推理
        image = torch.randn(3, 512, 512)
        
        with torch.no_grad():
            pred = tta(model, image, device='cpu')
        
        print(f"  - 输入形状: {image.shape}")
        print(f"  - 输出形状: {pred.shape}")
        print("✓ TTA测试通过\n")
        return True
    
    except Exception as e:
        print(f"✗ TTA测试失败: {e}\n")
        return False


def main():
    print("\n" + "=" * 60)
    print("🚀 裂纹检测系统 - 快速测试")
    print("=" * 60 + "\n")
    
    results = {
        "数据加载": test_data_loading(),
        "模型架构": test_model(),
        "损失函数": test_loss(),
        "滑窗推理": test_sliding_window(),
        "TTA增强": test_tta()
    }
    
    print("=" * 60)
    print("📊 测试结果汇总")
    print("=" * 60)
    
    for name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{name}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有测试通过！系统运行正常")
    else:
        print("⚠️  部分测试失败，请检查错误信息")
    print("=" * 60 + "\n")
    
    if all_passed:
        print("下一步:")
        print("1. 准备数据集（参考 README.md）")
        print("2. 配置训练参数（configs/train_config.yaml）")
        print("3. 开始训练: python train.py --config configs/train_config.yaml")
    
    return all_passed


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    success = main()
    sys.exit(0 if success else 1)

