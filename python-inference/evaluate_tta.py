"""
使用TTA评估已训练模型
"""

import torch
import yaml
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from models.convnext_upernet import create_model
from dataset.data_loader import create_dataloaders
from inference.tta_inference import EnhancedTTAInference, evaluate_with_tta, PostProcessor
from tqdm import tqdm
import numpy as np


def load_model(checkpoint_path: str, model_config: dict, device: str = 'cuda'):
    """加载模型"""
    model = create_model(model_config)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 尝试加载模型权重
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    print(f"✅ 模型加载成功: {checkpoint_path}")
    if 'metrics' in checkpoint:
        print(f"   训练时最佳指标: {checkpoint.get('metrics', {})}")
    
    return model


def evaluate_single_model(model, val_loader, device, use_tta=True, tta_config=None):
    """评估单个模型"""
    
    if use_tta:
        if tta_config is None:
            tta_config = {
                'scales': [0.75, 1.0, 1.25],
                'flip_h': True,
                'flip_v': True,
                'rotate_90': False,
                'use_edge_aware': True
            }
        
        metrics = evaluate_with_tta(model, val_loader, device, tta_config)
    else:
        # 无TTA评估
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                images = batch['image'].to(device)
                masks = batch['mask']
                
                outputs = model(images)
                pred = torch.sigmoid(outputs['out']).cpu()
                pred_binary = (pred > 0.5).float()
                
                all_preds.append(pred_binary)
                all_targets.append(masks)
        
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        intersection = (all_preds * all_targets).sum()
        union = all_preds.sum() + all_targets.sum() - intersection
        iou = (intersection + 1e-8) / (union + 1e-8)
        dice = (2 * intersection + 1e-8) / (all_preds.sum() + all_targets.sum() + 1e-8)
        
        metrics = {'iou': iou.item(), 'dice': dice.item()}
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='TTA Evaluation')
    parser.add_argument('--config', type=str, default='configs/train_optimized.yaml')
    parser.add_argument('--checkpoint', type=str, default='outputs/best.pth')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--no-tta', action='store_true', help='禁用TTA')
    parser.add_argument('--tta-scales', type=float, nargs='+', default=[0.75, 1.0, 1.25])
    parser.add_argument('--rotate-90', action='store_true', help='启用90度旋转TTA')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 创建数据加载器
    from dataset.data_loader import DatasetConfig
    dataset_config = DatasetConfig(**config['dataset'])
    
    _, val_loader = create_dataloaders(
        config=dataset_config,
        batch_size=1,  # TTA时使用batch_size=1
        num_workers=4
    )
    
    print(f"验证集样本数: {len(val_loader.dataset)}")
    
    # 加载模型
    model = load_model(args.checkpoint, config['model'], device)
    
    # 评估配置
    tta_config = {
        'scales': args.tta_scales,
        'flip_h': True,
        'flip_v': True,
        'rotate_90': args.rotate_90,
        'use_edge_aware': True
    }
    
    print("\n" + "="*50)
    print("开始评估...")
    print("="*50)
    
    # 无TTA评估
    print("\n📊 无TTA评估:")
    metrics_no_tta = evaluate_single_model(model, val_loader, device, use_tta=False)
    for k, v in metrics_no_tta.items():
        print(f"   {k}: {v:.4f}")
    
    # 有TTA评估
    if not args.no_tta:
        print(f"\n📊 TTA评估 (scales={args.tta_scales}, rotate_90={args.rotate_90}):")
        metrics_tta = evaluate_single_model(model, val_loader, device, use_tta=True, tta_config=tta_config)
        for k, v in metrics_tta.items():
            print(f"   {k}: {v:.4f}")
        
        # 对比提升
        print("\n📈 TTA提升:")
        iou_gain = metrics_tta['iou'] - metrics_no_tta['iou']
        print(f"   IoU: {metrics_no_tta['iou']:.4f} → {metrics_tta['iou']:.4f} ({'+' if iou_gain > 0 else ''}{iou_gain:.4f})")
        
        if 'dice' in metrics_tta:
            dice_gain = metrics_tta['dice'] - metrics_no_tta['dice']
            print(f"   Dice: {metrics_no_tta['dice']:.4f} → {metrics_tta['dice']:.4f} ({'+' if dice_gain > 0 else ''}{dice_gain:.4f})")
    
    print("\n" + "="*50)
    print("评估完成!")
    print("="*50)


if __name__ == "__main__":
    main()
