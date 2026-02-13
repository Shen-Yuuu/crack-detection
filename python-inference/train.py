"""
完整训练脚本
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    OneCycleLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    LinearLR,
    SequentialLR,
)
import yaml
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from dataset.data_loader import create_dataloaders, DatasetConfig
from models.convnext_upernet import create_model
from training.losses import create_loss
from training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train crack segmentation model')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    return parser.parse_args()


def create_optimizer(model: nn.Module, config: dict):
    """创建优化器"""
    optimizer_type = config.get('type', 'adamw')
    lr = config.get('lr', 1e-3)
    weight_decay = config.get('weight_decay', 1e-4)
    
    if optimizer_type == 'adamw':
        optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
    elif optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    return optimizer


def create_scheduler(optimizer, config: dict, steps_per_epoch: int, total_epochs: int):
    """创建学习率调度器"""
    scheduler_type = config.get('type', 'cosine')
    num_epochs = config.get('epochs', total_epochs)
    
    if scheduler_type == 'onecycle':
        scheduler = OneCycleLR(
            optimizer,
            max_lr=config.get('max_lr', 1e-3),
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=config.get('warmup_pct', 0.05)
        )
        step_mode = 'step'
    elif scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=config.get('min_lr', 1e-6)
        )
        step_mode = 'epoch'
    elif scheduler_type == 'cosine_warmup':
        warmup_epochs = int(config.get('warmup_epochs', max(1, int(0.05 * num_epochs))))
        warmup_epochs = max(warmup_epochs, 1)
        warmup_start = float(config.get('warmup_start_factor', 0.1))
        warmup_start = max(min(warmup_start, 1.0), 1e-4)

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=warmup_start,
            end_factor=1.0,
            total_iters=warmup_epochs
        )

        cosine_epochs = max(num_epochs - warmup_epochs, 1)
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cosine_epochs,
            eta_min=config.get('min_lr', 1e-6)
        )

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )
        step_mode = 'epoch'
    elif scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=config.get('mode', 'max'),
            factor=config.get('factor', 0.5),
            patience=config.get('patience', 5),
            verbose=True,
            min_lr=config.get('min_lr', 1e-7)
        )
        step_mode = 'metric'
    else:
        scheduler = None
        step_mode = None
    
    return scheduler, step_mode


def find_latest_checkpoint(output_dir):
    """查找最新的检查点"""
    output_path = Path(output_dir)
    last_checkpoint = output_path / 'last.pth'
    
    if last_checkpoint.exists():
        return str(last_checkpoint)
    return None


def main():
    args = parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("=" * 60)
    print("Configuration:")
    print(yaml.dump(config, default_flow_style=False))
    print("=" * 60)
    
    # 设置设备
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 自动检测检查点（如果未手动指定）
    resume_checkpoint = args.resume
    if not resume_checkpoint:
        auto_checkpoint = find_latest_checkpoint(config['training']['output_dir'])
        if auto_checkpoint:
            print(f"\n🔍 发现已存在的检查点: {auto_checkpoint}")
            response = input("是否从该检查点继续训练? [y/N]: ").strip().lower()
            if response in ['y', 'yes']:
                resume_checkpoint = auto_checkpoint
                print("✅ 将从检查点恢复训练")
            else:
                print("⚠️  将开始新的训练（已有检查点将被覆盖）")
    
    # 创建数据加载器
    print("\nCreating dataloaders...")
    dataset_config = DatasetConfig(**config['dataset'])
    
    train_loader, val_loader = create_dataloaders(
        config=dataset_config,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers']
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # 创建模型
    print("\nCreating model...")
    model = create_model(config['model'])
    print(f"Model: {config['model']['backbone']}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    
    # 创建损失函数
    print("\nCreating loss function...")
    loss_fn = create_loss(config['loss'])
    
    # 创建优化器
    print("\nCreating optimizer...")
    optimizer = create_optimizer(model, config['optimizer'])
    
    # 创建调度器
    scheduler, scheduler_step = create_scheduler(
        optimizer,
        config['scheduler'],
        len(train_loader),
        config['training']['epochs']
    )
    
    # 创建训练器
    print("\nCreating trainer...")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=scheduler,
        scheduler_step=scheduler_step or 'epoch',
        device=device,
        output_dir=config['training']['output_dir'],
        use_amp=config['training'].get('use_amp', True),
        use_ema=config['training'].get('use_ema', True),
        ema_decay=config['training'].get('ema_decay', 0.9995),
        use_swa=config['training'].get('use_swa', True),
        swa_start_epoch=config['training'].get('swa_start_epoch', 180),
        gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 1),
        max_grad_norm=config['training'].get('max_grad_norm', 1.0)
    )
    
    # 从检查点恢复
    if resume_checkpoint:
        print(f"\n{'='*60}")
        print(f"📥 正在从检查点恢复训练...")
        print(f"检查点路径: {resume_checkpoint}")
        trainer.load_checkpoint(resume_checkpoint)
        print(f"恢复epoch: {trainer.current_epoch + 1}")
        print(f"当前最佳IoU: {trainer.best_metric:.4f}")
        print(f"{'='*60}\n")
    
    # 开始训练
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")
    
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['epochs'],
        early_stopping_patience=config['training'].get('early_stopping_patience', 20)
    )
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()

