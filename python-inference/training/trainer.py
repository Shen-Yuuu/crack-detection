"""
高性能训练框架
包含EMA、SWA、混合精度、梯度累积等
"""

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
from collections import defaultdict
import time


class EMA:
    """指数移动平均（Exponential Moving Average）"""
    
    def __init__(self, model: nn.Module, decay: float = 0.9995):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        self.register()
    
    def register(self):
        """注册模型参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """更新EMA参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """应用EMA参数到模型"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """恢复原始参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class SWA:
    """随机权重平均（Stochastic Weight Averaging）"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.swa_model = None
        self.swa_n = 0
    
    def update(self):
        """更新SWA模型"""
        if self.swa_model is None:
            self.swa_model = {}
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.swa_model[name] = param.data.clone()
        else:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.swa_model[name] = (
                        self.swa_model[name] * self.swa_n + param.data
                    ) / (self.swa_n + 1)
        
        self.swa_n += 1
    
    def apply_swa(self):
        """应用SWA参数到模型"""
        if self.swa_model is not None:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.data = self.swa_model[name]


class MetricTracker:
    """指标追踪器"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.metrics = defaultdict(list)
    
    def update(self, metrics: Dict[str, float]):
        for k, v in metrics.items():
            self.metrics[k].append(v)
    
    def avg(self, key: str) -> float:
        if key in self.metrics and len(self.metrics[key]) > 0:
            return np.mean(self.metrics[key])
        return 0.0
    
    def get_summary(self) -> Dict[str, float]:
        return {k: np.mean(v) for k, v in self.metrics.items()}


class Trainer:
    """训练器"""
    
    def __init__(self,
                 model: nn.Module,
                 optimizer: Optimizer,
                 loss_fn: nn.Module,
                 scheduler: Optional[_LRScheduler] = None,
                 device: str = 'cuda',
                 output_dir: str = './outputs',
                 use_amp: bool = True,
                 use_ema: bool = True,
                 ema_decay: float = 0.9995,
                 use_swa: bool = True,
                 swa_start_epoch: int = 180,
                 gradient_accumulation_steps: int = 1,
                 max_grad_norm: float = 1.0,
                 log_interval: int = 50):
        
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 混合精度训练
        self.use_amp = use_amp
        self.scaler = GradScaler('cuda') if use_amp else None
        
        # EMA
        self.use_ema = use_ema
        self.ema = EMA(model, decay=ema_decay) if use_ema else None
        
        # SWA
        self.use_swa = use_swa
        self.swa = SWA(model) if use_swa else None
        self.swa_start_epoch = swa_start_epoch
        
        # 梯度累积
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        
        # 日志
        self.log_interval = log_interval
        self.logger = self._setup_logger()
        
        # 追踪
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = 0.0
        
        # 指标追踪
        self.train_tracker = MetricTracker()
        self.val_tracker = MetricTracker()
    
    def _setup_logger(self):
        """设置日志"""
        logger = logging.getLogger('Trainer')
        logger.setLevel(logging.INFO)
        
        # 文件处理器
        fh = logging.FileHandler(self.output_dir / 'train.log')
        fh.setLevel(logging.INFO)
        
        # 控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # 格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        self.train_tracker.reset()
        
        # 动态调整增强强度（通过epoch_ratio）
        epoch_ratio = epoch / 200  # total_epochs 默认为 200
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # 前向传播
            with autocast('cuda', enabled=self.use_amp):
                outputs = self.model(images)
                
                # 计算损失
                losses = self.loss_fn(
                    pred=outputs['out'],
                    target=masks,
                    edge_pred=outputs.get('edge'),
                    aux_preds=outputs.get('aux')
                )
                
                loss = losses['total'] / self.gradient_accumulation_steps
            
            # 反向传播
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 梯度累积
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    # 梯度裁剪
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.max_grad_norm
                    )
                    
                    # 优化器步进
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.max_grad_norm
                    )
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # 更新EMA
                if self.use_ema:
                    self.ema.update()
                
                self.global_step += 1
            
            # 记录指标
            metrics = {k: v.item() if isinstance(v, torch.Tensor) else v 
                      for k, v in losses.items()}
            self.train_tracker.update(metrics)
            
            # 更新进度条
            if batch_idx % self.log_interval == 0:
                pbar.set_postfix({
                    'loss': f"{metrics['total']:.4f}",
                    'dice': f"{metrics.get('dice', 0):.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
                })
        
        # 学习率调度
        if self.scheduler is not None:
            self.scheduler.step()
        
        # SWA更新
        if self.use_swa and epoch >= self.swa_start_epoch:
            self.swa.update()
            self.logger.info(f"SWA updated at epoch {epoch}")
        
        return self.train_tracker.get_summary()
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """验证"""
        # 如果使用EMA，应用EMA权重
        if self.use_ema:
            self.ema.apply_shadow()
        
        self.model.eval()
        self.val_tracker.reset()
        
        pbar = tqdm(val_loader, desc='Validation')
        
        all_preds = []
        all_targets = []
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            with autocast('cuda', enabled=self.use_amp):
                outputs = self.model(images)
                pred = outputs['out']
                
                # 计算损失
                losses = self.loss_fn(pred=pred, target=masks)
            
            # 记录指标
            metrics = {k: v.item() if isinstance(v, torch.Tensor) else v 
                      for k, v in losses.items()}
            self.val_tracker.update(metrics)
            
            # 保存预测用于计算IoU
            pred_binary = (torch.sigmoid(pred) > 0.5).float()
            all_preds.append(pred_binary.cpu())
            all_targets.append(masks.cpu())
        
        # 计算IoU
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        iou = self.compute_iou(all_preds, all_targets)
        
        val_metrics = self.val_tracker.get_summary()
        val_metrics['iou'] = iou
        
        # 恢复原始权重
        if self.use_ema:
            self.ema.restore()
        
        return val_metrics
    
    def compute_iou(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        """计算IoU"""
        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum() - intersection
        iou = (intersection + 1e-8) / (union + 1e-8)
        return iou.item()
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'best_metric': self.best_metric,
        }
        
        if self.use_ema:
            checkpoint['ema_shadow'] = self.ema.shadow
        
        if self.use_swa and self.swa.swa_model is not None:
            checkpoint['swa_model'] = self.swa.swa_model
        
        # 保存最新检查点
        torch.save(checkpoint, self.output_dir / 'last.pth')
        
        # 保存最佳检查点
        if is_best:
            torch.save(checkpoint, self.output_dir / 'best.pth')
            self.logger.info(f"Best model saved with IoU: {metrics['iou']:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch'] + 1  # 从下一个epoch开始
        self.best_metric = checkpoint.get('best_metric', 0.0)
        
        if self.use_ema and 'ema_shadow' in checkpoint:
            self.ema.shadow = checkpoint['ema_shadow']
        
        if self.use_swa and 'swa_model' in checkpoint:
            self.swa.swa_model = checkpoint['swa_model']
        
        self.logger.info(f"✅ 成功加载检查点: {checkpoint_path}")
        self.logger.info(f"   恢复到 Epoch {self.current_epoch}")
        self.logger.info(f"   当前最佳 IoU: {self.best_metric:.4f}")
        
        # 返回检查点信息
        return {
            'epoch': self.current_epoch,
            'best_metric': self.best_metric,
            'metrics': checkpoint.get('metrics', {})
        }
    
    def train(self, 
              train_loader: DataLoader, 
              val_loader: DataLoader, 
              num_epochs: int,
              early_stopping_patience: int = 20):
        """完整训练流程"""
        
        self.logger.info("Starting training...")
        self.logger.info(f"Total epochs: {num_epochs}")
        self.logger.info(f"Training samples: {len(train_loader.dataset)}")
        self.logger.info(f"Validation samples: {len(val_loader.dataset)}")
        
        patience_counter = 0
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # 训练
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # 验证
            val_metrics = self.validate(val_loader)
            
            # 日志
            self.logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_metrics['total']:.4f}, "
                f"Val Loss: {val_metrics['total']:.4f}, "
                f"Val IoU: {val_metrics['iou']:.4f}"
            )
            
            # 保存检查点
            is_best = val_metrics['iou'] > self.best_metric
            if is_best:
                self.best_metric = val_metrics['iou']
                patience_counter = 0
            else:
                patience_counter += 1
            
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # 早停
            if patience_counter >= early_stopping_patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # 应用SWA
        if self.use_swa and self.swa.swa_model is not None:
            self.logger.info("Applying SWA weights...")
            self.swa.apply_swa()
            
            # 验证SWA模型
            swa_metrics = self.validate(val_loader)
            self.logger.info(f"SWA Val IoU: {swa_metrics['iou']:.4f}")
            
            # 如果SWA更好，保存
            if swa_metrics['iou'] > self.best_metric:
                self.save_checkpoint(num_epochs, swa_metrics, is_best=True)
                self.logger.info("SWA model is better, saved as best model")
        
        self.logger.info("Training finished!")
        self.logger.info(f"Best Val IoU: {self.best_metric:.4f}")


if __name__ == "__main__":
    # 测试训练器
    from models.convnext_upernet import ConvNeXtUPerNet
    from losses import CombinedLoss
    
    model = ConvNeXtUPerNet(
        encoder_name='convnext_tiny',
        pretrained=False
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = CombinedLoss()
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        use_amp=True,
        use_ema=True,
        use_swa=True
    )
    
    print("Trainer initialized successfully!")

