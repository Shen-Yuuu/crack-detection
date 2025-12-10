"""
高级损失函数组合
包括Dice、Focal、BCE、Boundary、Lovasz等
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np
from scipy.ndimage import distance_transform_edt


class DiceLoss(nn.Module):
    """Soft Dice Loss"""
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, 1, H, W) logits
            target: (B, 1, H, W) binary labels
        """
        pred = torch.sigmoid(pred)
        
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        
        dice = (2. * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        
        return 1 - dice


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, 1, H, W) logits
            target: (B, 1, H, W) binary labels
        """
        # 使用 binary_cross_entropy_with_logits（AMP 安全）
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # 计算概率（用于 focal weight）
        pred_prob = torch.sigmoid(pred)
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
        focal_weight = (1 - p_t) ** self.gamma
        
        if self.alpha >= 0:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss
        
        return focal_loss.mean()


class TverskyLoss(nn.Module):
    """Tversky Loss - 可调节FP/FN权重"""
    
    def __init__(self, alpha: float = 0.5, beta: float = 0.7, smooth: float = 1.0):
        """
        Args:
            alpha: weight for FP
            beta: weight for FN
            通常设置 beta > alpha 来惩罚FN（漏检）
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        TP = (pred_flat * target_flat).sum()
        FP = ((1 - target_flat) * pred_flat).sum()
        FN = (target_flat * (1 - pred_flat)).sum()
        
        tversky_index = (TP + self.smooth) / (
            TP + self.alpha * FP + self.beta * FN + self.smooth
        )
        
        return 1 - tversky_index


class FocalTverskyLoss(nn.Module):
    """Focal Tversky Loss - 结合Focal和Tversky的优点"""
    
    def __init__(self, alpha: float = 0.5, beta: float = 0.7, 
                 gamma: float = 1.5, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        TP = (pred_flat * target_flat).sum()
        FP = ((1 - target_flat) * pred_flat).sum()
        FN = (target_flat * (1 - pred_flat)).sum()
        
        tversky_index = (TP + self.smooth) / (
            TP + self.alpha * FP + self.beta * FN + self.smooth
        )
        
        # Focal modulation
        focal_tversky = (1 - tversky_index) ** self.gamma
        
        return focal_tversky


class BoundaryLoss(nn.Module):
    """边界损失 - 基于距离变换"""
    
    def __init__(self):
        super().__init__()
    
    def compute_sdf(self, mask: np.ndarray) -> np.ndarray:
        """计算符号距离函数（Signed Distance Function）"""
        pos_mask = mask.astype(bool)
        neg_mask = ~pos_mask
        
        pos_dist = distance_transform_edt(pos_mask)
        neg_dist = distance_transform_edt(neg_mask)
        
        sdf = pos_dist - neg_dist
        return sdf
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, 1, H, W) logits
            target: (B, 1, H, W) binary labels
        """
        # 计算边界权重图
        boundary_weights = []
        
        for i in range(target.shape[0]):
            mask_np = target[i, 0].cpu().numpy()
            
            # 提取边界
            kernel = np.ones((3, 3), np.uint8)
            import cv2
            eroded = cv2.erode(mask_np.astype(np.uint8), kernel, iterations=1)
            boundary = mask_np - eroded
            
            # 距离变换权重
            dist = distance_transform_edt(1 - boundary)
            dist = 1.0 / (1.0 + dist)  # 边界附近权重大
            
            boundary_weights.append(torch.from_numpy(dist).float())
        
        boundary_weights = torch.stack(boundary_weights, dim=0).unsqueeze(1).to(pred.device)
        
        # 加权BCE（使用 logits，AMP 安全）
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        weighted_bce = bce * boundary_weights
        
        return weighted_bce.mean()


class LovaszHingeLoss(nn.Module):
    """Lovasz-Hinge Loss - IoU优化"""
    
    def __init__(self):
        super().__init__()
    
    def lovasz_grad(self, gt_sorted):
        """计算Lovasz扩展的梯度"""
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        
        if len(gt_sorted) > 1:
            jaccard[1:] = jaccard[1:] - jaccard[:-1]
        
        return jaccard
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, 1, H, W) logits
            target: (B, 1, H, W) binary labels
        """
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        errors = (pred_flat - target_flat).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        target_sorted = target_flat[perm]
        
        grad = self.lovasz_grad(target_sorted)
        loss = torch.dot(errors_sorted, grad)
        
        return loss


class CombinedLoss(nn.Module):
    """组合损失函数"""
    
    def __init__(self,
                 dice_weight: float = 0.4,
                 focal_weight: float = 0.3,
                 bce_weight: float = 0.2,
                 boundary_weight: float = 0.1,
                 focal_gamma: float = 2.0,
                 use_tversky: bool = False,
                 tversky_beta: float = 0.7):
        super().__init__()
        
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.bce_weight = bce_weight
        self.boundary_weight = boundary_weight
        
        # 损失组件
        self.dice_loss = DiceLoss()
        
        if use_tversky:
            self.focal_loss = FocalTverskyLoss(alpha=0.5, beta=tversky_beta, gamma=focal_gamma)
        else:
            self.focal_loss = FocalLoss(gamma=focal_gamma)
        
        self.boundary_loss = BoundaryLoss()
    
    def forward(self, 
                pred: torch.Tensor, 
                target: torch.Tensor,
                edge_pred: Optional[torch.Tensor] = None,
                aux_preds: Optional[list] = None) -> dict:
        """
        Args:
            pred: main prediction (B, 1, H, W)
            target: ground truth (B, 1, H, W)
            edge_pred: edge prediction (B, 1, H, W)
            aux_preds: auxiliary predictions for deep supervision
        """
        losses = {}
        
        # 主要损失
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        bce = F.binary_cross_entropy_with_logits(pred, target)
        boundary = self.boundary_loss(pred, target) if self.boundary_weight > 0 else 0
        
        main_loss = (
            self.dice_weight * dice +
            self.focal_weight * focal +
            self.bce_weight * bce +
            self.boundary_weight * boundary
        )
        
        losses['dice'] = dice
        losses['focal'] = focal
        losses['bce'] = bce
        if self.boundary_weight > 0:
            losses['boundary'] = boundary
        losses['main'] = main_loss
        
        # 边界损失
        if edge_pred is not None:
            # 从target提取边界
            kernel_size = 3
            kernel = torch.ones(1, 1, kernel_size, kernel_size).to(target.device)
            eroded = F.conv2d(target, kernel, padding=kernel_size//2)
            eroded = (eroded == kernel_size * kernel_size).float()
            
            edge_target = target - eroded
            
            # 对边界预测进行数值稳定性处理
            edge_pred_clamped = torch.clamp(edge_pred, min=-20.0, max=20.0)
            edge_loss = F.binary_cross_entropy_with_logits(edge_pred_clamped, edge_target)
            
            if torch.isfinite(edge_loss):
                losses['edge'] = edge_loss
                main_loss = main_loss + 0.5 * edge_loss
            else:
                losses['edge'] = torch.tensor(0.0, device=pred.device)
        
        # 深度监督损失（带数值稳定性保护）
        if aux_preds is not None:
            aux_loss = 0
            valid_aux_count = 0
            for i, aux_pred in enumerate(aux_preds):
                # 对辅助输出进行数值稳定性处理
                aux_pred_clamped = torch.clamp(aux_pred, min=-20.0, max=20.0)
                
                aux_dice = self.dice_loss(aux_pred_clamped, target)
                aux_focal = self.focal_loss(aux_pred_clamped, target)
                
                # 检查是否为有限值
                if torch.isfinite(aux_dice) and torch.isfinite(aux_focal):
                    weight = 0.4 / len(aux_preds)
                    aux_loss += weight * (aux_dice + aux_focal)
                    valid_aux_count += 1
            
            # 只有当存在有效的辅助损失时才添加
            if valid_aux_count > 0 and torch.isfinite(aux_loss):
                losses['aux'] = aux_loss
                main_loss = main_loss + aux_loss
            else:
                losses['aux'] = torch.tensor(0.0, device=pred.device)
        
        losses['total'] = main_loss
        
        return losses


class EdgeAwareLoss(nn.Module):
    """边缘感知损失"""
    
    def __init__(self, edge_weight: float = 2.0):
        super().__init__()
        self.edge_weight = edge_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """给边界区域更高的权重"""
        
        # Sobel算子提取边界
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32, device=target.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=torch.float32, device=target.device).view(1, 1, 3, 3)
        
        edge_x = F.conv2d(target, sobel_x, padding=1)
        edge_y = F.conv2d(target, sobel_y, padding=1)
        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2)
        
        # 边界权重图
        edge_weight_map = 1.0 + (self.edge_weight - 1.0) * (edge > 0.1).float()
        
        # 加权BCE（使用 logits，AMP 安全）
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        weighted_bce = bce * edge_weight_map
        
        return weighted_bce.mean()


def create_loss(loss_config: dict) -> nn.Module:
    """工厂函数创建损失"""
    
    loss_type = loss_config.get('type', 'combined')
    
    if loss_type == 'combined':
        return CombinedLoss(
            dice_weight=loss_config.get('dice', 0.4),
            focal_weight=loss_config.get('focal', 0.3),
            bce_weight=loss_config.get('bce', 0.2),
            boundary_weight=loss_config.get('boundary', 0.1),
            focal_gamma=loss_config.get('focal_gamma', 2.0),
            use_tversky=loss_config.get('use_tversky', False),
            tversky_beta=loss_config.get('tversky_beta', 0.7)
        )
    elif loss_type == 'dice':
        return DiceLoss()
    elif loss_type == 'focal':
        return FocalLoss(gamma=loss_config.get('gamma', 2.0))
    elif loss_type == 'tversky':
        return TverskyLoss(
            alpha=loss_config.get('alpha', 0.5),
            beta=loss_config.get('beta', 0.7)
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # 测试损失函数
    pred = torch.randn(2, 1, 256, 256)
    target = torch.randint(0, 2, (2, 1, 256, 256)).float()
    
    loss_fn = CombinedLoss()
    losses = loss_fn(pred, target)
    
    print("Losses:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")

