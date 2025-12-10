"""
增强版 TTA 推理模块
支持多尺度、多翻转、旋转增强
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional, Dict
from pathlib import Path
import cv2
from tqdm import tqdm


class EnhancedTTAInference:
    """增强版测试时增强推理
    
    特性:
    - 多尺度推理
    - 水平/垂直翻转
    - 90度旋转增强
    - 加权融合策略
    - 边界感知后处理
    """
    
    def __init__(self,
                 scales: List[float] = [0.75, 1.0, 1.25],
                 flip_h: bool = True,
                 flip_v: bool = True,
                 rotate_90: bool = True,
                 weight_by_scale: bool = True,
                 use_edge_aware: bool = True):
        """
        Args:
            scales: 多尺度列表
            flip_h: 是否水平翻转
            flip_v: 是否垂直翻转  
            rotate_90: 是否使用90度旋转增强
            weight_by_scale: 是否按尺度加权（原始尺度权重更高）
            use_edge_aware: 是否使用边界感知融合
        """
        self.scales = scales
        self.flip_h = flip_h
        self.flip_v = flip_v
        self.rotate_90 = rotate_90
        self.weight_by_scale = weight_by_scale
        self.use_edge_aware = use_edge_aware
        
        # 计算尺度权重（原始尺度1.0权重最高）
        if weight_by_scale:
            self.scale_weights = {s: 1.0 / (abs(s - 1.0) + 0.5) for s in scales}
        else:
            self.scale_weights = {s: 1.0 for s in scales}
    
    def _get_augmentation_configs(self) -> List[Dict]:
        """生成所有增强配置"""
        configs = []
        
        for scale in self.scales:
            # 原图
            configs.append({
                'scale': scale, 
                'flip_h': False, 
                'flip_v': False, 
                'rotate': 0,
                'weight': self.scale_weights[scale]
            })
            
            # 水平翻转
            if self.flip_h:
                configs.append({
                    'scale': scale, 
                    'flip_h': True, 
                    'flip_v': False, 
                    'rotate': 0,
                    'weight': self.scale_weights[scale] * 0.9
                })
            
            # 垂直翻转
            if self.flip_v:
                configs.append({
                    'scale': scale, 
                    'flip_h': False, 
                    'flip_v': True, 
                    'rotate': 0,
                    'weight': self.scale_weights[scale] * 0.9
                })
            
            # 水平+垂直翻转（等价于180度旋转）
            if self.flip_h and self.flip_v:
                configs.append({
                    'scale': scale, 
                    'flip_h': True, 
                    'flip_v': True, 
                    'rotate': 0,
                    'weight': self.scale_weights[scale] * 0.8
                })
            
            # 90度旋转增强
            if self.rotate_90:
                for angle in [90, 270]:
                    configs.append({
                        'scale': scale,
                        'flip_h': False,
                        'flip_v': False,
                        'rotate': angle,
                        'weight': self.scale_weights[scale] * 0.85
                    })
        
        return configs
    
    def _apply_transform(self, 
                        image: torch.Tensor,
                        config: Dict) -> torch.Tensor:
        """应用变换"""
        img = image.clone()
        
        # 缩放
        if config['scale'] != 1.0:
            c, h, w = img.shape
            new_h, new_w = int(h * config['scale']), int(w * config['scale'])
            img = F.interpolate(
                img.unsqueeze(0),
                size=(new_h, new_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        # 翻转
        if config['flip_h']:
            img = torch.flip(img, dims=[2])
        if config['flip_v']:
            img = torch.flip(img, dims=[1])
        
        # 旋转
        if config['rotate'] == 90:
            img = torch.rot90(img, k=1, dims=[1, 2])
        elif config['rotate'] == 270:
            img = torch.rot90(img, k=3, dims=[1, 2])
        
        return img
    
    def _reverse_transform(self,
                          pred: torch.Tensor,
                          original_size: Tuple[int, int],
                          config: Dict) -> torch.Tensor:
        """反向变换"""
        # 反向旋转
        if config['rotate'] == 90:
            pred = torch.rot90(pred, k=3, dims=[1, 2])
        elif config['rotate'] == 270:
            pred = torch.rot90(pred, k=1, dims=[1, 2])
        
        # 反向翻转
        if config['flip_v']:
            pred = torch.flip(pred, dims=[1])
        if config['flip_h']:
            pred = torch.flip(pred, dims=[2])
        
        # 反向缩放
        if config['scale'] != 1.0:
            pred = F.interpolate(
                pred.unsqueeze(0),
                size=original_size,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        return pred
    
    def _edge_aware_fusion(self, 
                          preds: List[torch.Tensor],
                          weights: List[float]) -> torch.Tensor:
        """边界感知融合 - 在边界区域使用更保守的融合"""
        # 加权平均
        weight_sum = sum(weights)
        weighted_preds = [p * w for p, w in zip(preds, weights)]
        mean_pred = torch.stack(weighted_preds).sum(dim=0) / weight_sum
        
        if not self.use_edge_aware:
            return mean_pred
        
        # 计算预测的不确定性（各增强预测的方差）
        pred_stack = torch.stack(preds)
        uncertainty = pred_stack.var(dim=0)
        
        # 在高不确定性区域（通常是边界），使用中位数而非均值
        median_pred = pred_stack.median(dim=0)[0]
        
        # 根据不确定性混合均值和中位数
        # 不确定性高时更倾向于中位数
        uncertainty_weight = torch.clamp(uncertainty * 5, 0, 1)
        fused_pred = mean_pred * (1 - uncertainty_weight) + median_pred * uncertainty_weight
        
        return fused_pred
    
    @torch.no_grad()
    def __call__(self,
                 model: nn.Module,
                 image: torch.Tensor,
                 device: str = 'cuda',
                 return_uncertainty: bool = False) -> torch.Tensor:
        """
        TTA推理
        
        Args:
            model: 分割模型
            image: (C, H, W) 单张图像（已归一化）
            device: 设备
            return_uncertainty: 是否返回不确定性图
            
        Returns:
            prediction: (1, H, W) 融合后的预测概率图
        """
        model.eval()
        
        original_size = image.shape[1:]
        configs = self._get_augmentation_configs()
        
        all_preds = []
        all_weights = []
        
        for config in configs:
            # 应用变换
            aug_image = self._apply_transform(image, config)
            
            # 推理
            aug_batch = aug_image.unsqueeze(0).to(device)
            outputs = model(aug_batch)
            pred = torch.sigmoid(outputs['out'][0]).cpu()
            
            # 反向变换
            pred = self._reverse_transform(pred, original_size, config)
            
            all_preds.append(pred)
            all_weights.append(config['weight'])
        
        # 融合预测
        final_pred = self._edge_aware_fusion(all_preds, all_weights)
        
        if return_uncertainty:
            pred_stack = torch.stack(all_preds)
            uncertainty = pred_stack.var(dim=0)
            return final_pred, uncertainty
        
        return final_pred
    
    def inference_with_sliding_window(self,
                                      model: nn.Module,
                                      image: torch.Tensor,
                                      window_size: Tuple[int, int] = (512, 512),
                                      overlap: float = 0.25,
                                      device: str = 'cuda') -> torch.Tensor:
        """结合滑窗的TTA推理（用于超高分辨率图像）"""
        from inference.sliding_window import SlidingWindowInference
        
        original_size = image.shape[1:]
        configs = self._get_augmentation_configs()
        
        sliding_window = SlidingWindowInference(
            window_size=window_size,
            overlap=overlap,
            batch_size=4,
            blend_mode='gaussian'
        )
        
        all_preds = []
        all_weights = []
        
        for config in tqdm(configs, desc="TTA + Sliding Window"):
            # 应用变换
            aug_image = self._apply_transform(image, config)
            
            # 滑窗推理
            pred = sliding_window(model, aug_image, device)
            
            # 反向变换
            pred = self._reverse_transform(pred, original_size, config)
            
            all_preds.append(pred)
            all_weights.append(config['weight'])
        
        # 融合
        final_pred = self._edge_aware_fusion(all_preds, all_weights)
        
        return final_pred


class PostProcessor:
    """后处理模块"""
    
    def __init__(self,
                 threshold: float = 0.5,
                 min_area: int = 50,
                 use_morphology: bool = True,
                 kernel_size: int = 3):
        self.threshold = threshold
        self.min_area = min_area
        self.use_morphology = use_morphology
        self.kernel_size = kernel_size
    
    def __call__(self, pred: torch.Tensor) -> np.ndarray:
        """
        后处理预测结果
        
        Args:
            pred: (1, H, W) 概率图
            
        Returns:
            binary_mask: (H, W) 二值化结果
        """
        # 转numpy
        pred_np = pred.squeeze().numpy()
        
        # 二值化
        binary = (pred_np > self.threshold).astype(np.uint8)
        
        if self.use_morphology:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, 
                (self.kernel_size, self.kernel_size)
            )
            
            # 开运算（去除小噪点）
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            # 闭运算（填充小孔洞）
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 去除小连通域
        if self.min_area > 0:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                binary, connectivity=8
            )
            
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] < self.min_area:
                    binary[labels == i] = 0
        
        return binary


def evaluate_with_tta(model: nn.Module,
                     val_loader,
                     device: str = 'cuda',
                     tta_config: Optional[Dict] = None) -> Dict[str, float]:
    """使用TTA评估模型
    
    Args:
        model: 分割模型
        val_loader: 验证数据加载器
        device: 设备
        tta_config: TTA配置
        
    Returns:
        metrics: 评估指标字典
    """
    if tta_config is None:
        tta_config = {
            'scales': [0.75, 1.0, 1.25],
            'flip_h': True,
            'flip_v': True,
            'rotate_90': False,
        }
    
    tta = EnhancedTTAInference(**tta_config)
    
    model.eval()
    
    all_preds = []
    all_targets = []
    
    for batch in tqdm(val_loader, desc="Evaluating with TTA"):
        images = batch['image']
        masks = batch['mask']
        
        for i in range(images.shape[0]):
            # 单张图像TTA推理
            pred = tta(model, images[i], device)
            pred_binary = (pred > 0.5).float()
            
            all_preds.append(pred_binary)
            all_targets.append(masks[i])
    
    # 计算指标
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # IoU
    intersection = (all_preds * all_targets).sum()
    union = all_preds.sum() + all_targets.sum() - intersection
    iou = (intersection + 1e-8) / (union + 1e-8)
    
    # Dice
    dice = (2 * intersection + 1e-8) / (all_preds.sum() + all_targets.sum() + 1e-8)
    
    # Precision & Recall
    tp = intersection
    fp = all_preds.sum() - tp
    fn = all_targets.sum() - tp
    
    precision = (tp + 1e-8) / (tp + fp + 1e-8)
    recall = (tp + 1e-8) / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        'iou': iou.item(),
        'dice': dice.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item()
    }


if __name__ == "__main__":
    # 测试
    print("Testing Enhanced TTA Inference...")
    
    # 模拟模型输出
    class DummyModel(nn.Module):
        def forward(self, x):
            return {'out': torch.randn(x.shape[0], 1, x.shape[2], x.shape[3])}
    
    model = DummyModel()
    image = torch.randn(3, 512, 512)
    
    # 测试TTA
    tta = EnhancedTTAInference(
        scales=[0.75, 1.0, 1.25],
        flip_h=True,
        flip_v=True,
        rotate_90=True,
        use_edge_aware=True
    )
    
    configs = tta._get_augmentation_configs()
    print(f"Total TTA configurations: {len(configs)}")
    
    pred = tta(model, image, device='cpu')
    print(f"TTA prediction shape: {pred.shape}")
    
    # 测试后处理
    post = PostProcessor(threshold=0.5, min_area=50)
    binary = post(torch.sigmoid(pred))
    print(f"Post-processed shape: {binary.shape}")
