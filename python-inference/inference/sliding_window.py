"""
高分辨率滑窗推理与TTA（Test Time Augmentation）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional
import cv2


class SlidingWindowInference:
    """滑窗推理（用于高分辨率图像）"""
    
    def __init__(self,
                 window_size: Tuple[int, int] = (1024, 1024),
                 overlap: float = 0.25,
                 batch_size: int = 4,
                 blend_mode: str = 'gaussian'):
        """
        Args:
            window_size: 滑窗大小
            overlap: 重叠比例 (0-1)
            batch_size: 批处理大小
            blend_mode: 融合模式 ('gaussian', 'linear', 'constant')
        """
        self.window_size = window_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.blend_mode = blend_mode
        
        # 生成融合权重图
        self.blend_weights = self._create_blend_weights()
    
    def _create_blend_weights(self) -> np.ndarray:
        """创建融合权重图"""
        h, w = self.window_size
        
        if self.blend_mode == 'gaussian':
            # 高斯权重（中心权重大，边缘权重小）
            center_y, center_x = h // 2, w // 2
            
            y = np.arange(h) - center_y
            x = np.arange(w) - center_x
            
            yy, xx = np.meshgrid(y, x, indexing='ij')
            
            sigma_y = h / 4
            sigma_x = w / 4
            
            weights = np.exp(-(yy**2 / (2 * sigma_y**2) + xx**2 / (2 * sigma_x**2)))
            
        elif self.blend_mode == 'linear':
            # 线性权重
            y = np.linspace(0, 1, h)
            x = np.linspace(0, 1, w)
            
            y_weights = np.minimum(y, 1 - y) * 2
            x_weights = np.minimum(x, 1 - x) * 2
            
            weights = np.outer(y_weights, x_weights)
            
        else:  # constant
            weights = np.ones((h, w))
        
        return weights
    
    def _compute_windows(self, image_shape: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
        """计算滑窗位置"""
        h, w = image_shape
        window_h, window_w = self.window_size
        
        stride_h = int(window_h * (1 - self.overlap))
        stride_w = int(window_w * (1 - self.overlap))
        
        windows = []
        
        for y in range(0, h, stride_h):
            for x in range(0, w, stride_w):
                # 确保窗口不超出图像边界
                y_end = min(y + window_h, h)
                x_end = min(x + window_w, w)
                
                # 调整起始位置确保窗口大小一致
                y_start = max(0, y_end - window_h)
                x_start = max(0, x_end - window_w)
                
                windows.append((y_start, y_end, x_start, x_end))
        
        return windows
    
    @torch.no_grad()
    def __call__(self, 
                 model: nn.Module,
                 image: torch.Tensor,
                 device: str = 'cuda') -> torch.Tensor:
        """
        Args:
            model: 分割模型
            image: (C, H, W) 单张图像
            device: 设备
        Returns:
            prediction: (1, H, W) 分割结果
        """
        model.eval()
        
        c, h, w = image.shape
        
        # 计算滑窗
        windows = self._compute_windows((h, w))
        
        # 累积预测和权重
        pred_sum = np.zeros((h, w), dtype=np.float32)
        weight_sum = np.zeros((h, w), dtype=np.float32)
        
        # 批处理窗口
        window_batches = [windows[i:i + self.batch_size] 
                         for i in range(0, len(windows), self.batch_size)]
        
        for window_batch in window_batches:
            batch_crops = []
            
            for y1, y2, x1, x2 in window_batch:
                crop = image[:, y1:y2, x1:x2]
                batch_crops.append(crop)
            
            # 批处理推理
            batch_tensor = torch.stack(batch_crops).to(device)
            
            outputs = model(batch_tensor)
            preds = torch.sigmoid(outputs['out']).cpu().numpy()
            
            # 融合预测
            for i, (y1, y2, x1, x2) in enumerate(window_batch):
                pred_window = preds[i, 0]
                
                # 应用融合权重
                weighted_pred = pred_window * self.blend_weights
                
                pred_sum[y1:y2, x1:x2] += weighted_pred
                weight_sum[y1:y2, x1:x2] += self.blend_weights
        
        # 归一化
        pred_avg = pred_sum / (weight_sum + 1e-8)
        
        return torch.from_numpy(pred_avg).unsqueeze(0)


class TTAInference:
    """测试时增强（Test Time Augmentation）"""
    
    def __init__(self,
                 scales: List[float] = [0.75, 1.0, 1.25],
                 flip_h: bool = True,
                 flip_v: bool = True,
                 use_sliding_window: bool = False,
                 window_config: Optional[dict] = None):
        """
        Args:
            scales: 多尺度列表
            flip_h: 是否水平翻转
            flip_v: 是否垂直翻转
            use_sliding_window: 是否使用滑窗
            window_config: 滑窗配置
        """
        self.scales = scales
        self.flip_h = flip_h
        self.flip_v = flip_v
        self.use_sliding_window = use_sliding_window
        
        if use_sliding_window and window_config:
            self.sliding_window = SlidingWindowInference(**window_config)
        else:
            self.sliding_window = None
    
    def _apply_augmentation(self, 
                           image: torch.Tensor,
                           scale: float = 1.0,
                           flip_h: bool = False,
                           flip_v: bool = False) -> torch.Tensor:
        """应用增强"""
        # 缩放
        if scale != 1.0:
            c, h, w = image.shape
            new_h, new_w = int(h * scale), int(w * scale)
            image = F.interpolate(
                image.unsqueeze(0),
                size=(new_h, new_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        # 翻转
        if flip_h:
            image = torch.flip(image, dims=[2])
        
        if flip_v:
            image = torch.flip(image, dims=[1])
        
        return image
    
    def _reverse_augmentation(self,
                            pred: torch.Tensor,
                            original_size: Tuple[int, int],
                            scale: float = 1.0,
                            flip_h: bool = False,
                            flip_v: bool = False) -> torch.Tensor:
        """反向增强"""
        # 反向翻转
        if flip_v:
            pred = torch.flip(pred, dims=[1])
        
        if flip_h:
            pred = torch.flip(pred, dims=[2])
        
        # 反向缩放
        if scale != 1.0:
            pred = F.interpolate(
                pred.unsqueeze(0),
                size=original_size,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        return pred
    
    @torch.no_grad()
    def __call__(self,
                 model: nn.Module,
                 image: torch.Tensor,
                 device: str = 'cuda') -> torch.Tensor:
        """
        Args:
            model: 分割模型
            image: (C, H, W) 单张图像
            device: 设备
        Returns:
            prediction: (1, H, W) 融合后的预测
        """
        model.eval()
        
        original_size = image.shape[1:]
        all_preds = []
        
        # 多尺度
        for scale in self.scales:
            # 基础预测
            aug_image = self._apply_augmentation(image, scale=scale)
            
            if self.use_sliding_window and self.sliding_window:
                pred = self.sliding_window(model, aug_image, device)
            else:
                aug_image_batch = aug_image.unsqueeze(0).to(device)
                outputs = model(aug_image_batch)
                pred = torch.sigmoid(outputs['out'][0]).cpu()
            
            pred = self._reverse_augmentation(pred, original_size, scale=scale)
            all_preds.append(pred)
            
            # 水平翻转
            if self.flip_h:
                aug_image = self._apply_augmentation(image, scale=scale, flip_h=True)
                
                if self.use_sliding_window and self.sliding_window:
                    pred = self.sliding_window(model, aug_image, device)
                else:
                    aug_image_batch = aug_image.unsqueeze(0).to(device)
                    outputs = model(aug_image_batch)
                    pred = torch.sigmoid(outputs['out'][0]).cpu()
                
                pred = self._reverse_augmentation(pred, original_size, scale=scale, flip_h=True)
                all_preds.append(pred)
            
            # 垂直翻转
            if self.flip_v:
                aug_image = self._apply_augmentation(image, scale=scale, flip_v=True)
                
                if self.use_sliding_window and self.sliding_window:
                    pred = self.sliding_window(model, aug_image, device)
                else:
                    aug_image_batch = aug_image.unsqueeze(0).to(device)
                    outputs = model(aug_image_batch)
                    pred = torch.sigmoid(outputs['out'][0]).cpu()
                
                pred = self._reverse_augmentation(pred, original_size, scale=scale, flip_v=True)
                all_preds.append(pred)
        
        # 平均融合
        final_pred = torch.stack(all_preds).mean(dim=0)
        
        return final_pred


class TemperatureScaling:
    """温度标定（Temperature Scaling）- 校准预测概率"""
    
    def __init__(self, temperature: float = 1.0):
        self.temperature = nn.Parameter(torch.ones(1) * temperature)
    
    def calibrate(self, 
                  model: nn.Module,
                  val_loader,
                  device: str = 'cuda',
                  max_iters: int = 50):
        """在验证集上校准温度"""
        model.eval()
        
        logits_list = []
        labels_list = []
        
        # 收集logits和labels
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                labels = batch['mask']
                
                outputs = model(images)
                logits = outputs['out'].cpu()
                
                logits_list.append(logits)
                labels_list.append(labels)
        
        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)
        
        # 优化温度
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iters)
        
        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = logits / self.temperature
            loss = F.binary_cross_entropy_with_logits(scaled_logits, labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        
        print(f"Optimal temperature: {self.temperature.item():.4f}")
    
    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """应用温度缩放"""
        return logits / self.temperature


if __name__ == "__main__":
    # 测试滑窗推理
    from models.convnext_upernet import ConvNeXtUPerNet
    
    model = ConvNeXtUPerNet(encoder_name='convnext_tiny', pretrained=False)
    model.eval()
    
    # 高分辨率图像
    image = torch.randn(3, 2048, 2048)
    
    # 滑窗推理
    sliding_window = SlidingWindowInference(
        window_size=(512, 512),
        overlap=0.25,
        batch_size=4
    )
    
    pred = sliding_window(model, image, device='cpu')
    print(f"Sliding window prediction shape: {pred.shape}")
    
    # TTA推理
    tta = TTAInference(
        scales=[0.75, 1.0, 1.25],
        flip_h=True,
        flip_v=True
    )
    
    small_image = torch.randn(3, 512, 512)
    pred_tta = tta(model, small_image, device='cpu')
    print(f"TTA prediction shape: {pred_tta.shape}")

