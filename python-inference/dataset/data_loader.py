"""
高性能数据加载模块
支持多种格式、多级缓存、难例挖掘
"""

import os
import json
import yaml
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import lmdb
import pickle


@dataclass
class DatasetConfig:
    """数据集配置"""
    root: str
    image_dir: str = "images"
    mask_dir: str = "masks"
    split: str = "train"  # train, val, test
    crop_size: Tuple[int, int] = (512, 512)
    train_scales: List[int] = (256, 384, 512)
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    use_cache: bool = True
    cache_dir: str = "./cache"


class AnnotationConverter:
    """多格式标注转换器"""
    
    @staticmethod
    def coco_to_mask(coco_annotation: Dict, image_shape: Tuple[int, int]) -> np.ndarray:
        """COCO格式 -> 二值掩码"""
        from pycocotools import mask as mask_utils
        
        h, w = image_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for ann in coco_annotation.get('annotations', []):
            if 'segmentation' in ann:
                # Polygon格式
                if isinstance(ann['segmentation'], list):
                    rles = mask_utils.frPyObjects(ann['segmentation'], h, w)
                    rle = mask_utils.merge(rles)
                # RLE格式
                else:
                    rle = ann['segmentation']
                m = mask_utils.decode(rle)
                mask = np.maximum(mask, m)
        
        return (mask > 0).astype(np.uint8) * 255
    
    @staticmethod
    def voc_to_mask(xml_path: str, image_shape: Tuple[int, int]) -> np.ndarray:
        """VOC XML -> 二值掩码"""
        import xml.etree.ElementTree as ET
        
        h, w = image_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        for obj in root.findall('object'):
            polygon = obj.find('polygon')
            if polygon is not None:
                points = []
                for pt in polygon:
                    x = int(pt.find('x').text)
                    y = int(pt.find('y').text)
                    points.append([x, y])
                
                points = np.array(points, dtype=np.int32)
                cv2.fillPoly(mask, [points], 255)
        
        return mask
    
    @staticmethod
    def yolo_to_mask(txt_path: str, image_shape: Tuple[int, int]) -> np.ndarray:
        """YOLO TXT -> 二值掩码（支持分割格式）"""
        h, w = image_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                # YOLO分割格式: class_id x1 y1 x2 y2 ... xn yn
                points = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
                points[:, 0] *= w
                points[:, 1] *= h
                points = points.astype(np.int32)
                
                cv2.fillPoly(mask, [points], 255)
        
        return mask


class QualityControl:
    """数据质量控制"""
    
    @staticmethod
    def check_size_consistency(image_path: str, mask_path: str) -> bool:
        """检查图像和掩码尺寸一致性"""
        img = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None or mask is None:
            return False
        
        return img.shape[:2] == mask.shape[:2]
    
    @staticmethod
    def filter_small_artifacts(mask: np.ndarray, 
                              min_area: int = 50,
                              min_aspect_ratio: float = 0.05) -> np.ndarray:
        """过滤小面积伪影"""
        # 连通域分析
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8)
        
        filtered_mask = np.zeros_like(mask)
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            # 过滤条件
            if area < min_area:
                continue
            
            aspect_ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 0
            if aspect_ratio < min_aspect_ratio:
                continue
            
            filtered_mask[labels == i] = 255
        
        return filtered_mask
    
    @staticmethod
    def detect_annotation_errors(image: np.ndarray, 
                                 mask: np.ndarray,
                                 threshold: float = 0.8) -> bool:
        """基于统计特征检测标注错误（简化版）"""
        # 检查掩码是否全黑或全白
        mask_ratio = np.sum(mask > 0) / mask.size
        if mask_ratio < 0.001 or mask_ratio > 0.95:
            return False
        
        # 检查掩码区域与图像的对比度
        crack_region = image[mask > 0]
        bg_region = image[mask == 0]
        
        if len(crack_region) > 0 and len(bg_region) > 0:
            crack_mean = np.mean(crack_region)
            bg_mean = np.mean(bg_region)
            contrast = abs(crack_mean - bg_mean) / 255.0
            
            # 裂纹应该与背景有明显对比
            if contrast < 0.1:
                return False
        
        return True


class CrackDataset(Dataset):
    """裂纹分割数据集"""
    
    def __init__(self, 
                 config: DatasetConfig,
                 transform: Optional[A.Compose] = None,
                 use_hard_mining: bool = False):
        """
        Args:
            config: 数据集配置
            transform: Albumentations变换
            use_hard_mining: 是否使用难例挖掘
        """
        self.config = config
        self.transform = transform
        self.use_hard_mining = use_hard_mining
        
        # 加载数据集索引
        self.samples = self._load_samples()
        
        # 难例权重（初始均匀）
        self.sample_weights = np.ones(len(self.samples))
        
        # LMDB缓存
        self.lmdb_env = None
        if config.use_cache:
            self._init_cache()
    
    def _load_samples(self) -> List[Dict]:
        """加载样本列表"""
        split_file = Path(self.config.root) / f"{self.config.split}.txt"
        
        print(f"\n{'='*60}")
        print(f"🔍 调试信息 - 加载 {self.config.split} 数据集")
        print(f"{'='*60}")
        print(f"📂 Root路径: {self.config.root}")
        print(f"📂 Root绝对路径: {Path(self.config.root).resolve()}")
        print(f"📄 Split文件: {split_file}")
        print(f"📄 Split文件存在: {split_file.exists()}")
        
        if not split_file.exists():
            print(f"❌ 错误: 找不到 split 文件!")
            print(f"   请检查路径: {split_file.resolve()}")
            return []
        
        samples = []
        checked_count = 0
        failed_checks = {
            'image_not_found': 0,
            'mask_not_found': 0,
            'size_mismatch': 0
        }
        
        with open(split_file, 'r') as f:
            lines = f.readlines()
            total = len(lines)
            print(f"📊 总样本数: {total}")
            
            for idx, line in enumerate(lines):
                sample_id = line.strip()
                if not sample_id:  # 跳过空行
                    continue
                
                # 只显示前3个样本的详细信息
                if idx < 3:
                    print(f"\n--- 样本 {idx+1}: {sample_id} ---")
                
                image_path = Path(self.config.root) / self.config.image_dir / self.config.split / f"{sample_id}.jpg"
                mask_path = Path(self.config.root) / self.config.mask_dir / self.config.split / f"{sample_id}.png"
                
                if idx < 3:
                    print(f"  图像路径: {image_path}")
                    print(f"  图像存在: {image_path.exists()}")
                    print(f"  掩码路径: {mask_path}")
                    print(f"  掩码存在: {mask_path.exists()}")
                
                if not image_path.exists():
                    failed_checks['image_not_found'] += 1
                    continue
                
                if not mask_path.exists():
                    failed_checks['mask_not_found'] += 1
                    continue
                
                # 质量检查（只检查前100个样本以加快速度）
                checked_count += 1
                if checked_count <= 100:
                    if QualityControl.check_size_consistency(
                        str(image_path), str(mask_path)):
                        samples.append({
                            'id': sample_id,
                            'image': str(image_path),
                            'mask': str(mask_path)
                        })
                        if idx < 3:
                            print(f"  ✅ 通过质量检查")
                    else:
                        failed_checks['size_mismatch'] += 1
                        if idx < 3:
                            print(f"  ❌ 质量检查失败（尺寸不匹配）")
                else:
                    # 后续样本跳过质量检查以加快速度
                    samples.append({
                        'id': sample_id,
                        'image': str(image_path),
                        'mask': str(mask_path)
                    })
                
                # 进度显示
                if (idx + 1) % 500 == 0:
                    print(f"  进度: {idx+1}/{total} ({(idx+1)*100//total}%) - 已加载 {len(samples)} 个有效样本")
        
        print(f"\n{'='*60}")
        print(f"📊 加载统计:")
        print(f"  - 总样本数: {total}")
        print(f"  - 图像未找到: {failed_checks['image_not_found']}")
        print(f"  - 掩码未找到: {failed_checks['mask_not_found']}")
        print(f"  - 尺寸不匹配: {failed_checks['size_mismatch']}")
        print(f"  - 成功加载: {len(samples)}")
        print(f"{'='*60}\n")
        
        if len(samples) == 0:
            print(f"⚠️  警告: 没有加载到任何样本!")
            print(f"   请检查:")
            print(f"   1. 数据集路径是否正确")
            print(f"   2. 图像和掩码文件是否存在")
            print(f"   3. 文件扩展名是否为 .jpg 和 .png")
        
        return samples
    
    def _init_cache(self):
        """初始化LMDB缓存"""
        cache_path = Path(self.config.cache_dir) / self.config.split
        cache_path.mkdir(parents=True, exist_ok=True)
        
        self.lmdb_env = lmdb.open(
            str(cache_path),
            map_size=5 * 1024 * 1024 * 1024,  # 10GB
            readonly=False,
            lock=False
        )
    
    def _get_from_cache(self, idx: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """从缓存读取"""
        if self.lmdb_env is None:
            return None
        
        with self.lmdb_env.begin() as txn:
            data = txn.get(str(idx).encode())
            if data is not None:
                return pickle.loads(data)
        return None
    
    def _put_to_cache(self, idx: int, image: np.ndarray, mask: np.ndarray):
        """写入缓存"""
        if self.lmdb_env is None:
            return
        
        with self.lmdb_env.begin(write=True) as txn:
            txn.put(str(idx).encode(), pickle.dumps((image, mask)))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取样本"""
        # 尝试从缓存加载
        cached = self._get_from_cache(idx)
        if cached is not None:
            image, mask = cached
        else:
            # 从磁盘加载
            sample = self.samples[idx]
            image = cv2.imread(sample['image'])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            mask = cv2.imread(sample['mask'], cv2.IMREAD_GRAYSCALE)
            
            # 质量控制：过滤小伪影
            mask = QualityControl.filter_small_artifacts(mask)
            
            # 写入缓存
            self._put_to_cache(idx, image, mask)
        
        # 数据增强
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
            
            # 归一化mask到0/1（已经是Tensor）
            mask = (mask > 0.5).float()
            # 添加 channel 维度: [H, W] -> [1, H, W]
            mask = mask.unsqueeze(0)
        else:
            # 如果没有transform，手动处理
            mask = (mask > 128).astype(np.float32)
            # 添加 channel 维度
            mask = np.expand_dims(mask, axis=0)
        
        return {
            'image': image,
            'mask': mask,
            'id': self.samples[idx]['id'],
            'weight': self.sample_weights[idx]
        }
    
    def update_sample_weights(self, losses: np.ndarray):
        """更新难例权重（Hard Example Mining）"""
        if not self.use_hard_mining:
            return
        
        # 基于损失值更新权重
        self.sample_weights = losses / (losses.mean() + 1e-8)
        self.sample_weights = np.clip(self.sample_weights, 0.5, 2.0)


def get_training_augmentation(config: DatasetConfig, epoch_ratio: float = 0.0) -> A.Compose:
    """
    获取训练增强策略
    epoch_ratio: 训练进度 0.0~1.0，用于动态调整增强强度
    """
    # 动态调整增强强度（前60% epoch强增强，后40%弱增强）
    strong_aug = epoch_ratio < 0.6
    
    transforms = []
    
    # 几何增强
    if strong_aug:
        transforms.extend([
            A.RandomScale(scale_limit=(-0.5, 1.0), p=0.5),
            A.Rotate(limit=90, p=0.5),
            A.Affine(scale=(0.8, 1.2), translate_percent=0.1, p=0.3),
            A.ElasticTransform(alpha=50, sigma=5, p=0.3),  # 移除 alpha_affine
        ])
    else:
        transforms.extend([
            A.RandomScale(scale_limit=(-0.2, 0.2), p=0.3),
            A.Rotate(limit=30, p=0.3),
        ])
    
    # 翻转
    transforms.extend([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
    ])
    
    # 颜色增强
    if strong_aug:
        transforms.extend([
            A.CLAHE(clip_limit=4.0, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
        ])
    else:
        transforms.extend([
            A.CLAHE(clip_limit=2.0, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        ])
    
    # 噪声与模糊
    transforms.extend([
        A.OneOf([
            A.GaussNoise(p=1.0),  # 使用默认参数
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        ], p=0.3),
        A.OneOf([
            A.MotionBlur(blur_limit=7, p=1.0),
            A.MedianBlur(blur_limit=7, p=1.0),
            A.GaussianBlur(blur_limit=7, p=1.0),
        ], p=0.2),
    ])
    
    # 裁剪（多尺度训练）
    if strong_aug and len(config.train_scales) > 1:
        crop_size = np.random.choice(config.train_scales)
    else:
        crop_size = config.crop_size[0]
    
    # 先确保图像大小足够进行裁剪
    # 使用 LongestMaxSize + PadIfNeeded 确保图像不会太小
    transforms.extend([
        A.LongestMaxSize(max_size=max(crop_size * 2, 1024), p=1.0),  # 确保图像足够大
        A.PadIfNeeded(
            min_height=crop_size,
            min_width=crop_size,
            border_mode=cv2.BORDER_REFLECT_101,
            p=1.0
        ),
        A.RandomCrop(height=crop_size, width=crop_size, p=1.0)
    ])
    
    # 归一化与转换
    transforms.extend([
        A.Normalize(mean=config.normalize_mean, std=config.normalize_std),
        ToTensorV2(),
    ])
    
    return A.Compose(transforms)


def get_validation_augmentation(config: DatasetConfig) -> A.Compose:
    """验证集增强（仅归一化）"""
    return A.Compose([
        A.Resize(config.crop_size[0], config.crop_size[1]),
        A.Normalize(mean=config.normalize_mean, std=config.normalize_std),
        ToTensorV2(),
    ])


def create_dataloaders(config: DatasetConfig,
                       batch_size: int = 16,
                       num_workers: int = 4,
                       epoch_ratio: float = 0.0) -> Tuple[DataLoader, DataLoader]:
    """创建训练和验证数据加载器"""
    
    # 训练集
    train_config = DatasetConfig(**{**config.__dict__, 'split': 'train'})
    train_dataset = CrackDataset(
        train_config,
        transform=get_training_augmentation(train_config, epoch_ratio),
        use_hard_mining=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,  # 只在多进程时设置
        persistent_workers=True if num_workers > 0 else False
    )
    
    # 验证集
    val_config = DatasetConfig(**{**config.__dict__, 'split': 'val'})
    val_dataset = CrackDataset(
        val_config,
        transform=get_validation_augmentation(val_config),
        use_hard_mining=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,  # 只在多进程时设置
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # 测试代码
    config = DatasetConfig(
        root="./data/processed",
        crop_size=(512, 512),
        train_scales=[256, 384, 512]
    )
    
    train_loader, val_loader = create_dataloaders(config, batch_size=4, num_workers=2)
    
    for batch in train_loader:
        print(f"Image shape: {batch['image'].shape}")
        print(f"Mask shape: {batch['mask'].shape}")
        print(f"Sample weights: {batch['weight']}")
        break

