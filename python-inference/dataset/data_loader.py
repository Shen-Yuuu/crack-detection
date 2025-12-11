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
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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
    min_mask_ratio: float = 0.001
    mixup_prob: float = 0.0
    cutmix_prob: float = 0.0
    mix_alpha: float = 0.4
    focus_crop_prob: float = 0.0
    hard_mining_update_interval: int = 256


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
        self.is_training = 'train' in self.config.split
        
        # 加载数据集索引
        self.samples = self._load_samples()
        
        # 难例权重（初始均匀）
        self.sample_weights = np.ones(len(self.samples), dtype=np.float32)
        self.sample_losses = np.ones(len(self.samples), dtype=np.float32)
        self._pending_updates = 0
        
        # LMDB缓存
        self.lmdb_env = None
        if config.use_cache:
            self._init_cache()
    
    def _load_samples(self) -> List[Dict]:
        """加载样本列表"""
        split_file = Path(self.config.root) / f"{self.config.split}.txt"
        
        print(f"\n{'='*60}")
        print(f"[DEBUG] Loading {self.config.split} dataset")
        print(f"{'='*60}")
        print(f"[PATH] Root: {self.config.root}")
        print(f"[PATH] Root (abs): {Path(self.config.root).resolve()}")
        print(f"[FILE] Split file: {split_file}")
        print(f"[FILE] Exists: {split_file.exists()}")
        
        if not split_file.exists():
            print(f"[ERROR] Split file not found!")
            print(f"        Check path: {split_file.resolve()}")
            return []
        
        samples = []
        checked_count = 0
        failed_checks = {
            'image_not_found': 0,
            'mask_not_found': 0,
            'size_mismatch': 0,
            'tiny_mask': 0
        }
        
        with open(split_file, 'r') as f:
            lines = f.readlines()
            total = len(lines)
            print(f"[STATS] Total samples: {total}")
            
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

                # 读取掩码以检查有效标注比例
                mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask_raw is None:
                    failed_checks['mask_not_found'] += 1
                    continue

                mask_ratio = float(np.count_nonzero(mask_raw)) / float(mask_raw.size)
                if mask_ratio < self.config.min_mask_ratio:
                    failed_checks['tiny_mask'] += 1
                    if idx < 3:
                        print(f"  [SKIP] Mask ratio too low ({mask_ratio:.6f})")
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
                            print(f"  [OK] Quality check passed")
                    else:
                        failed_checks['size_mismatch'] += 1
                        if idx < 3:
                            print(f"  [FAIL] Size mismatch")
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
        print(f"[STATS] Loading summary:")
        print(f"  - Total samples: {total}")
        print(f"  - Image not found: {failed_checks['image_not_found']}")
        print(f"  - Mask not found: {failed_checks['mask_not_found']}")
        print(f"  - Size mismatch: {failed_checks['size_mismatch']}")
        print(f"  - Mask ratio too low: {failed_checks['tiny_mask']}")
        print(f"  - Successfully loaded: {len(samples)}")
        print(f"{'='*60}\n")
        
        if len(samples) == 0:
            print(f"[WARNING] No samples loaded!")
            print(f"   Please check:")
            print(f"   1. Dataset path is correct")
            print(f"   2. Image and mask files exist")
            print(f"   3. File extensions are .jpg and .png")
        
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
    

    def _load_image_mask(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """加载单个样本的图像和掩码（带缓存）"""
        cached = self._get_from_cache(idx)
        if cached is not None:
            image, mask = cached
            return image.copy(), mask.copy()

        sample = self.samples[idx]
        image = cv2.imread(sample['image'])
        if image is None:
            raise FileNotFoundError(f"无法读取图像: {sample['image']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(sample['mask'], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"无法读取掩码: {sample['mask']}")

        mask = QualityControl.filter_small_artifacts(mask)

        # 写入缓存
        self._put_to_cache(idx, image, mask)

        return image.copy(), mask.copy()

    def _random_focus_crop(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """围绕裂缝区域随机裁剪"""
        if (not self.is_training or self.config.focus_crop_prob <= 0 or
                random.random() > self.config.focus_crop_prob):
            return image, mask

        coords = np.column_stack(np.where(mask > 0))
        if coords.size == 0:
            return image, mask

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        h, w = image.shape[:2]
        bbox_h = max(y_max - y_min + 1, 1)
        bbox_w = max(x_max - x_min + 1, 1)

        scale = random.uniform(1.1, 1.6)
        crop_h = min(int(bbox_h * scale), h)
        crop_w = min(int(bbox_w * scale), w)

        center_y = random.randint(max(y_min - bbox_h // 2, 0), min(y_max + bbox_h // 2, h - 1))
        center_x = random.randint(max(x_min - bbox_w // 2, 0), min(x_max + bbox_w // 2, w - 1))

        y1 = max(center_y - crop_h // 2, 0)
        x1 = max(center_x - crop_w // 2, 0)
        y2 = min(y1 + crop_h, h)
        x2 = min(x1 + crop_w, w)
        y1 = max(y2 - crop_h, 0)
        x1 = max(x2 - crop_w, 0)

        return image[y1:y2, x1:x2], mask[y1:y2, x1:x2]

    def _apply_mixup(self, image: np.ndarray, mask: np.ndarray, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if (not self.is_training or self.config.mixup_prob <= 0 or
                random.random() > self.config.mixup_prob or len(self.samples) < 2):
            return image, mask

        mix_idx = random.randrange(len(self.samples))
        if mix_idx == idx:
            return image, mask

        mix_image, mix_mask = self._load_image_mask(mix_idx)

        target_h, target_w = image.shape[:2]
        if mix_image.shape[:2] != (target_h, target_w):
            mix_image = cv2.resize(mix_image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            mix_mask = cv2.resize(mix_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        lam = np.random.beta(self.config.mix_alpha, self.config.mix_alpha)

        image_mix = lam * image.astype(np.float32) + (1 - lam) * mix_image.astype(np.float32)
        mask_mix = lam * (mask.astype(np.float32) / 255.0) + (1 - lam) * (mix_mask.astype(np.float32) / 255.0)

        image_mix = np.clip(image_mix, 0, 255).astype(np.uint8)
        mask_mix = np.clip(mask_mix, 0.0, 1.0)

        return image_mix, (mask_mix * 255.0).astype(np.uint8)

    def _apply_cutmix(self, image: np.ndarray, mask: np.ndarray, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if (not self.is_training or self.config.cutmix_prob <= 0 or
                random.random() > self.config.cutmix_prob or len(self.samples) < 2):
            return image, mask

        mix_idx = random.randrange(len(self.samples))
        if mix_idx == idx:
            return image, mask

        mix_image, mix_mask = self._load_image_mask(mix_idx)

        target_h, target_w = image.shape[:2]
        if mix_image.shape[:2] != (target_h, target_w):
            mix_image = cv2.resize(mix_image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            mix_mask = cv2.resize(mix_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        h, w = image.shape[:2]
        lam = np.random.beta(self.config.mix_alpha, self.config.mix_alpha)
        cut_ratio = np.sqrt(1 - lam)
        cut_h = max(1, int(h * cut_ratio))
        cut_w = max(1, int(w * cut_ratio))

        cy = random.randint(0, h - 1)
        cx = random.randint(0, w - 1)

        y1 = np.clip(cy - cut_h // 2, 0, h)
        x1 = np.clip(cx - cut_w // 2, 0, w)
        y2 = np.clip(cy + cut_h // 2, 0, h)
        x2 = np.clip(cx + cut_w // 2, 0, w)

        image[y1:y2, x1:x2] = mix_image[y1:y2, x1:x2]
        mask[y1:y2, x1:x2] = mix_mask[y1:y2, x1:x2]

        return image, mask

    def get_sampling_weights(self) -> torch.Tensor:
        return torch.as_tensor(self.sample_weights, dtype=torch.double)

    def on_epoch_start(self):
        if self.use_hard_mining and len(self.samples) > 0:
            self.update_sample_weights(self.sample_losses)
            self._pending_updates = 0

    
    def record_sample_losses(self, indices: np.ndarray, losses: torch.Tensor):
        if not self.use_hard_mining or len(self.samples) == 0:
            return

        losses_np = losses.detach().cpu().numpy().astype(np.float32)
        valid = np.isfinite(losses_np) & (losses_np >= 0.0)

        if np.any(valid):
            self.sample_losses[indices[valid]] = losses_np[valid]

        invalid = ~valid
        if np.any(invalid):
            # 对于 NaN/负值样本，保持原有损失并赋予中性权重
            self.sample_losses[indices[invalid]] = np.maximum(
                self.sample_losses[indices[invalid]], 1e-6
            )

        self._pending_updates += int(np.sum(valid))

        if self._pending_updates >= self.config.hard_mining_update_interval:
            self.update_sample_weights(self.sample_losses)
            self._pending_updates = 0

    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取样本"""
        image, mask = self._load_image_mask(idx)

        # 聚焦裁剪
        image, mask = self._random_focus_crop(image, mask)

        # MixUp / CutMix
        image, mask = self._apply_mixup(image, mask, idx)
        image, mask = self._apply_cutmix(image, mask, idx)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

            mask = mask.float().clamp_(0.0, 1.0)
            mask = mask.unsqueeze(0)
        else:
            mask = (mask > 128).astype(np.float32)
            mask = np.expand_dims(mask, axis=0)

        return {
            'image': image,
            'mask': mask,
            'id': self.samples[idx]['id'],
            'weight': float(self.sample_weights[idx]),
            'index': idx
        }

    def update_sample_weights(self, losses: np.ndarray):
        """更新难例权重（Hard Example Mining）- 增强稳定性版本"""
        if not self.use_hard_mining or len(self.samples) == 0:
            return

        losses = np.asarray(losses, dtype=np.float32)
        
        # 过滤无效值（增加上限检查）
        valid = np.isfinite(losses) & (losses > 0) & (losses < 1e6)
        if not np.any(valid):
            self.sample_weights = np.ones(len(self.samples), dtype=np.float32)
            return

        # 使用中位数而非均值，更鲁棒
        median_loss = np.median(losses[valid])
        if median_loss <= 0:
            median_loss = 1.0
            
        normalized = losses / (median_loss + 1e-8)
        
        # 更保守的裁剪范围
        normalized = np.clip(normalized, 0.5, 3.0)

        # 对无效样本使用中性权重
        normalized[~valid] = 1.0
        normalized = np.nan_to_num(normalized, nan=1.0, posinf=3.0, neginf=0.5)

        # 确保权重非负且和为正
        normalized = np.maximum(normalized, 1e-6)

        total = np.sum(normalized)
        if not np.isfinite(total) or total <= 0:
            normalized = np.ones_like(normalized, dtype=np.float32)

        self.sample_weights = normalized.astype(np.float32)

    def set_epoch_ratio(self, epoch_ratio: float):
        """根据训练进度动态调整增强策略"""
        if 'train' not in self.config.split:
            return

        epoch_ratio = float(np.clip(epoch_ratio, 0.0, 1.0))

        # 避免重复构建完全相同的pipeline
        current = getattr(self, "_current_epoch_ratio", None)
        if current is not None and abs(current - epoch_ratio) < 1e-4:
            return

        self.transform = get_training_augmentation(self.config, epoch_ratio)
        self._current_epoch_ratio = epoch_ratio


def get_deepcrack_augmentation(config: DatasetConfig) -> A.Compose:
    """
    DeepCrack论文风格的数据增强策略
    
    原论文方法（离线增强，生成35,100张图像）:
    - 9个旋转角度: 0°, 10°, 20°, 30°, 40°, 50°, 60°, 70°, 80°, 90°
    - 每个角度: 水平翻转 + 垂直翻转 (4种组合)
    - 每个翻转: 5个裁剪位置 (4角 + 中心)
    - 260 * 9 * 4 * 5 = 46,800 (论文说35,100可能是只用了部分组合)
    
    在线随机增强版本:
    - 随机选择一个旋转角度（0-90度，倾向于10度间隔的角度）
    - 随机翻转
    - 5个位置随机裁剪（4角 + 中心）
    """
    crop_h, crop_w = int(config.crop_size[0]), int(config.crop_size[1])
    
    transforms = [
        # 1. 确保图像足够大以进行裁剪
        A.LongestMaxSize(max_size=max(crop_h, crop_w) * 2, p=1.0),
        A.PadIfNeeded(
            min_height=crop_h + 100,  # 额外padding以支持角落裁剪
            min_width=crop_w + 100,
            border_mode=cv2.BORDER_REFLECT_101,
            p=1.0
        ),
        
        # 2. DeepCrack旋转策略: 0-90度，倾向于10度间隔
        # 使用90度范围的旋转
        A.Rotate(
            limit=90,  # -90到+90度
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_REFLECT_101,
            p=1.0  # 总是旋转
        ),
        
        # 3. 翻转 (水平+垂直的所有组合)
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        
        # 4. 五点裁剪策略 (4角+中心) - 使用RandomCrop模拟
        # RandomCrop会随机选择位置，但我们用CenterCrop和RandomCrop的组合
        A.OneOf([
            # 中心裁剪
            A.CenterCrop(height=crop_h, width=crop_w),
            # 随机裁剪 (模拟4角)
            A.RandomCrop(height=crop_h, width=crop_w),
            A.RandomCrop(height=crop_h, width=crop_w),
            A.RandomCrop(height=crop_h, width=crop_w),
            A.RandomCrop(height=crop_h, width=crop_w),
        ], p=1.0),
        
        # 5. 轻度颜色增强（保持与原图相似）
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
            A.CLAHE(clip_limit=2.0, p=1.0),
            A.RandomGamma(gamma_limit=(90, 110), p=1.0),
        ], p=0.3),
        
        # 6. 归一化与转换
        A.Normalize(mean=config.normalize_mean, std=config.normalize_std),
        ToTensorV2(),
    ]
    
    return A.Compose(transforms)


def get_training_augmentation(config: DatasetConfig, epoch_ratio: float = 0.0) -> A.Compose:
    """
    获取训练增强策略 - 使用DeepCrack风格增强
    epoch_ratio: 训练进度 0.0~1.0，用于动态调整增强强度
    """
    # 使用DeepCrack风格的增强策略
    return get_deepcrack_augmentation(config)


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
    
    train_sampler = None
    if train_dataset.use_hard_mining and len(train_dataset) > 0:
        train_dataset.on_epoch_start()
        train_sampler = WeightedRandomSampler(
            weights=train_dataset.get_sampling_weights(),
            num_samples=len(train_dataset),
            replacement=True
        )

    train_loader_kwargs = {
        'dataset': train_dataset,
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': True,
        'shuffle': train_sampler is None
    }

    if train_sampler is not None:
        train_loader_kwargs['sampler'] = train_sampler
        train_loader_kwargs['shuffle'] = False

    if num_workers > 0:
        train_loader_kwargs['prefetch_factor'] = 2
        train_loader_kwargs['persistent_workers'] = False  # 需让 worker 每轮重新载入增强配置

    train_loader = DataLoader(**train_loader_kwargs)
    
    # 验证集
    val_config = DatasetConfig(**{**config.__dict__, 'split': 'val'})
    val_dataset = CrackDataset(
        val_config,
        transform=get_validation_augmentation(val_config),
        use_hard_mining=False
    )
    
    val_loader_kwargs = {
        'dataset': val_dataset,
        'batch_size': batch_size,
        'shuffle': False,
        'num_workers': num_workers,
        'pin_memory': True
    }

    if num_workers > 0:
        val_loader_kwargs['prefetch_factor'] = 2
        val_loader_kwargs['persistent_workers'] = False

    val_loader = DataLoader(**val_loader_kwargs)
    
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

