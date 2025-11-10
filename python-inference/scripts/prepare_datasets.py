"""
数据集规范化脚本
将不同来源的裂纹数据集统一为标准格式

目标格式:
data/processed/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── masks/
│   ├── train/
│   ├── val/
│   └── test/
├── train.txt
├── val.txt
└── test.txt

支持的源数据集:
1. CrackDataset (AsphaltCrack300, CFD, Crack500)
2. DeepCrack (CrackLS315, CrackTree260, CRKWH100)
"""

import os
import sys
import io
import cv2
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm
import random
import json
from typing import List, Tuple, Dict
import argparse

# 设置标准输出编码为UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


class DatasetNormalizer:
    """数据集规范化器"""
    
    def __init__(self, 
                 source_root: str,
                 output_root: str,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 seed: int = 42):
        """
        Args:
            source_root: 源数据集根目录
            output_root: 输出目录
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            seed: 随机种子
        """
        self.source_root = Path(source_root)
        self.output_root = Path(output_root)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Split ratios must sum to 1.0"
        
        random.seed(seed)
        np.random.seed(seed)
        
        # 创建输出目录
        for split in ['train', 'val', 'test']:
            (self.output_root / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.output_root / 'masks' / split).mkdir(parents=True, exist_ok=True)
        
        self.sample_id = 0
        self.stats = {
            'train': 0,
            'val': 0,
            'test': 0,
            'total': 0
        }
    
    def normalize_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        标准化掩码为二值图像 (0/255)
        
        Args:
            mask: 输入掩码
        Returns:
            标准化后的掩码
        """
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        # 二值化
        _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        return mask_binary
    
    def assign_split(self, idx: int, total: int) -> str:
        """
        分配样本到train/val/test
        
        Args:
            idx: 当前索引
            total: 总数
        Returns:
            'train', 'val', or 'test'
        """
        ratio = idx / total
        
        if ratio < self.train_ratio:
            return 'train'
        elif ratio < self.train_ratio + self.val_ratio:
            return 'val'
        else:
            return 'test'
    
    def copy_sample(self, 
                   image_path: Path, 
                   mask_path: Path,
                   split: str,
                   prefix: str = "") -> bool:
        """
        复制并规范化单个样本
        
        Args:
            image_path: 图像路径
            mask_path: 掩码路径
            split: train/val/test
            prefix: 文件名前缀
        Returns:
            是否成功
        """
        try:
            # 读取图像
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Warning: Failed to read image {image_path}")
                return False
            
            # 读取掩码
            mask = cv2.imread(str(mask_path))
            if mask is None:
                print(f"Warning: Failed to read mask {mask_path}")
                return False
            
            # 检查尺寸一致性
            if image.shape[:2] != mask.shape[:2]:
                print(f"Warning: Size mismatch {image.shape[:2]} vs {mask.shape[:2]}")
                # 调整掩码尺寸
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), 
                                 interpolation=cv2.INTER_NEAREST)
            
            # 标准化掩码
            mask_normalized = self.normalize_mask(mask)
            
            # 生成新文件名
            new_name = f"{prefix}{self.sample_id:06d}"
            self.sample_id += 1
            
            # 保存
            image_out = self.output_root / 'images' / split / f"{new_name}.jpg"
            mask_out = self.output_root / 'masks' / split / f"{new_name}.png"
            
            # 统一为JPG（图像）和PNG（掩码）
            cv2.imwrite(str(image_out), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(mask_out), mask_normalized)
            
            self.stats[split] += 1
            self.stats['total'] += 1
            
            return True
        
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return False
    
    def process_crackdataset_subset(self, 
                                    subset_name: str,
                                    image_dir: str,
                                    label_dir: str,
                                    val_dir: str = None,
                                    val_label_dir: str = None):
        """
        处理CrackDataset的子集（AsphaltCrack300, CFD, Crack500）
        
        Args:
            subset_name: 子集名称
            image_dir: 训练图像目录
            label_dir: 训练标签目录
            val_dir: 验证图像目录（可选）
            val_label_dir: 验证标签目录（可选）
        """
        print(f"\n处理 {subset_name}...")
        
        base_path = self.source_root / 'CrackDataset-main' / subset_name
        
        # 处理训练集
        train_img_path = base_path / image_dir
        train_label_path = base_path / label_dir
        
        if not train_img_path.exists():
            print(f"Warning: {train_img_path} not found, skipping")
            return
        
        # 获取所有图像
        image_files = sorted(list(train_img_path.glob('*.jpg')))
        
        # 随机打乱
        random.shuffle(image_files)
        
        processed = 0
        for idx, img_file in enumerate(tqdm(image_files, desc=f"{subset_name} train")):
            # 查找对应的掩码
            mask_file = train_label_path / img_file.with_suffix('.png').name
            
            if not mask_file.exists():
                continue
            
            # 分配split
            split = self.assign_split(idx, len(image_files))
            
            # 复制样本
            if self.copy_sample(img_file, mask_file, split, f"{subset_name}_"):
                processed += 1
        
        # 处理验证集（如果存在）
        if val_dir and val_label_dir:
            val_img_path = base_path / val_dir
            val_label_path = base_path / val_label_dir
            
            if val_img_path.exists() and val_label_path.exists():
                val_image_files = sorted(list(val_img_path.glob('*.jpg')))
                
                for img_file in tqdm(val_image_files, desc=f"{subset_name} val"):
                    mask_file = val_label_path / img_file.with_suffix('.png').name
                    
                    if not mask_file.exists():
                        continue
                    
                    # 验证集统一放到test split
                    if self.copy_sample(img_file, mask_file, 'test', f"{subset_name}_val_"):
                        processed += 1
        
        print(f"✓ {subset_name}: 处理了 {processed} 个样本")
    
    def process_deepcrack_subset(self,
                                 subset_name: str,
                                 image_dir: str,
                                 mask_dir: str,
                                 image_ext: str = '.jpg',
                                 mask_ext: str = '.bmp'):
        """
        处理DeepCrack数据集子集（CrackLS315, CrackTree260, CRKWH100）
        
        Args:
            subset_name: 子集名称
            image_dir: 图像目录名
            mask_dir: 掩码目录名
            image_ext: 图像扩展名
            mask_ext: 掩码扩展名
        """
        print(f"\n处理 {subset_name}...")
        
        base_path = self.source_root / 'DeepCrack-datasets'
        img_path = base_path / image_dir
        mask_path = base_path / mask_dir
        
        if not img_path.exists():
            print(f"Warning: {img_path} not found, skipping")
            return
        
        # 获取所有图像
        image_files = sorted(list(img_path.glob(f'*{image_ext}')))
        
        # 随机打乱
        random.shuffle(image_files)
        
        processed = 0
        for idx, img_file in enumerate(tqdm(image_files, desc=subset_name)):
            # 查找对应的掩码
            mask_file = mask_path / img_file.with_suffix(mask_ext).name
            
            # 特殊处理：CrackTree260的掩码在gt子目录下
            if not mask_file.exists():
                mask_file = mask_path / 'gt' / img_file.with_suffix(mask_ext).name
            
            if not mask_file.exists():
                print(f"Warning: Mask not found for {img_file.name}")
                continue
            
            # 分配split
            split = self.assign_split(idx, len(image_files))
            
            # 复制样本
            if self.copy_sample(img_file, mask_file, split, f"{subset_name}_"):
                processed += 1
        
        print(f"✓ {subset_name}: 处理了 {processed} 个样本")
    
    def generate_split_files(self):
        """生成train.txt, val.txt, test.txt"""
        print("\n生成分割文件...")
        
        for split in ['train', 'val', 'test']:
            split_file = self.output_root / f"{split}.txt"
            
            image_dir = self.output_root / 'images' / split
            image_files = sorted(list(image_dir.glob('*.jpg')))
            
            with open(split_file, 'w') as f:
                for img_file in image_files:
                    # 写入不带扩展名的文件名
                    f.write(img_file.stem + '\n')
            
            print(f"  {split}.txt: {len(image_files)} 个样本")
    
    def generate_statistics(self):
        """生成数据集统计信息"""
        print("\n生成统计信息...")
        
        stats = {
            'total_samples': self.stats['total'],
            'splits': {
                'train': self.stats['train'],
                'val': self.stats['val'],
                'test': self.stats['test']
            },
            'split_ratios': {
                'train': self.stats['train'] / self.stats['total'] if self.stats['total'] > 0 else 0,
                'val': self.stats['val'] / self.stats['total'] if self.stats['total'] > 0 else 0,
                'test': self.stats['test'] / self.stats['total'] if self.stats['total'] > 0 else 0
            }
        }
        
        # 计算图像尺寸统计
        for split in ['train', 'val', 'test']:
            image_dir = self.output_root / 'images' / split
            image_files = list(image_dir.glob('*.jpg'))[:100]  # 采样100张
            
            sizes = []
            for img_file in image_files:
                img = cv2.imread(str(img_file))
                if img is not None:
                    sizes.append(img.shape[:2])
            
            if sizes:
                heights, widths = zip(*sizes)
                stats[f'{split}_image_stats'] = {
                    'mean_height': int(np.mean(heights)),
                    'mean_width': int(np.mean(widths)),
                    'min_height': int(np.min(heights)),
                    'max_height': int(np.max(heights)),
                    'min_width': int(np.min(widths)),
                    'max_width': int(np.max(widths))
                }
        
        # 保存统计信息
        stats_file = self.output_root / 'dataset_stats.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"\n统计信息已保存到: {stats_file}")
        print(f"\n数据集汇总:")
        print(f"  总样本数: {stats['total_samples']}")
        print(f"  训练集: {stats['splits']['train']} ({stats['split_ratios']['train']*100:.1f}%)")
        print(f"  验证集: {stats['splits']['val']} ({stats['split_ratios']['val']*100:.1f}%)")
        print(f"  测试集: {stats['splits']['test']} ({stats['split_ratios']['test']*100:.1f}%)")
    
    def process_all(self):
        """处理所有数据集"""
        print("=" * 60)
        print("开始数据集规范化")
        print("=" * 60)
        
        # 处理CrackDataset
        print("\n【1/2】处理 CrackDataset 系列")
        print("-" * 60)
        
        # AsphaltCrack300
        self.process_crackdataset_subset(
            subset_name='AsphaltCrack300',
            image_dir='train',
            label_dir='label'
        )
        
        # CFD
        self.process_crackdataset_subset(
            subset_name='CFD',
            image_dir='train',
            label_dir='label',
            val_dir='val',
            val_label_dir='val_label'
        )
        
        # Crack500
        self.process_crackdataset_subset(
            subset_name='crack500',
            image_dir='train',
            label_dir='label',
            val_dir='val',
            val_label_dir='val_label'
        )
        
        # 处理DeepCrack
        print("\n【2/2】处理 DeepCrack 系列")
        print("-" * 60)
        
        # CrackLS315
        self.process_deepcrack_subset(
            subset_name='CrackLS315',
            image_dir='CrackLS315',
            mask_dir='CrackLS315_gt',
            image_ext='.jpg',
            mask_ext='.bmp'
        )
        
        # CrackTree260
        self.process_deepcrack_subset(
            subset_name='CrackTree260',
            image_dir='CrackTree260',
            mask_dir='CrackTree260_gt',
            image_ext='.jpg',
            mask_ext='.bmp'
        )
        
        # CRKWH100
        self.process_deepcrack_subset(
            subset_name='CRKWH100',
            image_dir='CRKWH100',
            mask_dir='CRKWH100_gt',
            image_ext='.png',
            mask_ext='.bmp'
        )
        
        # 生成split文件
        self.generate_split_files()
        
        # 生成统计信息
        self.generate_statistics()
        
        print("\n" + "=" * 60)
        print("✅ 数据集规范化完成！")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='规范化裂纹检测数据集')
    parser.add_argument('--source', type=str, 
                       default='E:/desktop/Code/cloud/datasets',
                       help='源数据集目录')
    parser.add_argument('--output', type=str,
                       default='E:/desktop/Code/cloud/data/processed',
                       help='输出目录')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='训练集比例')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='验证集比例')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                       help='测试集比例')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    args = parser.parse_args()
    
    # 创建规范化器
    normalizer = DatasetNormalizer(
        source_root=args.source,
        output_root=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    # 处理所有数据集
    normalizer.process_all()
    
    print(f"\n规范化后的数据集位于: {args.output}")
    print("\n可以使用以下命令开始训练:")
    print(f"  cd python-inference")
    print(f"  python train.py --config configs/train_config.yaml")


if __name__ == '__main__':
    main()

