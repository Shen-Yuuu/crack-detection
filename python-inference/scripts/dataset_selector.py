"""
灵活数据集准备工具

支持选择单个或多个数据集进行训练

使用方法:
    # 查看所有数据集信息
    python dataset_selector.py --info
    
    # 准备单个数据集 (例如 CrackTree260，常用于论文对比)
    python dataset_selector.py --datasets CrackTree260 --output ../data/cracktree260
    
    # 准备多个数据集
    python dataset_selector.py --datasets Crack500 CFD CrackTree260 --output ../data/mixed
    
    # 准备所有数据集
    python dataset_selector.py --datasets all --output ../data/all

支持的数据集:
    - AsphaltCrack300: 沥青路面裂缝 (~300张)
    - CFD: 混凝土裂缝 (~118张)
    - Crack500: 路面裂缝 (~500张)
    - CrackTree260: 树状裂缝 (~260张) [DeepCrack论文常用]
    - CrackLS315: 多表面裂缝 (~315张)
    - CRKWH100: 武汉道路裂缝 (~100张)
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
from typing import List, Tuple, Dict, Optional
import argparse

# 设置标准输出编码为UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# ============== 数据集配置 ==============
DATASET_REGISTRY = {
    'AsphaltCrack300': {
        'description': '沥青路面裂缝数据集',
        'source': 'CrackDataset',
        'base_path': 'CrackDataset-main/AsphaltCrack300',
        'train_images': 'train',
        'train_masks': 'label',
        'val_images': None,
        'val_masks': None,
        'image_ext': '.jpg',
        'mask_ext': '.png',
        'expected_count': 300,
    },
    'CFD': {
        'description': '混凝土裂缝数据集 (Concrete Fracture Dataset)',
        'source': 'CrackDataset',
        'base_path': 'CrackDataset-main/CFD',
        'train_images': 'train',
        'train_masks': 'label',
        'val_images': 'val',
        'val_masks': 'val_label',
        'image_ext': '.jpg',
        'mask_ext': '.png',
        'expected_count': 118,
    },
    'Crack500': {
        'description': '路面裂缝数据集',
        'source': 'CrackDataset',
        'base_path': 'CrackDataset-main/crack500',
        'train_images': 'train',
        'train_masks': 'label',
        'val_images': 'val',
        'val_masks': 'val_label',
        'image_ext': '.jpg',
        'mask_ext': '.png',
        'expected_count': 500,
    },
    'CrackTree260': {
        'description': '树状裂缝数据集 (DeepCrack论文常用)',
        'source': 'DeepCrack',
        'base_path': 'DeepCrack-datasets',
        'train_images': 'CrackTree260',
        'train_masks': 'CrackTree260_gt/gt',
        'val_images': None,
        'val_masks': None,
        'image_ext': ['.jpg', '.JPG'],
        'mask_ext': '.png',
        'expected_count': 260,
        'mask_in_subdir': True,
    },
    'CrackLS315': {
        'description': '多表面裂缝数据集',
        'source': 'DeepCrack',
        'base_path': 'DeepCrack-datasets',
        'train_images': 'CrackLS315',
        'train_masks': 'CrackLS315_gt',
        'val_images': None,
        'val_masks': None,
        'image_ext': '.jpg',
        'mask_ext': '.png',
        'expected_count': 315,
    },
    'CRKWH100': {
        'description': '武汉道路裂缝数据集',
        'source': 'DeepCrack',
        'base_path': 'DeepCrack-datasets',
        'train_images': 'CRKWH100',
        'train_masks': 'CRKWH100_gt',
        'val_images': None,
        'val_masks': None,
        'image_ext': '.png',
        'mask_ext': '.png',
        'expected_count': 100,
    },
}


class DatasetSelector:
    """灵活数据集选择器"""
    
    def __init__(self, datasets_root: str):
        self.datasets_root = Path(datasets_root)
    
    def get_available_datasets(self) -> List[str]:
        """获取可用的数据集列表"""
        available = []
        for name, config in DATASET_REGISTRY.items():
            base = self.datasets_root / config['base_path']
            img_dir = base / config['train_images'] if config['train_images'] else base
            if img_dir.exists():
                available.append(name)
        return available
    
    def count_images(self, dataset_name: str) -> Dict[str, int]:
        """统计数据集图像数量"""
        if dataset_name not in DATASET_REGISTRY:
            return {'train': 0, 'val': 0, 'total': 0}
        
        config = DATASET_REGISTRY[dataset_name]
        base = self.datasets_root / config['base_path']
        
        train_count = 0
        val_count = 0
        
        # 训练集
        img_dir = base / config['train_images']
        if img_dir.exists():
            exts = config['image_ext'] if isinstance(config['image_ext'], list) else [config['image_ext']]
            for ext in exts:
                train_count += len(list(img_dir.glob(f'*{ext}')))
        
        # 验证集
        if config['val_images']:
            val_dir = base / config['val_images']
            if val_dir.exists():
                exts = config['image_ext'] if isinstance(config['image_ext'], list) else [config['image_ext']]
                for ext in exts:
                    val_count += len(list(val_dir.glob(f'*{ext}')))
        
        return {
            'train': train_count,
            'val': val_count,
            'total': train_count + val_count
        }
    
    def print_info(self):
        """打印所有数据集信息"""
        print("\n" + "=" * 70)
        print("📦 裂纹检测数据集概览")
        print("=" * 70)
        
        available = self.get_available_datasets()
        total_all = 0
        
        for name, config in DATASET_REGISTRY.items():
            status = "✅" if name in available else "❌"
            counts = self.count_images(name)
            
            print(f"\n{status} {name}")
            print(f"   📝 {config['description']}")
            print(f"   📁 来源: {config['source']}")
            print(f"   🖼️  数量: {counts['total']} 张", end="")
            if counts['val'] > 0:
                print(f" (训练: {counts['train']}, 验证: {counts['val']})")
            else:
                print()
            
            if name in available:
                total_all += counts['total']
        
        print("\n" + "-" * 70)
        print(f"📊 可用数据集总计: {len(available)} 个, {total_all} 张图像")
        print("=" * 70)
        
        print("\n💡 使用示例:")
        print("   # 单个数据集 (适合论文对比)")
        print("   python dataset_selector.py --datasets CrackTree260 --output ../data/cracktree260")
        print("\n   # 多个数据集")
        print("   python dataset_selector.py --datasets Crack500 CFD --output ../data/crack_cfd")
        print("\n   # 所有数据集")
        print("   python dataset_selector.py --datasets all --output ../data/all\n")
    
    def load_dataset_pairs(self, dataset_name: str) -> List[Tuple[Path, Path, str]]:
        """加载数据集的图像-mask对"""
        if dataset_name not in DATASET_REGISTRY:
            raise ValueError(f"未知数据集: {dataset_name}")
        
        config = DATASET_REGISTRY[dataset_name]
        base = self.datasets_root / config['base_path']
        
        pairs = []
        
        def find_mask(img_path: Path, mask_dir: Path, mask_ext: str) -> Optional[Path]:
            """查找对应的mask文件"""
            stem = img_path.stem
            
            # 尝试多种可能的mask命名
            candidates = [
                mask_dir / f"{stem}{mask_ext}",
                mask_dir / f"{stem}.png",
                mask_dir / f"{stem}.bmp",
                mask_dir / f"{stem}_mask{mask_ext}",
            ]
            
            for c in candidates:
                if c.exists():
                    return c
            return None
        
        def load_from_dir(img_dir: Path, mask_dir: Path, split_name: str):
            """从目录加载图像对"""
            if not img_dir.exists():
                return
            
            exts = config['image_ext'] if isinstance(config['image_ext'], list) else [config['image_ext']]
            mask_ext = config['mask_ext']
            
            for ext in exts:
                for img_path in img_dir.glob(f'*{ext}'):
                    mask_path = find_mask(img_path, mask_dir, mask_ext)
                    if mask_path:
                        pairs.append((img_path, mask_path, split_name))
        
        # 加载训练集
        train_img_dir = base / config['train_images']
        train_mask_dir = base / config['train_masks']
        load_from_dir(train_img_dir, train_mask_dir, 'train')
        
        # 加载验证集（如果有）
        if config['val_images'] and config['val_masks']:
            val_img_dir = base / config['val_images']
            val_mask_dir = base / config['val_masks']
            load_from_dir(val_img_dir, val_mask_dir, 'val')
        
        return pairs
    
    def prepare(self,
                dataset_names: List[str],
                output_dir: str,
                train_ratio: float = 0.7,
                val_ratio: float = 0.15,
                test_ratio: float = 0.15,
                target_size: Optional[Tuple[int, int]] = None,
                seed: int = 42):
        """
        准备选定的数据集
        
        Args:
            dataset_names: 数据集名称列表，或 ['all']
            output_dir: 输出目录
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            target_size: 目标尺寸 (H, W)
            seed: 随机种子
        """
        random.seed(seed)
        np.random.seed(seed)
        
        output_path = Path(output_dir)
        
        # 处理 'all'
        if 'all' in dataset_names:
            dataset_names = self.get_available_datasets()
        
        print(f"\n{'='*60}")
        print(f"📦 准备数据集: {', '.join(dataset_names)}")
        print(f"{'='*60}")
        
        # 收集所有样本
        all_pairs = []
        for name in dataset_names:
            print(f"\n加载 {name}...")
            pairs = self.load_dataset_pairs(name)
            # 添加数据集名称标记
            pairs_with_source = [(p[0], p[1], name) for p in pairs]
            all_pairs.extend(pairs_with_source)
            print(f"   找到 {len(pairs)} 个样本")
        
        if not all_pairs:
            print("❌ 没有找到任何有效样本!")
            return
        
        print(f"\n📊 总计: {len(all_pairs)} 个样本")
        
        # 打乱
        random.shuffle(all_pairs)
        
        # 划分
        n = len(all_pairs)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_pairs = all_pairs[:n_train]
        val_pairs = all_pairs[n_train:n_train + n_val]
        test_pairs = all_pairs[n_train + n_val:]
        
        splits = {
            'train': train_pairs,
            'val': val_pairs,
            'test': test_pairs,
        }
        
        # 创建目录
        for split in ['train', 'val', 'test']:
            (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
            (output_path / 'masks' / split).mkdir(parents=True, exist_ok=True)
        
        # 复制文件
        stats = {}
        file_lists = {}
        
        for split, pairs in splits.items():
            print(f"\n处理 {split} 集 ({len(pairs)} 样本)...")
            stats[split] = 0
            file_lists[split] = []
            
            for idx, (img_path, mask_path, source) in enumerate(tqdm(pairs, desc=f"  {split}")):
                try:
                    # 读取
                    img = cv2.imread(str(img_path))
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    
                    if img is None or mask is None:
                        continue
                    
                    # 尺寸检查
                    if img.shape[:2] != mask.shape[:2]:
                        mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                                         interpolation=cv2.INTER_NEAREST)
                    
                    # 调整尺寸
                    if target_size:
                        img = cv2.resize(img, (target_size[1], target_size[0]),
                                        interpolation=cv2.INTER_LINEAR)
                        mask = cv2.resize(mask, (target_size[1], target_size[0]),
                                         interpolation=cv2.INTER_NEAREST)
                    
                    # 二值化mask
                    mask = ((mask > 127) * 255).astype(np.uint8)
                    
                    # 保存
                    name = f"{source}_{idx:05d}"
                    cv2.imwrite(str(output_path / 'images' / split / f"{name}.png"), img)
                    cv2.imwrite(str(output_path / 'masks' / split / f"{name}.png"), mask)
                    
                    file_lists[split].append(name)
                    stats[split] += 1
                    
                except Exception as e:
                    print(f"   ⚠️ 跳过 {img_path}: {e}")
        
        # 保存文件列表
        for split, names in file_lists.items():
            with open(output_path / f"{split}.txt", 'w') as f:
                for name in names:
                    f.write(f"{name}\n")
        
        # 保存统计信息
        info = {
            'datasets_used': dataset_names,
            'total_samples': sum(stats.values()),
            'splits': stats,
            'split_ratios': {
                'train': train_ratio,
                'val': val_ratio,
                'test': test_ratio,
            },
            'target_size': target_size,
            'seed': seed,
        }
        
        with open(output_path / 'dataset_stats.json', 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        
        # 打印结果
        print(f"\n{'='*60}")
        print("✅ 数据集准备完成!")
        print(f"{'='*60}")
        print(f"   使用数据集: {', '.join(dataset_names)}")
        print(f"   输出目录: {output_path}")
        print(f"   训练集: {stats['train']} 样本")
        print(f"   验证集: {stats['val']} 样本")
        print(f"   测试集: {stats['test']} 样本")
        print(f"   总计: {sum(stats.values())} 样本")
        print(f"{'='*60}\n")
        
        return info


def main():
    parser = argparse.ArgumentParser(
        description='灵活数据集准备工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 查看数据集信息
  python dataset_selector.py --info
  
  # 准备单个数据集 (CrackTree260，论文常用)
  python dataset_selector.py --datasets CrackTree260 --output ../data/cracktree260
  
  # 准备多个数据集
  python dataset_selector.py --datasets Crack500 CFD CrackTree260 --output ../data/mixed
  
  # 准备所有数据集
  python dataset_selector.py --datasets all --output ../data/all
        """
    )
    
    parser.add_argument('--datasets-root', type=str, default='../../datasets',
                       help='数据集根目录')
    parser.add_argument('--info', action='store_true',
                       help='显示所有数据集信息')
    parser.add_argument('--datasets', type=str, nargs='+',
                       help='要使用的数据集: all, AsphaltCrack300, CFD, Crack500, CrackTree260, CrackLS315, CRKWH100')
    parser.add_argument('--output', type=str, default='../data/processed',
                       help='输出目录')
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--test-ratio', type=float, default=0.15)
    parser.add_argument('--size', type=int, nargs=2, default=None,
                       help='目标尺寸 H W')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    datasets_root = (script_dir / args.datasets_root).resolve()
    
    selector = DatasetSelector(str(datasets_root))
    
    if args.info:
        selector.print_info()
        return
    
    if not args.datasets:
        print("请指定数据集，或使用 --info 查看可用数据集")
        print("示例: python dataset_selector.py --datasets CrackTree260 --output ../data/cracktree260")
        return
    
    output_dir = (script_dir / args.output).resolve()
    target_size = tuple(args.size) if args.size else None
    
    selector.prepare(
        dataset_names=args.datasets,
        output_dir=str(output_dir),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        target_size=target_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
