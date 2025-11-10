"""
数据集可视化检查工具
用于验证规范化后的数据集质量
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
import argparse
from typing import List, Tuple


def visualize_sample(image_path: Path, mask_path: Path, save_path: Path = None):
    """
    可视化单个样本
    
    Args:
        image_path: 图像路径
        mask_path: 掩码路径
        save_path: 保存路径（可选）
    """
    # 读取图像和掩码
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    
    # 创建可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 原始图像
    axes[0, 0].imshow(image)
    axes[0, 0].set_title(f'Image\n{image.shape[1]}x{image.shape[0]}', fontsize=10)
    axes[0, 0].axis('off')
    
    # 掩码
    axes[0, 1].imshow(mask, cmap='gray')
    axes[0, 1].set_title(f'Mask\nValues: {mask.min()}-{mask.max()}', fontsize=10)
    axes[0, 1].axis('off')
    
    # 掩码统计
    crack_pixels = np.sum(mask > 0)
    total_pixels = mask.size
    crack_ratio = crack_pixels / total_pixels * 100
    
    axes[0, 2].text(0.1, 0.5, 
                   f'统计信息:\n'
                   f'图像尺寸: {image.shape[1]}×{image.shape[0]}\n'
                   f'掩码尺寸: {mask.shape[1]}×{mask.shape[0]}\n'
                   f'裂纹像素: {crack_pixels:,}\n'
                   f'总像素: {total_pixels:,}\n'
                   f'裂纹占比: {crack_ratio:.2f}%\n'
                   f'掩码值: {mask.min()}-{mask.max()}',
                   fontsize=12, verticalalignment='center')
    axes[0, 2].axis('off')
    
    # 红色叠加
    overlay_red = image.copy()
    overlay_red[mask > 0] = [255, 0, 0]  # 红色
    axes[1, 0].imshow(overlay_red)
    axes[1, 0].set_title('Red Overlay', fontsize=10)
    axes[1, 0].axis('off')
    
    # 半透明叠加
    overlay_alpha = image.copy().astype(float)
    alpha = 0.5
    overlay_alpha[mask > 0] = overlay_alpha[mask > 0] * (1 - alpha) + np.array([255, 0, 0]) * alpha
    overlay_alpha = overlay_alpha.astype(np.uint8)
    axes[1, 1].imshow(overlay_alpha)
    axes[1, 1].set_title('Alpha Blend (0.5)', fontsize=10)
    axes[1, 1].axis('off')
    
    # 边界检测
    edges = cv2.Canny(mask, 50, 150)
    axes[1, 2].imshow(edges, cmap='gray')
    axes[1, 2].set_title('Crack Edges', fontsize=10)
    axes[1, 2].axis('off')
    
    plt.suptitle(f'Sample: {image_path.stem}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def check_dataset_integrity(data_root: Path) -> dict:
    """
    检查数据集完整性
    
    Args:
        data_root: 数据集根目录
    Returns:
        检查结果字典
    """
    results = {
        'total_samples': 0,
        'valid_samples': 0,
        'invalid_samples': [],
        'size_mismatches': [],
        'missing_masks': [],
        'missing_images': [],
        'mask_value_issues': []
    }
    
    for split in ['train', 'val', 'test']:
        split_file = data_root / f'{split}.txt'
        
        if not split_file.exists():
            print(f"Warning: {split}.txt not found")
            continue
        
        with open(split_file) as f:
            sample_ids = [line.strip() for line in f]
        
        print(f"\n检查 {split} split ({len(sample_ids)} 个样本)...")
        
        for sample_id in sample_ids:
            results['total_samples'] += 1
            
            img_path = data_root / 'images' / split / f'{sample_id}.jpg'
            mask_path = data_root / 'masks' / split / f'{sample_id}.png'
            
            # 检查文件存在
            if not img_path.exists():
                results['missing_images'].append(str(img_path))
                continue
            
            if not mask_path.exists():
                results['missing_masks'].append(str(mask_path))
                continue
            
            # 读取并检查
            img = cv2.imread(str(img_path))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            if img is None or mask is None:
                results['invalid_samples'].append(sample_id)
                continue
            
            # 检查尺寸匹配
            if img.shape[:2] != mask.shape[:2]:
                results['size_mismatches'].append({
                    'id': sample_id,
                    'img_size': img.shape[:2],
                    'mask_size': mask.shape[:2]
                })
                continue
            
            # 检查掩码值
            unique_values = np.unique(mask)
            if not all(v in [0, 255] for v in unique_values):
                results['mask_value_issues'].append({
                    'id': sample_id,
                    'values': unique_values.tolist()
                })
                continue
            
            results['valid_samples'] += 1
    
    return results


def print_check_results(results: dict):
    """打印检查结果"""
    print("\n" + "=" * 60)
    print("数据集完整性检查结果")
    print("=" * 60)
    
    print(f"\n总样本数: {results['total_samples']}")
    print(f"有效样本: {results['valid_samples']} ✓")
    print(f"无效样本: {len(results['invalid_samples'])}")
    
    if results['missing_images']:
        print(f"\n缺失图像 ({len(results['missing_images'])}):")
        for path in results['missing_images'][:5]:
            print(f"  - {path}")
        if len(results['missing_images']) > 5:
            print(f"  ... 还有 {len(results['missing_images']) - 5} 个")
    
    if results['missing_masks']:
        print(f"\n缺失掩码 ({len(results['missing_masks'])}):")
        for path in results['missing_masks'][:5]:
            print(f"  - {path}")
        if len(results['missing_masks']) > 5:
            print(f"  ... 还有 {len(results['missing_masks']) - 5} 个")
    
    if results['size_mismatches']:
        print(f"\n尺寸不匹配 ({len(results['size_mismatches'])}):")
        for item in results['size_mismatches'][:5]:
            print(f"  - {item['id']}: 图像{item['img_size']} vs 掩码{item['mask_size']}")
        if len(results['size_mismatches']) > 5:
            print(f"  ... 还有 {len(results['size_mismatches']) - 5} 个")
    
    if results['mask_value_issues']:
        print(f"\n掩码值异常 ({len(results['mask_value_issues'])}):")
        for item in results['mask_value_issues'][:5]:
            print(f"  - {item['id']}: 值={item['values']}")
        if len(results['mask_value_issues']) > 5:
            print(f"  ... 还有 {len(results['mask_value_issues']) - 5} 个")
    
    print("\n" + "=" * 60)
    
    if results['valid_samples'] == results['total_samples']:
        print("✅ 数据集完整性检查通过！")
    else:
        print("⚠️  发现问题，请检查上述错误")
    
    print("=" * 60)


def visualize_random_samples(data_root: Path, 
                            split: str = 'train',
                            num_samples: int = 6,
                            output_dir: Path = None):
    """
    随机可视化多个样本
    
    Args:
        data_root: 数据集根目录
        split: train/val/test
        num_samples: 样本数量
        output_dir: 输出目录
    """
    split_file = data_root / f'{split}.txt'
    
    with open(split_file) as f:
        sample_ids = [line.strip() for line in f]
    
    # 随机选择样本
    random.shuffle(sample_ids)
    selected = sample_ids[:num_samples]
    
    print(f"\n可视化 {len(selected)} 个随机样本...")
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, sample_id in enumerate(selected, 1):
        img_path = data_root / 'images' / split / f'{sample_id}.jpg'
        mask_path = data_root / 'masks' / split / f'{sample_id}.png'
        
        save_path = output_dir / f'sample_{i:02d}_{sample_id}.png' if output_dir else None
        
        print(f"  [{i}/{len(selected)}] {sample_id}")
        visualize_sample(img_path, mask_path, save_path)


def analyze_dataset_statistics(data_root: Path):
    """分析数据集统计信息"""
    stats = {
        'splits': {},
        'image_sizes': [],
        'crack_ratios': [],
        'mask_values': []
    }
    
    for split in ['train', 'val', 'test']:
        split_file = data_root / f'{split}.txt'
        
        if not split_file.exists():
            continue
        
        with open(split_file) as f:
            sample_ids = [line.strip() for line in f]
        
        stats['splits'][split] = len(sample_ids)
        
        # 采样分析
        sample_count = min(100, len(sample_ids))
        sampled = random.sample(sample_ids, sample_count)
        
        for sample_id in sampled:
            img_path = data_root / 'images' / split / f'{sample_id}.jpg'
            mask_path = data_root / 'masks' / split / f'{sample_id}.png'
            
            img = cv2.imread(str(img_path))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            if img is not None and mask is not None:
                stats['image_sizes'].append(img.shape[:2])
                
                crack_ratio = np.sum(mask > 0) / mask.size
                stats['crack_ratios'].append(crack_ratio)
                
                stats['mask_values'].extend(np.unique(mask).tolist())
    
    # 打印统计
    print("\n" + "=" * 60)
    print("数据集统计分析")
    print("=" * 60)
    
    print("\n样本分布:")
    total = sum(stats['splits'].values())
    for split, count in stats['splits'].items():
        ratio = count / total * 100 if total > 0 else 0
        print(f"  {split}: {count} ({ratio:.1f}%)")
    print(f"  总计: {total}")
    
    if stats['image_sizes']:
        heights, widths = zip(*stats['image_sizes'])
        print("\n图像尺寸统计:")
        print(f"  高度: {np.min(heights)} ~ {np.max(heights)} (均值: {np.mean(heights):.0f})")
        print(f"  宽度: {np.min(widths)} ~ {np.max(widths)} (均值: {np.mean(widths):.0f})")
    
    if stats['crack_ratios']:
        print("\n裂纹占比:")
        print(f"  最小: {np.min(stats['crack_ratios'])*100:.2f}%")
        print(f"  最大: {np.max(stats['crack_ratios'])*100:.2f}%")
        print(f"  均值: {np.mean(stats['crack_ratios'])*100:.2f}%")
        print(f"  中位: {np.median(stats['crack_ratios'])*100:.2f}%")
    
    if stats['mask_values']:
        unique_values = sorted(set(stats['mask_values']))
        print(f"\n掩码唯一值: {unique_values}")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='数据集可视化检查工具')
    parser.add_argument('--data-root', type=str,
                       default='E:/desktop/Code/cloud/data/processed',
                       help='数据集根目录')
    parser.add_argument('--check-integrity', action='store_true',
                       help='检查数据集完整性')
    parser.add_argument('--visualize', action='store_true',
                       help='可视化随机样本')
    parser.add_argument('--analyze', action='store_true',
                       help='分析统计信息')
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'val', 'test'],
                       help='要可视化的split')
    parser.add_argument('--num-samples', type=int, default=6,
                       help='可视化样本数')
    parser.add_argument('--output-dir', type=str, default='./visualizations',
                       help='可视化输出目录')
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    
    if not data_root.exists():
        print(f"错误: 数据集目录不存在: {data_root}")
        return
    
    # 默认：全部检查
    if not (args.check_integrity or args.visualize or args.analyze):
        args.check_integrity = True
        args.visualize = True
        args.analyze = True
    
    # 完整性检查
    if args.check_integrity:
        results = check_dataset_integrity(data_root)
        print_check_results(results)
    
    # 统计分析
    if args.analyze:
        analyze_dataset_statistics(data_root)
    
    # 可视化
    if args.visualize:
        output_dir = Path(args.output_dir) if args.output_dir else None
        visualize_random_samples(
            data_root, 
            split=args.split,
            num_samples=args.num_samples,
            output_dir=output_dir
        )
        
        if output_dir:
            print(f"\n可视化结果保存在: {output_dir}")


if __name__ == '__main__':
    main()

