# 🚀 裂纹检测 - 数据处理与模型训练模块

基于 **ConvNeXt + UPerNet** 的高性能道路裂纹分割系统，采用最前沿的深度学习技术。

## ✨ 核心特性

### 📊 数据处理
- ✅ 多格式支持（COCO/VOC/YOLO → PNG Mask）
- ✅ 自动质量控制（尺寸检查、小伪影过滤、标注错误检测）
- ✅ 难例挖掘（Hard Example Mining）
- ✅ LMDB缓存加速
- ✅ 高级数据增强（Albumentations）
  - 几何增强：RandomScale、Rotate、Affine、ElasticTransform
  - 颜色增强：CLAHE、RandomBrightnessContrast、HueSaturationValue
  - 噪声/天气：GaussNoise、MotionBlur、Rain/Snow/Fog
  - Copy-Paste（细裂纹增强）

### 🎯 SOTA模型架构
- ✅ **Backbone**: ConvNeXt-T/S（ImageNet预训练）
- ✅ **Decoder**: UPerNet（金字塔池化 + FPN）
- ✅ **注意力机制**: CBAM（通道+空间注意力）
- ✅ **细长目标优化**: Strip Pooling
- ✅ **边界增强**: Edge Detection Branch（Sobel引导）
- ✅ **深度监督**: 多尺度辅助损失

### 🔥 高级损失函数
- ✅ Dice Loss（区域重叠优化）
- ✅ Focal Loss（类别不平衡处理）
- ✅ Tversky Loss（FP/FN权重可调）
- ✅ Boundary Loss（边界敏感）
- ✅ Lovasz-Hinge Loss（IoU直接优化）
- ✅ 组合损失（L = 0.4·Dice + 0.3·Focal + 0.2·BCE + 0.1·Boundary）

### 🚀 训练优化技术
- ✅ 混合精度训练（AMP）
- ✅ 指数移动平均（EMA, decay=0.9995）
- ✅ 随机权重平均（SWA, 最后10% epoch）
- ✅ 梯度累积与裁剪
- ✅ OneCycle / Cosine 学习率调度
- ✅ 早停机制（patience=20）

### 🎨 高性能推理
- ✅ 滑窗推理（高分辨率图像，Gaussian融合）
- ✅ 测试时增强（TTA）：多尺度 + 翻转
- ✅ 温度标定（Temperature Scaling）
- ✅ ONNX导出（opset 17+）
- ✅ 模型量化（INT8）
- ✅ TensorRT优化（FP16/INT8）

## 📁 项目结构

```
python-inference/
├── dataset/
│   └── data_loader.py          # 数据加载器、格式转换、质量控制
├── models/
│   └── convnext_upernet.py     # ConvNeXt + UPerNet模型
├── training/
│   ├── losses.py               # 损失函数集合
│   └── trainer.py              # 训练器（EMA、SWA、AMP）
├── inference/
│   └── sliding_window.py       # 滑窗推理、TTA
├── configs/
│   └── train_config.yaml       # 训练配置
├── train.py                    # 训练入口
├── export_onnx.py              # ONNX导出工具
├── requirements.txt            # 依赖包
└── README.md                   # 本文档
```

## 🛠️ 安装

```bash
# 创建虚拟环境
conda create -n crack-detection python=3.10
conda activate crack-detection

# 安装依赖
pip install -r requirements.txt

# 安装PyTorch（根据CUDA版本选择）
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## 📚 数据准备

### 1. 数据集目录结构

```
data/processed/
├── images/
│   ├── train/
│   │   ├── img001.jpg
│   │   └── ...
│   ├── val/
│   └── test/
├── masks/
│   ├── train/
│   │   ├── img001.png  # 二值掩码（0/255）
│   │   └── ...
│   ├── val/
│   └── test/
├── train.txt               # 训练集样本列表
├── val.txt                 # 验证集样本列表
└── test.txt                # 测试集样本列表
```

### 2. 样本列表格式

```txt
# train.txt
img001
img002
img003
...
```

## 🚀 训练

### 快速开始

```bash
# 使用默认配置训练
python train.py --config configs/train_config.yaml

# 从检查点恢复
python train.py --config configs/train_config.yaml --resume outputs/last.pth

# 指定GPU
python train.py --config configs/train_config.yaml --device cuda:0
```

### 配置说明

编辑 `configs/train_config.yaml`:

```yaml
# 模型配置
model:
  backbone: "convnext_tiny"      # convnext_tiny, small, base
  decoder_channels: 256
  deep_supervision: true
  edge_branch: true

# 训练配置
training:
  epochs: 200
  batch_size: 16
  use_amp: true                  # 混合精度
  use_ema: true                  # EMA
  use_swa: true                  # SWA
  swa_start_epoch: 180
```

### 训练技巧

1. **多尺度训练**：`train_scales: [256, 384, 512]`
2. **动态增强强度**：前60% epoch强增强，后40%弱增强
3. **难例挖掘**：自动根据损失值调整样本权重
4. **梯度累积**：小显存时增大 `gradient_accumulation_steps`

## 📊 模型评估

```python
from training.trainer import Trainer
from models.convnext_upernet import create_model
from training.losses import create_loss

# 加载模型
model = create_model(config['model'])
trainer = Trainer(model, optimizer, loss_fn)
trainer.load_checkpoint('outputs/best.pth')

# 验证
val_metrics = trainer.validate(val_loader)
print(f"Val IoU: {val_metrics['iou']:.4f}")
```

## 🎯 模型导出

### 导出ONNX

```bash
# 基础导出
python export_onnx.py \
  --checkpoint outputs/best.pth \
  --output model.onnx \
  --input-shape 1 3 512 512

# 导出 + 验证 + 优化 + 量化 + 基准测试
python export_onnx.py \
  --checkpoint outputs/best.pth \
  --output model.onnx \
  --verify \
  --optimize \
  --quantize \
  --benchmark
```

### ONNX推理

```python
import onnxruntime as ort
import numpy as np

# 创建会话
session = ort.InferenceSession('model.onnx')

# 推理
input_data = np.random.randn(1, 3, 512, 512).astype(np.float32)
output = session.run(None, {'input': input_data})[0]
```

## 🔬 高级推理

### 滑窗推理（高分辨率）

```python
from inference.sliding_window import SlidingWindowInference

sliding_window = SlidingWindowInference(
    window_size=(1024, 1024),
    overlap=0.25,
    batch_size=4,
    blend_mode='gaussian'
)

pred = sliding_window(model, high_res_image, device='cuda')
```

### 测试时增强（TTA）

```python
from inference.sliding_window import TTAInference

tta = TTAInference(
    scales=[0.75, 1.0, 1.25],
    flip_h=True,
    flip_v=True
)

pred = tta(model, image, device='cuda')
```

## 📈 性能指标

### 精度指标
- **mIoU**: ≥ 85%
- **Boundary F1**: 边界精度评估
- **Thin-Region IoU**: 细裂纹（宽度<3px）评估

### 速度指标
- **训练速度**: ~200 images/s (V100, batch=16, AMP)
- **推理速度**: 
  - FP32: ~50 ms/image (512×512)
  - FP16: ~30 ms/image
  - INT8: ~20 ms/image

## 🎓 参考论文

1. **ConvNeXt**: A ConvNet for the 2020s ([arXiv](https://arxiv.org/abs/2201.03545))
2. **UPerNet**: Unified Perceptual Parsing ([arXiv](https://arxiv.org/abs/1807.10221))
3. **CBAM**: Convolutional Block Attention Module ([arXiv](https://arxiv.org/abs/1807.06521))
4. **Focal Loss**: Dense Object Detection ([arXiv](https://arxiv.org/abs/1708.02002))
5. **Lovasz Loss**: A tractable surrogate for IoU optimization ([arXiv](https://arxiv.org/abs/1705.08790))

## 📚 详细文档

- [实现总结](docs/实现总结.md) - 技术实现细节和算法原理
- [数据处理指南](docs/数据处理指南.md) - 数据集规范化和处理流程
- [项目主文档](../README.md) - 项目概述和快速开始
- [系统设计方案](../docs/系统设计方案.md) - 完整系统架构
- [使用指南](../docs/使用指南.md) - 详细使用教程

## 📝 许可证

MIT License

## 👥 联系方式

- 作者：高绅语
- 指导教师：杨飞
- 学校：山东大学 低空科学与工程学院
- 专业：计算机科学与技术（未来网络）

---

**注意**：本项目为毕业设计项目，代码仅供学习参考。

