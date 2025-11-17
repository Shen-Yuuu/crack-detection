"""
SOTA模型架构: ConvNeXt + UPerNet
结合边界感知分支和深度监督
支持本地权重路径字符串（pretrained 可为 True/False 或 本地路径）
"""
import os
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ConvNeXtEncoder(nn.Module):
    """ConvNeXt编码器（使用timm预训练模型）。

    pretrained 参数可以是:
      - True/False: 由 timm 决定是否从远端加载（会触发网络访问）
      - str: 本地权重文件路径，若文件存在则优先从该路径加载（不会再远端下载）
    """

    def __init__(self,
                 model_name: str = 'convnext_tiny',
                 pretrained: bool = True,
                 in_channels: int = 3):
        super().__init__()

        # 如果 pretrained 是字符串且文件存在，则先创建不带预训练加载的模型
        use_local = isinstance(pretrained, str) and os.path.isfile(pretrained)
        timm_pretrained_flag = False if use_local else bool(pretrained)

        # 创建 backbone（如果 use_local=True 则不会尝试远端下载）
        self.backbone = timm.create_model(
            model_name,
            pretrained=timm_pretrained_flag,
            features_only=True,
            in_chans=in_channels,
            out_indices=(0, 1, 2, 3)  # 4个特征层
        )

        # 若指定本地权重路径，尝试加载
        if use_local:
            path = pretrained
            try:
                state = torch.load(path, map_location='cpu', weights_only=False)
            except Exception as e:
                print(f"[ConvNeXtEncoder] 无法加载本地权重文件 {path}: {e}")
                state = None

            if isinstance(state, dict):
                # 支持常见包装形式: {'model':..., 'state_dict':...} 或 直接 state_dict
                if 'model' in state and isinstance(state['model'], dict):
                    sd = state['model']
                elif 'state_dict' in state and isinstance(state['state_dict'], dict):
                    sd = state['state_dict']
                else:
                    sd = state
            else:
                sd = state

            if isinstance(sd, dict):
                # 清理常见前缀（module., model., backbone., encoder.）
                cleaned = {}
                for k, v in sd.items():
                    new_k = k
                    for prefix in ('module.', 'model.', 'backbone.', 'encoder.'):
                        if new_k.startswith(prefix):
                            new_k = new_k[len(prefix):]
                    cleaned[new_k] = v

                # 尝试加载到 backbone（宽松模式 strict=False）
                try:
                    self.backbone.load_state_dict(cleaned, strict=False)
                    print(f"[ConvNeXtEncoder] 成功从本地文件加载权重: {path}")
                except Exception as e:
                    print(f"[ConvNeXtEncoder] 加载本地权重时发生异常: {e}")
                    try:
                        self.backbone.load_state_dict(cleaned, strict=False)
                        print(f"[ConvNeXtEncoder] 宽松加载成功（strict=False）")
                    except Exception as ee:
                        print(f"[ConvNeXtEncoder] 宽松加载失败: {ee}")
            else:
                print(f"[ConvNeXtEncoder] 本地权重文件内容无法识别，跳过加载: {path}")

        # 获取特征通道数
        self.feature_info = self.backbone.feature_info
        self.out_channels = [info['num_chs'] for info in self.feature_info]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            features: List[(B, C1, H/4, W/4), (B, C2, H/8, W/8),
                          (B, C3, H/16, W/16), (B, C4, H/32, W/32)]
        """
        features = self.backbone(x)
        return features


class PSPModule(nn.Module):
    """金字塔池化模块（Pyramid Pooling Module）"""

    def __init__(self, in_channels: int, out_channels: int, pool_sizes: Tuple[int] = (1, 2, 3, 6)):
        super().__init__()

        # 计算合适的group数
        psp_channels = out_channels // len(pool_sizes)
        psp_groups = min(32, psp_channels) if psp_channels >= 32 else psp_channels
        out_groups = min(32, out_channels) if out_channels >= 32 else out_channels

        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(size),
                nn.Conv2d(in_channels, psp_channels, 1, bias=False),
                nn.GroupNorm(psp_groups, psp_channels),
                nn.ReLU(inplace=True)
            )
            for size in pool_sizes
        ])

        self.conv_out = nn.Sequential(
            nn.Conv2d(in_channels + out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(out_groups, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.size()[2:]

        # 多尺度池化
        pyramids = [x]
        for stage in self.stages:
            feat = stage(x)
            feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
            pyramids.append(feat)

        # 拼接
        out = torch.cat(pyramids, dim=1)
        out = self.conv_out(out)

        return out


class StripPooling(nn.Module):
    """条形池化（用于细长裂纹）

    注意：PyTorch 的 AdaptiveAvgPool2d 无法使用 None 动态维度，因此这里用沿宽/高方向做平均来实现“条形”池化。
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # 使用 1x1 卷积投影
        self.conv_h = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv_w = nn.Conv2d(in_channels, out_channels, 1, bias=False)

        # 计算合适的group数
        out_groups = min(32, out_channels) if out_channels >= 32 else out_channels

        self.conv_out = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(out_groups, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape

        # 沿宽度方向平均，得到 (B, C, H, 1)
        feat_h = x.mean(dim=3, keepdim=True)
        feat_h = self.conv_h(feat_h)  # (B, out_channels, H, 1)
        feat_h = F.interpolate(feat_h, size=(H, W), mode='bilinear', align_corners=False)

        # 沿高度方向平均，得到 (B, C, 1, W)
        feat_w = x.mean(dim=2, keepdim=True)
        feat_w = self.conv_w(feat_w)  # (B, out_channels, 1, W)
        feat_w = F.interpolate(feat_w, size=(H, W), mode='bilinear', align_corners=False)

        out = torch.cat([feat_h, feat_w], dim=1)
        out = self.conv_out(out)
        return out


class CBAM(nn.Module):
    """卷积块注意力模块（Convolutional Block Attention Module）"""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()

        # Channel Attention
        self.channel_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )

        # Spatial Attention
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel Attention
        avg_out = self.channel_fc(self.channel_avg_pool(x))
        max_out = self.channel_fc(self.channel_max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        x = x * channel_att

        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial_conv(torch.cat([avg_out, max_out], dim=1))
        x = x * spatial_att

        return x


class UPerNetDecoder(nn.Module):
    """UPerNet解码器"""

    def __init__(self,
                 encoder_channels: List[int],
                 decoder_channels: int = 256,
                 num_classes: int = 1,
                 head_dropout: float = 0.1):
        super().__init__()

        # PPM模块（应用于最深层特征）
        self.ppm = PSPModule(encoder_channels[-1], decoder_channels)

        # 计算合适的group数
        dec_groups = min(32, decoder_channels) if decoder_channels >= 32 else decoder_channels

        # FPN侧向连接
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, decoder_channels, 1, bias=False),
                nn.GroupNorm(dec_groups, decoder_channels),
                nn.ReLU(inplace=True)
            )
            for ch in encoder_channels
        ])

        # FPN输出卷积
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1, bias=False),
                nn.GroupNorm(dec_groups, decoder_channels),
                nn.ReLU(inplace=True),
                CBAM(decoder_channels)  # 添加注意力
            )
            for _ in encoder_channels
        ])

        # 条形池化（针对细长裂纹）
        self.strip_pool = StripPooling(decoder_channels * len(encoder_channels), decoder_channels)

        # 最终分类头
        head_channels = decoder_channels // 2
        head_groups = min(16, head_channels) if head_channels >= 16 else head_channels

        self.cls_head = nn.Sequential(
            nn.Conv2d(decoder_channels, head_channels, 3, padding=1, bias=False),
            nn.GroupNorm(head_groups, head_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(head_dropout),
            nn.Conv2d(head_channels, num_classes, 1)
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of encoder features
        Returns:
            output: (B, num_classes, H, W)
        """
        # PPM处理最深层特征
        fpn_features = [self.ppm(features[-1])]

        # 构建FPN（自顶向下）
        for i in range(len(features) - 2, -1, -1):
            lateral = self.lateral_convs[i](features[i])

            # 上采样并相加
            top_down = F.interpolate(
                fpn_features[0],
                size=lateral.shape[2:],
                mode='bilinear',
                align_corners=False
            )

            fpn_feat = lateral + top_down
            fpn_feat = self.fpn_convs[i](fpn_feat)
            fpn_features.insert(0, fpn_feat)

        # 统一尺度并融合
        target_size = fpn_features[0].shape[2:]
        upsampled_features = [fpn_features[0]]

        for feat in fpn_features[1:]:
            upsampled = F.interpolate(
                feat,
                size=target_size,
                mode='bilinear',
                align_corners=False
            )
            upsampled_features.append(upsampled)

        # 拼接所有特征
        fused = torch.cat(upsampled_features, dim=1)

        # 条形池化增强细长特征
        fused = self.strip_pool(fused)

        # 分类
        output = self.cls_head(fused)

        return output


class EdgeDetectionBranch(nn.Module):
    """边界检测分支"""

    def __init__(self, in_channels: int, proj_channels: int = 256):
        super().__init__()

        # 投影层（将输入特征投影到固定通道数）
        proj_groups = min(32, proj_channels) if proj_channels >= 32 else proj_channels
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, proj_channels, 1, bias=False),
            nn.GroupNorm(proj_groups, proj_channels),
            nn.ReLU(inplace=True)
        )

        # Sobel算子提取边界（采用深度卷积实现）
        self.sobel_x = nn.Conv2d(proj_channels, proj_channels, 3, padding=1, bias=False, groups=proj_channels)
        self.sobel_y = nn.Conv2d(proj_channels, proj_channels, 3, padding=1, bias=False, groups=proj_channels)

        # 初始化Sobel核
        sobel_x_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y_kernel = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

        # 为每个通道赋相同的核（使用 no_grad）
        with torch.no_grad():
            # shape expected: (proj_channels, 1, 3, 3)
            for i in range(proj_channels):
                self.sobel_x.weight.data[i, 0].copy_(sobel_x_kernel)
                self.sobel_y.weight.data[i, 0].copy_(sobel_y_kernel)
            self.sobel_x.weight.requires_grad = False
            self.sobel_y.weight.requires_grad = False

        # 边界特征增强
        edge_groups = min(32, proj_channels) if proj_channels >= 32 else proj_channels

        self.edge_conv = nn.Sequential(
            nn.Conv2d(proj_channels * 2, proj_channels, 3, padding=1, bias=False),
            nn.GroupNorm(edge_groups, proj_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(proj_channels, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 投影到固定通道数
        x = self.proj(x)

        # 提取边界
        edge_x = self.sobel_x(x)
        edge_y = self.sobel_y(x)

        edges = torch.cat([edge_x, edge_y], dim=1)
        edge_map = self.edge_conv(edges)

        return edge_map


class ConvNeXtUPerNet(nn.Module):
    """完整的ConvNeXt + UPerNet模型（带边界分支和深度监督）"""

    def __init__(self,
                 encoder_name: str = 'convnext_tiny',
                 pretrained: bool = True,
                 num_classes: int = 1,
                 decoder_channels: int = 256,
                 deep_supervision: bool = True,
                 edge_branch: bool = True,
                 head_dropout: float = 0.1):
        super().__init__()

        self.deep_supervision = deep_supervision
        self.edge_branch = edge_branch

        # 编码器
        self.encoder = ConvNeXtEncoder(encoder_name, pretrained)
        encoder_channels = self.encoder.out_channels

        # 解码器
        self.decoder = UPerNetDecoder(encoder_channels, decoder_channels, num_classes, head_dropout=head_dropout)

        # 边界分支
        if edge_branch:
            # 使用倒数第二层特征的通道数
            edge_in_channels = encoder_channels[-2]
            self.edge_head = EdgeDetectionBranch(edge_in_channels, decoder_channels)

        # 深度监督辅助头
        if deep_supervision:
            aux_channels = decoder_channels // 2
            aux_groups = min(16, aux_channels) if aux_channels >= 16 else aux_channels

            self.aux_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(ch, aux_channels, 3, padding=1),
                    nn.GroupNorm(aux_groups, aux_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(aux_channels, num_classes, 1)
                )
                for ch in encoder_channels[-3:]  # 使用后3层特征
            ])

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            dict with keys:
                - 'out': main output (B, num_classes, H, W)
                - 'edge': edge output (B, 1, H, W) if edge_branch
                - 'aux': list of auxiliary outputs if deep_supervision
        """
        input_size = x.shape[2:]

        # 编码
        features = self.encoder(x)

        # 解码
        output = self.decoder(features)

        # 上采样到输入尺寸
        output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=False)

        results = {'out': output}

        # 边界分支
        if self.edge_branch:
            # 使用中间层特征
            edge_feat = features[-2]  # 1/16尺度
            edge_output = self.edge_head(edge_feat)
            edge_output = F.interpolate(edge_output, size=input_size, mode='bilinear', align_corners=False)
            results['edge'] = edge_output

        # 深度监督
        if self.deep_supervision and self.training:
            aux_outputs = []
            # encoder_channels[-3:] 创建了3个辅助头，分别对应 features[-3], features[-2], features[-1]
            # aux_heads[0] 对应 encoder_channels[-3] -> features[-3]
            # aux_heads[1] 对应 encoder_channels[-2] -> features[-2]
            # aux_heads[2] 对应 encoder_channels[-1] -> features[-1]
            for i, aux_head in enumerate(self.aux_heads):
                # 从 features[-3] 开始，依次使用 features[-3], features[-2], features[-1]
                aux_feat = features[-3 + i]
                aux_out = aux_head(aux_feat)
                aux_out = F.interpolate(aux_out, size=input_size, mode='bilinear', align_corners=False)
                aux_outputs.append(aux_out)
            results['aux'] = aux_outputs

        return results


def create_model(model_config: dict) -> nn.Module:
    """工厂函数创建模型"""

    model_type = model_config.get('backbone', 'convnext_tiny')

    if 'convnext' in model_type:
        model = ConvNeXtUPerNet(
            encoder_name=model_type,
            pretrained=model_config.get('pretrained', True),
            num_classes=model_config.get('num_classes', 1),
            decoder_channels=model_config.get('decoder_channels', 256),
            deep_supervision=model_config.get('deep_supervision', True),
            edge_branch=model_config.get('edge_branch', True),
            head_dropout=model_config.get('head_dropout', 0.1)
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return model


if __name__ == "__main__":
    # 测试模型
    model = ConvNeXtUPerNet(
        encoder_name='convnext_tiny',
        pretrained=False,
        deep_supervision=True,
        edge_branch=True
    )

    model.eval()

    x = torch.randn(2, 3, 512, 512)

    with torch.no_grad():
        outputs = model(x)

    print(f"Main output shape: {outputs['out'].shape}")
    if 'edge' in outputs:
        print(f"Edge output shape: {outputs['edge'].shape}")
    if 'aux' in outputs:
        print(f"Auxiliary outputs: {len(outputs['aux'])}")

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")