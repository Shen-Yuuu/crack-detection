"""
SOTA模型架构: ConvNeXt + UPerNet
结合边界感知分支和深度监督
支持本地权重路径字符串（pretrained 可为 True/False 或 本地路径）

优化版本：
- 添加 DropPath (Stochastic Depth)
- 升级注意力机制 (ECA + Coordinate Attention)
- 改进特征融合 (ASPP-lite)
- 可学习边界检测
"""
import os
import math
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


def drop_path(x: torch.Tensor, drop_prob: float = 0., training: bool = False) -> torch.Tensor:
    """DropPath (Stochastic Depth) 正则化"""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """DropPath (Stochastic Depth) 模块"""
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class EfficientChannelAttention(nn.Module):
    """高效通道注意力 (ECA) - 比SE更轻量且效果更好"""
    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        # 自适应计算kernel size
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        k = max(3, k)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, 1, 1) -> (B, 1, C)
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y


class CoordinateAttention(nn.Module):
    """坐标注意力 - 同时编码通道和空间位置信息"""
    def __init__(self, in_channels: int, reduction: int = 32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        mid_channels = max(8, in_channels // reduction)
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = nn.SiLU(inplace=True)  # Swish激活
        
        self.conv_h = nn.Conv2d(mid_channels, in_channels, 1, bias=False)
        self.conv_w = nn.Conv2d(mid_channels, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # 沿宽度和高度分别池化
        x_h = self.pool_h(x)  # (B, C, H, 1)
        x_w = self.pool_w(x).transpose(-1, -2)  # (B, C, 1, W) -> (B, C, W, 1)
        
        # 拼接并通过共享卷积
        y = torch.cat([x_h, x_w], dim=2)  # (B, C, H+W, 1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        # 分离并生成注意力权重
        x_h, x_w = torch.split(y, [H, W], dim=2)
        x_w = x_w.transpose(-1, -2)
        
        a_h = self.sigmoid(self.conv_h(x_h))
        a_w = self.sigmoid(self.conv_w(x_w))
        
        return x * a_h * a_w


class ASPPLite(nn.Module):
    """轻量级ASPP - 多尺度空洞卷积特征融合"""
    def __init__(self, in_channels: int, out_channels: int, dilations: Tuple[int] = (1, 6, 12)):
        super().__init__()
        
        # 确保通道数能被整除
        branch_channels = out_channels // len(dilations)
        # 计算合适的group数
        branch_groups = min(8, branch_channels) if branch_channels >= 8 else max(1, branch_channels)
        # 确保branch_channels能被branch_groups整除
        while branch_channels % branch_groups != 0 and branch_groups > 1:
            branch_groups -= 1
        
        self.convs = nn.ModuleList()
        for dilation in dilations:
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, branch_channels, 3, 
                              padding=dilation, dilation=dilation, bias=False),
                    nn.GroupNorm(branch_groups, branch_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # 全局上下文
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, branch_channels, 1, bias=False),
            nn.ReLU(inplace=True)
        )
        
        # 融合
        total_channels = branch_channels * (len(dilations) + 1)
        out_groups = min(32, out_channels) if out_channels >= 32 else max(1, out_channels)
        while out_channels % out_groups != 0 and out_groups > 1:
            out_groups -= 1
            
        self.fuse = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, 1, bias=False),
            nn.GroupNorm(out_groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[2:]
        
        features = [conv(x) for conv in self.convs]
        
        global_feat = self.global_pool(x)
        global_feat = F.interpolate(global_feat, size=size, mode='bilinear', align_corners=False)
        features.append(global_feat)
        
        return self.fuse(torch.cat(features, dim=1))


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
        psp_groups = self._select_group_count(psp_channels)
        out_groups = self._select_group_count(out_channels)

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

    @staticmethod
    def _select_group_count(channels: int, max_groups: int = 32) -> int:
        """选择能够整除通道数的最大分组数，避免 GroupNorm 抛出异常。"""
        channels = max(channels, 1)
        max_g = min(max_groups, channels)
        for group in range(max_g, 0, -1):
            if channels % group == 0:
                return group
        return 1


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
    """增强版CBAM - 使用ECA替代原始通道注意力"""

    def __init__(self, channels: int, reduction: int = 16, use_coord_att: bool = False):
        super().__init__()
        
        self.use_coord_att = use_coord_att
        
        if use_coord_att:
            # 使用坐标注意力（同时编码通道和空间）
            self.attention = CoordinateAttention(channels, reduction)
        else:
            # ECA通道注意力 + 空间注意力
            self.channel_att = EfficientChannelAttention(channels)
            
            # 增强的空间注意力（使用深度可分离卷积）
            self.spatial_conv = nn.Sequential(
                nn.Conv2d(2, 8, 3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(8, 1, 3, padding=1, bias=False),
                nn.Sigmoid()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_coord_att:
            return self.attention(x)
        
        # ECA通道注意力
        x = self.channel_att(x)

        # 增强空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial_conv(torch.cat([avg_out, max_out], dim=1))
        x = x * spatial_att

        return x


class UPerNetDecoder(nn.Module):
    """增强版UPerNet解码器 - 添加ASPP和DropPath"""

    def __init__(self,
                 encoder_channels: List[int],
                 decoder_channels: int = 256,
                 num_classes: int = 1,
                 head_dropout: float = 0.1,
                 drop_path_rate: float = 0.1):
        super().__init__()

        # PPM模块（应用于最深层特征）
        self.ppm = PSPModule(encoder_channels[-1], decoder_channels)
        
        # 添加ASPP进行多尺度融合
        self.aspp = ASPPLite(encoder_channels[-1], decoder_channels, dilations=(1, 6, 12))

        # 计算合适的group数
        dec_groups = min(32, decoder_channels) if decoder_channels >= 32 else decoder_channels

        # FPN侧向连接（添加坐标注意力）
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, decoder_channels, 1, bias=False),
                nn.GroupNorm(dec_groups, decoder_channels),
                nn.ReLU(inplace=True),
                CoordinateAttention(decoder_channels, reduction=16)  # 添加坐标注意力
            )
            for ch in encoder_channels
        ])

        # FPN输出卷积（添加DropPath）
        self.fpn_convs = nn.ModuleList()
        self.drop_paths = nn.ModuleList()
        for i in range(len(encoder_channels)):
            self.fpn_convs.append(
                nn.Sequential(
                    nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1, bias=False),
                    nn.GroupNorm(dec_groups, decoder_channels),
                    nn.ReLU(inplace=True),
                    CBAM(decoder_channels, use_coord_att=False)
                )
            )
            # 逐层递增的drop path率
            dp_rate = drop_path_rate * i / max(len(encoder_channels) - 1, 1)
            self.drop_paths.append(DropPath(dp_rate) if dp_rate > 0 else nn.Identity())

        # 条形池化（针对细长裂纹）
        self.strip_pool = StripPooling(decoder_channels * len(encoder_channels), decoder_channels)
        
        # 特征精炼模块
        self.refine = nn.Sequential(
            nn.Conv2d(decoder_channels * 2, decoder_channels, 1, bias=False),  # PPM + ASPP
            nn.GroupNorm(dec_groups, decoder_channels),
            nn.ReLU(inplace=True),
            CBAM(decoder_channels, use_coord_att=True)
        )

        # 最终分类头（增强版）
        head_channels = decoder_channels // 2
        head_groups = min(16, head_channels) if head_channels >= 16 else head_channels

        self.cls_head = nn.Sequential(
            nn.Conv2d(decoder_channels, head_channels, 3, padding=1, bias=False),
            nn.GroupNorm(head_groups, head_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(head_dropout),
            nn.Conv2d(head_channels, head_channels, 3, padding=1, bias=False),  # 额外一层
            nn.GroupNorm(head_groups, head_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_channels, num_classes, 1)
        )
        
        # 初始化最后一层
        nn.init.zeros_(self.cls_head[-1].weight)
        nn.init.zeros_(self.cls_head[-1].bias)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of encoder features
        Returns:
            output: (B, num_classes, H, W)
        """
        # PPM和ASPP处理最深层特征
        ppm_out = self.ppm(features[-1])
        aspp_out = self.aspp(features[-1])
        
        # 融合PPM和ASPP
        deep_feat = self.refine(torch.cat([ppm_out, aspp_out], dim=1))
        
        fpn_features = [deep_feat]

        # 构建FPN（自顶向下，带残差连接和DropPath）
        for i in range(len(features) - 2, -1, -1):
            lateral = self.lateral_convs[i](features[i])

            # 上采样并相加
            top_down = F.interpolate(
                fpn_features[0],
                size=lateral.shape[2:],
                mode='bilinear',
                align_corners=False
            )

            # 残差连接 + DropPath
            fpn_feat = lateral + self.drop_paths[i](top_down)
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
    """增强版边界检测分支 - 结合固定Sobel和可学习边界检测"""

    def __init__(self, in_channels: int, proj_channels: int = 256):
        super().__init__()

        # 投影层
        proj_groups = min(32, proj_channels) if proj_channels >= 32 else proj_channels
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, proj_channels, 1, bias=False),
            nn.GroupNorm(proj_groups, proj_channels),
            nn.ReLU(inplace=True)
        )

        # 固定Sobel算子（提供先验边界信息）
        self.sobel_x = nn.Conv2d(proj_channels, proj_channels, 3, padding=1, bias=False, groups=proj_channels)
        self.sobel_y = nn.Conv2d(proj_channels, proj_channels, 3, padding=1, bias=False, groups=proj_channels)

        sobel_x_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y_kernel = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

        with torch.no_grad():
            for i in range(proj_channels):
                self.sobel_x.weight.data[i, 0].copy_(sobel_x_kernel)
                self.sobel_y.weight.data[i, 0].copy_(sobel_y_kernel)
            self.sobel_x.weight.requires_grad = False
            self.sobel_y.weight.requires_grad = False

        # 可学习边界检测器（多尺度）
        self.learnable_edge = nn.ModuleList([
            nn.Conv2d(proj_channels, proj_channels // 4, k, padding=k//2, bias=False, groups=proj_channels // 4)
            for k in [3, 5, 7]  # 多尺度卷积
        ])
        
        # 边界特征增强（融合固定+可学习）
        edge_groups = min(32, proj_channels) if proj_channels >= 32 else proj_channels
        fuse_channels = proj_channels * 2 + proj_channels // 4 * 3  # sobel_x + sobel_y + 3个可学习

        self.edge_fuse = nn.Sequential(
            nn.Conv2d(fuse_channels, proj_channels, 1, bias=False),
            nn.GroupNorm(edge_groups, proj_channels),
            nn.ReLU(inplace=True),
            EfficientChannelAttention(proj_channels),  # 添加通道注意力
        )
        
        self.edge_head = nn.Sequential(
            nn.Conv2d(proj_channels, proj_channels // 2, 3, padding=1, bias=False),
            nn.GroupNorm(proj_channels // 2, proj_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(proj_channels // 2, 1, 1)
        )
        
        # 初始化输出层
        nn.init.zeros_(self.edge_head[-1].weight)
        nn.init.zeros_(self.edge_head[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)

        # 固定Sobel边界
        edge_x = self.sobel_x(x)
        edge_y = self.sobel_y(x)

        # 可学习多尺度边界
        learned_edges = [conv(x) for conv in self.learnable_edge]

        # 融合所有边界特征
        all_edges = torch.cat([edge_x, edge_y] + learned_edges, dim=1)
        fused = self.edge_fuse(all_edges)
        
        edge_map = self.edge_head(fused)
        return edge_map


class ConvNeXtUPerNet(nn.Module):
    """完整的ConvNeXt + UPerNet模型（带边界分支和深度监督）
    
    增强版特性:
    - DropPath随机深度正则化
    - ECA/CoordinateAttention通道注意力
    - ASPP多尺度特征融合
    - 可学习边界检测分支
    - 多尺度边界金字塔
    """

    def __init__(self,
                 encoder_name: str = 'convnext_tiny',
                 pretrained: bool = True,
                 num_classes: int = 1,
                 decoder_channels: int = 256,
                 deep_supervision: bool = True,
                 edge_branch: bool = True,
                 head_dropout: float = 0.1,
                 drop_path_rate: float = 0.1):
        super().__init__()

        self.deep_supervision = deep_supervision
        self.edge_branch = edge_branch

        # 编码器
        self.encoder = ConvNeXtEncoder(encoder_name, pretrained)
        encoder_channels = self.encoder.out_channels

        # 解码器 - 传入drop_path_rate
        self.decoder = UPerNetDecoder(
            encoder_channels, 
            decoder_channels, 
            num_classes, 
            head_dropout=head_dropout,
            drop_path_rate=drop_path_rate
        )

        # 边界分支
        if edge_branch:
            # 使用倒数第二层特征的通道数
            edge_in_channels = encoder_channels[-2]
            self.edge_head = EdgeDetectionBranch(edge_in_channels, decoder_channels)

        # 深度监督辅助头（增强稳定性）
        if deep_supervision:
            aux_channels = decoder_channels // 2
            aux_groups = min(16, aux_channels) if aux_channels >= 16 else aux_channels

            self.aux_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(ch, aux_channels, 3, padding=1, bias=False),
                    nn.GroupNorm(aux_groups, aux_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(0.1),  # 添加Dropout提高稳定性
                    nn.Conv2d(aux_channels, num_classes, 1)
                )
                for ch in encoder_channels[-3:]  # 使用后3层特征
            ])
            
            # 初始化辅助头的最后一层，使输出接近0
            for aux_head in self.aux_heads:
                nn.init.zeros_(aux_head[-1].weight)
                nn.init.zeros_(aux_head[-1].bias)

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

        # 深度监督（带数值稳定性）
        if self.deep_supervision and self.training:
            aux_outputs = []
            for i, aux_head in enumerate(self.aux_heads):
                aux_feat = features[-3 + i]
                aux_out = aux_head(aux_feat)
                aux_out = F.interpolate(aux_out, size=input_size, mode='bilinear', align_corners=False)
                # 对辅助输出进行clamp，防止极端值导致损失爆炸
                aux_out = torch.clamp(aux_out, min=-20.0, max=20.0)
                aux_outputs.append(aux_out)
            results['aux'] = aux_outputs

        return results


def create_model(model_config: dict) -> nn.Module:
    """工厂函数创建模型
    
    支持的配置参数:
    - backbone: 编码器名称 (convnext_tiny/small/base)
    - pretrained: 是否使用预训练权重
    - num_classes: 输出类别数
    - decoder_channels: 解码器通道数
    - deep_supervision: 是否使用深度监督
    - edge_branch: 是否使用边界分支
    - head_dropout: 分类头dropout率
    - drop_path_rate: 随机深度丢弃率
    """

    model_type = model_config.get('backbone', 'convnext_tiny')

    if 'convnext' in model_type:
        model = ConvNeXtUPerNet(
            encoder_name=model_type,
            pretrained=model_config.get('pretrained', True),
            num_classes=model_config.get('num_classes', 1),
            decoder_channels=model_config.get('decoder_channels', 256),
            deep_supervision=model_config.get('deep_supervision', True),
            edge_branch=model_config.get('edge_branch', True),
            head_dropout=model_config.get('head_dropout', 0.1),
            drop_path_rate=model_config.get('drop_path_rate', 0.1)
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