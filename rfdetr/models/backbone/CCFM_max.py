import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union, List, Tuple, Dict


# ===== 增强的基础组件 =====
def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, norm_type='bn'):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.norm = self._get_norm(norm_type, c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    # 多归一化支持 BatchNorm、LayerNorm、GroupNorm、InstanceNorm、RMSNorm
    # 架构兼容：LayerNorm 特别适合与 Transformer架构集成
    # 动态选择：每个模块可独立配置归一化类型
    # 性能优化：RMSNorm 提供更快收敛，适合大模型训练

    def _get_norm(self, norm_type, num_features):
        """获取不同类型的归一化层"""
        if norm_type == 'bn':
            return nn.BatchNorm2d(num_features)
        elif norm_type == 'ln':
            return LayerNorm2d(num_features)
        elif norm_type == 'gn':
            return nn.GroupNorm(num_groups=32, num_channels=num_features)
        elif norm_type == 'in':
            return nn.InstanceNorm2d(num_features)
        elif norm_type == 'rms':
            return RMSNorm2d(num_features)
        else:
            return nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class LayerNorm2d(nn.Module):
    """LayerNorm for channels of 2D spatial inputs"""

    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class RMSNorm2d(nn.Module):
    """Root Mean Square Layer Normalization for 2D inputs"""

    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(num_channels))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=1, keepdim=True) + self.eps)
        x = x / rms
        return self.scale[:, None, None] * x


class EnhancedCBAM(nn.Module):
    """增强型CBAM注意力模块"""

    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        # 通道注意力
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction_ratio, 1),
            nn.SiLU(),
            nn.Conv2d(channels // reduction_ratio, channels, 1),
            nn.Sigmoid()
        )

        # 空间注意力
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        # 深度可分离卷积增强
        self.enhance = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )

    def forward(self, x):
        # 通道注意力
        ca = self.channel_att(x)
        x = x * ca

        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_att(sa)
        x = x * sa

        # 特征增强
        x = x + self.enhance(x)
        return x


# ===== 增强版 RepConv 实现 =====
class AdvancedRepConv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True,
                 norm_type='bn', use_identity=True, attention=False):
        super().__init__()
        assert k == 3 and p == 1, "只支持3x3卷积核和p=1的配置"
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.norm_type = norm_type
        self.use_identity = use_identity
        self.deploy = False
        self.attention = attention and c1 == c2  # 仅当c1==c2时应用注意力

        # 激活函数
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        # 身份分支 - 只有在通道数匹配且步长为1时才启用
        if use_identity and c2 == c1 and s == 1:
            if norm_type == 'bn':
                self.bn_identity = nn.BatchNorm2d(num_features=c1)
            elif norm_type == 'ln':
                self.bn_identity = LayerNorm2d(c1)
            else:
                self.bn_identity = nn.Identity()
        else:
            self.bn_identity = None

        # 3x3卷积分支
        self.conv3x3 = Conv(c1, c2, k, s, p=p, g=g, act=False, norm_type=norm_type)

        # 1x1卷积分支
        self.conv1x1 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False, norm_type=norm_type)

        # 5x5等效分支（通过两个3x3实现）
        self.conv5x5_1 = Conv(c1, c2, 3, 1, p=1, g=g, act=False, norm_type=norm_type)
        self.conv5x5_2 = Conv(c2, c2, 3, s, p=1, g=g, act=False, norm_type=norm_type)

        # 注意力模块
        if self.attention:
            self.attn = EnhancedCBAM(c2)
        else:
            self.attn = nn.Identity()

    def forward(self, x):
        if self.deploy and hasattr(self, 'conv_fused'):
            return self.act(self.attn(self.conv_fused(x)))

        # 身份分支
        identity_out = self.bn_identity(x) if self.bn_identity is not None else 0

        # 3x3 + 1x1分支
        conv_out = self.conv3x3(x) + self.conv1x1(x)

        # 5x5等效分支
        conv5x5_out = self.conv5x5_2(self.conv5x5_1(x))

        # 融合所有分支
        out = conv_out + conv5x5_out + identity_out

        # 注意力增强
        out = self.attn(out)

        return self.act(out)

    def _fuse_bn_tensor(self, branch):
        """融合卷积和BN层"""
        if branch is None:
            return 0, 0

        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.norm.running_mean
            running_var = branch.norm.running_var
            gamma = branch.norm.weight
            beta = branch.norm.bias
            eps = branch.norm.eps

            std = (running_var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            return kernel * t, beta - running_mean * gamma / std

        elif isinstance(branch.norm, (nn.BatchNorm2d, LayerNorm2d)) and hasattr(self, 'id_tensor'):
            kernel = self.id_tensor
            running_mean = branch.norm.running_mean
            running_var = branch.norm.running_var
            gamma = branch.norm.weight
            beta = branch.norm.bias
            eps = branch.norm.eps

            std = (running_var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            return kernel * t, beta - running_mean * gamma / std

        return 0, 0

    def _pad_tensor(self, kernel, target_size):
        """将卷积核填充到目标大小"""
        pad_size = (target_size - kernel.shape[2]) // 2
        if pad_size <= 0:
            return kernel
        return nn.functional.pad(kernel, (pad_size, pad_size, pad_size, pad_size))

    def fuse_convs(self):
        """融合所有卷积分支为单个卷积层"""
        if self.deploy:
            return

        # 获取各分支的等效卷积核和偏置
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv3x3)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv1x1)
        kernel5x5_1, bias5x5_1 = self._fuse_bn_tensor(self.conv5x5_1)
        kernel5x5_2, bias5x5_2 = self._fuse_bn_tensor(self.conv5x5_2)

        # 处理身份分支
        if self.bn_identity is not None and hasattr(self, 'id_tensor'):
            kernel_id, bias_id = self._fuse_bn_tensor(nn.Module())
            kernel_id, bias_id = self._fuse_bn_tensor(self)
        else:
            kernel_id, bias_id = 0, 0

        # 融合5x5分支（两个3x3卷积的等效）
        kernel5x5 = nn.functional.conv2d(
            self._pad_tensor(kernel5x5_2, 5),
            kernel5x5_1.permute(1, 0, 2, 3),
            padding=2
        )
        bias5x5 = bias5x5_2 + (bias5x5_1 * kernel5x5_2.sum(dim=(1, 2, 3))).sum()

        # 填充1x1卷积核到3x3
        kernel1x1_padded = self._pad_tensor(kernel1x1, 3)

        # 合并所有卷积核
        fused_kernel = kernel3x3 + kernel1x1_padded + self._pad_tensor(kernel5x5, 3) + kernel_id
        fused_bias = bias3x3 + bias1x1 + bias5x5 + bias_id

        # 创建融合卷积层
        self.conv_fused = nn.Conv2d(
            self.c1, self.c2, 3, stride=self.conv3x3.conv.stride,
            padding=1, groups=self.g, bias=True
        )

        self.conv_fused.weight.data = fused_kernel
        self.conv_fused.bias.data = fused_bias

        # 标记为已部署
        self.deploy = True

        # 清理未使用的参数
        for para in self.parameters():
            para.detach_()

        return self


# ===== 核心：增强版 RepC3 模块 =====
class AdvancedRepC3(nn.Module):
    """
    高性能增强版 RepC3 模块，专为最大化精度设计
    """

    def __init__(
            self,
            c1,
            c2,
            n=3,
            e=1.0,
            e_dynamic=True,
            norm_type='bn',
            attention=True,
            multi_scale=True,
            residual_strength=1.0,
            activation='silu'
    ):
        """
        初始化增强版 RepC3
        Args:
            c1 (int): 输入通道数
            c2 (int): 输出通道数
            n (int): RepConv 层数量
            e (float): 扩展比例基础值
            e_dynamic (bool): 是否使用动态扩展比例
            norm_type (str): 归一化类型 'bn', 'ln', 'gn', 'in', 'rms'
            attention (bool): 是否使用注意力机制
            multi_scale (bool): 是否启用多尺度特征融合
            residual_strength (float): 残差连接强度
            activation (str): 激活函数类型
        """
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.n = n
        self.e = e
        self.e_dynamic = e_dynamic
        self.norm_type = norm_type
        self.attention = attention
        self.multi_scale = multi_scale
        self.residual_strength = residual_strength
        self.deploy = False

        # 动态扩展比例 - 中间层使用更大扩展比
        if self.e_dynamic:
            base_e = e
            self.e_values = [base_e * (1 + 0.5 * i / (max(1, n - 1))) for i in range(n)]
            c_ = int(c2 * max(self.e_values))
        else:
            self.e_values = [e] * n
            c_ = int(c2 * e)

        # 主要特征处理路径
        self.cv1 = Conv(c1, c_, 1, 1, norm_type=norm_type)

        # 残差路径 - 改进为自适应残差连接
        if c1 != c2:
            self.residual_path = Conv(c1, c2, 1, 1, norm_type=norm_type)
        else:
            if norm_type == 'bn':
                self.residual_path = nn.BatchNorm2d(c1)
            elif norm_type == 'ln':
                self.residual_path = LayerNorm2d(c1)
            else:
                self.residual_path = nn.Identity()

        # 中间处理模块 - 使用不同扩展比例
        modules = []
        curr_channels = c_
        for i in range(n):
            next_channels = int(c_ * (self.e_values[i] / e))
            modules.append(AdvancedRepConv(
                curr_channels, next_channels,
                norm_type=norm_type,
                attention=self.attention,
                use_identity=True
            ))
            curr_channels = next_channels

        self.m = nn.Sequential(*modules)

        # 输出投影层
        self.cv3 = Conv(curr_channels, c2, 1, 1, norm_type=norm_type) if curr_channels != c2 else nn.Identity()

        # 多尺度特征融合
        if self.multi_scale:
            self.scale_fusion = nn.ModuleList([
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(output_size),
                    nn.Conv2d(c2, c2 // 4, 1),
                    nn.Upsample(size=None, scale_factor=1, mode='nearest')
                ) for output_size in [4, 8, 16]
            ])
            self.scale_combine = Conv(c2 + 3 * (c2 // 4), c2, 1, 1, norm_type=norm_type)

        # 最终注意力模块
        if self.attention:
            self.final_attn = EnhancedCBAM(c2)

        # 自适应激活函数
        if activation == 'silu':
            self.final_act = nn.SiLU()
        elif activation == 'gelu':
            self.final_act = nn.GELU()
        elif activation == 'relu':
            self.final_act = nn.ReLU()
        else:
            self.final_act = nn.Identity()

    def forward(self, x):
        """前向传播"""
        # 主要特征处理路径
        y = self.cv1(x)
        y = self.m(y)
        y = self.cv3(y)

        # 残差连接
        residual = self.residual_path(x)
        if residual.shape[1:] != y.shape[1:]:
            # 通道数不匹配时的调整
            if residual.shape[1] < y.shape[1]:
                pad = torch.zeros_like(y)
                pad[:, :residual.shape[1]] = residual
                residual = pad
            else:
                residual = residual[:, :y.shape[1]]

        # 自适应残差强度
        out = y + residual * self.residual_strength

        # 多尺度特征融合
        if self.multi_scale and not self.deploy:
            scales = [out]
            for sf in self.scale_fusion:
                scale_feat = sf(out)
                # 确保与输出特征图大小匹配
                if scale_feat.shape[2:] != out.shape[2:]:
                    scale_feat = nn.functional.interpolate(scale_feat, size=out.shape[2:], mode='nearest')
                scales.append(scale_feat)
            out = torch.cat(scales, dim=1)
            out = self.scale_combine(out)

        # 最终注意力
        if self.attention and not self.deploy:
            out = self.final_attn(out)

        # 最终激活
        out = self.final_act(out)
        return out

    def fuse(self):
        """融合模块以优化推理"""
        self.deploy = True

        # 融合所有AdvancedRepConv模块
        for module in self.m:
            if hasattr(module, 'fuse_convs'):
                module.fuse_convs()

        # 禁用多尺度处理和注意力（部署时）
        if hasattr(self, 'scale_fusion'):
            del self.scale_fusion
            del self.scale_combine
        if hasattr(self, 'final_attn'):
            del self.final_attn

        return self


# ===== 实用函数和测试代码 =====
def get_model_complexity(model, input_shape=(3, 224, 224)):
    """计算模型复杂度"""
    from thop import profile
    input = torch.randn(1, *input_shape)
    flops, params = profile(model, inputs=(input,), verbose=False)
    return flops, params


if __name__ == "__main__":

    # 测试配置
    input_channels = 64
    output_channels = 128
    input_size = (4, input_channels, 32, 32)  # batch, channels, height, width

    # 创建输入
    x = torch.randn(*input_size)

    print("AdvancedRepC3 高性能模块测试")

    # 测试不同配置
    configs = [
        {
            'name': '基础配置 (BN)',
            'norm_type': 'bn',
            'e': 0.5,
            'e_dynamic': False,
            'attention': False,
            'multi_scale': False
        },
        {
            'name': '标准增强 (LN)',
            'norm_type': 'ln',
            'e': 0.75,
            'e_dynamic': True,
            'attention': True,
            'multi_scale': False
        },
        {
            'name': '最大性能配置',
            'norm_type': 'ln',
            'e': 1.0,
            'e_dynamic': True,
            'attention': True,
            'multi_scale': True
        }
    ]

    results = []

    for config in configs:
        print(f"\n{'-' * 40}")
        print(f"测试: {config['name']}")
        print(f"配置: {config}")

        # 创建模型
        model = AdvancedRepC3(
            c1=input_channels,
            c2=output_channels,
            n=3,
            e=config['e'],
            e_dynamic=config['e_dynamic'],
            norm_type=config['norm_type'],
            attention=config['attention'],
            multi_scale=config['multi_scale']
        )

        # 前向传播
        with torch.no_grad():
            output = model(x)

        # 模型复杂度
        flops, params = get_model_complexity(model, (input_channels, 32, 32))

    # 结果对比
    print("\n" + "=" * 80)
    print("结果对比")
    print("=" * 80)
    for result in results:
        print(f"\n{result['config']}:")
        print(f"  输出形状: {result['output_stats']['shape']}")
        print(f"  均值/标准差: {result['output_stats']['mean']:.4f}/{result['output_stats']['std']:.4f}")
        print(f"  复杂度: {result['flops'] / 1e6:.2f}M FLOPs, {result['params'] / 1e6:.2f}M 参数")


    # 基础场景：使用标准增强配置（LayerNorm + 动态扩展 + 基础注意力）
    #model = AdvancedRepC3(c1, c2, n=3, e=0.75, norm_type='ln', e_dynamic=True, attention=True)
    # 高精度场景：启用全部增强特性
    #model = AdvancedRepC3(c1, c2, n=4, e=1.0, norm_type='ln', e_dynamic=True, attention=True, multi_scale=True)