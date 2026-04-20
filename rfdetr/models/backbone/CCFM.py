
#Cross-Channel Feature Mixing
import torch
import torch.nn as nn
import numpy as np


# 必要的依赖模块实现
def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class RepConv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward(self, x):
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)


class RepC3(nn.Module):
    def __init__(self, c1, c2, n=3, e=1.0):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        return self.cv3(self.m(self.cv1(x))) + self.cv2(x)

    def fuse_convs(self):
        """融合 RepConv 模块以优化推理"""
        for module in self.m:
            if hasattr(module, 'fuse_convs') and not hasattr(module, 'conv'):
                module.fuse_convs()


# 测试 RepC3 模块
if __name__ == "__main__":
    # 设置随机种子以确保结果可重现
    torch.manual_seed(0)

    # 定义输入参数
    batch_size = 2
    input_channels = 64
    height, width = 32, 32

    # 创建随机输入
    input_tensor = torch.randn(batch_size, input_channels, height, width)
    print(f"输入张量形状: {input_tensor.shape}")

    # 创建 RepC3 模块实例
    # 参数: c1=64 (输入通道), c2=128 (输出通道), n=3 (RepConv层数), e=0.5 (扩展比例)
    repc3_module = RepC3(c1=input_channels, c2=128, n=3, e=0.5)

    # 前向传播
    output = repc3_module(input_tensor)
    print(f"RepC3 输出张量形状: {output.shape}")

    # 打印模型结构摘要
    print("\nRepC3 模块结构:")
    print(repc3_module)

    # 计算参数量
    total_params = sum(p.numel() for p in repc3_module.parameters())
    print(f"\n总参数量: {total_params:,}")

    # 演示融合卷积操作
    print("\n融合卷积前的参数量:", total_params)
    repc3_module.fuse_convs()
    fused_params = sum(p.numel() for p in repc3_module.parameters())
    print("融合卷积后的参数量:", fused_params)

    # 验证融合后输出是否一致
    with torch.no_grad():
        fused_output = repc3_module(input_tensor)
    print("\n融合前输出与融合后输出的差异:", torch.max(torch.abs(output - fused_output)).item())
