import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C2f

class RFAConv(nn.Module):
    """RFAConv: Receptive-Field Attention Convolution (Fixed Dimension Logic)"""
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1):
        super().__init__()
        # 自动处理传入 kernel_size 为 tuple 的情况 (修复之前的 TypeError)
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
            
        self.kernel_size = kernel_size
        
        # 1. 权重生成分支 (Attention Branch)
        # [核心修复] 使用 AdaptiveAvgPool2d(1) 获取全局上下文，并使用 Group Conv 为每个通道生成独立的权重
        self.get_weight = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channel, in_channel * (kernel_size * kernel_size), kernel_size=1, groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel * (kernel_size * kernel_size)),
            nn.ReLU(inplace=True)
        )
        
        # 2. 特征生成分支 (Feature Generation Branch)
        # 通过 Group Conv 模拟 unfold 操作，提取空间特征
        self.generate_feature = nn.Sequential(
            nn.Conv2d(in_channel, in_channel * (kernel_size * kernel_size), kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel * (kernel_size * kernel_size)),
            nn.ReLU(inplace=True)
        )
        
        # 3. 最终投影
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1)

    def forward(self, x):
        b, c = x.shape[0:2]
        
        # --- 计算注意力权重 ---
        weight = self.get_weight(x) # 输出: (B, C*9, 1, 1)
        # 变换维度以便进行 Softmax: (B, C, 9)
        weight = weight.view(b, c, self.kernel_size * self.kernel_size)
        weight = weight.softmax(2)
        # 再次变换维度以便与特征相乘: (B, C, 9, 1, 1)
        weight = weight.view(b, c, self.kernel_size * self.kernel_size, 1, 1)
        
        # --- 计算展开的特征 ---
        feature = self.generate_feature(x) # 输出: (B, C*9, H, W)
        # 变换维度: (B, C, 9, H, W)
        feature = feature.view(b, c, self.kernel_size * self.kernel_size, feature.shape[2], feature.shape[3])
        
        # --- 加权求和 (Receptive Field Attention) ---
        # (B, C, 9, H, W) * (B, C, 9, 1, 1) -> sum(dim=2) -> (B, C, H, W)
        weighted_feature = (feature * weight).sum(2)
        
        # --- 最终输出 ---
        return self.conv(weighted_feature)

class Bottleneck_RFA(nn.Module):
    """Standard Bottleneck with RFAConv"""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        # 处理 k 可能为嵌套 tuple 的情况，增强鲁棒性
        k1 = k[0] if isinstance(k[0], int) else k[0][0]
        k2 = k[1] if isinstance(k[1], int) else k[1][0]
        
        self.cv1 = Conv(c1, c_, k1, 1)
        self.cv2 = RFAConv(c_, c2, k2, 1) 
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f_RFAConv(C2f):
    """Inherit from C2f but use RFAConv in the Bottleneck"""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        # 使用修正后的 Bottleneck_RFA
        self.m = nn.ModuleList(Bottleneck_RFA(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))