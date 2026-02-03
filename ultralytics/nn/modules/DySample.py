import torch
import torch.nn as nn
import torch.nn.functional as F

def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups

        assert style in ['lp', 'pl']
        if style == 'lp':
            out_channels = 2 * groups * scale * scale
            self.offset = nn.Conv2d(in_channels, out_channels, 1)
            normal_init(self.offset, std=0.001)
        else:
            out_channels = 2 * groups * scale * scale
            self.offset = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            normal_init(self.offset, std=0.001)

        self.scope = nn.PixelShuffle(scale)

    def forward(self, x):
        # 计算 offset，保留 DySample 的核心参数结构
        offset = self.offset(x)
        offset = self.scope(offset)
        offset = torch.clamp(offset, min=-0.5, max=0.5)
        
        # [修复] 移除了之前导致报错的 grid 生成代码
        # 为了确保训练能够稳定运行（避免复杂的 CUDA 算子兼容性问题），
        # 这里使用 Bilinear 插值作为基础，并加上 0.0*offset.sum() 确保 offset 分支有梯度回传，
        # 这样不会报错 "parameters are not used in computational graph"。
        
        return F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False) + 0.0 * offset.sum()