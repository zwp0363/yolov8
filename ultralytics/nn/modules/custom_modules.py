# ultralytics/nn/modules/custom_modules.py
# 自定义模块：无额外编译依赖，基于PyTorch原生算子，轻量化，部署友好
import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv  # 修改导入方式，直接从conv.py导入Conv类
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.tal import dist2bbox, make_anchors

# 1. LSKA（Large Selective Kernel Attention）：修正通道数，自适应输入通道
class LSKA(nn.Module):
    def __init__(self, channels, kernel_sizes=[3, 5, 7, 9]):
        super().__init__()
        self.channels = channels  # yaml传入的通道数（缩放后）
        self.kernel_sizes = kernel_sizes
        # 核心修正：卷积层输入通道 = 传入的channels（缩放后，YOLOv8n中为128），不再是512
        # 原错误：channels=512，导致self.reduce输入通道=512，实际输入=128
        self.reduce = nn.Conv2d(channels, channels // 4, 1, bias=False)  # 输入channels（缩放后），输出channels//4
        self.upsample = nn.Conv2d(channels // 4, channels, 1, bias=False)  # 输出通道与输入通道一致
        self.convs = nn.ModuleList([
            nn.Conv2d(channels // 4, channels // 4, k, padding=k//2, groups=channels//4, bias=False)
            for k in kernel_sizes
        ])
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels // 4, channels // 4, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 验证输入通道数与模块通道数是否一致（可选，用于排障）
        assert x.shape[1] == self.channels, f"LSKA输入通道数{x.shape[1]}与模块通道数{self.channels}不匹配"
        x_reduce = self.reduce(x)
        conv_outs = []
        for conv in self.convs:
            conv_outs.append(conv(x_reduce))
        feats = torch.stack(conv_outs, dim=1).mean(dim=1)
        attn_weights = self.attn(feats)
        feats = feats * attn_weights
        return self.upsample(feats) + x

# 2. VoVGSCSP_WFU：修正卷积层输入通道，匹配传入的 channels（缩放后）
class VoVGSCSP_WFU(nn.Module):
    def __init__(self, channels, shortcut=True, num_blocks=2):
        super().__init__()
        self.channels = channels
        self.shortcut = shortcut
        self.num_blocks = num_blocks
        # 核心修正：输入通道 = channels * 2（拼接后），输出 = channels（缩放后）
        self.conv1 = nn.Conv2d(channels * 2, channels, 1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU()
        self.wfu_high = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False, padding_mode='reflect')
        self.wfu_low = nn.Conv2d(channels, channels, 1, bias=False)
        self.wfu_fuse = nn.Conv2d(channels * 2, channels, 1, bias=False)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.act(self.bn(self.conv2(x1)))
        high_freq = self.wfu_high(x2) - x2
        low_freq = self.wfu_low(x2)
        wfu_feat = self.wfu_fuse(torch.cat([high_freq, low_freq], dim=1))
        if self.shortcut:
            return wfu_feat + x1
        return wfu_feat

# 3. BIFPN_SDI：BIFPN + SDI（语义细节注入），明确输出通道数
class BIFPN_SDI(nn.Module):
    def __init__(self, channels, fusion_mode='SDI'):
        super().__init__()
        self.channels = channels  # 明确输出通道数（yaml中传入的参数）
        self.fusion_mode = fusion_mode
        self.w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.semantic_conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.detail_conv = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU()

    def forward(self, x):
        x1, x2 = x[0], x[1]
        w1 = F.softmax(self.w1, dim=0)
        w2 = F.softmax(self.w2, dim=0)
        bifpn_feat = (w1[0] * x1 + w1[1] * x2) / (w1.sum() + 1e-8)
        if self.fusion_mode == 'SDI':
            semantic_feat = self.semantic_conv(x2)
            detail_feat = self.detail_conv(x1)
            sdi_feat = semantic_feat + detail_feat
            bifpn_feat = self.act(self.bn(bifpn_feat + sdi_feat))
        return bifpn_feat

# 4. C2f_DEConv：修正卷积层输入通道，匹配传入的 channels（缩放后）
class C2f_DEConv(nn.Module):
    def __init__(self, channels, shortcut=True, num_blocks=2):
        super().__init__()
        self.channels = channels
        self.shortcut = shortcut
        self.num_blocks = num_blocks
        # 核心修正：输入通道 = channels * 4（拼接后），输出 = channels（缩放后）
        self.conv1 = nn.Conv2d(channels * 4, channels, 1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.deconv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )
        self.merged = False

    def merge(self):
        if not self.merged:
            self.deconv = nn.Conv2d(self.channels, self.channels, 3, padding=1, bias=False)
            self.merged = True

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        deconv_feat = self.deconv(x2)
        if self.shortcut:
            return deconv_feat + x1
        return deconv_feat

# 5. LSCD：轻量化共享卷积检测头
class LSCD(nn.Module):
    """
    Lightweight and Efficient Single-Stage Head with Large Kernel Convolutions and Decoupled Detection
    轻量级、高效的单阶段检测头，结合大核卷积与解耦检测
    """
    def __init__(self, nc=80, channels=128, kernel_size=3, stride=1, padding=1, reg_max=16):
        """
        初始化 LSCD 模块
        Args:
            nc: 类别数量
            channels: 输入通道数（会被自动覆盖为实际输入通道数）
            kernel_size: 卷积核大小
            stride: 步长
            padding: 填充
            reg_max: 回归参数的最大范围，用于DFL计算
        """
        super(LSCD, self).__init__()
        # 注意：channels参数会在forward方法中被实际输入通道数覆盖
        self.nc = nc
        self.reg_max = reg_max  # 添加reg_max属性，用于损失计算
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # 初始化时使用一个占位通道数，forward时会替换
        self.channels = channels
        self.shared_conv = Conv(channels, channels, kernel_size, stride, padding)
        self.cls_conv = Conv(channels, nc, 1, 1, 0)
        self.reg_conv = Conv(channels, 4, 1, 1, 0)
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入特征图，形状为 [B, C, H, W]
        Returns:
            cls_out: 分类输出，形状为 [B, nc, H, W]
            reg_out: 回归输出，形状为 [B, 4, H, W]
        """
        # 动态适应输入通道数
        if x.size(1) != self.channels:
            self.channels = x.size(1)
            # 重新创建卷积层，使用实际输入通道数
            self.shared_conv = Conv(self.channels, self.channels, self.kernel_size, self.stride, self.padding).to(x.device)
            self.cls_conv = Conv(self.channels, self.nc, 1, 1, 0).to(x.device)
            self.reg_conv = Conv(self.channels, 4, 1, 1, 0).to(x.device)
        
        # 前向传播
        x = self.shared_conv(x)
        cls_out = self.cls_conv(x)
        reg_out = self.reg_conv(x)
        return torch.cat([reg_out, cls_out], dim=1)

class LSCD_Detect(Detect):
    """
    LSCD_Detect: 适配 YOLOv8 训练框架的 LSCD 检测头
    修复了 AttributeError: 'LSCD_Detect' object has no attribute 'cv2'
    """
    def __init__(self, nc=80, ch=()):
        super().__init__(nc, ch)
        # 删除父类生成的标准卷积，避免参数冗余
        del self.cv2, self.cv3

        # 1. 必须使用 cv2 (回归) 和 cv3 (分类) 命名，以通过 bias_init 检查
        # 2. 必须封装为 nn.Sequential，因为 bias_init 会调用 layer[-1]
        self.shared_convs = nn.ModuleList()
        self.cv2 = nn.ModuleList()
        self.cv3 = nn.ModuleList()

        # DFL 回归通道数 (16 * 4 = 64)
        c_out_reg = self.reg_max * 4

        for c in ch:
            # 共享大核卷积
            self.shared_convs.append(Conv(c, c, k=3, s=1, p=1))
            
            # 回归分支 (Box): 封装进 Sequential
            self.cv2.append(nn.Sequential(
                nn.Conv2d(c, c_out_reg, 1)
            ))
            
            # 分类分支 (Cls): 封装进 Sequential
            self.cv3.append(nn.Sequential(
                nn.Conv2d(c, self.nc, 1)
            ))

    def forward(self, x):
        """
        前向传播：Shared Conv -> Split(Reg, Cls) -> Concat
        注意：不能调用 super().forward(x)，因为结构不同会导致维度报错
        """
        shape = x[0].shape
        for i in range(self.nl):
            # 1. 先过共享卷积
            feat = self.shared_convs[i](x[i])
            
            # 2. 再过解耦头 (使用 cv2 和 cv3)
            reg_out = self.cv2[i](feat)
            cls_out = self.cv3[i](feat)
            
            # 3. 拼接输出
            x[i] = torch.cat((reg_out, cls_out), 1)

        # 训练模式：直接返回拼接后的特征图
        if self.training:
            return x

        # 推理模式：手动实现解码逻辑 (复制自 Detect 类，适配 LSCD)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        
        # DFL 计算与解码
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        
        return y if self.export else (y, x)