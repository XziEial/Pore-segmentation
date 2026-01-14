import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, resnet50


# 双卷积块，添加残差连接
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 残差连接适配：如果输入输出通道一致，直接残差相加；否则用 1x1 卷积调整通道
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        residual = x  # 保存输入，用于残差连接
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        
        # 残差连接：shortcut 调整通道后与卷积结果相加
        residual = self.shortcut(residual)
        x += residual  
        x = self.relu(x)
        return x


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv with attention gate"""
    def __init__(self, in_channels, skip_channels, out_channels, bilinear=True):
        super().__init__()
        # 如果使用双线性插值上采样，则使用1x1卷积减少通道数
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels + skip_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)
            
        # 注意力门控机制
        self.attention_gate = AttentionGate(skip_channels, in_channels if bilinear else in_channels // 2, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 处理尺寸不匹配问题
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        # 应用注意力门控
        x2_attended = self.attention_gate(x2, x1)
        
        x = torch.cat([x2_attended, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """1x1卷积，输出最终分割图"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class AttentionGate(nn.Module):
    """注意力门控机制，增强特征融合"""
    def __init__(self, g_channels, x_channels, inter_channels):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(g_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(x_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class MultiScaleFeatureExtractor(nn.Module):
    """多尺度特征提取模块，使用不同膨胀率的空洞卷积"""
    def __init__(self, in_channels, out_channels):
        super(MultiScaleFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3, dilation=3)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=5, dilation=5)
        self.bn = nn.BatchNorm2d(out_channels * 3)
        self.relu = nn.ReLU(inplace=True)
        self.conv_out = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        out = torch.cat([x1, x3, x5], dim=1)
        out = self.bn(out)
        out = self.relu(out)
        out = self.conv_out(out)
        return out


class RMA_UNet(nn.Module):
    """针对固化土孔隙分割优化的U-Net模型"""
    def __init__(self, n_channels=3, n_classes=1, bilinear=True, pretrained_encoder=False):
        super(SoilPorosityUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1
        
        # 编码器部分
        if pretrained_encoder:
            # 使用预训练的ResNet作为编码器
            resnet = resnet34(pretrained=True)
            self.inc = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu
            )
            self.down1 = nn.Sequential(
                resnet.maxpool,
                resnet.layer1
            )
            self.down2 = resnet.layer2
            self.down3 = resnet.layer3
            self.down4 = resnet.layer4
            
            # 调整通道数以匹配原始U-Net设计
            encoder_channels = [64, 64, 128, 256, 512]
        else:
            # 标准U-Net编码器
            self.inc = DoubleConv(n_channels, 64)
            self.down1 = Down(64, 128)
            self.down2 = Down(128, 256)
            self.down3 = Down(256, 512)
            self.down4 = Down(512, 1024 // factor)
            
            encoder_channels = [64, 128, 256, 512, 1024 // factor]
        
        # 多尺度特征提取器
        self.multi_scale = MultiScaleFeatureExtractor(encoder_channels[4], encoder_channels[4])
        
        # 解码器部分
        self.up1 = Up(encoder_channels[4], encoder_channels[3], 512 // factor, bilinear)
        self.up2 = Up(512 // factor, encoder_channels[2], 256 // factor, bilinear)
        self.up3 = Up(256 // factor, encoder_channels[1], 128 // factor, bilinear)
        self.up4 = Up(128 // factor, encoder_channels[0], 64, bilinear)
        
        
        # 输出
        self.outc = OutConv(64, n_classes)
 

    def forward(self, x):
        # 编码器前向传播
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # 多尺度特征融合
        x5_ms = self.multi_scale(x5)
        
        # 解码器前向传播
        x = self.up1(x5_ms, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # 输出
        logits = self.outc(x)
        
        return logits
