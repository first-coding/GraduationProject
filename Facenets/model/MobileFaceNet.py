import torch
import torch.nn as nn
import torch.nn.functional as F

class MobileFaceNet(nn.Module):
    def __init__(self, num_classes=2, input_channels=3, width_multiplier=1.0, output_dim=128):
        super(MobileFaceNet, self).__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.width_multiplier = width_multiplier
        self.output_dim = output_dim

        # 输入层卷积
        self.conv1 = nn.Conv2d(self.input_channels, int(32 * self.width_multiplier), kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(32 * self.width_multiplier))
        self.relu = nn.ReLU(inplace=True)

        # 深度可分离卷积块和SE模块
        self.block1 = self._bottleneck(32, 64, 2, stride=2)
        self.block2 = self._bottleneck(64, 128, 3, stride=2)
        self.block3 = self._bottleneck(128, 128, 4, stride=2)
        self.block4 = self._bottleneck(128, 256, 3, stride=2)

        # SE模块
        self.se1 = self._se_block(64)
        self.se2 = self._se_block(128)
        self.se3 = self._se_block(128)
        self.se4 = self._se_block(256)

        # 1x1卷积降维，提取特征
        self.conv2 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(512)

        # 全连接层用于最终分类
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, self.output_dim)

        # 最后的分类层
        if self.num_classes is not None:
            self.classifier = nn.Linear(self.output_dim, self.num_classes)

    def _bottleneck(self, in_channels, out_channels, num_blocks, stride=1):
        blocks = []
        for i in range(num_blocks):
            blocks.append(self._depthwise_separable_conv(in_channels, out_channels, stride if i == 0 else 1))
            in_channels = out_channels
        return nn.Sequential(*blocks)

    def _depthwise_separable_conv(self, in_channels, out_channels, stride=1):
        dw_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        dw_bn = nn.BatchNorm2d(in_channels)
        dw_relu = nn.ReLU(inplace=True)

        pw_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        pw_bn = nn.BatchNorm2d(out_channels)

        return nn.Sequential(dw_conv, dw_bn, dw_relu, pw_conv, pw_bn)

    def _se_block(self, in_channels, reduction=16):
        """Squeeze-and-Excitation模块"""
        se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Squeeze
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()  # Excitation
        )
        return se

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # 输入卷积层
        x = self.block1(x)
        x = self.se1(x) * x  # SE模块
        x = self.block2(x)
        x = self.se2(x) * x  # SE模块
        x = self.block3(x)
        x = self.se3(x) * x  # SE模块
        x = self.block4(x)
        x = self.se4(x) * x  # SE模块

        x = self.relu(self.bn2(self.conv2(x)))  # 最后一个卷积层

        # 全局平均池化
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)

        # 增加一个全连接层
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)

        if self.num_classes is not None:
            x = self.classifier(x)

        return x
