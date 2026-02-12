"""
CNN (卷积神经网络) 示例

核心思想：局部感受野 + 权重共享
- 卷积层：提取局部特征
- 池化层：降维，增强平移不变性
- 全连接层：分类
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """
    简单的 CNN 分类网络

    结构: Conv → ReLU → MaxPool → Conv → ReLU → MaxPool → FC
    """

    def __init__(self, num_classes=10):
        super().__init__()

        # 第一层卷积
        # 输入 [B, 1, 28, 28] (MNIST)
        # 输出 [B, 32, 26, 26]
        self.conv1 = nn.Conv2d(
            in_channels=1,      # 灰度图
            out_channels=32,    # 32 个卷积核
            kernel_size=3,      # 3x3 卷积核
            stride=1,
            padding=1           # 保持尺寸
        )

        # 第二层卷积
        # 输入 [B, 32, 13, 13] (pooling 后)
        # 输出 [B, 64, 11, 11]
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # 最大池化
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全连接层
        # 输入特征图大小: 64 * 7 * 7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        前向传播

        Args:
            x: [B, 1, 28, 28] 输入图像

        Returns:
            logits: [B, num_classes] 分类 logits
        """
        # 第一层: Conv + ReLU + Pool
        # [B, 1, 28, 28] → [B, 32, 28, 28] → [B, 32, 14, 14]
        x = self.pool(F.relu(self.conv1(x)))

        # 第二层: Conv + ReLU + Pool
        # [B, 32, 14, 14] → [B, 64, 14, 14] → [B, 64, 7, 7]
        x = self.pool(F.relu(self.conv2(x)))

        # 展平
        # [B, 64, 7, 7] → [B, 3136]
        x = x.view(x.size(0), -1)

        # 全连接层
        # [B, 3136] → [B, 128] → [B, num_classes]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class LeNet5(nn.Module):
    """
    LeNet-5: 经典的 CNN 架构 (1998)

    结构: Conv → ReLU → Pool → Conv → ReLU → Pool → FC → FC
    """

    def __init__(self, num_classes=10):
        super().__init__()

        # C1: 卷积层
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)

        # C3: 卷积层
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=2)

        # F5: 全连接层
        self.fc1 = nn.Linear(16 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # C1 + ReLU + S2 (池化)
        # [B, 1, 28, 28] → [B, 6, 14, 14]
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)

        # C3 + ReLU + S4 (池化)
        # [B, 6, 14, 14] → [B, 16, 5, 5]
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        # 展平
        # [B, 16, 5, 5] → [B, 400]
        x = x.view(x.size(0), -1)

        # F5 + F6
        # [B, 400] → [B, 120] → [B, 84] → [B, 10]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


# 测试代码
if __name__ == "__main__":
    # 测试前向传播
    model = CNN(num_classes=10)
    x = torch.randn(2, 1, 28, 28)  # batch=2

    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")

    # 统计参数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {num_params:,}")
