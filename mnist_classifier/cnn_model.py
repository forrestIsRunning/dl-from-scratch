"""
MNIST CNN 分类器

网络结构:
- Conv2d Layer × 3 + ReLU × 3 + MaxPooling × 3
- Linear Layer × 3

支持多种配置变体进行比较
"""
import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class CNNConfig:
    """CNN 配置参数"""
    name: str
    conv_channels: List[int]       # 各卷积层的输出通道数
    kernel_sizes: List[int]        # 各卷积层的卷积核大小
    pool_sizes: List[int]          # 各池化层的池化大小
    fc_hidden: List[int]           # 全连接层隐藏单元数
    use_batchnorm: bool = False    # 是否使用 BatchNorm
    use_dropout: bool = False      # 是否使用 Dropout
    dropout_rate: float = 0.25     # Dropout 比率


# 预定义的配置变体
CONFIGS = {
    # 基础配置: 小通道数
    "small": CNNConfig(
        name="Small",
        conv_channels=[16, 32, 64],
        kernel_sizes=[3, 3, 3],
        pool_sizes=[2, 2, 2],
        fc_hidden=[128, 64],
        use_batchnorm=False,
        use_dropout=False,
    ),

    # 中等配置: 更多通道
    "medium": CNNConfig(
        name="Medium",
        conv_channels=[32, 64, 128],
        kernel_sizes=[3, 3, 3],
        pool_sizes=[2, 2, 2],
        fc_hidden=[256, 128],
        use_batchnorm=False,
        use_dropout=False,
    ),

    # 大配置: 更多通道和更大全连接层
    "large": CNNConfig(
        name="Large",
        conv_channels=[64, 128, 256],
        kernel_sizes=[3, 3, 3],
        pool_sizes=[2, 2, 2],
        fc_hidden=[512, 256],
        use_batchnorm=False,
        use_dropout=False,
    ),

    # 带 BatchNorm
    "medium_bn": CNNConfig(
        name="Medium + BatchNorm",
        conv_channels=[32, 64, 128],
        kernel_sizes=[3, 3, 3],
        pool_sizes=[2, 2, 2],
        fc_hidden=[256, 128],
        use_batchnorm=True,
        use_dropout=False,
    ),

    # 带 Dropout
    "medium_dropout": CNNConfig(
        name="Medium + Dropout",
        conv_channels=[32, 64, 128],
        kernel_sizes=[3, 3, 3],
        pool_sizes=[2, 2, 2],
        fc_hidden=[256, 128],
        use_batchnorm=False,
        use_dropout=True,
        dropout_rate=0.25,
    ),

    # BatchNorm + Dropout
    "medium_bn_dropout": CNNConfig(
        name="Medium + BN + Dropout",
        conv_channels=[32, 64, 128],
        kernel_sizes=[3, 3, 3],
        pool_sizes=[2, 2, 2],
        fc_hidden=[256, 128],
        use_batchnorm=True,
        use_dropout=True,
        dropout_rate=0.25,
    ),

    # 大卷积核
    "large_kernel": CNNConfig(
        name="Large Kernel (5x5)",
        conv_channels=[32, 64, 128],
        kernel_sizes=[5, 5, 5],
        pool_sizes=[2, 2, 2],
        fc_hidden=[256, 128],
        use_batchnorm=False,
        use_dropout=False,
    ),

    # 更深的全连接层
    "deeper_fc": CNNConfig(
        name="Deeper FC",
        conv_channels=[32, 64, 128],
        kernel_sizes=[3, 3, 3],
        pool_sizes=[2, 2, 2],
        fc_hidden=[512, 256, 128],
        use_batchnorm=False,
        use_dropout=False,
    ),
}


class MNISTCNN(nn.Module):
    """
    MNIST CNN 分类器

    结构:
    - 3个卷积块: Conv2d → [BatchNorm] → ReLU → MaxPool → [Dropout]
    - 3个全连接层: Flatten → Linear → ReLU → ... → Linear (输出)

    输入: [N, 1, 28, 28]
    输出: [N, 10] (logits)
    """

    def __init__(self, config: CNNConfig):
        super(MNISTCNN, self).__init__()
        self.config = config

        # 构建卷积层
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList() if config.use_batchnorm else None

        in_channels = 1  # MNIST 是单通道灰度图
        for i, (out_channels, kernel_size, pool_size) in enumerate(
            zip(config.conv_channels, config.kernel_sizes, config.pool_sizes)
        ):
            # 卷积层
            conv = nn.Conv2d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2  # 保持空间维度
            )
            self.conv_layers.append(conv)

            # BatchNorm
            if config.use_batchnorm:
                self.bn_layers.append(nn.BatchNorm2d(out_channels))

            in_channels = out_channels

        # 计算卷积后的特征图大小
        # 28x28 → 3次 2x2 pooling → 3x3 (28 / 2 / 2 / 2 = 3.5 → 3)
        # 实际计算: floor((28 - kernel + 2*padding) / 1 + 1) / pool_size
        self.feature_size = self._calculate_feature_size()

        # 构建全连接层
        fc_layers = []
        fc_input = config.conv_channels[-1] * self.feature_size * self.feature_size

        # 隐藏层
        for hidden_size in config.fc_hidden:
            fc_layers.append(nn.Linear(fc_input, hidden_size))
            fc_layers.append(nn.ReLU())
            if config.use_dropout:
                fc_layers.append(nn.Dropout(config.dropout_rate))
            fc_input = hidden_size

        # 输出层 (无激活函数, CrossEntropyLoss 内置 Softmax)
        fc_layers.append(nn.Linear(fc_input, 10))

        self.fc = nn.Sequential(*fc_layers)

        # Dropout (用于卷积层后)
        self.dropout = nn.Dropout(config.dropout_rate) if config.use_dropout else None

    def _calculate_feature_size(self) -> int:
        """计算经过卷积和池化后的特征图大小"""
        size = 28
        for kernel_size, pool_size in zip(self.config.kernel_sizes, self.config.pool_sizes):
            # Conv with padding=kernel//2 保持大小不变
            # MaxPool 减半
            size = size // pool_size
        return size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入图像, shape=(N, 1, 28, 28)

        Returns:
            logits, shape=(N, 10)
        """
        # 卷积块
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)

            if self.config.use_batchnorm:
                x = self.bn_layers[i](x)

            x = torch.relu(x)
            x = torch.max_pool2d(x, self.config.pool_sizes[i])

            if self.dropout is not None:
                x = self.dropout(x)

        # 展平
        x = x.view(x.size(0), -1)

        # 全连接层
        x = self.fc(x)

        return x

    def count_parameters(self) -> int:
        """计算模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_config(name: str) -> CNNConfig:
    """获取预定义配置"""
    if name not in CONFIGS:
        available = ", ".join(CONFIGS.keys())
        raise ValueError(f"Unknown config: {name}. Available: {available}")
    return CONFIGS[name]


def list_configs() -> List[str]:
    """列出所有可用配置"""
    return list(CONFIGS.keys())


if __name__ == "__main__":
    # 测试不同配置
    print("=" * 60)
    print("MNIST CNN 模型配置测试")
    print("=" * 60)

    for config_name in list_configs():
        config = get_config(config_name)
        model = MNISTCNN(config)

        # 测试前向传播
        x = torch.randn(2, 1, 28, 28)
        y = model(x)

        print(f"\n{config.name}:")
        print(f"  参数量: {model.count_parameters():,}")
        print(f"  输出形状: {y.shape}")
        print(f"  卷积通道: {config.conv_channels}")
        print(f"  卷积核: {config.kernel_sizes}")
        print(f"  全连接: {config.fc_hidden}")
        print(f"  BatchNorm: {config.use_batchnorm}, Dropout: {config.use_dropout}")
