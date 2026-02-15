"""
MNIST 手写数字分类器 - 多层神经网络

网络结构: 784 → 512 → 256 → 128 → 64 → 10
- 输入: 28x28 灰度图展平为 784 维向量
- 输出: 10 维向量 (对应数字 0-9 的概率)

注意: 最后一层无需 Softmax，CrossEntropyLoss 内置 LogSoftmax
"""
import torch


class MNISTClassifier(torch.nn.Module):
    """
    MNIST 多分类网络

    维度变换链:
    [N,1,28,28] → flatten → [N,784] → [N,512] → [N,256] → [N,128] → [N,64] → [N,10]
    """

    def __init__(self):
        super(MNISTClassifier, self).__init__()

        # 五层线性层
        self.fc1 = torch.nn.Linear(784, 512)  # 输入层: 784 → 512
        self.fc2 = torch.nn.Linear(512, 256)  # 隐藏层1: 512 → 256
        self.fc3 = torch.nn.Linear(256, 128)  # 隐藏层2: 256 → 128
        self.fc4 = torch.nn.Linear(128, 64)   # 隐藏层3: 128 → 64
        self.fc5 = torch.nn.Linear(64, 10)    # 输出层: 64 → 10 (10个类别)

        # ReLU 激活函数
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入图像, shape=(N, 1, 28, 28)

        Returns:
            输出 logits, shape=(N, 10)
            注意: 返回的是线性输出，不是概率分布
                  CrossEntropyLoss 会自动处理 Softmax
        """
        # 展平图像: [N, 1, 28, 28] → [N, 784]
        x = x.view(-1, 784)

        # 逐层传播 (输出层无激活函数)
        x = self.relu(self.fc1(x))  # [N, 784] → [N, 512]
        x = self.relu(self.fc2(x))  # [N, 512] → [N, 256]
        x = self.relu(self.fc3(x))  # [N, 256] → [N, 128]
        x = self.relu(self.fc4(x))  # [N, 128] → [N, 64]
        x = self.fc5(x)             # [N, 64] → [N, 10] (无激活)

        return x
