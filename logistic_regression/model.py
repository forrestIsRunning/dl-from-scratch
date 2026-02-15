"""
逻辑回归模型实现

逻辑回归是一种二分类算法，通过线性组合 + Sigmoid 函数将输入映射到 [0, 1] 区间的概率值。
"""
import torch


class LogisticRegressionModel(torch.nn.Module):
    """
    逻辑回归模型：线性层 + Sigmoid 激活函数

    数学公式：y = sigmoid(wx + b)
    其中 sigmoid(x) = 1 / (1 + exp(-x))
    """

    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        # 线性层：输入维度1，输出维度1
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：线性计算 -> Sigmoid 映射

        Args:
            x: 输入张量，shape=(batch_size, 1)

        Returns:
            预测概率，shape=(batch_size, 1)，值域 [0, 1]
        """
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred