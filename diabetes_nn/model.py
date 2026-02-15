"""
多层神经网络模型 - 糖尿病预测

维度变换链: 8 → 6 → 4 → 1
- 输入层: 8维特征 (糖尿病数据集的8个指标)
- 隐藏层1: 6维 (特征抽象)
- 隐藏层2: 4维 (进一步压缩)
- 输出层: 1维 (患病概率)

每层后接Sigmoid激活函数，引入非线性拟合能力。
"""
import torch


class DiabetesNN(torch.nn.Module):
    """
    多层神经网络：3层全连接层 + Sigmoid激活

    维度变换: [N,8] → [N,6] → [N,4] → [N,1]

    公式:
        O1 = Sigmoid(X · W1^T + b1)   # [N,8] → [N,6]
        O2 = Sigmoid(O1 · W2^T + b2)  # [N,6] → [N,4]
        Ŷ = Sigmoid(O2 · W3^T + b3)   # [N,4] → [N,1]
    """

    def __init__(self):
        super(DiabetesNN, self).__init__()

        # 第1层: 8维 → 6维
        # 权重矩阵W1形状: [6, 8]
        # 矩阵乘法: [N,8] × [8,6] = [N,6]
        self.linear1 = torch.nn.Linear(8, 6)

        # 第2层: 6维 → 4维
        # 权重矩阵W2形状: [4, 6]
        # 矩阵乘法: [N,6] × [6,4] = [N,4]
        self.linear2 = torch.nn.Linear(6, 4)

        # 第3层: 4维 → 1维
        # 权重矩阵W3形状: [1, 4]
        # 矩阵乘法: [N,4] × [4,1] = [N,1]
        self.linear3 = torch.nn.Linear(4, 1)

        # Sigmoid激活函数 (按元素计算，不改变维度)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播: 三层线性变换 + Sigmoid激活

        Args:
            x: 输入张量，shape=(N, 8)，N为样本数

        Returns:
            预测概率，shape=(N, 1)，值域 [0, 1]
        """
        # 第1层: [N,8] → [N,6]
        x = self.sigmoid(self.linear1(x))

        # 第2层: [N,6] → [N,4]
        x = self.sigmoid(self.linear2(x))

        # 第3层: [N,4] → [N,1]
        x = self.sigmoid(self.linear3(x))

        return x
