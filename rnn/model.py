"""
RNN/LSTM (循环神经网络) 示例

核心思想：序列建模，保持隐藏状态
- RNN: 基础循环神经网络
- LSTM: 长短期记忆网络（解决梯度消失）
- GRU: 门控循环单元（LSTM 简化版）
"""

import torch
import torch.nn as nn


class RNN(nn.Module):
    """
    基础 RNN

    h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b)

    问题：梯度消失/爆炸，难以学习长距离依赖
    """

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size

        # 输入到隐藏层
        self.W_xh = nn.Linear(input_size, hidden_size)
        # 隐藏层到隐藏层
        self.W_hh = nn.Linear(hidden_size, hidden_size)
        # 隐藏层到输出
        self.W_hy = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Args:
            x: [seq_len, batch, input_size]
        Returns:
            output: [seq_len, batch, output_size]
        """
        seq_len, batch, _ = x.shape

        # 初始化隐藏状态
        h = torch.zeros(batch, self.hidden_size).to(x.device)

        outputs = []
        for t in range(seq_len):
            # h_t = tanh(W_xh * x_t + W_hh * h_{t-1})
            h = torch.tanh(self.W_xh(x[t]) + self.W_hh(h))
            # y_t = W_hy * h_t
            y = self.W_hy(h)
            outputs.append(y)

        return torch.stack(outputs)


class LSTM(nn.Module):
    """
    长短期记忆网络 (Long Short-Term Memory)

    解决 RNN 的梯度消失问题，通过门控机制：
    - 遗忘门 (Forget Gate): 控制保留多少旧信息
    - 输入门 (Input Gate): 控制写入多少新信息
    - 输出门 (Output Gate): 控制输出多少信息
    - 细胞状态 (Cell State): 长期记忆的"高速公路"
    """

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size

        # 输入门: i_t = σ(W_xi * x_t + W_hi * h_{t-1})
        self.W_xi = nn.Linear(input_size, hidden_size)
        self.W_hi = nn.Linear(hidden_size, hidden_size)

        # 遗忘门: f_t = σ(W_xf * x_t + W_hf * h_{t-1})
        self.W_xf = nn.Linear(input_size, hidden_size)
        self.W_hf = nn.Linear(hidden_size, hidden_size)

        # 候选细胞状态: C̃_t = tanh(W_xc * x_t + W_hc * h_{t-1})
        self.W_xc = nn.Linear(input_size, hidden_size)
        self.W_hc = nn.Linear(hidden_size, hidden_size)

        # 输出门: o_t = σ(W_xo * x_t + W_ho * h_{t-1})
        self.W_xo = nn.Linear(input_size, hidden_size)
        self.W_ho = nn.Linear(hidden_size, hidden_size)

        # 输出层
        self.W_hy = nn.Linear(hidden_size, output_size)

    def _lstm_cell(self, x, h, c):
        """单步 LSTM 计算"""
        # 门控计算
        i = torch.sigmoid(self.W_xi(x) + self.W_hi(h))  # 输入门
        f = torch.sigmoid(self.W_xf(x) + self.W_hf(h))  # 遗忘门
        o = torch.sigmoid(self.W_xo(x) + self.W_ho(h))  # 输出门

        # 细胞状态更新
        # C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
        c_candidate = torch.tanh(self.W_xc(x) + self.W_hc(h))
        c_new = f * c + i * c_candidate

        # 隐藏状态更新
        # h_t = o_t ⊙ tanh(C_t)
        h_new = o * torch.tanh(c_new)

        return h_new, c_new

    def forward(self, x):
        """
        Args:
            x: [seq_len, batch, input_size]
        Returns:
            output: [seq_len, batch, output_size]
        """
        seq_len, batch, _ = x.shape

        # 初始化隐藏状态和细胞状态
        h = torch.zeros(batch, self.hidden_size).to(x.device)
        c = torch.zeros(batch, self.hidden_size).to(x.device)

        outputs = []
        for t in range(seq_len):
            h, c = self._lstm_cell(x[t], h, c)
            y = self.W_hy(h)
            outputs.append(y)

        return torch.stack(outputs)


class GRU(nn.Module):
    """
    门控循环单元 (Gated Recurrent Unit)

    LSTM 的简化版：
    - 合并输入门和遗忘门为"更新门"
    - 没有单独的细胞状态
    - 参数更少，计算更快
    """

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size

        # 更新门: z_t = σ(W_xz * x_t + W_hz * h_{t-1})
        self.W_xz = nn.Linear(input_size, hidden_size)
        self.W_hz = nn.Linear(hidden_size, hidden_size)

        # 重置门: r_t = σ(W_xr * x_t + W_hr * h_{t-1})
        self.W_xr = nn.Linear(input_size, hidden_size)
        self.W_hr = nn.Linear(hidden_size, hidden_size)

        # 候选隐藏状态
        self.W_xh = nn.Linear(input_size, hidden_size)
        self.W_hh = nn.Linear(hidden_size, hidden_size)

        # 输出层
        self.W_hy = nn.Linear(hidden_size, output_size)

    def _gru_cell(self, x, h):
        """单步 GRU 计算"""
        # 更新门和重置门
        z = torch.sigmoid(self.W_xz(x) + self.W_hz(h))  # 更新门
        r = torch.sigmoid(self.W_xr(x) + self.W_hr(h))  # 重置门

        # 候选隐藏状态
        # h̃_t = tanh(W_xh * x_t + r_t ⊙ (W_hh * h_{t-1}))
        h_candidate = torch.tanh(self.W_xh(x) + r * self.W_hh(h))

        # 隐藏状态更新
        # h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
        h_new = (1 - z) * h + z * h_candidate

        return h_new

    def forward(self, x):
        """
        Args:
            x: [seq_len, batch, input_size]
        Returns:
            output: [seq_len, batch, output_size]
        """
        seq_len, batch, _ = x.shape

        # 初始化隐藏状态
        h = torch.zeros(batch, self.hidden_size).to(x.device)

        outputs = []
        for t in range(seq_len):
            h = self._gru_cell(x[t], h)
            y = self.W_hy(h)
            outputs.append(y)

        return torch.stack(outputs)


# 使用 PyTorch 内置的 LSTM（更高效）
class OptimizedLSTM(nn.Module):
    """
    使用 nn.LSTM 的优化版本
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size

        # PyTorch LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False  # 输入格式: [seq, batch, features]
        )

        # 输出层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Args:
            x: [seq_len, batch, input_size]
        Returns:
            output: [seq_len, batch, output_size]
        """
        # LSTM 输出: (output, (h_n, c_n))
        lstm_out, _ = self.lstm(x)

        # 通过全连接层
        output = self.fc(lstm_out)

        return output


# 测试代码
if __name__ == "__main__":
    # 测试不同模型
    seq_len, batch, input_size, hidden_size, output_size = 10, 2, 16, 32, 10

    x = torch.randn(seq_len, batch, input_size)

    print("=" * 60)
    print("RNN/LSTM/GRU 模型对比")
    print("=" * 60)

    # RNN
    rnn = RNN(input_size, hidden_size, output_size)
    rnn_out = rnn(x)
    print(f"RNN 输出形状: {rnn_out.shape}")

    # LSTM (手写)
    lstm = LSTM(input_size, hidden_size, output_size)
    lstm_out = lstm(x)
    print(f"LSTM 输出形状: {lstm_out.shape}")

    # GRU
    gru = GRU(input_size, hidden_size, output_size)
    gru_out = gru(x)
    print(f"GRU 输出形状: {gru_out.shape}")

    # Optimized LSTM
    opt_lstm = OptimizedLSTM(input_size, hidden_size, output_size)
    opt_lstm_out = opt_lstm(x)
    print(f"Optimized LSTM 输出形状: {opt_lstm_out.shape}")

    # 参数量对比
    print("\n参数量对比:")
    print(f"  RNN: {sum(p.numel() for p in rnn.parameters()):,}")
    print(f"  LSTM: {sum(p.numel() for p in lstm.parameters()):,}")
    print(f"  GRU: {sum(p.numel() for p in gru.parameters()):,}")
    print(f"  OptimizedLSTM: {sum(p.numel() for p in opt_lstm.parameters()):,}")
