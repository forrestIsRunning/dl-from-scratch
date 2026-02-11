"""
Transformer 模型定义

从零实现一个完整的 Transformer Decoder-only 模型（类似 GPT）。

模型结构:
- Token Embedding
- Positional Encoding
- N × Transformer Block
- Final LayerNorm
- Output Linear Layer
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F


# =============================================================================
# Feed Forward Network (前馈网络)
# =============================================================================

class FeedForwardNetwork(nn.Module):
    """
    前馈神经网络 (FFN)

    结构: Linear → ReLU → Linear → Dropout

    维度变化: [batch, seq_len, d_model] → [batch, seq_len, d_model*4] → [batch, seq_len, d_model]

    为什么先扩大 4 倍再缩回来？
    - 扩大维度增加模型的表达能力
    - 类似于给模型更多"思考空间"
    """

    def __init__(self, d_model: int, dropout: float):
        """
        初始化 FFN

        Args:
            d_model: 模型维度
            dropout: Dropout 概率
        """
        super().__init__()
        self.ffn = nn.Sequential(
            # 第一层: d_model → d_model * 4
            nn.Linear(d_model, d_model * 4),
            # 激活函数: 引入非线性
            nn.ReLU(),
            # 第二层: d_model * 4 → d_model
            nn.Linear(d_model * 4, d_model),
            # Dropout: 防止过拟合
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 [batch, seq_len, d_model]

        Returns:
            输出张量 [batch, seq_len, d_model]
        """
        return self.ffn(x)


# =============================================================================
# Scaled Dot-Product Attention (缩放点积注意力)
# =============================================================================

class Attention(nn.Module):
    """
    单头注意力机制

    公式: Attention(Q, K, V) = softmax(QK^T / √d_k) V

    Causal Mask: 防止模型"看到未来"的信息
    """

    def __init__(self, d_model: int, head_size: int, context_length: int, dropout: float):
        """
        初始化单头注意力

        Args:
            d_model: 模型维度
            head_size: 每个头的维度
            context_length: 最大上下文长度
            dropout: Dropout 概率
        """
        super().__init__()
        self.head_size = head_size

        # Q、K、V 的线性变换
        # 注意: bias=False 是常见的做法
        self.Wq = nn.Linear(d_model, head_size, bias=False)
        self.Wk = nn.Linear(d_model, head_size, bias=False)
        self.Wv = nn.Linear(d_model, head_size, bias=False)

        # Causal Mask: 下三角矩阵
        # 1 表示可见，0 表示被 mask 掉
        self.register_buffer(
            'mask',
            torch.tril(torch.ones(context_length, context_length))
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 [batch, seq_len, d_model]

        Returns:
            输出张量 [batch, seq_len, head_size]
        """
        B, T, C = x.shape  # Batch, Time(seq_len), Channels(d_model)

        # 1. 生成 Q, K, V
        q = self.Wq(x)  # [B, T, head_size]
        k = self.Wk(x)  # [B, T, head_size]
        v = self.Wv(x)  # [B, T, head_size]

        # 2. 计算注意力分数: Q @ K^T
        # 结果形状: [B, T, T]
        weights = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_size)

        # 3. 应用 Causal Mask
        # 将未来位置设为 -inf，这样 softmax 后就是 0
        weights = weights.masked_fill(
            self.mask[:T, :T] == 0,
            float('-inf')
        )

        # 4. Softmax 归一化
        # 现在每个位置的权重加起来等于 1
        weights = F.softmax(weights, dim=-1)

        # 5. Dropout
        weights = self.dropout(weights)

        # 6. 与 V 相乘得到输出
        # 加权求和: 每个位置的输出是所有位置值的加权平均
        output = weights @ v  # [B, T, head_size]

        return output


# =============================================================================
# Multi-Head Attention (多头注意力)
# =============================================================================

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制

    多头注意力的思路是: 让每个头关注不同的"视角"
    - 有的头可能关注语法结构
    - 有的头可能关注语义关系
    - 有的头可能关注上下文引用

    最后把所有头的输出拼接起来，得到更丰富的表示。
    """

    def __init__(self, d_model: int, num_heads: int, head_size: int, context_length: int, dropout: float):
        """
        初始化多头注意力

        Args:
            d_model: 模型维度
            num_heads: 注意力头数量
            head_size: 每个头的维度 (d_model // num_heads)
            context_length: 最大上下文长度
            dropout: Dropout 概率
        """
        super().__init__()

        # 创建多个注意力头
        # 这里是"物理分开"的实现，每个头有独立的 Wq、Wk、Wv
        self.heads = nn.ModuleList([
            Attention(d_model, head_size, context_length, dropout)
            for _ in range(num_heads)
        ])

        # 输出投影层 (Wo)
        # 将所有头的输出拼接后，再通过一个线性层
        self.projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 [batch, seq_len, d_model]

        Returns:
            输出张量 [batch, seq_len, d_model]
        """
        # 1. 并行运行所有头
        head_outputs = [head(x) for head in self.heads]

        # 2. 拼接所有头的输出
        # 每个头输出 [B, T, head_size]，拼接后是 [B, T, d_model]
        x = torch.cat(head_outputs, dim=-1)

        # 3. 输出投影
        x = self.dropout(self.projection(x))

        return x


# =============================================================================
# Transformer Block (Transformer 块)
# =============================================================================

class TransformerBlock(nn.Module):
    """
    Transformer Block

    Pre-Norm 结构 (GPT-2 使用):
    x = x + MHA(LayerNorm(x))
    x = x + FFN(LayerNorm(x))

    Pre-Norm vs Post-Norm:
    - Pre-Norm: 先 Norm 再 Attention，训练更稳定
    - Post-Norm: 先 Attention 再 Norm（原始 Transformer）
    """

    def __init__(self, d_model: int, num_heads: int, head_size: int, context_length: int, dropout: float):
        """
        初始化 Transformer Block

        Args:
            d_model: 模型维度
            num_heads: 注意力头数量
            head_size: 每个头的维度
            context_length: 最大上下文长度
            dropout: Dropout 概率
        """
        super().__init__()

        # 两个 LayerNorm
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        # 多头注意力
        self.mha = MultiHeadAttention(d_model, num_heads, head_size, context_length, dropout)

        # 前馈网络
        self.ffn = FeedForwardNetwork(d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 [batch, seq_len, d_model]

        Returns:
            输出张量 [batch, seq_len, d_model]
        """
        # Attention + 残差连接
        x = x + self.mha(self.ln1(x))

        # FFN + 残差连接
        x = x + self.ffn(self.ln2(x))

        return x


# =============================================================================
# 完整 Transformer 模型
# =============================================================================

class GPTModel(nn.Module):
    """
    完整的 GPT 风格 Transformer 模型

    结构:
    1. Token Embedding: 将 token ID 转换为向量
    2. Positional Encoding: 添加位置信息
    3. N × Transformer Block: 处理序列
    4. Final LayerNorm: 归一化
    5. Output Linear: 投影到词汇表
    """

    def __init__(self, config: dict):
        """
        初始化模型

        Args:
            config: 超参数字典，包含:
                - d_model: 模型维度
                - num_heads: 注意力头数量
                - num_blocks: Transformer Block 数量
                - head_size: 每个头的维度
                - context_length: 最大上下文长度
                - dropout: Dropout 概率
                - vocab_size: 词汇表大小
                - device: 计算设备
        """
        super().__init__()

        # 保存配置
        self.context_length = config["context_length"]
        self.d_model = config["d_model"]
        self.device = config["device"]

        # Token Embedding
        # 将 token ID (整数) 转换为 d_model 维向量
        self.token_embedding = nn.Embedding(config["vocab_size"], self.d_model)

        # Transformer Blocks + 最终 LayerNorm
        self.blocks = nn.Sequential(*(
            [TransformerBlock(
                self.d_model,
                config["num_heads"],
                config["head_size"],
                self.context_length,
                config["dropout"]
            ) for _ in range(config["num_blocks"])] +
            [nn.LayerNorm(self.d_model)]
        ))

        # 输出投影层
        # 将 d_model 维投影到 vocab_size 维（得到每个 token 的 logits）
        self.output_projection = nn.Linear(self.d_model, config["vocab_size"])

        # 权重共享: Token embedding 和输出投影共享权重
        # 这是一个常见技巧，可以减少参数量
        self.output_projection.weight = self.token_embedding.weight

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """
        初始化模型权重

        使用 PyTorch 默认初始化，对 Linear 层应用 Xavier 初始化。
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot 初始化
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _get_positional_encoding(self, seq_len: int) -> torch.Tensor:
        """
        计算位置编码

        使用正弦/余弦位置编码（论文原始方法）

        公式:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

        Args:
            seq_len: 序列长度

        Returns:
            位置编码张量 [seq_len, d_model]
        """
        # 创建位置索引 [0, 1, 2, ..., seq_len-1]
        position = torch.arange(seq_len, dtype=torch.float, device=self.device).unsqueeze(1)  # [seq_len, 1]

        # 计算频率项
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float, device=self.device) *
            (-math.log(10000.0) / self.d_model)
        )  # [d_model/2]

        # 创建位置编码矩阵
        pe = torch.zeros(seq_len, self.d_model, device=self.device)

        # 偶数维度用 sin
        pe[:, 0::2] = torch.sin(position * div_term)

        # 奇数维度用 cos
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """
        前向传播

        Args:
            idx: 输入 token IDs [batch, seq_len]
            targets: 目标 token IDs [batch, seq_len]（训练时需要）

        Returns:
            (logits, loss)
            - logits: [batch, seq_len, vocab_size]
            - loss: 标量张量（训练时）或 None（推理时）
        """
        B, T = idx.shape

        # 1. Token Embedding
        tok_emb = self.token_embedding(idx)  # [B, T, d_model]

        # 2. Positional Encoding
        pos_emb = self._get_positional_encoding(T)  # [T, d_model]

        # 3. 相加得到输入表示
        x = tok_emb + pos_emb  # [B, T, d_model]

        # 4. 通过所有 Transformer Blocks
        x = self.blocks(x)  # [B, T, d_model]

        # 5. 输出投影到词汇表
        logits = self.output_projection(x)  # [B, T, vocab_size]

        # 6. 如果有目标，计算损失
        if targets is not None:
            # 重塑为 [B*T, vocab_size] 和 [B*T]
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
        else:
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = None) -> torch.Tensor:
        """
        自回归生成文本

        Args:
            idx: 初始 token IDs [batch, seq_len]
            max_new_tokens: 最多生成多少个新 token
            temperature: 温度参数，控制随机性
                - < 1: 更确定（概率集中）
                - = 1: 原始概率
                - > 1: 更随机（概率分散）
            top_k: 只从 top-k 个最高概率的词中采样

        Returns:
            生成的 token IDs [batch, seq_len + max_new_tokens]
        """
        for _ in range(max_new_tokens):
            # 1. 截断到最大上下文长度
            idx_crop = idx[:, -self.context_length:]

            # 2. 前向传播
            logits, _ = self.forward(idx_crop)

            # 3. 只取最后一个位置的 logits
            # 这是我们需要预测的下一个 token
            logits = logits[:, -1, :] / temperature

            # 4. 可选: Top-K 采样
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # 5. Softmax 得到概率分布
            probs = F.softmax(logits, dim=-1)

            # 6. 采样
            idx_next = torch.multinomial(probs, num_samples=1)  # [B, 1]

            # 7. 拼接到序列
            idx = torch.cat((idx, idx_next), dim=1)  # [B, T+1]

        return idx

    def get_num_params(self):
        """
        获取模型参数量

        Returns:
            总参数数量
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    from config import get_model_params, get_device

    # 创建测试配置
    vocab_size = 100
    config = get_model_params(vocab_size)

    # 创建模型
    model = GPTModel(config).to(config["device"])
    print(f"模型参数量: {model.get_num_params():,}")

    # 测试前向传播
    batch_size = 2
    seq_len = 64
    idx = torch.randint(0, vocab_size, (batch_size, seq_len)).to(config["device"])
    targets = torch.randint(0, vocab_size, (batch_size, seq_len)).to(config["device"])

    logits, loss = model(idx, targets)
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item()}")

    # 测试生成
    generated = model.generate(idx[:, :8], max_new_tokens=10, temperature=0.8)
    print(f"Generated shape: {generated.shape}")
    print(f"Generated: {generated}")
