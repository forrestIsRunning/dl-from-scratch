"""
LoRA (Low-Rank Adaptation) 实现

LoRA 核心思想：
1. 冻结预训练模型的原始权重 W
2. 在旁路添加低秩分解：W' = A × B
3. 输出 = 原始输出 + LoRA 输出

参数量对比：
- 原始：d_model × d_model
- LoRA：d_model × rank + d_model × rank = 2 × d_model × rank
- 当 rank << d_model 时，参数量大幅减少
"""

import torch
import torch.nn as nn
import math
from typing import List


class LoRALayer(nn.Module):
    """
    LoRA 层

    用低秩矩阵 A × B 替代原始权重 W 的更新

    h = W₀ x + LoRA(x) = W₀ x + BA(x)

    其中：
    - W₀: 冻结的预训练权重
    - B: (rank, d_model) 下投影
    - A: (d_model, rank) 上投影
    - rank: 低秩维度（通常 8-64）
    """

    def __init__(self, original_layer: nn.Linear, rank: int = 8, alpha: float = 1.0):
        """
        Args:
            original_layer: 原始的 nn.Linear 层
            rank: LoRA 秩（控制参数量）
            alpha: 缩放因子（初始化强度）
        """
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha

        # 获取原始层维度
        in_features = original_layer.in_features
        out_features = original_layer.out_features

        # 冻结原始权重
        for param in original_layer.parameters():
            param.requires_grad = False

        # LoRA A: (d_model, rank) 上投影
        self.lora_A = nn.Parameter(torch.randn(in_features, rank))

        # LoRA B: (rank, d_model) 下投影
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        # 缩放因子
        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        原始输出 + LoRA 增量

        h = W₀ x + (alpha/rank) × BA(x)
        """
        # 原始层输出
        original_output = self.original_layer(x)

        # LoRA 增量: BA(x)
        # [B, T, out] = [B, T, rank] @ [rank, out] = [B, T, out]
        lora_output = (x @ self.lora_A) @ self.lora_B

        # 合并：原始 + 缩放后的 LoRA
        return original_output + lora_output * self.scaling

    def merge_to(self):
        """
        将 LoRA 权重合并回原始层

        用于推理时减少计算开销
        """
        # W_new = W₀ + (alpha/rank) × B × A
        delta_w = (self.lora_B @ self.lora_A) * self.scaling

        # 更新原始层权重
        with torch.no_grad():
            self.original_layer.weight += delta_w.T

        # 删除 LoRA 参数
        del self.lora_A
        del self.lora_B

    def unmerge(self):
        """
        从原始层移除 LoRA 权重

        用于恢复纯 LoRA 推理
        """
        delta_w = (self.lora_B @ self.lora_A) * self.scaling

        with torch.no_grad():
            self.original_layer.weight -= delta_w.T


class LoRALinear(nn.Module):
    """
    封装的 LoRA Linear 层

    提供标准 nn.Linear 的接口，可以直接替换
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 8):
        super().__init__()
        self.rank = rank

        # 原始 Linear 层
        self.linear = nn.Linear(in_features, out_features, bias=False)

        # LoRA 层
        self.lora = LoRALayer(self.linear, rank=rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora(x)


def apply_lora_to_qkv(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 1.0,
    target_modules: List[str] = ["q_proj", "v_proj"]
) -> nn.Module:
    """
    将 LoRA 应用到模型的 QKV 投影层

    这是 LLaMA 的典型用法：只对 Attention 的 Q 和 V 使用 LoRA

    Args:
        model: 原始模型
        rank: LoRA 秩
        alpha: 缩放因子
        target_modules: 要替换的模块名

    Returns:
        应用 LoRA 后的模型
    """
    # 找到所有需要替换的 Linear 层
    for name, module in model.named_modules():
        # 检查是否是目标模块
        if any(target in name for target in target_modules):
            # 获取原始层维度
            original_linear = module

            # 替换为 LoRA 封装层
            lora_linear = LoRALinear(
                in_features=original_linear.in_features,
                out_features=original_linear.out_features,
                rank=rank
            )

            # 复制权重
            with torch.no_grad():
                lora_linear.linear.weight.copy_(original_linear.weight)

            # 替换模块
            parent_name = name.rsplit('.', 1)[0]
            setattr(model, parent_name, lora_linear)

    return model


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("LoRA 层测试")
    print("=" * 60)

    # 创建测试数据
    batch_size, seq_len, d_model = 2, 10, 64

    # 原始 Linear 层
    original_layer = nn.Linear(d_model, d_model, bias=False)

    # LoRA 层
    lora_layer = LoRALayer(original_layer, rank=8)

    # 输入
    x = torch.randn(batch_size, seq_len, d_model)

    # 前向传播
    output = lora_layer(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")

    # 参数量对比
    original_params = sum(p.numel() for p in original_layer.parameters())
    lora_params = sum(p.numel() for p in lora_layer.parameters())

    print(f"\n原始参数量: {original_params:,}")
    print(f"LoRA 参数量: {lora_params:,}")
    print(f"总参数量: {original_params + lora_params:,}")
    print(f"参数增加: {lora_params / original_params * 100:.2f}%")

    # 打印权重形状
    print(f"\n权重形状:")
    print(f"  原始 W: {original_layer.weight.shape}")
    print(f"  LoRA A: {lora_layer.lora_A.shape}")
    print(f"  LoRA B: {lora_layer.lora_B.shape}")
