# LoRA (Low-Rank Adaptation) 示例

## 核心思想

**LoRA = 低秩适应 (Low-Rank Adaptation)**

不是直接微调预训练模型的大权重矩阵，而是：
1. **冻结原始权重** W₀（不做梯度更新）
2. **添加旁路**：W = W₀ + ΔW
3. **低秩分解**：ΔW = B × A（rank << d_model）

```
原始: d_model × d_model = 4096 × 4096 = 16,777,216 参数
LoRA: 4096 × 8 + 8 × 4096 = 65,536 + 32,768 = 98,304 参数
节省: 99.4%
```

## LoRA 的优势

| 特性 | 全量微调 | LoRA |
|------|-----------|------|
| **参数量** | 全部更新 | 仅 ~1% 可训练 |
| **显存占用** | 高 | 低（可用更大 batch） |
| **训练速度** | 慢 | 快 |
| **存储空间** | 每个模型一份 | 共享基础 + 小 LoRA 权重 |
| **切换成本** | 高（加载整个模型） | 低（切换 LoRA 权重） |

## LoRA 数学原理

### 标准线性层

```
y = Wx + b
```

### LoRA 线性层

```
  冻结           可训练
    W₀              B (r×d)      A (d×r)
     ↓                ↓              ↓
     └─────────────×──────────────→ ΔW (r×d)
          ↓                        ↓
          y = W₀x + (α/r) × ΔWx
```

### 为什么有效？

**低秩假设**：预训练权重是在低维流形上，微调只需要在这个流形上做小的移动。

**矩阵分解**：ΔW = BA 是一个秩为 r 的矩阵（r << d），参数大幅减少。

## 项目结构

```
lora/
├── model.py        # LoRA 层实现
├── train.py        # LLaMA LoRA 微调脚本
├── requirements.txt # Python 依赖
└── README.md       # 本文件
```

## 快速开始

### 1. 安装依赖

```bash
pip install transformers accelerate datasets peft
```

### 2. 训练 LoRA

```bash
cd lora

# 基础训练（未量化，适合快速验证）
python train.py --model huggyllama/llama-7b --rank 8

# 4-bit 量化训练（节省显存，7B 模型可用 8-12GB 显存）
python train.py --model huggyllama/llama-7b --rank 8 --use-4bit
```

### 3. 生成文本

```bash
python train.py --model huggyllama/llama-7b --rank 8 --prompt "The future of AI is"
```

## 支持的模型

| 模型系列 | 参数量 | 7B 4-bit 显存 |
|---------|-------|----------------|
| LLaMA 7B | 6.7B | ~6GB |
| LLaMA 13B | 13B | ~10GB |
| LLaMA 2 7B/13B | 6.7B/13B | ~6-10GB |

## 训练参数说明

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--rank` | 8 | LoRA 秩（越大参数越多，效果可能越好） |
| `--alpha` | 16 | 缩放因子（通常 = 2 × rank） |
| `--use-4bit` | False | 启用 4-bit 量化（BNB） |
| `--no-4bit` | - | 禁用 4-bit 量化 |

## 常见问题

**Q: rank 选择多少合适？**
```
A: 经验法则
  - 7B 模型: rank 8-16
  - 13B 模型: rank 16-32
  - 70B 模型: rank 32-64
  - 显存不足：降低 rank
```

**Q: alpha 是什么？**
```
A: 缩放初始化的 LoRA 权重
  alpha = 2 × rank 是常见选择
  alpha 越大，LoRA 影响越大
  训练时可以调整，推理时固定
```

**Q: 为什么只对 QV 做 LoRA？**
```
A: LLaMA 论文发现：
  - 对 QV 做 LoRA 效果最好
  - 对 K 做效果提升不明显
  - 所以默认只微调 QV 投影层
```

## 参考资料

- [LoRA 论文](https://arxiv.org/abs/2106.09685)
- [Hugging Face PEFT](https://huggingface.co/docs/peft)
- [LLaMA 论文](https://arxiv.org/abs/2302.13971)
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) - 4-bit 量化
