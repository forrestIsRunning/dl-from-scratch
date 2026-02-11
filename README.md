# Transformer From Scratch

从零实现一个完整的 Transformer Decoder-only 模型（类似 GPT）。

这是一个教学项目，目标是帮助理解 Transformer 的核心原理。所有代码都有详细的中文注释，适合学习和扩展。

## 项目特点

- **完整实现**：包含数据加载、模型定义、训练和推理
- **详细注释**：每个模块都有清晰的中文注释
- **即用即走**：一条命令即可开始训练
- **Mac 友好**：支持 MPS 加速（M系列芯片）

## 快速开始

### 1. 安装依赖

```bash
# 使用 uv
uv sync

# 或使用 pip
pip install torch numpy tqdm requests
```

### 2. 开始训练

```bash
python main.py train
```

训练会自动：
- 下载莎士比亚数据集
- 创建模型
- 开始训练并保存检查点

### 3. 生成文本

```bash
python main.py generate
```

## 项目结构

```
torch/
├── config.py      # 超参数配置
├── data.py        # 数据加载和处理
├── model.py       # Transformer 模型定义
├── train.py       # 训练脚本
├── inference.py   # 推理脚本
├── main.py        # 主入口
├── checkpoints/   # 模型检查点（训练后生成）
└── data/          # 数据文件（训练后生成）
```

## 核心概念

### Transformer 模型结构

```
输入 Token IDs
    ↓
Token Embedding (词嵌入)
    ↓
Positional Encoding (位置编码)
    ↓
┌─────────────────────────────────┐
│  Transformer Block × N           │
│  ┌─────────────────────────────┐│
│  │ LayerNorm                   ││
│  │ Multi-Head Attention        ││
│  │ + 残差连接                  ││
│  │                             ││
│  │ LayerNorm                   ││
│  │ Feed Forward Network        ││
│  │ + 残差连接                  ││
│  └─────────────────────────────┘│
└─────────────────────────────────┘
    ↓
Final LayerNorm
    ↓
Output Linear (投影到词表)
    ↓
Logits (每个位置的概率分布)
```

### 模块说明

| 模块 | 说明 |
|------|------|
| `FeedForwardNetwork` | 前馈网络，两层全连接 + ReLU |
| `Attention` | 单头注意力，实现 QKV 变换和 Causal Mask |
| `MultiHeadAttention` | 多头注意力，并行多个注意力头 |
| `TransformerBlock` | Transformer 块，包含 Attention 和 FFN |
| `GPTModel` | 完整模型，组合所有组件 |

## 超参数配置

所有超参数在 `config.py` 中定义：

### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `D_MODEL` | 256 | 模型维度（原论文 512） |
| `NUM_HEADS` | 8 | 注意力头数量 |
| `NUM_BLOCKS` | 6 | Transformer Block 数量 |
| `CONTEXT_LENGTH` | 256 | 最大序列长度 |
| `DROPOUT` | 0.1 | Dropout 概率 |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `BATCH_SIZE` | 64 | 批次大小 |
| `LEARNING_RATE` | 3e-4 | 学习率 |
| `MAX_STEPS` | 5000 | 最大训练步数 |
| `WARMUP_STEPS` | 100 | 学习率预热步数 |
| `GRAD_CLIP` | 1.0 | 梯度裁剪阈值 |

### 推理参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MAX_NEW_TOKENS` | 500 | 最大生成 token 数 |
| `TEMPERATURE` | 0.8 | 温度参数（越小越确定） |
| `TOP_K` | 50 | Top-K 采样 |

## 使用指南

### 训练

```bash
# 基础训练
python main.py train

# 修改超参数后训练
# 编辑 config.py，然后运行
python main.py train
```

训练过程中会：
- 定期在验证集上评估（每 500 步）
- 保存最佳模型
- 保存定期检查点（每 1000 步）
- 显示训练进度和损失

### 推理

```bash
python main.py generate
```

推理模式支持：

1. **单次生成**：输入提示文本，生成一次
2. **交互模式**：持续输入提示文本生成

交互模式命令：
- `temp <值>` - 设置温度
- `topk <值>` - 设置 top-k
- `len <值>` - 设置生成长度
- `quit` - 退出

## 参数调整指南

### 调整模型大小

更大的模型 = 更强的表达能力，但训练更慢

```python
# 小模型（快速验证）
D_MODEL = 128
NUM_BLOCKS = 4
NUM_HEADS = 4

# 中等模型（默认）
D_MODEL = 256
NUM_BLOCKS = 6
NUM_HEADS = 8

# 大模型（更好效果）
D_MODEL = 512
NUM_BLOCKS = 12
NUM_HEADS = 8
```

### 调整训练速度

```python
# 更快训练（减少步数）
MAX_STEPS = 2000
BATCH_SIZE = 128

# 更好效果（增加步数）
MAX_STEPS = 10000
BATCH_SIZE = 64
```

### 调整生成质量

```python
# 更确定的生成
TEMPERATURE = 0.5
TOP_K = 30

# 更随机的生成
TEMPERATURE = 1.2
TOP_K = 100
```

## 常见问题

### Q: 训练需要多长时间？

A: 取决于硬件和配置。使用默认配置：
- Mac M1/M2: 约 10-20 分钟
- GPU: 约 5-10 分钟
- CPU: 约 30-60 分钟

### Q: 如何检查模型是否在学习？

A: 观察训练损失：
- 初始损失：约 4.0（随机预测）
- 良好训练：约 1.5-2.0
- 完美训练：约 1.0 以下

### Q: 可以用自己的数据训练吗？

A: 可以。修改 `data.py` 中的 `load_text()` 函数，加载你自己的文本数据。

### Q: 如何在 GPU 上训练？

A: 代码会自动检测并使用 CUDA。如果检测到 GPU，会自动使用。

### Q: Mac M系列芯片支持吗？

A: 完全支持。代码会自动使用 MPS 加速。

## 技术细节

### Tokenization

当前使用字符级 tokenization（最简单）。每个字符对应一个 token。

升级路径：字符级 → BPE → SentencePiece

### 位置编码

使用正弦/余弦位置编码（论文原版）。优点是不需要训练，且可处理任意长度。

### 初始化

使用 PyTorch 默认初始化（Xavier for Linear, Normal for Embedding）。

### 优化器

AdamW + 学习率调度（Warmup + 余弦退火）

## 下一步扩展

- [ ] 实现 BPE tokenization
- [ ] 添加 Flash Attention
- [ ] 实现 KV Cache（加速推理）
- [ ] 添加 LoRA 支持
- [ ] 实现模型量化
- [ ] 添加更多评估指标

## 参考资料

- [Attention Is All You Need (原论文)](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Andrej Karpathy's GPT Video](https://www.youtube.com/watch?v=kCc8FmEb1nY)

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！
