# Transformer From Scratch

从零实现一个完整的 Transformer Decoder-only 模型（类似 GPT）。

## 项目特点

- **完整实现**：包含数据加载、模型定义、训练和推理
- **详细注释**：每个模块都有清晰的中文注释
- **即用即走**：一条命令即可开始训练
- **Mac 友好**：支持 MPS 加速（M系列芯片）

## 快速开始

```bash
# 进入 transformer 目录
cd transformer

# 安装依赖
pip install torch numpy tqdm requests

# 开始训练
python main.py train

# 生成文本
python main.py generate
```

## 项目结构

```
transformer/
├── config.py      # 超参数配置
├── data.py        # 数据加载和处理
├── model.py       # Transformer 模型定义
├── train.py       # 训练脚本
├── inference.py   # 推理脚本
├── main.py        # 主入口
├── ARCHITECTURE.md # 调用逻辑详解
├── data/          # 数据目录
└── checkpoints/   # 模型检查点
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
│  Transformer Block × N         │
│  ┌─────────────────────────┐  │
│  │ LayerNorm               │  │
│  │ Multi-Head Attention    │  │
│  │ + 残差连接              │  │
│  │                         │  │
│  │ LayerNorm               │  │
│  │ Feed Forward Network    │  │
│  │ + 残差连接              │  │
│  └─────────────────────────┘  │
└─────────────────────────────────┘
    ↓
Final LayerNorm
    ↓
Output Linear (投影到词表)
    ↓
Logits (每个位置的概率分布)
```

## 详细文档

查看 [ARCHITECTURE.md](ARCHITECTURE.md) 了解：
- 调用逻辑与数据流转
- 训练 vs 推理流程
- 模块调用关系
- 数据维度变化
