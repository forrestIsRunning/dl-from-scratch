# 🚀 深度学习从零实现

> **CNN、RNN、Transformer、LoRA** 与 **CUDA 编程** 的完整示例集合

![Python](https://img.shields.io/badge/python-3.12+-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## ✨ 特性

- ✅ **从零实现**：完整代码，详细中文注释
- 📚 **渐进学习**：CNN → RNN → Transformer → LoRA
- 🚀 **CUDA 教程**：GPU 并行编程基础
- 🛠️ **MLOps 流程**：实验跟踪、部署、监控
- 📖 **示例驱动**：每个模型都可以直接运行

---

## 📁 项目结构

```
dl-from-scratch/
├── pyproject.toml         # 项目配置（uv 管理）
├── uv.lock              # 依赖锁定
│
├── docs/               # 📚 文档
│   └── MODEL_COMPARISON.md    # 模型对比详解
│
├── cnn/               # 🔷 CNN 示例
│   ├── model.py        # CNN/LeNet-5 定义
│   ├── train.py        # MNIST 训练
│   └── README.md
│
├── rnn/               # 🔄 RNN/LSTM 示例
│   ├── model.py        # RNN/LSTM/GRU 实现
│   ├── train.py        # 序列预测训练
│   └── README.md
│
├── transformer/        # 🤖 Transformer 示例
│   ├── config.py
│   ├── data.py
│   ├── model.py        # GPT 风格 Transformer
│   ├── train.py
│   ├── inference.py
│   ├── main.py
│   ├── ARCHITECTURE.md  # 架构详解
│   └── README.md
│
├── cuda/              # ⚡ CUDA 编程示例
│   ├── tutorial.py     # 6 个基础示例
│   └── README.md
│
├── lora/              # 🎯 LoRA 微调
│   ├── model.py        # LoRA 层实现
│   ├── train.py        # LLaMA 微调
│   ├── README.md
│   └── （依赖统一在根目录）
│
└── mlops-example/     # 🛠️ MLOps
    ├── mlflow_tracking_example.py
    ├── docker-compose.yml
    └── README.md
```

---

## 🚀 快速开始

### 环境配置

```bash
cd /Users/Zhuanz/go/src/dl-from-scratch

# 安装基础依赖
uv sync
```

### 运行示例

| 任务 | 命令 |
|------|------|
| **CNN 训练** | `python cnn/train.py --model cnn --epochs 5` |
| **RNN 训练** | `python rnn/train.py --model lstm --epochs 10` |
| **文本生成** | `cd transformer && python main.py train` |
| **CUDA 教程** | `python cuda/tutorial.py` |
| **LoRA 微调** | `uv pip install "transformers[accelerate]" peft && python lora/train.py` |

---

## 📚 子项目指南

### 🔷 CNN (卷积神经网络)

**核心思想**：局部感受野 + 权重共享

- `model.py` - CNN 和 LeNet-5 定义
- `train.py` - MNIST 训练脚本
- 参数量：~28K (CNN), ~60K (LeNet-5)

```bash
cd cnn
python train.py --model cnn --epochs 5
```

### 🔄 RNN/LSTM (循环神经网络)

**核心思想**：序列建模，保持隐藏状态

- `model.py` - RNN、LSTM、GRU 手写实现
- `train.py` - 序列预测训练
- 支持三种模型对比学习

```bash
cd rnn
python train.py --model lstm --epochs 10
```

### 🤖 Transformer

**核心思想**：自注意力机制 + 全局上下文

- 完整 GPT 风格 Transformer 实现
- 支持莎士比亚文本生成
- 包含训练、推理、CLI

```bash
cd transformer
python main.py train      # 训练
python main.py generate   # 推理
```

**相关文档**：[ARCHITECTURE.md](transformer/ARCHITECTURE.md) - 调用逻辑详解

### ⚡ CUDA 编程

**核心概念**：设备管理、并行计算、内存管理

- 6 个基础示例
- GPU vs CPU 性能对比
- 自定义 CUDA 内核

```bash
cd cuda
python tutorial.py
```

### 🎯 LoRA 微调

**核心思想**：低秩适应 (Low-Rank Adaptation)

- LoRA 层实现（含详细数学公式）
- LLaMA 系列微调
- 支持 4-bit 量化

```bash
# 安装 LoRA 依赖（可选）
uv pip install "transformers[accelerate]" peft bitsandbytes

# 训练
python lora/train.py --model huggyllama/llama-7b --rank 8
```

**参数对比**：
- 全量微调：~6.7B 参数
- LoRA (rank=8)：~66M 参数 (仅 1%)
- 节省显存：~50%

### 🛠️ MLOps

**核心流程**：训练 → 跟踪 → 注册 → 部署

- MLflow 端到端示例
- Docker 容器化
- 模型版本管理

```bash
cd mlops-example
docker-compose -f docker-compose.mlflow.yml up -d
python mlflow_tracking_example.py train
```

---

## 📖 模型对比速查

| 模型 | 适用场景 | 并行度 | 长距离 | 参数量 | 推理速度 |
|------|----------|---------|---------|---------|---------|
| **CNN** | 图像 | 高 | 无 | 中 | 快 |
| **RNN/LSTM** | 时序 | 低 | 弱 | 慢 |
| **Transformer** | 文本/NLP | 极高 | 强 | 快 |

**选型建议**：
- 图像任务 → CNN
- 序列数据 → LSTM/GRU
- 文本生成 → Transformer
- 大模型微调 → LoRA

详见：[docs/MODEL_COMPARISON.md](docs/MODEL_COMPARISON.md)

---

## 📚 进阶学习路径

```
1️⃣ CNN 基础
   理解卷积、池化、局部连接
   ↓
2️⃣ RNN/LSTM
   理解序列建模、梯度消失问题
   ↓
3️⃣ Transformer
   理解注意力机制、并行计算
   ↓
4️⃣ CUDA 编程
   理解 GPU 并行、内存优化
   ↓
5️⃣ LoRA 微调
   理解参数高效微调
   ↓
6️⃣ MLOps
   理解工程化部署
```

---

## 📋 依赖说明

### 基础依赖（必需）

```
torch          # PyTorch 框架
numpy         # 数值计算
tqdm          # 进度条
matplotlib    # 可视化
pillow        # 图像处理
```

### LoRA/微调依赖（可选）

按需安装，运行 LoRA 示例时：
```bash
uv pip install "transformers[accelerate]" peft bitsandbytes
```

这些依赖较大，只有需要 LLaMA 微调时才安装。

---

## 🌐 环境要求

- Python >= 3.12
- PyTorch >= 2.0.0
- CUDA (可选，用于 GPU 加速)

支持平台：macOS (MPS)、Linux、Windows

---

## 📄 文档

| 文档 | 说明 |
|------|------|
| **[docs/MODEL_COMPARISON.md](docs/MODEL_COMPARISON.md)** | CNN vs RNN vs Transformer 深入对比 |
| **[transformer/ARCHITECTURE.md](transformer/ARCHITECTURE.md)** | Transformer 调用逻辑详解 |
| **[lora/README.md](lora/README.md)** | LoRA 原理与使用说明 |

---

## 🏷️ License

MIT License - 自由使用和修改

---

## 🙏 致谢

- [PyTorch](https://pytorch.org/)
- [Hugging Face](https://huggingface.co/)
- [Andrej Karpathy](https://www.youtube.com/watch?v=VMj-3S1dg0)
- [MLflow](https://mlflow.org/)

---

## 🌟 Star History

如果这个项目对你有帮助，请给个 Star ⭐

[![GitHub](https://img.shields.io/badge/github-actions%20by%20actions-1-success?style=flat-square)](https://github.com/forrestIsRunning/dl-from-scratch)
