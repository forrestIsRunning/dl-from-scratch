# 深度学习从零实现

> CNN、RNN/LSTM、Transformer 与 CUDA 编程的完整示例集合

![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 项目简介

本项目通过**代码实现 + 详细文档**，帮助深入理解深度学习三大模型家族：**CNN、RNN、Transformer**，以及 **CUDA 并行编程**基础。

## 目录

```
dl-from-scratch/
├── docs/                           # 文档目录
│   └── MODEL_COMPARISON.md       # 模型对比详解
│
├── cnn/                            # CNN (卷积神经网络) 示例
│   ├── model.py                   # CNN/LeNet-5 模型定义
│   ├── train.py                   # MNIST 训练脚本
│   └── README.md                  # 使用说明
│
├── rnn/                            # RNN/LSTM 示例
│   ├── model.py                   # RNN/LSTM/GRU 模型
│   ├── train.py                   # 序列预测训练
│   └── README.md                  # 使用说明
│
├── transformer/                    # Transformer 示例
│   ├── config.py                  # 超参数配置
│   ├── data.py                    # 数据加载
│   ├── model.py                   # Transformer 模型定义
│   ├── train.py                   # 训练脚本
│   ├── inference.py                # 推理脚本
│   ├── main.py                    # CLI 入口
│   ├── ARCHITECTURE.md            # 调用逻辑详解
│   └── README.md                  # 使用说明
│
├── cuda/                           # CUDA 编程示例
│   ├── tutorial.py                # CUDA 基础教程
│   └── README.md                  # 使用说明
│
├── mlops-example/                  # MLOps 示例
│   ├── mlflow_tracking_example.py # MLflow 端到端示例
│   ├── Dockerfile.mlflow          # 容器化定义
│   ├── docker-compose.mlflow.yml  # 服务编排
│   └── README.md                # 使用说明
│
└── MLOPS_INTERVIEW_GUIDE.md       # MLOps 面试指南
```

## 快速开始

### 环境配置

```bash
# 使用 uv (推荐)
pip install uv
uv sync

# 或使用 pip
pip install torch numpy tqdm matplotlib pillow requests mlflow scikit-learn
```

### 运行示例

```bash
# CNN 示例 (MNIST 分类)
cd cnn && python train.py --model cnn --epochs 5

# RNN/LSTM 示例 (序列预测)
cd rnn && python train.py --model lstm --epochs 10

# Transformer 示例 (莎士比亚文本生成)
cd transformer && python main.py train

# CUDA 编程教程
cd cuda && python tutorial.py
```

## 子项目说明

| 目录 | 说明 | 参数量 |
|------|------|---------|
| **cnn/** | 卷积神经网络 | ~28K-60K |
| **rnn/** | 循环神经网络 | 可变 |
| **transformer/** | Transformer 实现 | ~4.7M |
| **cuda/** | CUDA 编程基础 | - |
| **mlops-example/** | MLOps 完整流程 | - |

## 文档

- [模型对比详解](docs/MODEL_COMPARISON.md) - CNN vs RNN vs Transformer 对比
- [MLOps 面试指南](MLOPS_INTERVIEW_GUIDE.md) - 完整面试准备指南

## 模型对比速查

| 特性 | CNN | RNN/LSTM | Transformer |
|------|-----|----------|-------------|
| 适用场景 | 图像 | 时序数据 | 文本/序列 |
| 并行度 | 高 | 低 | 极高 |
| 长距离建模 | 无 | 弱 | 强 |
| 参数量 | 中 | 中 | 高 |
| 推理速度 | 快 | 慢 | 快 |

## 学习路径

```
1. CNN 基础 → 理解局部连接和权重共享
   ↓
2. RNN/LSTM → 理解序列建模和梯度问题
   ↓
3. Transformer → 理解注意力机制和并行计算
   ↓
4. CUDA 编程 → 理解 GPU 并行和内存管理
   ↓
5. MLOps → 理解实验跟踪、部署、监控
```

## 环境要求

- Python >= 3.12
- PyTorch >= 2.0.0
- CUDA (可选，用于 GPU 加速)

## License

MIT License - 自由使用和修改

## 致谢

- [PyTorch](https://pytorch.org/)
- [MLflow](https://mlflow.org/)
- [Andrej Karpathy](https://www.youtube.com/watch?v=VMj-3S1dg0)
