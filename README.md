# ML Projects

一个包含 Transformer 实现和 MLOps 示例的机器学习项目集合。

## 项目结构

```
.
├── transformer/           # Transformer From Scratch 实现
│   ├── config.py        # 超参数配置
│   ├── data.py          # 数据加载和处理
│   ├── model.py         # Transformer 模型定义
│   ├── train.py         # 训练脚本
│   ├── inference.py     # 推理脚本
│   ├── main.py          # 主入口
│   └── ARCHITECTURE.md  # 调用逻辑详解
│
└── mlops-example/        # MLflow 端到端示例
    ├── mlflow_tracking_example.py  # 主程序
    ├── Dockerfile.mlflow           # 容器化定义
    ├── docker-compose.mlflow.yml     # 服务编排
    ├── mlflow-requirements.txt       # Python 依赖
    └── MLFLOW_EXAMPLE_README.md    # 使用说明
```

## 子项目

### 1. Transformer From Scratch

从零实现一个完整的 Transformer Decoder-only 模型（类似 GPT）。

**特点**：
- 完整实现（数据加载、模型定义、训练和推理）
- 详细中文注释
- 支持 Mac M系列芯片 MPS 加速

**快速开始**：
```bash
cd transformer
python main.py train
```

详见 [transformer/README.md](transformer/README.md)

### 2. MLOps Example

使用 MLflow 进行模型训练、跟踪和部署的端到端示例。

**特点**：
- 完整的实验跟踪
- 模型注册和版本管理
- 容器化部署
- REST API 服务

**快速开始**：
```bash
cd mlops-example
docker-compose -f docker-compose.mlflow.yml up -d
python mlflow_tracking_example.py train
```

详见 [mlops-example/README.md](mlops-example/README.md)

## 文档

- **[MLOPS_INTERVIEW_GUIDE.md](MLOPS_INTERVIEW_GUIDE.md)** - MLOps 面试完整指南

## 环境配置

```bash
# 安装依赖
pip install torch numpy tqdm requests mlflow scikit-learn

# 或使用 uv
uv sync
```

## License

MIT License
