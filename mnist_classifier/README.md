# MNIST 手写数字分类器

基于课件 Lecture 09 实现的多分类神经网络，使用 Softmax + CrossEntropyLoss。

## 核心概念

### Softmax 层
将线性输出转换为概率分布:
```
P(y=i) = exp(z_i) / sum(exp(z_j))
```
- 保证每个类别概率 ≥ 0
- 所有类别概率和 = 1

### CrossEntropyLoss
PyTorch 内置 LogSoftmax + NLLLoss:
```python
criterion = torch.nn.CrossEntropyLoss()
```

**注意**:
- 模型最后一层**无需手动加 Softmax**
- 标签使用**整数类型** (0-9)，无需 One-hot 编码

## 网络结构

```
输入 [N,1,28,28] → flatten → [N,784]
    ↓ Linear(784,512) + ReLU
   [N,512]
    ↓ Linear(512,256) + ReLU
   [N,256]
    ↓ Linear(256,128) + ReLU
   [N,128]
    ↓ Linear(128,64) + ReLU
   [N,64]
    ↓ Linear(64,10)  (无激活)
输出 [N,10] → CrossEntropyLoss → 预测类别
```

## 运行

```bash
cd mnist_classifier
python train.py
```

## 核心要点

| 项目 | 二分类 | 多分类 (MNIST) |
|------|--------|----------------|
| 输出层激活 | Sigmoid | 无 (CrossEntropyLoss 内置) |
| 损失函数 | BCELoss | CrossEntropyLoss |
| 标签格式 | 0/1 浮点 | 0-9 整数 |

## 文件结构

- `model.py` - 网络模型 (784→512→256→128→64→10)
- `train.py` - 训练和测试脚本
- `data/` - MNIST 数据集 (自动下载)
