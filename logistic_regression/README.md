# Logistic Regression (逻辑回归)

逻辑回归是一种经典的二分类算法，广泛应用于各种分类场景。

## 数学原理

逻辑回归通过线性组合 + Sigmoid 函数将输入映射到 [0, 1] 区间的概率值：

```
y = sigmoid(wx + b)
```

其中 Sigmoid 函数定义为：

```
sigmoid(x) = 1 / (1 + exp(-x))
```

## 损失函数

使用二分类交叉熵损失 (Binary Cross Entropy Loss)：

```
Loss = -[y * log(y_pred) + (1 - y) * log(1 - y_pred)]
```

## 运行示例

```bash
cd logistic_regression
python train.py
```

## 训练数据

| 学习小时数 | 是否及格 |
|-----------|---------|
| 1.0       | 0 (不及格) |
| 2.0       | 0 (不及格) |
| 3.0       | 1 (及格) |

## 文件结构

- `model.py` - 逻辑回归模型定义
- `train.py` - 训练脚本（含可视化）
