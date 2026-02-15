"""
逻辑回归训练脚本

数据集：学习小时数(x) -> 是否及格(y: 0=不及格, 1=及格)
训练目标：学习输入到概率的映射关系
"""
import os
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from model import LogisticRegressionModel


def main():
    # 准备数据集
    x_data = torch.Tensor([[1.0], [2.0], [3.0]])  # 学习小时数
    y_data = torch.Tensor([[0], [0], [1]])        # 及格标签

    # 实例化模型
    model = LogisticRegressionModel()

    # 定义损失函数和优化器
    # BCELoss：二分类交叉熵损失
    criterion = torch.nn.BCELoss(reduction='sum')
    # SGD 优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # 训练循环
    for epoch in range(10000):
        # 前向传播
        y_pred = model(x_data)
        loss = criterion(y_pred, y_data)

        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item():.4f}")

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 模型预测
    print("\n===== Prediction Results =====")
    test_hours = [0.5, 1.5, 2.5, 3.5, 4.0]
    for hour in test_hours:
        x_test = torch.Tensor([[hour]])
        prob = model(x_test).item()
        result = "Pass" if prob >= 0.5 else "Fail"
        print(f"Study {hour} hours -> Pass probability: {prob:.4f} -> {result}")

    # 可视化
    x = np.linspace(0, 10, 200)
    x_t = torch.Tensor(x).view((200, 1))
    y_t = model(x_t).data.numpy()

    plt.figure(figsize=(8, 5))
    plt.plot(x, y_t, 'b-', label='Probability Curve')
    plt.plot([0, 10], [0.5, 0.5], 'r-', label='Threshold (0.5)')
    plt.scatter(x_data.numpy(), y_data.numpy(), c='black', s=100, label='Training Data')
    plt.xlabel('Study Hours', fontsize=12)
    plt.ylabel('Probability of Pass', fontsize=12)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.title('Logistic Regression: Study Hours vs Pass Probability', fontsize=14)

    # Save figure
    output_path = os.path.join(os.path.dirname(__file__), 'logistic_regression.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to: {output_path}")


if __name__ == '__main__':
    main()
