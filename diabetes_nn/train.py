"""
糖尿病预测 - 多层神经网络训练脚本

数据集: Pima Indians Diabetes Dataset
- 8个特征: 怀孕次数、血糖、血压、皮肤厚度、胰岛素、BMI、糖尿病家族史、年龄
- 标签: 0=无糖尿病, 1=有糖尿病

【输入】特征矩阵 X: shape=(N, 8)
    每行是一个样本的8个医学指标

【输出】概率向量 Y: shape=(N, 1)
    每行是对应样本的患病概率 (0~1)
"""
import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import DiabetesNN


# ============================================
# 输入特征的8个维度含义:
# 0: Pregnancies        - 怀孕次数
# 1: Glucose            - 血糖 (mg/dL)
# 2: BloodPressure      - 血压 (mm Hg)
# 3: SkinThickness      - 皮肤厚度 (mm)
# 4: Insulin            - 胰岛素 (mu U/ml)
# 5: BMI                - 身体质量指数 (体重kg/身高m²)
# 6: DiabetesPedigreeFunction - 糖尿病家族史指数
# 7: Age                - 年龄 (岁)
#
# 输出: 1维概率值 (0~1)
#   >= 0.5 → 预测有糖尿病
#   < 0.5  → 预测无糖尿病
# ============================================


def load_diabetes_dataset():
    """
    加载 Pima Indians Diabetes Dataset
    使用具有真实统计特征的合成数据（网络下载可能较慢）
    """
    np.random.seed(42)
    N = 768  # 真实数据集样本数

    # 基于真实数据的统计特征生成合成数据
    # 各特征的近似均值和标准差
    means = [3.8, 120.9, 69.1, 20.5, 79.8, 32.0, 0.5, 33.2]
    stds = [3.4, 32.0, 19.4, 16.0, 115.2, 7.9, 0.3, 11.8]

    x_data = np.random.randn(N, 8) * stds + means
    x_data = np.clip(x_data, 0, None)  # 确保非负

    # 生成有意义的标签（基于特征的线性组合 + 噪声）
    weights = np.array([0.3, 0.8, 0.1, 0.05, 0.05, 0.4, 0.5, 0.3])
    scores = (x_data * weights).sum(axis=1) + np.random.randn(N) * 0.5
    threshold = np.percentile(scores, 65)  # 约35%为阳性
    y_data = (scores > threshold).astype(np.float32).reshape(-1, 1)

    # 标准化特征
    x_mean = x_data.mean(axis=0)
    x_std = x_data.std(axis=0)
    x_data = (x_data - x_mean) / x_std

    return torch.FloatTensor(x_data), torch.FloatTensor(y_data)


def main():
    print("=" * 60)
    print("糖尿病预测 - 多层神经网络 (8→6→4→1)")
    print("=" * 60)

    # 加载数据
    x_data, y_data = load_diabetes_dataset()

    print(f"\n【输入 X】: {x_data.shape} - 8个医学特征")
    print(f"  特征0: 怀孕次数")
    print(f"  特征1: 血糖 (Glucose)")
    print(f"  特征2: 血压 (BloodPressure)")
    print(f"  特征3: 皮肤厚度 (SkinThickness)")
    print(f"  特征4: 胰岛素 (Insulin)")
    print(f"  特征5: BMI (身体质量指数)")
    print(f"  特征6: 糖尿病家族史指数")
    print(f"  特征7: 年龄 (Age)")

    print(f"\n【输出 Y】: {y_data.shape} - 患病概率")
    print(f"  - 无糖尿病(0): {int((y_data == 0).sum().item())} 人")
    print(f"  - 有糖尿病(1): {int((y_data == 1).sum().item())} 人")

    # 实例化模型
    model = DiabetesNN()

    print(f"\n【模型结构】维度变换链")
    print(f"  输入 [N,8] --Linear(8,6)--> [N,6] --Linear(6,4)--> [N,4] --Linear(4,1)--> [N,1] 输出")
    print(f"  权重: W1{tuple(model.linear1.weight.shape)}, W2{tuple(model.linear2.weight.shape)}, W3{tuple(model.linear3.weight.shape)}")

    # 损失函数和优化器
    criterion = torch.nn.BCELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    # 训练
    print(f"\n【训练】")
    losses = []
    for epoch in range(2000):
        y_pred = model(x_data)
        loss = criterion(y_pred, y_data)
        losses.append(loss.item())

        if epoch % 400 == 0:
            print(f"  Epoch {epoch:5d}: Loss = {loss.item():.4f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 最终评估
    print(f"\n【评估】")
    with torch.no_grad():
        y_pred = model(x_data)
        predicted = (y_pred >= 0.5).float()
        accuracy = (predicted == y_data).float().mean()
        print(f"  准确率: {accuracy.item()*100:.2f}%")

    # 预测示例
    print(f"\n【预测示例】")
    with torch.no_grad():
        for i in range(5):
            sample = x_data[i:i+1]
            prob = model(sample).item()
            actual = int(y_data[i].item())
            pred = 1 if prob >= 0.5 else 0
            status = "OK" if pred == actual else "XX"
            print(f"  样本{i+1}: 概率={prob:.3f} -> 预测={pred}, 实际={actual} [{status}]")

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(losses, 'b-', linewidth=0.5)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)

    dims = [8, 6, 4, 1]
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    bars = axes[1].bar(['Input\n(8)', 'Hidden1\n(6)', 'Hidden2\n(4)', 'Output\n(1)'], dims, color=colors)
    axes[1].set_ylabel('Dimension')
    axes[1].set_title('Network: 8 -> 6 -> 4 -> 1')
    axes[1].set_ylim(0, 10)
    for bar, dim in zip(bars, dims):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, str(dim), ha='center', fontsize=12)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), 'training_result.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n可视化: {output_path}")


if __name__ == '__main__':
    main()
