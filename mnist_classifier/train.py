"""
MNIST 手写数字分类器 - 训练脚本

多分类任务核心:
1. Softmax: 将线性输出转为概率分布 (CrossEntropyLoss 内置)
2. CrossEntropyLoss: 多分类交叉熵损失
3. 整数标签: 无需 One-hot 编码

【输入】图像张量: shape=(N, 1, 28, 28), 像素值 [0, 1]
【输出】类别预测: shape=(N,), 值为 0-9 的整数
"""
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from model import MNISTClassifier


def load_mnist(batch_size=64):
    """
    加载 MNIST 数据集

    预处理:
    1. ToTensor: PIL 图像 → Tensor, 像素值归一化到 [0, 1]
    2. Normalize: 标准化 (使用 MNIST 全局均值 0.1307 和标准差 0.3081)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 数据目录
    data_dir = os.path.join(os.path.dirname(__file__), 'data')

    # 训练集
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 测试集
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train(model, device, train_loader, optimizer, epoch, criterion):
    """训练一个 epoch"""
    model.train()
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # 前向传播
        output = model(data)
        loss = criterion(output, target)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 300 == 0:
            avg_loss = running_loss / (batch_idx + 1)
            print(f'  Batch {batch_idx:4d}/{len(train_loader)}: Loss = {avg_loss:.4f}')

    return running_loss / len(train_loader)


def test(model, device, test_loader, criterion):
    """测试模型"""
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():  # 关闭自动求导
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target).item()

            # 获取预测类别: torch.max 返回 (最大值, 索引)
            _, predicted = torch.max(output.data, dim=1)
            correct += (predicted == target).sum().item()

    avg_loss = test_loss / len(test_loader)
    accuracy = 100.0 * correct / len(test_loader.dataset)

    return avg_loss, accuracy


def main():
    print("=" * 60)
    print("MNIST 手写数字分类器 (Softmax + CrossEntropyLoss)")
    print("=" * 60)

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n【设备】{device}")

    # 加载数据
    print("\n【加载数据】")
    train_loader, test_loader = load_mnist(batch_size=64)
    print(f"  训练集: {len(train_loader.dataset)} 样本")
    print(f"  测试集: {len(test_loader.dataset)} 样本")

    # 模型
    model = MNISTClassifier().to(device)
    print(f"\n【模型结构】784 → 512 → 256 → 128 → 64 → 10")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 损失函数和优化器
    # CrossEntropyLoss = LogSoftmax + NLLLoss (内置 Softmax)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    # 训练
    print("\n【训练】")
    num_epochs = 10
    train_losses = []
    test_losses = []
    accuracies = []

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        train_loss = train(model, device, train_loader, optimizer, epoch, criterion)
        test_loss, accuracy = test(model, device, test_loader, criterion)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        accuracies.append(accuracy)

        print(f"  训练损失: {train_loss:.4f} | 测试损失: {test_loss:.4f} | 准确率: {accuracy:.2f}%")

    # 最终评估
    print(f"\n【最终结果】")
    print(f"  测试集准确率: {accuracies[-1]:.2f}%")

    # 预测示例
    print(f"\n【预测示例】")
    model.eval()
    with torch.no_grad():
        data, target = next(iter(test_loader))
        data, target = data.to(device), target.to(device)
        output = model(data[:5])
        _, predicted = torch.max(output.data, dim=1)

        for i in range(5):
            status = "OK" if predicted[i].item() == target[i].item() else "XX"
            print(f"  样本{i+1}: 预测={predicted[i].item()}, 实际={target[i].item()} [{status}]")

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 损失曲线
    axes[0].plot(range(1, num_epochs+1), train_losses, 'b-', label='Train Loss')
    axes[0].plot(range(1, num_epochs+1), test_losses, 'r-', label='Test Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Progress')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 准确率曲线
    axes[1].plot(range(1, num_epochs+1), accuracies, 'g-', marker='o')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Test Accuracy')
    axes[1].set_ylim(90, 100)
    axes[1].grid(True, alpha=0.3)

    # 网络结构
    dims = [784, 512, 256, 128, 64, 10]
    colors = ['#3498db'] * 4 + ['#e74c3c', '#9b59b6']
    axes[2].bar(range(len(dims)), dims, color=colors)
    axes[2].set_xticks(range(len(dims)))
    axes[2].set_xticklabels(['Input\n(784)', 'H1\n(512)', 'H2\n(256)', 'H3\n(128)', 'H4\n(64)', 'Output\n(10)'])
    axes[2].set_ylabel('Dimension')
    axes[2].set_title('Network Architecture')
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), 'training_result.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n可视化: {output_path}")


if __name__ == '__main__':
    main()
