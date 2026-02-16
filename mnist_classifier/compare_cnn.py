"""
CNN 模型配置比较实验

训练不同配置的 CNN 模型，比较其性能:
- 准确率
- 训练时间
- 参数量
- 收敛速度
"""
import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from cnn_model import MNISTCNN, get_config, list_configs, CNNConfig


def load_mnist(batch_size=64):
    """加载 MNIST 数据集"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    data_dir = os.path.join(os.path.dirname(__file__), 'data')

    train_dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train_epoch(model, device, train_loader, optimizer, criterion):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        _, predicted = output.max(1)
        correct += predicted.eq(target).sum().item()
        total += target.size(0)

    return total_loss / total, correct / total


def evaluate(model, device, test_loader, criterion):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item() * data.size(0)
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()
            total += target.size(0)

    return total_loss / total, correct / total


def train_model(config: CNNConfig, device, train_loader, test_loader,
                num_epochs: int = 10, lr: float = 0.001):
    """训练单个模型"""
    model = MNISTCNN(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'epoch_time': []
    }

    print(f"\n训练 {config.name}...")
    print(f"  参数量: {model.count_parameters():,}")

    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_epoch(
            model, device, train_loader, optimizer, criterion
        )
        test_loss, test_acc = evaluate(
            model, device, test_loader, criterion
        )

        epoch_time = time.time() - epoch_start

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['epoch_time'].append(epoch_time)

        print(f"  Epoch {epoch:2d}: "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc*100:.2f}%, "
              f"Test Loss={test_loss:.4f}, Test Acc={test_acc*100:.2f}%, "
              f"Time={epoch_time:.1f}s")

    total_time = time.time() - start_time

    return {
        'config': config,
        'model': model,
        'history': history,
        'total_time': total_time,
        'final_test_acc': history['test_acc'][-1],
        'params': model.count_parameters()
    }


def run_comparison(config_names: list, num_epochs: int = 10, batch_size: int = 64):
    """运行多个配置的比较实验"""
    print("=" * 70)
    print("MNIST CNN 配置比较实验")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n设备: {device}")

    # 加载数据
    print("\n加载数据...")
    train_loader, test_loader = load_mnist(batch_size)
    print(f"  训练集: {len(train_loader.dataset)} 样本")
    print(f"  测试集: {len(test_loader.dataset)} 样本")

    # 训练所有配置
    results = []
    for config_name in config_names:
        config = get_config(config_name)
        result = train_model(
            config, device, train_loader, test_loader, num_epochs
        )
        results.append(result)

    # 打印汇总
    print("\n" + "=" * 70)
    print("结果汇总")
    print("=" * 70)
    print(f"\n{'配置':<25} {'参数量':>12} {'测试准确率':>12} {'训练时间':>12}")
    print("-" * 70)
    for r in results:
        print(f"{r['config'].name:<25} {r['params']:>12,} "
              f"{r['final_test_acc']*100:>11.2f}% {r['total_time']:>11.1f}s")

    # 生成可视化
    plot_results(results, num_epochs)

    # 保存最佳模型
    best = max(results, key=lambda x: x['final_test_acc'])
    model_path = os.path.join(os.path.dirname(__file__), 'cnn_best_model.pth')
    torch.save(best['model'].state_dict(), model_path)
    print(f"\n最佳模型 ({best['config'].name}) 保存至: {model_path}")

    return results


def plot_results(results: list, num_epochs: int):
    """生成比较图表"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    # 1. 测试准确率曲线
    ax1 = axes[0, 0]
    for i, r in enumerate(results):
        ax1.plot(range(1, num_epochs + 1),
                [acc * 100 for acc in r['history']['test_acc']],
                marker='o', markersize=3,
                color=colors[i], label=r['config'].name)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('Test Accuracy Comparison')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(95, 100)

    # 2. 训练损失曲线
    ax2 = axes[0, 1]
    for i, r in enumerate(results):
        ax2.plot(range(1, num_epochs + 1),
                r['history']['train_loss'],
                color=colors[i], label=r['config'].name)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Training Loss')
    ax2.set_title('Training Loss Comparison')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3. 参数量 vs 准确率
    ax3 = axes[1, 0]
    params = [r['params'] / 1000 for r in results]  # K parameters
    accs = [r['final_test_acc'] * 100 for r in results]
    names = [r['config'].name for r in results]
    bars = ax3.barh(names, accs, color=colors)
    ax3.set_xlabel('Test Accuracy (%)')
    ax3.set_title('Final Test Accuracy by Configuration')
    ax3.set_xlim(95, 100)
    for bar, acc in zip(bars, accs):
        ax3.text(acc + 0.05, bar.get_y() + bar.get_height()/2,
                f'{acc:.2f}%', va='center', fontsize=9)

    # 4. 参数量 vs 训练时间
    ax4 = axes[1, 1]
    times = [r['total_time'] for r in results]
    scatter = ax4.scatter(params, times, c=accs, cmap='RdYlGn',
                         s=200, edgecolors='black', linewidths=1)
    for i, name in enumerate(names):
        ax4.annotate(name, (params[i], times[i]),
                    fontsize=8, ha='center', va='bottom')
    ax4.set_xlabel('Parameters (K)')
    ax4.set_ylabel('Training Time (s)')
    ax4.set_title('Parameters vs Training Time')
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Accuracy (%)')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), 'cnn_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n可视化保存至: {output_path}")


def main():
    # 选择要比较的配置
    # 可以选择全部配置或子集
    all_configs = list_configs()

    print("可用配置:", ", ".join(all_configs))

    # 默认比较这几种代表性配置
    compare_configs = [
        "small",           # 小模型
        "medium",          # 中等模型
        "large",           # 大模型
        "medium_bn",       # 带 BatchNorm
        "medium_dropout",  # 带 Dropout
        "medium_bn_dropout",  # BatchNorm + Dropout
    ]

    # 如果命令行指定了配置，使用命令行的
    if len(sys.argv) > 1:
        compare_configs = sys.argv[1:]

    run_comparison(compare_configs, num_epochs=10)


if __name__ == "__main__":
    main()
