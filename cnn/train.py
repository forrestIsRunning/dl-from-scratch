"""
CNN 训练示例

使用 MNIST 数据集训练 CNN 分类器。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from model import CNN, LeNet5


def get_dataloader(batch_size=64):
    """获取 MNIST 数据加载器"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    return train_loader, test_loader


def train_epoch(model, device, train_loader, optimizer, epoch):
    """训练一个 epoch"""
    model.train()
    train_loss = 0
    correct = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        # 前向传播
        optimizer.zero_grad()
        output = model(data)

        # 计算损失
        loss = nn.CrossEntropyLoss()(output, target)
        train_loss += loss.item()

        # 计算准确率
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()

        # 反向传播
        loss.backward()
        optimizer.step()

        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / ((batch_idx + 1) * len(data)):.2f}%'
        })

    avg_loss = train_loss / len(train_loader)
    accuracy = 100. * correct / len(train_loader.dataset)

    return avg_loss, accuracy


def test(model, device, test_loader):
    """测试模型"""
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += nn.CrossEntropyLoss()(output, target).item()

            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f"\n测试集: 平均损失: {test_loss:.4f}, 准确率: {accuracy:.2f}%")

    return test_loss, accuracy


def train(
    model_name="cnn",
    epochs=5,
    batch_size=64,
    lr=0.01,
    device=None
):
    """训练函数"""
    print(f"\n{'='*60}")
    print(f"训练 {model_name.upper()}")
    print(f"{'='*60}\n")

    # 设备
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 数据
    train_loader, test_loader = get_dataloader(batch_size)

    # 模型
    if model_name == "lenet":
        model = LeNet5().to(device)
    else:
        model = CNN().to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {num_params:,}")

    # 优化器
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # 训练循环
    best_acc = 0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, device, train_loader, optimizer, epoch)
        test_loss, test_acc = test(model, device, test_loader)

        print(f"Epoch {epoch}: 训练损失={train_loss:.4f}, 训练准确率={train_acc:.2f}%, "
              f"测试准确率={test_acc:.2f}%")

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f"cnn_{model_name}_best.pth")
            print(f"→ 保存最佳模型 (准确率: {best_acc:.2f}%)")

    print(f"\n训练完成！最佳测试准确率: {best_acc:.2f}%")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "lenet"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()

    train(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
