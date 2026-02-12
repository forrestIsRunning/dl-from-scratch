"""
RNN/LSTM 训练示例

使用合成数据训练序列预测模型。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model import RNN, LSTM, GRU, OptimizedLSTM


def generate_data(num_samples=1000, seq_len=20, input_size=16):
    """
    生成合成序列数据

    任务: 给定序列，预测下一个值
    """
    X = []
    y = []

    for _ in range(num_samples):
        # 随机序列
        seq = torch.randn(seq_len, input_size)

        # 目标: 序列最后一个值的符号
        target = (seq[-1, 0] > 0).long()

        X.append(seq)
        y.append(target)

    X = torch.stack(X)  # [num_samples, seq_len, input_size]
    y = torch.stack(y)  # [num_samples]

    return X, y


def train_epoch(model, device, train_loader, optimizer):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="训练")
    for X, y in pbar:
        X, y = X.to(device), y.to(device)

        # 前向传播
        optimizer.zero_grad()

        # 取最后一个时间步的输出
        output = model(X.transpose(0, 1))  # [batch, seq, input_size] -> [seq, batch, input_size]
        output = output[-1]  # [batch, output_size]

        loss = nn.CrossEntropyLoss()(output, y)

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(y).sum().item()
        total += y.size(0)

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(train_loader), 100. * correct / total


def test(model, device, test_loader):
    """测试模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)

            output = model(X.transpose(0, 1))
            output = output[-1]

            total_loss += nn.CrossEntropyLoss()(output, y).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)

    return total_loss / len(test_loader), 100. * correct / total


def train(
    model_type="lstm",
    seq_len=20,
    input_size=16,
    hidden_size=32,
    epochs=10,
    batch_size=64,
    lr=0.001,
    device=None
):
    """训练函数"""
    print(f"\n{'='*60}")
    print(f"训练 {model_type.upper()} 序列预测")
    print(f"{'='*60}\n")

    # 设备
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 数据
    X_train, y_train = generate_data(1000, seq_len, input_size)
    X_test, y_test = generate_data(200, seq_len, input_size)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    # 模型
    if model_type == "rnn":
        model = RNN(input_size, hidden_size, 2).to(device)
    elif model_type == "lstm":
        model = LSTM(input_size, hidden_size, 2).to(device)
    elif model_type == "gru":
        model = GRU(input_size, hidden_size, 2).to(device)
    else:  # optimized_lstm
        model = OptimizedLSTM(input_size, hidden_size, 2).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {num_params:,}")

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练循环
    best_acc = 0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, device, train_loader, optimizer)
        test_loss, test_acc = test(model, device, test_loader)

        print(f"Epoch {epoch}: 训练损失={train_loss:.4f}, 训练准确率={train_acc:.2f}%, "
              f"测试准确率={test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f"rnn_{model_type}_best.pth")

    print(f"\n训练完成！最佳测试准确率: {best_acc:.2f}%")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="lstm",
                       choices=["rnn", "lstm", "gru", "optimized_lstm"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    train(
        model_type=args.model,
        epochs=args.epochs,
        hidden_size=args.hidden_size,
        lr=args.lr
    )
