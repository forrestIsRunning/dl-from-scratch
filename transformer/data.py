"""
数据加载和处理模块

实现功能：
- 加载内置莎士比亚数据集
- 字符级 Tokenization
- 创建训练/验证 batches
"""

import os
import torch
import requests
from typing import Tuple


# =============================================================================
# 内置莎士比亚数据集
# =============================================================================

# 莎士比亚文本的 URL（Andrej Karpathy 的 char-rnn 项目）
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

# 数据目录
DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "input.txt")


def download_shakespeare() -> str:
    """
    下载莎士比亚数据集

    Returns:
        数据文件路径
    """
    # 创建数据目录
    os.makedirs(DATA_DIR, exist_ok=True)

    # 如果文件已存在，直接返回
    if os.path.exists(DATA_FILE):
        print(f"数据文件已存在: {DATA_FILE}")
        return DATA_FILE

    # 下载数据
    print(f"正在下载莎士比亚数据集...")
    response = requests.get(SHAKESPEARE_URL, timeout=30)
    response.raise_for_status()

    # 保存到文件
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        f.write(response.text)

    print(f"下载完成，保存到: {DATA_FILE}")
    return DATA_FILE


def load_text() -> str:
    """
    加载文本数据

    Returns:
        完整的文本字符串
    """
    file_path = download_shakespeare()

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"数据加载完成，共 {len(text)} 个字符")
    return text


# =============================================================================
# 字符级 Tokenization
# =============================================================================

class CharTokenizer:
    """
    字符级 Tokenizer

    将每个字符映射为一个唯一的整数 ID。
    这是最简单的 tokenization 方式，适合学习和快速验证。
    """

    def __init__(self, text: str):
        """
        初始化 Tokenizer

        Args:
            text: 训练文本，用于构建词汇表
        """
        # 获取所有唯一字符（这就是我们的词汇表）
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)

        # 创建字符到整数的映射
        self.char_to_int = {ch: i for i, ch in enumerate(self.chars)}

        # 创建整数到字符的映射
        self.int_to_char = {i: ch for i, ch in enumerate(self.chars)}

        print(f"词汇表大小: {self.vocab_size}")
        print(f"词汇表: {''.join(self.chars)}")

    def encode(self, text: str) -> list:
        """
        将文本编码为整数序列

        Args:
            text: 输入文本

        Returns:
            整数列表
        """
        return [self.char_to_int[ch] for ch in text]

    def decode(self, indices: list) -> str:
        """
        将整数序列解码为文本

        Args:
            indices: 整数列表

        Returns:
            解码后的文本
        """
        return "".join([self.int_to_char[i] for i in indices])


# =============================================================================
# 数据集类
# =============================================================================

class TextDataset:
    """
    文本数据集

    处理文本数据并生成训练 batches。
    """

    def __init__(self, text: str, tokenizer: CharTokenizer, context_length: int):
        """
        初始化数据集

        Args:
            text: 完整文本
            tokenizer: Tokenizer 实例
            context_length: 上下文长度（序列长度）
        """
        self.tokenizer = tokenizer
        self.context_length = context_length

        # 将整个文本编码为整数
        self.data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

        print(f"数据集大小: {len(self.data)} tokens")

    def get_batch(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取一个训练 batch

        训练数据的形式是: 给定前 n 个 token，预测第 n+1 个 token。
        因此每个样本包含：
        - x: 输入序列 [batch_size, context_length]
        - y: 目标序列 [batch_size, context_length]（x 向右移一位）

        Args:
            batch_size: batch 大小
            device: 计算设备

        Returns:
            (输入序列, 目标序列)
        """
        # 随机选择起始位置
        # 我们需要确保有足够的数据来构建 batch
        max_start_idx = len(self.data) - self.context_length - 1
        start_indices = torch.randint(0, max_start_idx, (batch_size,))

        # 构建 batch
        x = torch.stack([self.data[i:i + self.context_length] for i in start_indices])
        y = torch.stack([self.data[i + 1:i + self.context_length + 1] for i in start_indices])

        # 移动到指定设备
        x, y = x.to(device), y.to(device)

        return x, y

    def split(self, split_ratio: float = 0.9) -> Tuple["TextDataset", "TextDataset"]:
        """
        划分训练集和验证集

        Args:
            split_ratio: 训练集比例

        Returns:
            (训练集, 验证集)
        """
        # 计算划分点
        split_idx = int(split_ratio * len(self.data))

        # 划分数据
        train_data = self.data[:split_idx]
        val_data = self.data[split_idx:]

        # 创建新的数据集
        train_dataset = TextDataset.__new__(TextDataset)
        train_dataset.tokenizer = self.tokenizer
        train_dataset.context_length = self.context_length
        train_dataset.data = train_data

        val_dataset = TextDataset.__new__(TextDataset)
        val_dataset.tokenizer = self.tokenizer
        val_dataset.context_length = self.context_length
        val_dataset.data = val_data

        print(f"训练集大小: {len(train_data)} tokens")
        print(f"验证集大小: {len(val_data)} tokens")

        return train_dataset, val_dataset


# =============================================================================
# 数据加载函数
# =============================================================================

def load_data(context_length: int, device: torch.device) -> Tuple[TextDataset, TextDataset, CharTokenizer]:
    """
    加载并预处理数据

    这是数据加载的主入口函数。

    Args:
        context_length: 上下文长度
        device: 计算设备

    Returns:
        (训练集, 验证集, tokenizer)
    """
    # 1. 加载文本
    text = load_text()

    # 2. 创建 tokenizer
    tokenizer = CharTokenizer(text)

    # 3. 创建数据集
    full_dataset = TextDataset(text, tokenizer, context_length)

    # 4. 划分训练集和验证集
    train_dataset, val_dataset = full_dataset.split(split_ratio=0.9)

    return train_dataset, val_dataset, tokenizer


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    from config import get_device, CONTEXT_LENGTH

    # 加载数据
    train_dataset, val_dataset, tokenizer = load_data(CONTEXT_LENGTH, get_device())

    # 测试编码/解码
    test_text = "Hello, world!"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"\n编码测试:")
    print(f"原文: {test_text}")
    print(f"编码: {encoded}")
    print(f"解码: {decoded}")

    # 测试获取 batch
    x, y = train_dataset.get_batch(2, get_device())
    print(f"\nBatch 测试:")
    print(f"x shape: {x.shape}")
    print(f"y shape: {y.shape}")
    print(f"x:\n{x}")
    print(f"y:\n{y}")
