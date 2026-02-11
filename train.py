"""
训练脚本

实现功能：
- 加载数据和模型
- 训练循环
- 学习率调度（带 warmup 的余弦退火）
- 梯度裁剪
- 检查点保存和加载
- 验证集评估
- 日志记录
"""

import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from config import (
    DEVICE, BATCH_SIZE, LEARNING_RATE, MAX_STEPS,
    EVAL_INTERVAL, EVAL_STEPS, GRAD_CLIP, SAVE_INTERVAL,
    WARMUP_STEPS, CONTEXT_LENGTH, CHECKPOINT_DIR, get_device, get_model_params
)
from data import load_data
from model import GPTModel


# =============================================================================
# 学习率调度器
# =============================================================================

def get_lr_scheduler(optimizer, warmup_steps: int, max_steps: int):
    """
    创建学习率调度器

    使用带 warmup 的余弦退火调度:
    1. Warmup 阶段: 学习率线性增长
    2. 衰减阶段: 学习率按余弦曲线衰减

    Args:
        optimizer: 优化器
        warmup_steps: Warmup 步数
        max_steps: 总训练步数

    Returns:
        学习率调度器
    """

    def lr_lambda(current_step: int) -> float:
        """
        计算当前步的学习率倍数

        Args:
            current_step: 当前训练步

        Returns:
            学习率倍数
        """
        # Warmup 阶段
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        # 衰减阶段: 余弦退火
        progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265359))).item())

    return LambdaLR(optimizer, lr_lambda)


# =============================================================================
# 评估函数
# =============================================================================

@torch.no_grad()
def evaluate(model: GPTModel, val_dataset, eval_steps: int) -> float:
    """
    在验证集上评估模型

    Args:
        model: 模型
        val_dataset: 验证数据集
        eval_steps: 评估步数

    Returns:
        平均损失
    """
    model.eval()
    total_loss = 0.0

    for _ in range(eval_steps):
        x, y = val_dataset.get_batch(BATCH_SIZE, DEVICE)
        _, loss = model(x, y)
        total_loss += loss.item()

    model.train()
    return total_loss / eval_steps


# =============================================================================
# 训练函数
# =============================================================================

def train():
    """
    主训练函数
    """
    print("=" * 60)
    print("开始训练")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/6] 加载数据...")
    train_dataset, val_dataset, tokenizer = load_data(CONTEXT_LENGTH, DEVICE)
    vocab_size = tokenizer.vocab_size
    print(f"词汇表大小: {vocab_size}")

    # 2. 创建模型
    print("\n[2/6] 创建模型...")
    config = get_model_params(vocab_size)
    model = GPTModel(config).to(DEVICE)
    num_params = model.get_num_params()
    print(f"模型参数量: {num_params:,}")

    # 3. 创建优化器和学习率调度器
    print("\n[3/6] 创建优化器和学习率调度器...")
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_lr_scheduler(optimizer, WARMUP_STEPS, MAX_STEPS)
    print(f"优化器: AdamW (lr={LEARNING_RATE})")
    print(f"学习率调度: Warmup {WARMUP_STEPS} 步 + 余弦退火")

    # 4. 创建检查点目录
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # 5. 训练循环
    print("\n[4/6] 开始训练...")
    print(f"配置: batch_size={BATCH_SIZE}, max_steps={MAX_STEPS}")
    print(f"设备: {DEVICE}")
    print()

    # 训练状态
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    start_time = time.time()

    # 创建进度条
    pbar = tqdm(range(MAX_STEPS), desc="Training")

    for step in pbar:
        # === 训练步骤 ===

        # 1. 获取 batch
        x, y = train_dataset.get_batch(BATCH_SIZE, DEVICE)

        # 2. 前向传播
        logits, loss = model(x, y)
        train_losses.append(loss.item())

        # 3. 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 4. 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        # 5. 更新参数
        optimizer.step()

        # 6. 更新学习率
        scheduler.step()

        # === 记录和评估 ===

        # 更新进度条
        current_lr = scheduler.get_last_lr()[0]
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{current_lr:.2e}'
        })

        # 定期评估
        if (step + 1) % EVAL_INTERVAL == 0 or step == 0:
            val_loss = evaluate(model, val_dataset, EVAL_STEPS)
            val_losses.append(val_loss)

            # 计算训练损失（最近 EVAL_INTERVAL 步的平均）
            avg_train_loss = sum(train_losses[-EVAL_INTERVAL:]) / min(len(train_losses), EVAL_INTERVAL)

            elapsed = time.time() - start_time
            print(f"\n{'='*60}")
            print(f"Step {step + 1}/{MAX_STEPS}")
            print(f"训练损失: {avg_train_loss:.4f}")
            print(f"验证损失: {val_loss:.4f}")
            print(f"学习率: {current_lr:.2e}")
            print(f"已用时间: {elapsed//60:.0f}m{elapsed%60:.0f}s")
            print(f"{'='*60}\n")

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, scheduler, step, val_loss, "best_model.pt")
                print(f"保存最佳模型 (val_loss={val_loss:.4f})")

        # 定期保存检查点
        if (step + 1) % SAVE_INTERVAL == 0:
            save_checkpoint(model, optimizer, scheduler, step, val_loss, f"checkpoint_step_{step+1}.pt")

    # 6. 训练完成
    print("\n[5/6] 训练完成！")
    elapsed = time.time() - start_time
    print(f"总训练时间: {elapsed//60:.0f}m{elapsed%60:.0f}s")
    print(f"最佳验证损失: {best_val_loss:.4f}")

    # 7. 保存最终模型
    print("\n[6/6] 保存最终模型...")
    save_checkpoint(model, optimizer, scheduler, MAX_STEPS, val_losses[-1], "final_model.pt")

    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)


# =============================================================================
# 检查点保存和加载
# =============================================================================

def save_checkpoint(model: GPTModel, optimizer, scheduler, step: int, loss: float, filename: str):
    """
    保存模型检查点

    Args:
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器
        step: 当前训练步
        loss: 当前损失
        filename: 保存文件名
    """
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'config': {
            'd_model': model.d_model,
            'context_length': model.context_length,
            'vocab_size': model.token_embedding.num_embeddings,
        }
    }

    filepath = os.path.join(CHECKPOINT_DIR, filename)
    torch.save(checkpoint, filepath)
    print(f"检查点已保存: {filepath}")


def load_checkpoint(filename: str, device: torch.device) -> GPTModel:
    """
    加载模型检查点

    Args:
        filename: 检查点文件名
        device: 计算设备

    Returns:
        加载好的模型
    """
    filepath = os.path.join(CHECKPOINT_DIR, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"检查点文件不存在: {filepath}")

    checkpoint = torch.load(filepath, map_location=device)

    # 创建模型
    config = checkpoint['config']
    config['device'] = device
    model = GPTModel(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"模型已加载: {filepath}")
    print(f"训练步数: {checkpoint['step']}")
    print(f"损失: {checkpoint['loss']:.4f}")

    return model


# =============================================================================
# 主程序
# =============================================================================

if __name__ == "__main__":
    train()
