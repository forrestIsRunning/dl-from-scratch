"""
推理脚本

实现功能：
- 加载训练好的模型
- 文本生成（支持 temperature 和 top-k 采样）
- 交互式生成模式
"""

import torch
from train import load_checkpoint
from data import load_data, CharTokenizer
from config import DEVICE, CONTEXT_LENGTH, TEMPERATURE, TOP_K, MAX_NEW_TOKENS, get_device


# =============================================================================
# 生成函数
# =============================================================================

@torch.no_grad()
def generate_text(
    model,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    top_k: int = TOP_K,
) -> str:
    """
    使用模型生成文本

    Args:
        model: 训练好的模型
        tokenizer: Tokenizer
        prompt: 提示文本（生成起点）
        max_new_tokens: 最多生成多少个新 token
        temperature: 温度参数（越小越确定，越大越随机）
        top_k: Top-K 采样（只从概率最高的 k 个词中采样）

    Returns:
        生成的文本
    """
    model.eval()

    # 将 prompt 编码为 token IDs
    idx = tokenizer.encode(prompt)
    idx = torch.tensor([idx], dtype=torch.long).to(DEVICE)  # [1, seq_len]

    # 生成
    generated_idx = model.generate(idx, max_new_tokens, temperature, top_k)

    # 解码为文本
    generated_text = tokenizer.decode(generated_idx[0].tolist())

    return generated_text


# =============================================================================
# 交互式生成模式
# =============================================================================

def interactive_mode(model, tokenizer: CharTokenizer):
    """
    交互式生成模式

    用户可以输入提示文本，模型会持续生成。

    Args:
        model: 训练好的模型
        tokenizer: Tokenizer
    """
    print("\n" + "=" * 60)
    print("交互式生成模式")
    print("=" * 60)
    print("\n命令:")
    print("  - 输入文本作为提示，按回车生成")
    print("  - 输入 'temp <值>' 设置温度（如: temp 0.7）")
    print("  - 输入 'topk <值>' 设置 top-k（如: topk 50）")
    print("  - 输入 'len <值>' 设置生成长度（如: len 500）")
    print("  - 输入 'quit' 或 'exit' 退出")
    print("\n" + "-" * 60)

    # 默认参数
    temperature = TEMPERATURE
    top_k = TOP_K
    max_new_tokens = MAX_NEW_TOKENS

    while True:
        print(f"\n当前参数: temp={temperature}, topk={top_k}, len={max_new_tokens}")
        user_input = input("请输入提示文本 > ").strip()

        # 处理命令
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("退出交互模式")
            break

        if user_input.lower().startswith('temp '):
            try:
                temperature = float(user_input.split()[1])
                print(f"温度已设置为: {temperature}")
            except:
                print("无效的温度值")
            continue

        if user_input.lower().startswith('topk '):
            try:
                top_k = int(user_input.split()[1])
                print(f"Top-K 已设置为: {top_k}")
            except:
                print("无效的 top-k 值")
            continue

        if user_input.lower().startswith('len '):
            try:
                max_new_tokens = int(user_input.split()[1])
                print(f"生成长度已设置为: {max_new_tokens}")
            except:
                print("无效的长度值")
            continue

        # 空输入则跳过
        if not user_input:
            continue

        # 生成文本
        print("\n生成中...")
        try:
            generated = generate_text(
                model,
                tokenizer,
                user_input,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
            print("\n" + "=" * 60)
            print(generated)
            print("=" * 60)
        except Exception as e:
            print(f"生成失败: {e}")


# =============================================================================
# 单次生成
# =============================================================================

def single_generation(model, tokenizer: CharTokenizer, checkpoint_name: str):
    """
    单次生成模式

    Args:
        model: 训练好的模型
        tokenizer: Tokenizer
        checkpoint_name: 检查点名称
    """
    print("\n" + "=" * 60)
    print("单次生成模式")
    print("=" * 60)

    # 获取参数
    print("\n请输入生成参数（直接回车使用默认值）:")

    prompt = input(f"\n提示文本 (默认: 'First Citizen'): ") or "First Citizen"
    max_tokens_input = input(f"最大生成长度 (默认: {MAX_NEW_TOKENS}): ")
    max_new_tokens = int(max_tokens_input) if max_tokens_input else MAX_NEW_TOKENS

    temp_input = input(f"温度 Temperature (默认: {TEMPERATURE}): ")
    temperature = float(temp_input) if temp_input else TEMPERATURE

    topk_input = input(f"Top-K (默认: {TOP_K}, 输入 0 禁用): ")
    top_k = int(topk_input) if topk_input else TOP_K
    if top_k == 0:
        top_k = None

    # 生成
    print("\n生成中...")
    generated = generate_text(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )

    print("\n" + "=" * 60)
    print("生成结果:")
    print("=" * 60)
    print(generated)
    print("=" * 60)


# =============================================================================
# 主程序
# =============================================================================

def main():
    """
    主函数
    """
    print("=" * 60)
    print("Transformer 推理")
    print("=" * 60)

    # 1. 加载数据（获取 tokenizer）
    print("\n[1/4] 加载数据...")
    _, _, tokenizer = load_data(CONTEXT_LENGTH, DEVICE)
    vocab_size = tokenizer.vocab_size
    print(f"词汇表大小: {vocab_size}")

    # 2. 加载模型
    print("\n[2/4] 加载模型...")
    checkpoint_file = input("请输入检查点文件名 (默认: best_model.pt): ") or "best_model.pt"

    try:
        model = load_checkpoint(checkpoint_file, DEVICE)
    except FileNotFoundError as e:
        print(f"\n错误: {e}")
        print("\n可用的检查点文件:")

        # 列出可用的检查点
        import os
        checkpoint_dir = "checkpoints"
        if os.path.exists(checkpoint_dir):
            files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
            if files:
                for f in files:
                    print(f"  - {f}")
            else:
                print("  (没有找到检查点文件)")
        return

    # 3. 选择模式
    print("\n[3/4] 选择模式:")
    print("  1. 单次生成")
    print("  2. 交互式生成")

    mode = input("\n请选择 (1 或 2，默认: 1): ") or "1"

    # 4. 运行
    print("\n[4/4] 开始生成...\n")

    if mode == "2":
        interactive_mode(model, tokenizer)
    else:
        single_generation(model, tokenizer, checkpoint_file)

    print("\n程序结束。")


if __name__ == "__main__":
    main()
