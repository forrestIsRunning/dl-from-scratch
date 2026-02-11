#!/usr/bin/env python3
"""
Transformer From Scratch - 主入口文件

命令:
    python main.py train      - 开始训练
    python main.py generate   - 运行推理
    python main.py help       - 显示帮助信息
"""

import sys


def print_banner():
    """打印欢迎横幅"""
    print("""
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   Transformer From Scratch - 手写 GPT 风格模型             ║
║                                                           ║
║   一个完整的教学项目，从零实现 Transformer 模型           ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
    """)


def print_help():
    """打印帮助信息"""
    print("""
用法: python main.py <命令>

可用命令:

    train        开始训练模型
                 - 使用内置莎士比亚数据集
                 - 训练过程中会定期保存检查点
                 - 检查点保存在 checkpoints/ 目录

    generate     运行推理生成文本
                 - 加载训练好的模型
                 - 支持交互式和单次生成模式
                 - 可调整 temperature 和 top-k 参数

    help         显示此帮助信息

示例:

    # 开始训练
    python main.py train

    # 生成文本
    python main.py generate

项目结构:

    ├── config.py      # 超参数配置
    ├── data.py        # 数据加载和处理
    ├── model.py       # Transformer 模型定义
    ├── train.py       # 训练脚本
    ├── inference.py   # 推理脚本
    └── main.py        # 本文件（主入口）

更多信息请查看 README.md
    """)


def cmd_train():
    """训练命令"""
    print_banner()
    print("命令: 训练模型")
    print("=" * 60)

    from train import train
    train()


def cmd_generate():
    """推理命令"""
    print_banner()
    print("命令: 生成文本")
    print("=" * 60)

    from inference import main as inference_main
    inference_main()


def main():
    """主函数"""
    # 获取命令
    if len(sys.argv) < 2:
        cmd = "help"
    else:
        cmd = sys.argv[1].lower()

    # 执行命令
    if cmd in ["train", "t"]:
        cmd_train()
    elif cmd in ["generate", "gen", "g"]:
        cmd_generate()
    elif cmd in ["help", "h", "-h", "--help"]:
        print_help()
    else:
        print(f"未知命令: {cmd}")
        print("\n运行 'python main.py help' 查看可用命令")
        sys.exit(1)


if __name__ == "__main__":
    main()
