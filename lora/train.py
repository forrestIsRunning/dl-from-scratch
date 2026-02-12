"""
LLaMA LoRA 微调示例

使用 Hugging Face Transformers 进行 LLaMA 模型的 LoRA 微调。

支持：LLaMA、LLaMA-2、LLaMA-3 等系列

注意：依赖已在根目录 pyproject.toml 中管理
使用前先运行：uv sync
"""

import os
import torch
from dataclasses import dataclass
from typing import Optional, Dict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from transformers import BitsAndBytesConfig  # 用于 4-bit 量化
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training


@dataclass
class LoraTrainingArguments:
    """LoRA 微调参数"""

    # 模型相关
    model_name: str = "huggyllama/llama-7b"  # 或其他 LLaMA 变体
    max_length: int = 128
    cache_dir: str = "./models"

    # LoRA 相关
    lora_r: int = 8  # LoRA 秩（推荐 8-64）
    lora_alpha: int = 16  # 缩放因子（通常 = 2 × r）
    lora_dropout: float = 0.05

    # 训练相关
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    num_train_epochs: int = 3
    learning_rate: float = 3e-4
    warmup_steps: int = 100

    # 量化相关（节省显存）
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    # LoRA 目标模块
    # LLaMA 的 Attention 投影层命名
    target_modules: list = None

    def __post_init__(self):
        if self.target_modules is None:
            # LLaMA 的 QKV 投影层命名模式
            self.target_modules = [
                "q_proj",
                "v_proj",
            ]


def load_model_and_tokenizer(args: LoraTrainingArguments):
    """
    加载 LLaMA 模型和分词器

    支持 4-bit 量化以减少显存占用
    """
    print(f"加载模型: {args.model_name}")

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True
    )

    # 加载模型（可能带 4-bit 量化）
    if args.use_4bit:
        print("使用 4-bit 量化...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.bnb_4bit_compute_dtype,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type="nf4",
        )

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            device_map="auto",
            trust_remote_code=True
        )

    # 打印参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"可训练比例: {trainable_params / total_params * 100:.4f}%")

    return model, tokenizer


def setup_lora(model, args: LoraTrainingArguments):
    """
    配置 LoRA

    将 LoRA 适配器添加到预训练模型
    """
    print(f"配置 LoRA (rank={args.lora_r}, alpha={args.lora_alpha})...")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # 准备模型进行 k-bit 训练
    if args.use_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True
        )

    # 应用 LoRA
    model = get_peft_model(model, lora_config)

    # 打印可训练参数
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LoRA 后可训练参数: {trainable:,}")

    return model


def create_training_data(tokenizer, max_length: int):
    """
    创建训练数据（使用合成数据作为示例）

    实际使用时替换为真实数据集
    """
    from datasets import Dataset

    # 合成数据示例
    class DummyDataset(Dataset):
        def __init__(self, tokenizer, max_length, num_samples=1000):
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.num_samples = num_samples

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            # 随机生成一些文本
            texts = [
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning is fascinating.",
                "Deep learning models are getting larger.",
                "Transformers revolutionized NLP.",
            ]
            text = texts[idx % len(texts)]

            # 编码
            encodings = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length"
            )

            return {
                "input_ids": encodings["input_ids"].flatten(),
                "attention_mask": encodings["attention_mask"].flatten(),
                "labels": encodings["input_ids"].flatten(),  # 因果 LM: 预测下一个 token
            }

    train_dataset = DummyDataset(tokenizer, max_length)

    # 简单的数据整理函数
    def data_collator(examples):
        return {
            "input_ids": torch.stack([torch.LongTensor(e["input_ids"]) for e in examples]),
            "attention_mask": torch.stack([torch.LongTensor(e["attention_mask"]) for e in examples]),
            "labels": torch.stack([torch.LongTensor(e["labels"]) for e in examples]),
        }

    return train_dataset, data_collator


def train_lora_model(args: LoraTrainingArguments):
    """
    训练 LoRA 模型
    """
    print("\n" + "=" * 60)
    print("开始 LoRA 微调")
    print("=" * 60)

    # 1. 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(args)

    # 2. 配置 LoRA
    model = setup_lora(model, args)

    # 3. 准备训练数据
    train_dataset, data_collator = create_training_data(tokenizer, args.max_length)

    # 4. 训练参数
    training_args = TrainingArguments(
        output_dir=f"./outputs/{args.model_name.split('/')[-1]}",
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        fp16=True if not args.use_4bit else False,  # 4-bit 时不启用 fp16
        report_to=["tensorboard"],
        ddp_find_unused_parameters=False,
    )

    # 5. 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # 6. 开始训练
    print("\n开始训练...")
    trainer.train()

    # 7. 保存模型
    print("\n保存模型...")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)

    print(f"\n训练完成！模型保存在: {training_args.output_dir}")

    return model, tokenizer, training_args.output_dir


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 50):
    """
    使用微调后的模型生成文本
    """
    model.eval()

    # 编码提示
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            temperature=0.8,
        )

    # 解码
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="huggyllama/llama-7b")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    # 更新参数
    training_args = LoraTrainingArguments(
        model_name=args.model,
        lora_r=args.rank,
        lora_alpha=args.alpha,
        use_4bit=not args.no_4bit,
    )

    # 训练
    model, tokenizer, output_dir = train_lora_model(training_args)

    # 测试生成
    print("\n" + "=" * 60)
    print("生成示例")
    print("=" * 60)

    prompt = "The future of AI is"
    generated = generate_text(model, tokenizer, prompt, max_new_tokens=30)

    print(f"提示: {prompt}")
    print(f"生成: {generated}")
