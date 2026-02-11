# 调用逻辑与数据流转详解

## 目录
- [整体架构](#整体架构)
- [训练流程](#训练流程)
- [推理流程](#推理流程)
- [数据维度变化](#数据维度变化)
- [模块调用关系](#模块调用关系)

---

## 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         入口层 (main.py)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│    python main.py train    ──→    train.py                     │
│    python main.py generate ──→    inference.py                 │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         配置层 (config.py)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│    get_model_params(vocab_size) ──→ config 字典                 │
│                                                                   │
│    {                                                              │
│      d_model: 256,                                               │
│      num_heads: 8,                                               │
│      num_blocks: 6,                                              │
│      head_size: 32,                                              │
│      context_length: 256,                                        │
│      dropout: 0.1,                                               │
│      vocab_size: 65,                                             │
│      device: mpu/cuda/cpu                                        │
│    }                                                              │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         数据层 (data.py)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│    下载莎士比亚文本                                               │
│           │                                                       │
│           ▼                                                       │
│    CharTokenizer.encode/decode                                   │
│    (字符级 tokenization)                                          │
│           │                                                       │
│           ▼                                                       │
│    TextDataset.get_batch() ──→ (x, y)                            │
│    x: [batch, seq_len]      输入序列                              │
│    y: [batch, seq_len]      目标序列 (x 向右移一位)               │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         模型层 (model.py)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│    GPTModel.forward(idx, targets=None)                            │
│                                                                   │
│    输入: idx [B, T]  ← batch_size, seq_len                        │
│           │                                                       │
│           ├─→ Token Embedding    ──→ [B, T, d_model]             │
│           │                                                       │
│           ├─→ Positional Encoding  ──→ [T, d_model]              │
│           │                                                       │
│           ├─→ 相加                   → [B, T, d_model]            │
│           │                                                       │
│           ├─→ N × TransformerBlock  → [B, T, d_model]            │
│           │    │                                                   │
│           │    ├─→ LayerNorm                                      │
│           │    ├─→ MultiHeadAttention                            │
│           │    │    ├─→ Attention (× num_heads)                  │
│           │    │    │    └─→ QKV 计算 + Causal Mask              │
│           │    │    └─→ 输出投影                                  │
│           │    └─→ FeedForwardNetwork                            │
│           │                                                       │
│           ├─→ Final LayerNorm     ──→ [B, T, d_model]            │
│           │                                                       │
│           └─→ Output Linear       ──→ [B, T, vocab_size]         │
│                  logits                                            │
│                                                                   │
│    输出: logits [B, T, vocab_size]                                │
│          loss (可选，训练时计算)                                   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 训练流程

### 1. 调用链路

```
main.py
    │
    └─→ cmd_train()
            │
            └─→ train.train()
                    │
                    ├─→ load_data()
                    │       │
                    │       ├─→ download_shakespeare()
                    │       ├─→ CharTokenizer()
                    │       └─→ TextDataset.split()
                    │               │
                    │               ├─→ train_dataset
                    │               └─→ val_dataset
                    │
                    ├─→ GPTModel(config)
                    │       │
                    │       └─→ 初始化所有组件
                    │
                    ├─→ AdamW optimizer
                    ├─→ get_lr_scheduler()
                    │
                    └─→ 训练循环 (for step in range(MAX_STEPS))
                            │
                            ├─→ train_dataset.get_batch() ──→ (x, y)
                            │       │
                            │       └─→ x: [64, 256], y: [64, 256]
                            │
                            ├─→ model(x, y) ──→ (logits, loss)
                            │       │
                            │       └─→ GPTModel.forward()
                            │
                            ├─→ loss.backward()
                            ├─→ clip_grad_norm_()
                            ├─→ optimizer.step()
                            ├─→ scheduler.step()
                            │
                            ├─→ (定期) evaluate(model, val_dataset)
                            │       │
                            │       └─→ model.eval()
                            │           for _ in range(eval_steps):
                            │               x, y = val_dataset.get_batch()
                            │               _, loss = model(x, y)
                            │           model.train()
                            │
                            └─→ (定期) save_checkpoint()
```

### 2. 训练数据流

```
原始文本 (1115394 字符)
    │
    ▼
CharTokenizer.encode()
    │
    ▼
整数序列 [1115394]  ← 每个字符对应一个整数 ID
    │
    ▼
TextDataset
    │
    ▼
get_batch(batch_size=64)
    │
    ├─→ 随机选择 64 个起始位置
    │
    ├─→ x = data[i : i+256]      [64, 256]
    │    (当前序列)
    │
    └─→ y = data[i+1 : i+257]    [64, 256]
         (下一个 token，目标)
         │
         ▼
    model(x, y)
         │
         ▼
    logits = model.forward(x)    [64, 256, 65]
         │
         ▼
    loss = CrossEntropy(logits, y)
         │
         ▼
    loss.backward()  ← 反向传播，计算梯度
         │
         ▼
    optimizer.step()  ← 更新参数
```

### 3. 训练关键代码解析

```python
# train.py: 训练循环核心
for step in range(MAX_STEPS):
    # 1. 获取数据
    x, y = train_dataset.get_batch(BATCH_SIZE, DEVICE)
    # x: [64, 256]  ← 64 个样本，每个 256 个 token
    # y: [64, 256]  ← 对应的目标（向右移一位）

    # 2. 前向传播
    logits, loss = model(x, y)
    # logits: [64, 256, 65]  ← 每个位置对 65 个字符的预测
    # loss: 标量

    # 3. 反向传播
    optimizer.zero_grad()  # 清空梯度
    loss.backward()        # 计算梯度

    # 4. 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

    # 5. 更新参数
    optimizer.step()

    # 6. 更新学习率
    scheduler.step()
```

---

## 推理流程

### 1. 调用链路

```
main.py
    │
    └─→ cmd_generate()
            │
            └─→ inference.main()
                    │
                    ├─→ load_data() ──→ tokenizer
                    │       (只获取 tokenizer，不重新训练)
                    │
                    ├─→ load_checkpoint(checkpoint_file)
                    │       │
                    │       └─→ GPTModel(config) + 加载权重
                    │
                    └─→ 生成模式
                            │
                            ├─→ single_generation()
                            │       │
                            │       └─→ generate_text()
                            │               │
                            │               ├─→ tokenizer.encode(prompt)
                            │               ├─→ model.generate()
                            │               └─→ tokenizer.decode()
                            │
                            └─→ interactive_mode()
                                    │
                                    └─→ 循环: generate_text()
```

### 2. 推理数据流

```
用户输入 prompt: "First Citizen"
    │
    ▼
tokenizer.encode(prompt)
    │
    ▼
idx = [20, 43, 50, 50, 53, 6, 1, 61, 46, 39, 58, 5, 57]
    │
    ▼
idx = tensor([[20, 43, 50, ...]])  [1, 13]
    │
    ▼
model.generate(idx, max_new_tokens=500)
    │
    └─→ for _ in range(500):
            │
            ├─→ idx_crop = idx[:, -256:]  (截取最后 256 个)
            │
            ├─→ logits, _ = model(idx_crop)
            │    logits: [1, T, 65]
            │
            ├─→ logits = logits[:, -1, :] / temperature
            │    (只取最后一个位置的 logits)
            │
            ├─→ probs = softmax(logits)
            │
            ├─→ idx_next = multinomial(probs, 1)
            │    (采样得到下一个 token)
            │
            └─→ idx = cat([idx, idx_next])
                 (拼接到序列)
    │
    ▼
generated_idx: [1, 513]  (原 13 + 新 500)
    │
    ▼
tokenizer.decode(generated_idx[0])
    │
    ▼
"First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n\nFirst Citizen:\nYou are all resolved rather to die than to famish?\n\n..."
```

### 3. 推理关键代码解析

```python
# model.py: generate() 方法
@torch.no_grad()
def generate(self, idx, max_new_tokens, temperature, top_k):
    """
    自回归生成：一次生成一个 token，然后拼接到序列中
    """
    for _ in range(max_new_tokens):
        # 1. 截取最后 context_length 个 token
        idx_crop = idx[:, -self.context_length:]
        # [batch, min(T, context_length)]

        # 2. 前向传播
        logits, _ = self.forward(idx_crop)
        # [batch, T, vocab_size]

        # 3. 只取最后一个位置的 logits (这是我们要预测的)
        logits = logits[:, -1, :] / temperature
        # [batch, vocab_size]

        # 4. Top-K 采样 (可选)
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')

        # 5. Softmax 得到概率
        probs = F.softmax(logits, dim=-1)
        # [batch, vocab_size]

        # 6. 采样
        idx_next = torch.multinomial(probs, num_samples=1)
        # [batch, 1]

        # 7. 拼接到序列
        idx = torch.cat((idx, idx_next), dim=1)
        # [batch, T+1]

    return idx
```

---

## 数据维度变化

### 模型前向传播维度变化

```
输入: idx [B, T]  ← batch_size=64, seq_len=256
     │
     ▼
Token Embedding
     [B, T, d_model=256]
     │
     ▼
+ Positional Encoding [T, d_model=256]  (广播到 [B, T, d_model])
     │
     ▼
x = [B, T, 256]
     │
     ▼
┌─────────────────────────────────────────┐
│  TransformerBlock × 6                   │
│                                         │
│  每个 Block:                            │
│    x: [B, T, 256]                      │
│      │                                  │
│      ├─→ LayerNorm → [B, T, 256]       │
│      ├─→ MultiHeadAttention            │
│      │    │                             │
│      │    ├─→ Attention (8 个头)       │
│      │    │    每个: [B, T, 32]        │
│      │    │    拼接: [B, T, 256]       │
│      │    │                             │
│      │    └─→ Linear → [B, T, 256]     │
│      │                                 │
│      └─→ x = x + output (残差)          │
│          [B, T, 256]                   │
│          │                             │
│          ├─→ LayerNorm → [B, T, 256]   │
│          ├─→ FFN                       │
│          │    │                         │
│          │    ├─→ Linear → [B,T,1024]  │
│          │    ├─→ ReLU                 │
│          │    ├─→ Linear → [B,T,256]   │
│          │    │                         │
│          └─→ x = x + output (残差)      │
│              [B, T, 256]               │
└─────────────────────────────────────────┘
     │
     ▼
Final LayerNorm → [B, T, 256]
     │
     ▼
Output Linear → [B, T, vocab_size=65]
     │
     ▼
logits: [B, T, 65]
     │
     ▼
训练时: loss = CrossEntropy(logits.view(-1, 65), targets.view(-1))
       [B*T, 65]           [B*T]
```

### 注意力机制维度变化

```
输入: x [B, T, d_model=256]
     │
     ▼
QKV 变换 (8 个头，每个 head_size=32)
     │
     ▼
┌─────────────────────────────────────────┐
│  Head 1:                                │
│    q = x @ Wq → [B, T, 32]             │
│    k = x @ Wk → [B, T, 32]             │
│    v = x @ Wv → [B, T, 32]             │
│                                         │
│    weights = (q @ k^T) / sqrt(32)      │
│             [B, T, 32] @ [B, 32, T]     │
│             = [B, T, T]                 │
│                                         │
│    应用 Causal Mask (下三角)            │
│                                         │
│    weights = softmax(weights)          │
│             [B, T, T]                   │
│                                         │
│    output = weights @ v                 │
│            [B, T, T] @ [B, T, 32]      │
│            = [B, T, 32]                 │
│                                         │
│  Head 2-8: 相同过程...                  │
└─────────────────────────────────────────┘
     │
     ▼
拼接 8 个头的输出
[B, T, 32] × 8 → cat → [B, T, 256]
     │
     ▼
输出投影 Linear → [B, T, 256]
```

---

## 模块调用关系

```
┌─────────────────────────────────────────────────────────────────┐
│                        main.py                                  │
│                                                                   │
│    cmd_train() ─────────────────────────────────┐                │
│    cmd_generate() ───────────────────────┐      │                │
│                                             │      │                │
└─────────────────────────────────────────────┼──────┼────────────────┘
                                              │      │
                              ┌───────────────┘      └───────────────┐
                              │                                      │
                     ┌────────▼────────┐                   ┌────────▼────────┐
                     │   train.py      │                   │  inference.py   │
                     └────────┬────────┘                   └────────┬────────┘
                              │                                      │
         ┌────────────────────┼────────────────────┐       ┌─────────┼─────────┐
         │                    │                    │       │         │           │
    ┌────▼────┐        ┌─────▼─────┐         ┌────▼────┐  ┌───▼───┐ ┌───▼────┐
    │ data.py │        │  config.py │         │ model.py│  │data.py│ │train.py│
    └────┬────┘        └───────────┘         └─────────┘  └───────┘ └───┬────┘
         │                                                    │         │
    ┌────▼────────────────────────────────────────────────────┴─────────┘
    │
    │  data.py 导出:
    │    - load_data() → (train_dataset, val_dataset, tokenizer)
    │    - CharTokenizer.encode() / decode()
    │    - TextDataset.get_batch() → (x, y)
    │
    └─────────────────────────────────────────────────────────────────┐
    │                                                                  │
    │  model.py 导出:                                                 │
    │    - GPTModel(config)                                           │
    │    - GPTModel.forward(idx, targets) → (logits, loss)            │
    │    - GPTModel.generate(idx, ...) → generated_idx                │
    │                                                                  │
    │  内部类 (train.py 通过 GPTModel 使用):                           │
    │    - FeedForwardNetwork                                         │
    │    - Attention                                                  │
    │    - MultiHeadAttention                                        │
    │    - TransformerBlock                                           │
    │                                                                  │
    └─────────────────────────────────────────────────────────────────┘
```

---

## 训练 vs 推理对比

| 方面 | 训练 | 推理 |
|------|------|------|
| **入口** | `python main.py train` | `python main.py generate` |
| **数据来源** | 随机采样 batch | 用户输入 prompt |
| **模型状态** | `model.train()` | `model.eval()` |
| **输入格式** | `x, y = get_batch()` | `idx = encode(prompt)` |
| **前向传播** | `model(x, y)` 返回 logits, loss | `model(idx)` 返回 logits (loss=None) |
| **梯度计算** | `loss.backward()` | `@torch.no_grad()` |
| **参数更新** | `optimizer.step()` | 不更新 |
| **输出** | 损失值、检查点 | 生成的文本 |
| **并行度** | 批量处理 (batch_size=64) | 单样本或小批量 |
| **Causal Mask** | 防止看到未来 | 防止看到未来 |
| **生成方式** | 不生成，只训练当前位置 | 自回归逐个生成 |

---

## 核心设计思想

### 1. 预测下一个 token

无论是训练还是推理，核心思想都是：**给定前面的序列，预测下一个 token**

```
训练: "Hello, wor" → 预测 "l"
推理: "Hello, wor" → 采样 "l" → 拼接 → "Hello, worl" → 继续预测...
```

### 2. Causal Mask 的作用

```
位置 0 只能看到位置 0
位置 1 只能看到位置 0, 1
位置 2 只能看到位置 0, 1, 2
...

这确保了模型"只能看到过去，不能看到未来"
```

### 3. 残差连接的作用

```
每个 TransformerBlock 中:
x = x + Attention(x)
x = x + FFN(x)

残差连接让梯度更容易流动，训练更稳定
```

### 4. 权重共享

```
Token Embedding 和 Output Linear 共享权重:

self.output_projection.weight = self.token_embedding.weight

好处:
1. 减少参数量
2. 理论上: 编码和解码使用相同的语义空间
```
