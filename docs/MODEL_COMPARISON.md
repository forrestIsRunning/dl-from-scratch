# 模型架构对比：CNN vs RNN vs Transformer

## 核心问题：哪种模型最适合你的任务？

| 任务类型 | 最适合的模型 | 备选方案 |
|----------|--------------|----------|
| 图像分类 | CNN | Vision Transformer |
| 文本分类 | Transformer | CNN (字符级) |
| 序列预测 | Transformer/LSTM | GRU |
| 生成任务 | Transformer | RNN |
| 时序数据 | LSTM/GRU | Temporal CNN |

---

## 一、CNN (卷积神经网络)

### 核心思想

**局部感受野 + 权重共享**

```
图像 → [Conv → ReLU → Pool] × N → FC → 输出
```

### 优势

1. **参数高效**：比全连接层少 10-100 倍参数
2. **平移不变性**：池化层带来位置鲁棒性
3. **特征层次**：低层特征 → 高层语义
4. **GPU 友好**：卷积操作高度并行

### 劣势

1. **固定感受野**：卷积核大小限制上下文
2. **位置不敏感**：池化丢失精确位置信息
3. **不适合长序列**：处理文本时全局建模能力弱

### 适用场景

- **图像分类**：MNIST, ImageNet
- **目标检测**：YOLO, Faster R-CNN
- **图像分割**：U-Net, SegNet
- **视频处理**：3D CNN, ConvLSTM

---

## 二、RNN/LSTM (循环神经网络)

### 核心思想

**序列建模，保持隐藏状态**

```
x_1 → [RNN 单元] → h_1 ─┐
x_2 → [RNN 单元] → h_2 ─┤
x_3 → [RNN 单元] → h_3 ─┼→ h_final → 输出
...                         │
x_t → [RNN 单元] → h_t ─┘
```

### LSTM 的门控机制

```
输入门 i_t = σ(W_xi * x_t + W_hi * h_{t-1})
    ↓ 控制：写入多少新信息
遗忘门 f_t = σ(W_xf * x_t + W_hf * h_{t-1})
    ↓ 控制：保留多少旧信息
细胞状态 C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
    ↓ 长期记忆的"高速公路"
输出门 o_t = σ(W_xo * x_t + W_ho * h_{t-1})
    ↓ 控制：输出多少信息
隐藏状态 h_t = o_t ⊙ tanh(C_t)
```

### 优势

1. **变长处理**：自然处理不同长度序列
2. **时序建模**：捕捉时间依赖关系
3. **参数共享**：时间步间共享权重
4. **在线推理**：可以流式处理

### 劣势

1. **顺序计算**：无法并行，训练慢
2. **长距离遗忘**：即使 LSTM 也难以记住很早的信息
3. **梯度问题**：虽然缓解但仍存在
4. **单向性**：标准 RNN 只能看过去

### 适用场景

- **时序预测**：股票价格、天气
- **语音识别**：声学模型
- **文本生成**：字符级语言模型
- **序列标注**：POS tagging、NER

---

## 三、Transformer

### 核心思想

**自注意力机制 + 全局上下文**

```
输入 → Embedding → [Attention × N] → 输出
                    ↑
            Q, K, V 计算
```

### 注意力机制

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V

Q (Query): 我在找什么？
K (Key): 你有什么？
V (Value): 你的内容是什么？
```

### 优势

1. **并行计算**：所有位置同时处理（不像 RNN）
2. **全局建模**：任意两个位置直接连接
3. **长距离依赖**：完美捕捉远距离关系
4. **可解释性**：注意力图可视化理解模型

### 劣势

1. **计算复杂度**：O(n²) 序列长度平方
2. **位置无感**：需要位置编码
3. **数据饥渴**：需要大量数据训练
4. **资源消耗**：参数多，显存占用大

### 适用场景

- **NLP**：翻译、摘要、对话
- **大语言模型**：GPT, BERT, LLaMA
- **计算机视觉**：Vision Transformer
- **多模态**：图文匹配、视频理解

---

## 四、为什么 Transformer 更优秀？

### 1. 并行度对比

```
RNN/LSTM:        O(T) 顺序计算
    │
    ▼
[x_1] → [h_1] → [y_1]
    │
    ▼
[x_2] → [h_2] → [y_2]
    │
    ▼
[x_T] → [h_T] → [y_T]

总时间: O(T) × 计算时间


Transformer:       O(1) 并行计算
    │
    ▼
[x_1, x_2, ..., x_T] → [Attention] → [y_1, y_2, ..., y_T]

总时间: O(1) × 计算时间
```

### 2. 长距离依赖对比

```
RNN: 信息传递链路

"The" → [h1] ─→ [h2] ─→ ... ─→ [h50] ─→ "roi"

问题: 路径越长，信息越容易丢失


Transformer: 直接连接

"The" ─────────────────────────────┐
                                    ├→ Attention ─→ "roi"
"is"  ─────────────────────────────┤
"king" ────────────────────────────┘

任意两个位置直接通信，信息无损
```

### 3. 实验对比（论文结果）

| 模型 | BLEU (翻译) | Perplexity (语言模型) | 训练时间 |
|------|-------------|---------------------|----------|
| LSTM (2017) | 26.0 | - | 基线 |
| Transformer (2017) | **28.4** | - | 慢 1/4 |
| Transformer (GPU) | 28.4 | - | **快 8x** |

### 4. 为什么 Transformer 赢了？

```
         计算资源 (FLOPs)
              ↑
              │
    RNN     │    Transformer
    ┌────────┴────────┐
    │                 │
  O(T)              O(T²)
    │                 │
    │    长距离依赖质量
    │                 │
  差                优秀
    │                 │
    └────────┬────────┘
             │
         综合性能
```

Transformer 用更多计算换取了：
1. 更好的长距离建模
2. 完全并行的训练
3. 更强的表达能力

在 GPU 时代，这个交易是值得的！

---

## 五、选型决策树

```
开始
  │
  ├─ 数据类型?
  │   ├─ 图像 → CNN
  │   ├─ 音频 → CNN + RNN
  │   └─ 文本/序列 → 继续
  │
  ├─ 序列长度?
  │   ├─ 短 (< 50) → LSTM/GRU
  │   ├─ 中 (50-500) → Transformer
  │   └─ 长 (> 500) → Transformer + 优化
  │
  ├─ 计算资源?
  │   ├─ 受限 → LSTM/GRU
  │   └─ 充足 → Transformer
  │
  └─ 数据量?
      ├─ 小 (< 1M) → LSTM/GRU
      └─ 大 (> 1M) → Transformer
```

---

## 六、实践建议

### 小项目 / 学习阶段

```
推荐: CNN → LSTM → Transformer

原因: 循序渐进，理解演进过程
```

### 生产环境

```
NLP: Transformer (BERT 系列)
CV: CNN (ResNet, EfficientNet) + ViT
时序: LSTM/GRU (简单任务), Transformer (复杂任务)
```

### 边缘设备

```
推荐: 量化后的 CNN/LSTM

原因: Transformer 对资源要求高
```

---

## 七、代码对比示例

### 同一个任务，三种实现

**任务：序列到序列 (seq2seq) 翻译**

```python
# CNN: 简单粗暴，直接拼接
cnn_output = CNN(encoder_input)
decoder_output = CNN(decoder_input)
final = torch.cat([cnn_output, decoder_output])

# LSTM: 顺序编码，顺序解码
encoder_hidden, encoder_cell = LSTM_encoder(encoder_input)
decoder_output = LSTM_decoder(decoder_input,
                              (encoder_hidden, encoder_cell))

# Transformer: 并行编码，并行解码 + 注意力
encoder_output = Transformer_encoder(encoder_input)
decoder_output = Transformer_decoder(decoder_input,
                                  encoder_output)  # Cross-Attention!
```

---

## 总结

| 维度 | CNN | RNN/LSTM | Transformer |
|------|-----|----------|-------------|
| **并行度** | 高 | 低 | 极高 |
| **长距离建模** | 无 | 弱 | 强 |
| **参数量** | 中 | 中 | 高 |
| **数据需求** | 低 | 中 | 高 |
| **推理速度** | 快 | 慢 | 快 |
| **可解释性** | 中 | 低 | 高 (Attention) |
| **成熟度** | 非常成熟 | 成熟 | 快速发展 |

**最终建议**：
- 学习：全部学习
- 实践：Transformer 为主，其他为辅
- 面试：深入理解各自优劣
