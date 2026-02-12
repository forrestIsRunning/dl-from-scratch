# CUDA 编程基础示例

## 核心概念

1. **设备管理**：CPU 和 GPU 之间数据传输
2. **并行计算**：GPU 同时处理多个数据
3. **内存管理**：GPU 内存分配和释放
4. **自定义内核**：用 CUDA C++ 写自定义操作

## 为什么需要 CUDA？

### CPU vs GPU

| 特性 | CPU | GPU |
|------|------|------|
| 核心数 | 几十个 | 几千个 |
| 单核速度 | 快 (3-4GHz) | 慢 (~1GHz) |
| 内存带宽 | ~50 GB/s | ~600 GB/s |
| 适用任务 | 串行、复杂逻辑 | 并行、简单计算 |

**GPU 加速场景**：矩阵运算、神经网络训练、图像处理

## 快速开始

```bash
cd cuda
python tutorial.py
```

## 示例内容

### 1. 张量设备传输

```python
# CPU → GPU
cpu_tensor = torch.randn(1000, 1000)
gpu_tensor = cpu_tensor.cuda()  # 或 .to("cuda:0")

# GPU → CPU
back_to_cpu = gpu_tensor.cpu()
```

### 2. 性能对比

```python
# 大矩阵乘法
# CPU: ~100 ms
# GPU: ~5 ms
# 加速比: ~20x
```

### 3. 自定义 CUDA 内核

```python
@torch.jit.script
def custom_kernel(x):
    # JIT 编译为 CUDA
    return x * 2 + 1
```

### 4. 多 GPU (DataParallel)

```python
model = torch.nn.DataParallel(model)
# 自动使用所有可用 GPU
```

## GPU 内存优化

1. **及时清理**：`del tensor; torch.cuda.empty_cache()`
2. **批处理**：增大 batch size，充分利用 GPU
3. **梯度累积**：模拟大 batch
4. **混合精度**：FP16 减少内存占用

## CUDA 层级

```
高级: CUDA C++ 编写自定义内核
        ↑
中级: torch.jit.script 编译优化
        ↑
初级: .to("cuda") 设备迁移
```
