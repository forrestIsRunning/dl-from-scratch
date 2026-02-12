"""
CUDA 编程基础示例

展示 PyTorch CUDA 的核心概念：
1. 张量在 GPU 和 CPU 之间移动
2. CUDA 内核编写（用 PyTorch JIT）
3. GPU 内存管理
4. 并行计算模式
"""

import torch
import time


# =============================================================================
# 基础：检查 CUDA 可用性
# =============================================================================

def check_cuda():
    """检查 CUDA 环境"""
    print("=" * 60)
    print("CUDA 环境检查")
    print("=" * 60)

    print(f"CUDA 可用: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
        print(f"GPU 数量: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {props.name}")
            print(f"  总内存: {props.total_memory / 1024**3:.2f} GB")
            print(f"  计算能力: {props.major}.{props.minor}")
    else:
        print("CUDA 不可用，将使用 CPU")

    print()


# =============================================================================
# 1. 张量在设备间移动
# =============================================================================

def tensor_device_transfer():
    """演示 CPU <-> GPU 张量传输"""

    print("=" * 60)
    print("1. 张量设备间传输")
    print("=" * 60)

    # CPU 张量
    cpu_tensor = torch.randn(1000, 1000)
    print(f"CPU 张量: 设备={cpu_tensor.device}, 形状={cpu_tensor.shape}")

    if torch.cuda.is_available():
        # 移动到 GPU
        gpu_tensor = cpu_tensor.cuda()  # 或 .to("cuda:0")
        print(f"GPU 张量: 设备={gpu_tensor.device}, 形状={gpu_tensor.shape}")

        # 移回 CPU
        cpu_tensor_back = gpu_tensor.cpu()
        print(f"回 CPU 张量: 设备={cpu_tensor_back.device}")

        # 指定 GPU
        gpu_1 = cpu_tensor.to("cuda:0")  # 第一块 GPU
        print(f"指定 GPU 张量: 设备={gpu_1.device}")

    print()


# =============================================================================
# 2. GPU vs CPU 性能对比
# =============================================================================

def performance_comparison():
    """演示 GPU 加速效果"""

    print("=" * 60)
    print("2. GPU vs CPU 性能对比")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA 不可用，跳过性能对比")
        return

    # 大矩阵乘法
    size = 2000
    a = torch.randn(size, size)
    b = torch.randn(size, size)

    # CPU 计算
    a_cpu, b_cpu = a.to("cpu"), b.to("cpu")
    start = time.time()
    for _ in range(10):
        c_cpu = a_cpu @ b_cpu
    cpu_time = (time.time() - start) / 10 * 1000
    print(f"CPU 平均时间: {cpu_time:.2f} ms")

    # GPU 计算
    a_gpu, b_gpu = a.cuda(), b.cuda()
    torch.cuda.synchronize()  # 等待 GPU 完成
    start = time.time()
    for _ in range(10):
        c_gpu = a_gpu @ b_gpu
        torch.cuda.synchronize()
    gpu_time = (time.time() - start) / 10 * 1000
    print(f"GPU 平均时间: {gpu_time:.2f} ms")

    print(f"加速比: {cpu_time / gpu_time:.2f}x")
    print()


# =============================================================================
# 3. 自定义 CUDA 内核（使用 torch.jit）
# =============================================================================

def custom_cuda_kernel():
    """演示自定义 CUDA 内核"""

    print("=" * 60)
    print("3. 自定义 CUDA 内核（torch.jit.script）")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA 不可用，跳过")
        return

    # JIT 编译的函数
    @torch.jit.script
    def cuda_kernel(x: torch.Tensor) -> torch.Tensor:
        """
        自定义 CUDA 核函数（通过 JIT 编译）
        PyTorch 会自动将其编译为 CUDA
        """
        # 逐元素操作
        y = torch.zeros_like(x)
        for i in range(x.size(0)):
            y[i] = x[i] * 2 + 1
        return y

    # 测试
    x = torch.randn(1000).cuda()

    # 调用 JIT 编译的函数
    y = cuda_kernel(x)

    print(f"输入: {x[:5]}")
    print(f"输出: {y[:5]}")
    print(f"验证: {x[0] * 2 + 1:.4f}, {y[0]:.4f}")
    print()


# =============================================================================
# 4. 批量并行计算
# =============================================================================

def parallel_computation():
    """演示 GPU 并行计算"""

    print("=" * 60)
    print("4. 批量并行计算")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA 不可用，跳过")
        return

    # CPU: 顺序处理
    batch_size = 64
    feature_size = 512

    x_cpu = torch.randn(batch_size, feature_size)
    w_cpu = torch.randn(feature_size, feature_size)

    start = time.time()
    result_cpu = x_cpu @ w_cpu
    cpu_time = (time.time() - start) * 1000

    # GPU: 并行处理
    x_gpu = x_cpu.cuda()
    w_gpu = w_cpu.cuda()

    torch.cuda.synchronize()
    start = time.time()
    result_gpu = x_gpu @ w_gpu
    torch.cuda.synchronize()
    gpu_time = (start - time.time()) * 1000

    print(f"CPU 时间: {cpu_time:.4f} ms")
    print(f"GPU 时间: {gpu_time:.4f} ms")
    print(f"加速比: {cpu_time / gpu_time:.2f}x")

    # 验证结果一致
    print(f"结果一致: {torch.allclose(result_cpu, result_gpu.cpu())}")
    print()


# =============================================================================
# 5. GPU 内存管理
# =============================================================================

def memory_management():
    """演示 GPU 内存管理"""

    print("=" * 60)
    print("5. GPU 内存管理")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA 不可用，跳过")
        return

    # 查看当前内存使用
    print(f"当前内存分配: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"当前内存缓存: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

    # 分配大张量
    large_tensor = torch.randn(10000, 10000).cuda()

    print(f"分配后: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"峰值内存: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

    # 清理缓存
    del large_tensor
    torch.cuda.empty_cache()

    print(f"清理后: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"清理后缓存: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    print()


# =============================================================================
# 6. 多 GPU 使用
# =============================================================================

def multi_gpu_usage():
    """演示多 GPU 使用"""

    print("=" * 60)
    print("6. 多 GPU 使用 (DataParallel)")
    print("=" * 60)

    if torch.cuda.device_count() < 2:
        print(f"只有 {torch.cuda.device_count()} 块 GPU，跳过多 GPU 示例")
        return

    # 定义简单模型
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(1000, 10)

        def forward(self, x):
            return self.fc(x)

    # 创建模型并包装为 DataParallel
    model = SimpleModel()

    print(f"原始模型设备: {next(model.parameters()).device}")

    # 包装为 DataParallel（自动使用所有可用 GPU）
    model = torch.nn.DataParallel(model)

    # 数据
    x = torch.randn(64, 1000).cuda()

    # 前向传播
    output = model(x)

    print(f"DataParallel 模型")
    print(f"  GPU 数量: {model.device_ids}")
    print(f"  输出设备: {output.device}")
    print(f"  输出形状: {output.shape}")
    print()


# =============================================================================
# 主函数
# =============================================================================

def main():
    """运行所有示例"""
    check_cuda()
    tensor_device_transfer()
    performance_comparison()
    custom_cuda_kernel()
    parallel_computation()
    memory_management()
    multi_gpu_usage()

    print("=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
