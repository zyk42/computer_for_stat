import time
import numpy as np
import torch
import matplotlib.pyplot as plt

def benchmark_summation():
    # 使用较小的 N 以避免等待过久，但足够大以显示差异
    N = 10_000_000
    print(f"Benchmarking summation of {N:,} integers...")
    
    # 1. Pure Python Loop
    # 这是最慢的，因为每一行代码都需要解释器逐行执行，且涉及 PyObject 的频繁创建和销毁
    start = time.time()
    total = 0
    for i in range(N):
        total += i
    python_time = time.time() - start
    print(f"Pure Python Loop: {python_time:.4f}s")
    
    # 2. Python Built-in sum()
    # 虽然是 Python 内置函数，但底层是 C 实现，比显式循环快得多
    start = time.time()
    sum(range(N))
    builtin_time = time.time() - start
    print(f"Python Built-in sum(): {builtin_time:.4f}s")
    
    # 3. NumPy
    # 核心运算在 C 中完成，且利用了 SIMD 指令
    arr = np.arange(N, dtype=np.int64)
    start = time.time()
    np.sum(arr)
    numpy_time = time.time() - start
    print(f"NumPy: {numpy_time:.4f}s")
    
    # 4. PyTorch (CPU)
    # 类似 NumPy，但通常有多线程优化
    tensor = torch.arange(N, dtype=torch.int64)
    start = time.time()
    torch.sum(tensor)
    torch_time = time.time() - start
    print(f"PyTorch (CPU): {torch_time:.4f}s")
    
    # --- 绘图 ---
    labels = ['Python Loop', 'Built-in sum()', 'NumPy', 'PyTorch']
    times = [python_time, builtin_time, numpy_time, torch_time]
    colors = ['#FF6B6B', '#FFD93D', '#6BCB77', '#4D96FF'] # Red, Yellow, Green, Blue
    
    plt.figure(figsize=(12, 7))
    bars = plt.bar(labels, times, color=colors, alpha=0.8, width=0.6)
    
    plt.ylabel('Execution Time (seconds)', fontsize=12)
    plt.title(f'Summation of {N:,} Integers: The Cost of Interpretation', fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # 在柱状图上方添加具体时间
    for bar, t in zip(bars, times):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(times),
                 f'{t:.4f}s',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 添加加速比标注 (相对于 Python Loop)
    # NumPy Speedup
    speedup_numpy = python_time / numpy_time
    plt.annotate(f'{speedup_numpy:.1f}x Faster!',
                 xy=(bars[2].get_x() + bars[2].get_width()/2, bars[2].get_height()),
                 xytext=(bars[2].get_x() + bars[2].get_width()/2, bars[0].get_height()/2),
                 arrowprops=dict(arrowstyle="->", color='black', connectionstyle="arc3,rad=.2"),
                 ha='center', fontsize=12, color='darkgreen', fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.8))

    # PyTorch Speedup
    speedup_torch = python_time / torch_time
    plt.annotate(f'{speedup_torch:.1f}x Faster!',
                 xy=(bars[3].get_x() + bars[3].get_width()/2, bars[3].get_height()),
                 xytext=(bars[3].get_x() + bars[3].get_width()/2, bars[0].get_height()/3),
                 arrowprops=dict(arrowstyle="->", color='black', connectionstyle="arc3,rad=-.2"),
                 ha='center', fontsize=12, color='darkblue', fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.8))

    # 添加说明文字
    plt.text(0.65, 0.8, 
             "Why so slow?\n"
             "1. Type Checking per iteration\n"
             "2. PyObject Creation (Boxing)\n"
             "3. GC Overhead", 
             transform=plt.gca().transAxes,
             fontsize=12, bbox=dict(facecolor='red', alpha=0.1),
             verticalalignment='top')

    plt.tight_layout()
    output_path = 'assets/python_overhead.png'
    plt.savefig(output_path, dpi=150)
    print(f"Benchmark plot saved to {output_path}")

if __name__ == "__main__":
    benchmark_summation()
