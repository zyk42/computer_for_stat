import numpy as np
import matplotlib.pyplot as plt

def plot_float_distribution():
    """
    绘制浮点数在数轴上的分布，展示精度随数值增大而降低的特性
    使用一个假想的 "Mini-Float" (类似于 FP8 的简化版)
    假设：1位符号，3位指数，2位尾数 (Total 6 bits)
    """
    
    # 定义 Mini-Float 的参数
    EXP_BITS = 3
    MANT_BITS = 2
    BIAS = 2**(EXP_BITS-1) - 1  # 3
    
    values = []
    
    # 遍历所有可能的位组合 (忽略符号位，只看正数)
    # 指数 e: 0 到 2^3 - 1 = 7
    # 尾数 m: 0 到 2^2 - 1 = 3
    
    for e in range(1, 2**EXP_BITS - 1): # 规范化数: exp != 0 and exp != max
        for m in range(2**MANT_BITS):
            # Value = (1 + m / 2^MANT_BITS) * 2^(e - BIAS)
            mantissa = 1 + m / (2**MANT_BITS)
            exponent = e - BIAS
            val = mantissa * (2**exponent)
            values.append(val)
            
    # 非规范化数 (Denormalized): exp = 0
    # Value = (0 + m / 2^MANT_BITS) * 2^(1 - BIAS)
    for m in range(1, 2**MANT_BITS): # m=0 is zero
        mantissa = m / (2**MANT_BITS)
        exponent = 1 - BIAS
        val = mantissa * (2**exponent)
        values.append(val)
        
    values = sorted(list(set(values)))
    values = np.array(values)
    
    # 绘图
    plt.figure(figsize=(12, 6))
    
    # 1. 数轴上的点
    plt.scatter(values, np.zeros_like(values), marker='|', s=1000, color='blue', alpha=0.6, label='Representable Numbers')
    
    # 2. 标注 Gap
    # 选取几个区间展示 Gap 的变化
    # 区间 [0.5, 1], [1, 2], [2, 4], [4, 8]
    
    y_text = 0.1
    
    # 辅助函数：绘制区间和Gap
    def annotate_gap(start_idx, end_idx, text_y):
        v1 = values[start_idx]
        v2 = values[start_idx+1]
        gap = v2 - v1
        mid = (v1 + v2) / 2
        
        plt.annotate(f'Gap: {gap:.4f}', 
                     xy=(mid, 0), xytext=(mid, text_y),
                     arrowprops=dict(arrowstyle='->', color='red'),
                     ha='center', color='red', fontsize=9)

    # 找几个典型的点
    # 找到 1.0 附近的点
    idx_1 = np.argmin(np.abs(values - 1.0))
    annotate_gap(idx_1, idx_1+1, 0.2) # Gap at 1.0
    
    # 找到 2.0 附近的点
    idx_2 = np.argmin(np.abs(values - 2.0))
    annotate_gap(idx_2, idx_2+1, 0.2) # Gap at 2.0
    
    # 找到 4.0 附近的点
    idx_4 = np.argmin(np.abs(values - 4.0))
    annotate_gap(idx_4, idx_4+1, 0.2) # Gap at 4.0
    
    # 找到 8.0 附近的点
    idx_8 = np.argmin(np.abs(values - 8.0))
    if idx_8 < len(values) - 1:
        annotate_gap(idx_8, idx_8+1, 0.2) # Gap at 8.0

    plt.title("Visualizing Floating Point Density (Mini-Float 6-bit)", fontsize=16)
    plt.xlabel("Real Number Line", fontsize=12)
    plt.yticks([])
    plt.ylim(-0.1, 0.5)
    plt.xlim(0, 15)  # Show full range up to 14
    
    # 添加文字说明
    plt.text(8, 0.4, 
             "NOTICE:\n"
             "Density decreases as magnitude increases.\n"
             "Precision is NOT uniform!\n"
             "Gap doubles every time you cross a power of 2.", 
             fontsize=10, bbox=dict(facecolor='yellow', alpha=0.2))

    plt.tight_layout()
    plt.savefig('assets/float_precision.png', dpi=300)
    print("Image saved to assets/float_precision.png")

if __name__ == "__main__":
    plot_float_distribution()
