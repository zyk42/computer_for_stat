import matplotlib.pyplot as plt
import numpy as np
import os

def plot_roofline():
    # Hardware specs for NVIDIA A100 (Approximate values for demonstration)
    # Peak FP16 Tensor Core Performance: ~312 TFLOPS (Non-Sparse)
    # Note: With Sparsity, it can go up to 624 TFLOPS, but let's use dense for standard comparison.
    peak_flops = 312 * 10**12  
    # Peak HBM2e Memory Bandwidth: ~1555 GB/s (40GB Model) / ~2039 GB/s (80GB Model)
    # Let's use 1555 GB/s
    peak_bandwidth = 1555 * 10**9 

    # Arithmetic Intensity range (log scale)
    # Extend range to make plot look better
    intensities = np.logspace(-1.5, 3.5, 200)
    
    # Roofline function: Performance = min(Peak FLOPS, Bandwidth * Intensity)
    attainable_performance = np.minimum(peak_flops, intensities * peak_bandwidth)
    
    # Create plot with larger size to avoid text overlap
    plt.figure(figsize=(12, 8))
    
    # Plot the roofline
    plt.loglog(intensities, attainable_performance, 'b-', linewidth=3, label='A100 Roofline (FP16 Dense)')
    
    # Calculate the "Knee" point (Ridge point)
    # This is where the system transitions from memory-bound to compute-bound
    knee_intensity = peak_flops / peak_bandwidth
    knee_performance = peak_flops
    
    # Mark the Knee point
    plt.scatter([knee_intensity], [knee_performance], color='red', s=100, zorder=10, label='Knee Point')
    
    # Annotate Knee Point with arrow to avoid clutter
    plt.annotate(f'Knee Point\n{knee_intensity:.1f} FLOPs/Byte', 
                 xy=(knee_intensity, knee_performance), 
                 xytext=(knee_intensity * 3, knee_performance / 5),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=2),
                 fontsize=12,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

    # Add regions with better positioning
    # Memory Bound Region: Position text parallel to the slope
    # Slope in log-log plot looks like 45 degrees if aspect ratio is 1, but we adjust position manually.
    mid_mem_intensity = knee_intensity / 10
    mid_mem_perf = mid_mem_intensity * peak_bandwidth
    plt.text(mid_mem_intensity, mid_mem_perf * 2, 'Memory Bound Region\n(Performance limited by Bandwidth)', 
             rotation=45, ha='center', va='bottom', fontsize=12, color='darkgreen', fontweight='bold')
    
    # Compute Bound Region: Position text above the flat line
    mid_comp_intensity = knee_intensity * 10
    plt.text(mid_comp_intensity, peak_flops * 1.5, 'Compute Bound Region\n(Performance limited by Peak FLOPS)', 
             ha='center', va='bottom', fontsize=12, color='darkred', fontweight='bold')

    # Example Algorithms
    # Vector Add: Intensity ~ 0.16 FLOPs/Byte
    vec_add_intensity = 1/6
    vec_add_perf = min(peak_flops, vec_add_intensity * peak_bandwidth)
    plt.scatter([vec_add_intensity], [vec_add_perf], color='orange', s=120, marker='s', zorder=10, label='Vector Add (Memory Bound)')
    plt.annotate('Vector Add\n(Low Intensity)', 
                 xy=(vec_add_intensity, vec_add_perf), 
                 xytext=(vec_add_intensity / 5, vec_add_perf * 5),
                 arrowprops=dict(facecolor='orange', shrink=0.05),
                 fontsize=10)
    
    # Matrix Multiply (Large N): Intensity increases with N, eventually compute bound
    # Let's pick a point deep in compute bound region
    matmul_intensity = 500 
    matmul_perf = min(peak_flops, matmul_intensity * peak_bandwidth)
    plt.scatter([matmul_intensity], [matmul_perf], color='purple', s=120, marker='^', zorder=10, label='Matrix Mul (Compute Bound)')
    plt.annotate('Matrix Mul (Large N)\n(High Intensity)', 
                 xy=(matmul_intensity, matmul_perf), 
                 xytext=(matmul_intensity / 5, matmul_perf / 5),
                 arrowprops=dict(facecolor='purple', shrink=0.05),
                 fontsize=10)

    # FlashAttention (Optimized) - Just conceptually
    # It increases effective intensity by tiling, moving right on the X-axis compared to standard Attention
    # Let's place it near the knee or slightly to the right to show "optimization"
    flash_intensity = knee_intensity * 1.5
    flash_perf = peak_flops # Ideally
    plt.scatter([flash_intensity], [flash_perf], color='green', s=120, marker='*', zorder=10, label='FlashAttention (Optimized)')
    plt.annotate('FlashAttention\n(Tiling moves it right)', 
                 xy=(flash_intensity, flash_perf), 
                 xytext=(flash_intensity * 2, flash_perf / 2),
                 arrowprops=dict(facecolor='green', shrink=0.05),
                 fontsize=10)

    # Formatting
    plt.xlabel("Arithmetic Intensity (FLOPs/Byte) [Log Scale]", fontsize=14)
    plt.ylabel("Performance (FLOPS) [Log Scale]", fontsize=14)
    plt.title("Roofline Model: The Physics of Computing Performance (A100 Example)", fontsize=16)
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend(loc='lower right', fontsize=12, framealpha=0.9)
    
    # Set limits to make sure annotations are visible
    plt.ylim(bottom=10**11, top=peak_flops * 10)
    plt.xlim(left=0.05, right=3000)

    # Add Bandwidth and Peak FLOPS labels on axes
    plt.text(0.06, peak_bandwidth * 0.08, f'Slope = Bandwidth\n({peak_bandwidth/1e9:.0f} GB/s)', fontsize=10, color='blue')
    plt.axhline(y=peak_flops, color='gray', linestyle='--', alpha=0.5)
    plt.text(3000, peak_flops * 1.1, f'Peak = {peak_flops/1e12:.0f} TFLOPS', ha='right', fontsize=10, color='blue')

    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(os.path.dirname(__file__), 'roofline_model.png')
    plt.savefig(output_path, dpi=300)
    print(f"Roofline plot saved to: {output_path}")

if __name__ == "__main__":
    plot_roofline()
