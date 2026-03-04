import torch
import torch.utils.data as data
import time
import os
import psutil

# 模拟一个耗时的数据加载过程
class FakeDataset(data.Dataset):
    def __init__(self, size=1000, sleep_time=0.01):
        self.size = size
        self.sleep_time = sleep_time
        # 创建一些假数据 (3通道 224x224 图像)
        self.data = torch.randn(size, 3, 224, 224)

    def __getitem__(self, index):
        # 模拟磁盘 I/O 或 数据增强的耗时
        time.sleep(self.sleep_time)
        return self.data[index]

    def __len__(self):
        return self.size

def measure_dataloader_speed(num_workers, pin_memory, batch_size=32):
    dataset = FakeDataset(size=500, sleep_time=0.01) # 500 个样本，每个样本耗时 10ms
    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )

    start_time = time.time()
    
    # 模拟训练循环：只取数据，不做计算 (Pure I/O bound)
    for batch in loader:
        if pin_memory:
            # 如果开启 pin_memory，通常会立即把数据转到 GPU
            batch = batch.cuda(non_blocking=True)
        pass

    end_time = time.time()
    return end_time - start_time

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("This demo requires a GPU to demonstrate pin_memory effect fully.")
        # Fallback for CPU-only environments (pin_memory won't show speedup)
        device = "cpu"
    else:
        device = "cuda"

    print(f"Running on: {device.upper()}")
    print("-" * 60)
    print(f"{'Workers':<10} | {'Pin Memory':<12} | {'Time (s)':<10} | {'Speedup'}")
    print("-" * 60)

    # Baseline: Single process, no pin memory
    baseline_time = measure_dataloader_speed(num_workers=0, pin_memory=False)
    print(f"{0:<10} | {'False':<12} | {baseline_time:.4f}     | 1.0x (Baseline)")

    # Experiment 1: Multi-process (2 workers)
    t1 = measure_dataloader_speed(num_workers=2, pin_memory=False)
    print(f"{2:<10} | {'False':<12} | {t1:.4f}     | {baseline_time/t1:.2f}x")

    # Experiment 2: Multi-process (4 workers)
    t2 = measure_dataloader_speed(num_workers=4, pin_memory=False)
    print(f"{4:<10} | {'False':<12} | {t2:.4f}     | {baseline_time/t2:.2f}x")

    if device == "cuda":
        # Experiment 3: Multi-process + Pin Memory
        t3 = measure_dataloader_speed(num_workers=4, pin_memory=True)
        print(f"{4:<10} | {'True':<12} | {t3:.4f}     | {baseline_time/t3:.2f}x (Best)")
    else:
        print("\nSkipping Pin Memory test (No GPU detected).")

    print("-" * 60)
    print("\nAnalysis:")
    print("1. num_workers > 0: Prefetching works! The main process doesn't wait for data.")
    print("2. pin_memory=True: Faster transfer to GPU (Host -> Device).")
