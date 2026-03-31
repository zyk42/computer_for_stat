# 书名：《stats2algo-cs-guide统计转算法要学的计算机知识》

**目标读者**：数学/统计学/理论计算机科学背景的学生与研究者
**核心痛点**：精通公式推导与算法理论，但缺乏计算机系统底层认知，导致模型跑不快、显存总溢出、分布式训练配不通、工程代码难以维护。
**本书宗旨**：用数学的思维理解计算机系统，补齐从「数学公式」到「高效物理实现」之间的工程鸿沟。

---

## 第一部分：计算的物理边界 (The Physics of Computing)
**核心理念**：计算机不是无限资源的数学抽象，而是受物理限制（时间、空间、能耗、带宽）的实体。任何算法的复杂度分析如果不考虑物理代价，都是耍流氓。

### 第 1 章：存储金字塔与数据搬运 (The Memory Hierarchy)
- 1.1 **从 SRAM 到 HDD：速度与容量的权衡**
    - **L1/L2/L3 Cache**：CPU 内部的极速缓存（ns 级），为什么 Cache Miss 是性能杀手？
    - **DRAM (内存)**：CPU 的工作台，带宽 (~100GB/s) 与延迟 (~100ns) 的物理限制。
    - **HBM (高带宽内存)**：GPU 的专用高速公路，通过 TSV 堆叠实现 TB/s 级带宽，它是大模型训练的核心瓶颈。
    - **SSD/HDD**：数据的永久居所，IOPS 与顺序读写的巨大差异。
- 1.2 **数据搬运的代价**
    - **冯·诺依曼瓶颈**：计算速度 ($10^{12}$ FLOPS) 远快于数据搬运速度 ($10^9$ Bytes/s)。
    - **Roofline 模型**：你的模型是 Compute-bound（算力受限）还是 Memory-bound（带宽受限）？
    - **算术强度 (Arithmetic Intensity)**：每字节数据能进行多少次浮点运算？为什么 Transformer 的 Attention 是 Memory-bound？

### 第 2 章：计算核心与并行范式 (The Compute Engines)
- 2.1 **CPU：逻辑复杂的控制者**
    - **流水线 (Pipeline)** 与 **超标量 (Superscalar)**：如何让一条指令跑得更快。
    - **分支预测 (Branch Prediction)**：为了避免流水线中断，CPU 必须“猜”代码的走向。
    - **SIMD (AVX/NEON)**：单指令多数据，CPU 向量化的基础。
- 2.2 **GPU：吞吐为王的暴力美学**
    - **SIMT (Single Instruction, Multiple Threads)**：成千上万个线程执行同一行代码。
    - **Warp 与 Divergence**：为什么 GPU 害怕 `if-else` 分支？32 个线程必须同进退。
    - **Tensor Core**：专为矩阵乘法设计的硬件单元，混合精度计算的物理基础。
- 2.3 **异构计算与 PCIe/NVLink**
    - **PCIe**：CPU 与 GPU 之间的细水管，数据传输的瓶颈。
    - **NVLink**：GPU 之间的宽带互联，多卡训练的基石。

### 第 3 章：精度与数值的艺术 (Numerical Precision)
- 3.1 **浮点数陷阱 (IEEE 754)**
    - **精度丢失**：大数吃小数，`a + b == a` 的物理现象。
    - **下溢出 (Underflow)** 与 **上溢出 (Overflow)**：梯度消失与爆炸的数值原因。
- 3.2 **混合精度训练 (Mixed Precision)**
    - **FP32 vs FP16 vs BF16**：为什么 BF16 (Brain Floating Point) 成为大模型训练的主流？
    - **Loss Scaling**：如何在低精度下防止梯度归零。
    - **FP8 与量化 (Quantization)**：W4A16, INT8 量化原理与精度损失分析。

---

## 第二部分：从算法到指令——代码是如何运行的 (Execution & Abstraction)
**核心理念**：高级语言（Python）提供了便利的数学抽象，但隐藏了巨大的性能开销。理解“解释”与“编译”的区别，是优化的第一步。

### 第 4 章：Python 的性能真相 (The Interpreter Overhead)
- 4.1 **解释器与动态类型**
    - **PyObject**：一个整数在 Python 中为什么占用 28 字节？
    - **解释开销**：逐行翻译执行的代价。
- 4.2 **全局解释器锁 (GIL)**
    - **多线程的假象**：为什么 Python 多线程跑不满 CPU？
    - **多进程 (Multiprocessing)**：绕过 GIL 的代价（进程间通信 IPC）。
- 4.3 **C++ Extension 与 PyTorch**
    - **Python 胶水层**：如何通过 C++/CUDA 扩展让 Python 变快。
    - **JIT (Just-In-Time) 编译**：`torch.compile` 如何在运行时优化代码。

### 第 5 章：张量的物理视图 (The Physics of Tensors)
- 5.1 **内存布局 (Memory Layout)**
    - **Stride (步长)**：张量在内存中真的是多维数组吗？
    - **Contiguous (连续性)**：`view`, `permute`, `transpose` 的零拷贝原理与潜在陷阱。
    - **Row-major vs Col-major**：矩阵乘法对缓存友好性的影响。
- 5.2 **广播机制 (Broadcasting)**
    - **虚拟扩展**：如何在不复制数据的情况下实现 `(3, 1) + (3)`？
    - **硬件实现**：广播对缓存和内存带宽的节省。

### 第 6 章：数据流水线与 I/O 优化 (Data Pipeline Optimization)
- 6.1 **PyTorch DataLoader 深度解析**
    - **多进程预取 (Prefetching)**：掩盖 I/O 延迟的艺术。
    - **Pin Memory (锁页内存)**：加速 Host 到 Device 的数据传输。
    - **Collate_fn**：Batch 组装的 CPU 瓶颈。
- 6.2 **高性能文件格式与数据表示**
    - **JSON / JSONL**：文本格式的通用性与解析开销（JSONL 对流式读取的友好性）。
    - **CSV / TSV**：扁平数据的经典交换格式。
    - **Parquet / Arrow**：列式存储与零拷贝读取。
    - **HDF5 / NPY**：科学计算的标准容器。
    - **Safetensors**：为什么它比 Pickle 更安全、更快？(mmap 与零拷贝)。
- 6.3 **Linux 系统的 I/O 机制**
    - **Page Cache**：Linux 如何利用空闲内存加速文件读取。
    - **DMA (直接内存访问)**：解放 CPU 的数据搬运工。

---

## 第三部分：AI 加速核心——GPU 编程与算子优化 (GPU Kernel Optimization)
**核心理念**：深度学习框架的核心是算子 (Operator)。理解算子的实现原理，才能理解 FlashAttention 等前沿技术。

### 第 7 章：GPU 架构深度解析 (Deep Dive into GPU)
- 7.1 **SM (Streaming Multiprocessor)**
    - **资源分配**：寄存器 (Register)、共享内存 (Shared Memory) 与线程块 (Thread Block) 的调度限制。
    - **Occupancy (占用率)**：如何让 GPU 保持忙碌？
- 7.2 **显存访问模式**
    - **Coalesced Access (合并访问)**：为什么读写内存要对齐？
    - **Bank Conflict**：共享内存访问冲突导致的串行化。

### 第 8 章：计算图与自动微分 (Computational Graph)
- 8.1 **Autograd 原理**
    - **DAG (有向无环图)**：动态图构建与拓扑排序。
    - **反向传播的内存开销**：为了计算梯度，我们需要保存哪些中间激活值 (Activations)？
- 8.2 **算子融合 (Operator Fusion)**
    - **Kernel Launch Overhead**：启动一个 GPU 核心的固定开销。
    - **带宽节省**：`ReLU(Conv(x))` 融合后减少了一次显存读写。
- 8.3 **FlashAttention 详解**
    - **数学等价性**：如何在不存储 $N \times N$ Attention Matrix 的情况下计算 Attention？
    - **Tiling (分块)**：利用 SRAM (Shared Memory) 减少 HBM 访问次数的经典案例。

### 第 9 章：编译器技术 (Compiler Technologies)
- 9.1 **Triton 语言**
    - **块级编程 (Block-level Programming)**：介于 CUDA 与 Python 之间的抽象。
    - **自动化优化**：Triton 编译器如何处理合并访问与冲突。
- 9.2 **PyTorch 2.0 Inductor**
    - **图捕获 (Graph Capture)**：从动态图到静态图。
    - **代码生成**：自动生成 Triton Kernel 替代手写算子。

---

## 第四部分：打破单机枷锁——分布式训练 (Distributed Training)
**核心理念**：当单机显存和算力不足以支撑大模型时，我们需要通过网络将成千上万个 GPU 连接起来。这需要解决通信、同步与一致性问题。

### 第 10 章：通信的物理限制 (Communication Physics)
- 10.1 **网络拓扑与硬件**
    - **Infiniband vs Ethernet**：RDMA (远程直接内存访问) 为什么重要？
    - **Fat-Tree / Torus / Mesh**：集群网络拓扑对通信效率的影响。
- 10.2 **集合通信原语 (Collectives)**
    - **AllReduce**：数据并行的核心操作。
    - **AllGather / ReduceScatter**：模型并行与 ZeRO 的基础。
    - **Ring / Tree 算法**：如何高效地在 N 个节点间同步数据？

### 第 11 章：并行策略的数学描述 (Parallelism Strategies)
- 11.1 **数据并行 (Data Parallelism, DP)**
    - **DDP (Distributed Data Parallel)**：梯度桶 (Gradient Bucketing) 与计算通信重叠。
    - **ZeRO (Zero Redundancy Optimizer)**：切分优化器状态、梯度与参数，打破显存墙。
- 11.2 **模型并行 (Tensor Parallelism, TP)**
    - **矩阵切分**：Row-wise 与 Column-wise 切分。
    - **通信代价**：每一层 Transformer 需要多少次 AllReduce？
- 11.3 **流水线并行 (Pipeline Parallelism, PP)**
    - **GPipe / 1F1B**：如何减少气泡 (Bubble) 提高设备利用率？
    - **激活重算 (Activation Checkpointing)**：用计算换显存。

### 第 12 章：大模型训练实战 (LLM Training Practice)
- 12.1 **3D 并行混合**
    - **DP + TP + PP**：如何根据集群规模选择最佳组合？
- 12.2 **混合精度训练稳定性**
    - **NaN 调试**：梯度裁剪 (Gradient Clipping) 与溢出检测。
- 12.3 **长序列训练**
    - **Sequence Parallelism**：突破 Sequence Length 限制。

---

## 第五部分：全链路实战——从工程到生产 (Engineering & Production)
**核心理念**：算法只是系统的一部分。工程能力决定了模型能否真正落地。

### 第 13 章：Linux 环境与高性能运维 (Linux & DevOps)
- 13.1 **Shell 编程与自动化**
    - **文本处理**：grep, awk, sed 在日志分析中的应用。
    - **任务调度**：Slurm 集群调度基础。
- 13.2 **性能分析工具 (Profiling)**
    - **系统级**：top, htop, nvtop, dstat。
    - **代码级**：cProfile, py-spy, line_profiler。
    - **GPU 级**：Nsight Systems, PyTorch Profiler (Trace 分析)。

### 第 14 章：工程化生存指南 (Project Engineering)
- 14.1 **解剖 GitHub 项目：如何读懂别人的代码 (Anatomy of a ML Project)**
    - **标准目录心理模型**：
        - `src/` 或 `package_name/`：核心算法逻辑，数学公式的物理实现地。
        - `configs/`：所有超参数的控制台 (YAML/Hydra)，修改模型行为不需要改代码。
        - `scripts/` 或 `tools/`：一键启动的脚本 (Bash/Python)，从这里寻找 `entry point`。
        - `tests/`：理解代码预期行为的最佳文档，看测试用例比看注释更管用。
    - **环境还原 (Reproduction)**：
        - 看到 `environment.yaml` $\rightarrow$ `conda env create`
        - 看到 `poetry.lock` $\rightarrow$ `poetry install`
        - 看到 `Dockerfile` $\rightarrow$ 最安全但最重的运行方式。
- 14.2 **从 Clone 到 Run：运行与调试策略 (Debugging Strategy)**
    - **寻找入口 (Entry Point)**：不要只盯着 `README`，去 `setup.py` 看 `entry_points`，或去 `scripts/train.sh` 看实际命令。
    - **最小化运行 (Sanity Check)**：修改配置，将 `batch_size=1`, `max_steps=10`，在 CPU 上跑通完整流程，确保环境无误。
    - **Overfit Tiny Batch**：在 10 个样本上训练，确保 Loss 能迅速下降到 0，验证代码逻辑与梯度计算的正确性。
- 14.3 **Git 与版本控制**
    - **Git Flow**：分支管理策略。
    - **Pre-commit Hook**：代码质量的守门员。
- 14.4 **依赖与环境管理**
    - **Docker**：容器化与环境隔离，解决 "It works on my machine" 问题。
    - **Poetry / Conda**：Python 依赖管理的最佳实践。
- 14.5 **配置与实验管理**
    - **Hydra / OmegaConf**：层次化配置管理。
    - **WandB / MLflow**：实验追踪与可视化。

### 第 15 章：推理与服务化 (Inference & Serving)
- 15.1 **大模型推理优化**
    - **KV Cache**：用显存换计算，加速自回归生成。
    - **PagedAttention**：vLLM 的核心，解决 KV Cache 的内存碎片化。
    - **Continuous Batching**：动态批处理提高吞吐量。
- 15.2 **模型压缩与量化**
    - **PTQ (Post-Training Quantization)** vs **QAT (Quantization Aware Training)**。
    - **AWQ / GPTQ**：权重与激活的量化策略。
- 15.3 **服务化架构**
    - **Triton Inference Server**：多模型部署与并发处理。
    - **gRPC vs HTTP**：通信协议的选择。

---

## 附录 (Appendices)
- **A. 常用 Linux 命令速查表 (AI 工程师版)**
- **B. 性能分析工具图谱**
- **C. 推荐阅读书单 (CSAPP, OSTEP, DDIA 等)**
- **D. 关键数学公式与物理实现的对照表**
