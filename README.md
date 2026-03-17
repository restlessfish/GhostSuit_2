## In-Run Data Shapley（一阶，CIFAR-10N / ResNet18）

本仓库现在只关注 **一阶 In-Run Data Shapley** 在 CIFAR-10N / ResNet18 上的实验复现，方便你下次一键跑通并对照结果。

### 1. 环境准备

```bash
conda create -n shapley311 python=3.11 -y
conda activate shapley311
pip install -r requirements.txt
```

假设：
- 当前工作目录为 `GhostSuit_2` 根目录
- 有可用的 CUDA GPU（用于加速 ResNet18 训练）

### 2. 一阶 In-Run Shapley 运行配置（“原始”一阶实现）

**训练命令（5000 步，CIFAR-10N / ResNet18）：**

```bash
cd /home/mengkahan3/GhostSuit_2

conda run -n shapley311 python examples/InRunShapley_LM/scripts/train_cifar10_resnet18_inrun_shapley.py \
  --data_root ./data/cifar10 \
  --output_dir ./results/cifar10_resnet18_full_5000_order1_base \
  --num_steps 5000 \
  --batch_size 128 \
  --val_batch_size 32 \
  --learning_rate 1e-3 \
  --noise_type aggre_label \
  --shapley_order 1 \
  --shapley_save_interval 1000 \
  --eval_interval 500 \
  --num_workers 4 \
  --device cuda
```

训练脚本会：
- 自动下载 CIFAR-10 与 CIFAR-10N (`aggre_label`) 噪声标签到 `./data/cifar10`
- 训练 ResNet18 共 5000 步，并在训练过程中实时计算一阶 In-Run Shapley 值
- 周期性保存 Shapley 输出到  
  `./results/cifar10_resnet18_full_5000_order1_base/grad_dotprods/`
- 保存真实错标索引 `mislabeled_indices.npy`

**评估命令（Mislabeled Detection AUROC）：**

```bash
conda run -n shapley311 python examples/InRunShapley_LM/scripts/evaluate_tasks.py \
  --result_dir ./results/cifar10_resnet18_full_5000_order1_base \
  --mislabeled_indices ./results/cifar10_resnet18_full_5000_order1_base/mislabeled_indices.npy
```

### 3. 本次一阶 Shapley 的运行结果

在上述配置和“未做 PSE 式 grad_val 重用”的一阶实现下，本环境中得到的结果（使用真实 CIFAR-10N 噪声标签）：

- **模型性能（ResNet18）：**
  - 最佳测试准确率：≈ **0.6171**

- **Mislabeled Data Detection（基于一阶 In-Run Shapley）：**
  - **AUROC: 0.7217**
  - PR-AUC: 0.1721
  - Accuracy: 0.5552
  - Shapley 值统计（最终迭代 `iter_5000`）：
    - mean ≈ −1.4×10⁻⁵
    - std ≈ 4.6×10⁻⁵
    - min ≈ −3.28×10⁻⁴
    - max ≈ 1.15×10⁻⁴

与论文 *“Data Shapley in One Training Run”* 中报告的一阶 In-Run Shapley AUROC=0.678 相比，
当前 **原始一阶实现** 在同一 CIFAR-10N / ResNet18 错标检测任务上已经取得更优表现（0.7217 > 0.678）。

### 4. 与一阶 Shapley 直接相关的核心代码

- `ghostEngines/`
  - `graddotprod_engine.py`：梯度点积引擎（挂载到优化器，负责保存 dot product 日志）
  - `autograd_grad_sample_dotprod.py`：逐样本梯度 / dot-product 钩子实现；当某些层无法捕获 activation 时，会安全 fallback（该层点积视为 0）以保证训练不中断。
  - `supported_layers_grad_samplers_dotprod.py`：
    - `_compute_linear_dot_product`：线性层的一阶 `<g_i, g_val>` 计算（此处保持“原始版本”，未做 grad_val 单次重用的 PSE 改动）
    - `_compute_conv2d_dot_product`：Conv2d 点积实现（当前强制使用 materialize 路径，避免 ghost 分支形状不匹配）。

- `examples/InRunShapley_LM/`
  - `src/inrun_shapley_engine.py`：In-Run Shapley 引擎（基于 dot-product 聚合，并在训练过程中累积一阶 Shapley 值）
  - `scripts/train_cifar10_resnet18_inrun_shapley.py`：本 README 中使用的 CIFAR-10N / ResNet18 训练 + In-Run Shapley 主脚本
  - `scripts/evaluate_tasks.py`：从 `grad_dotprods` 目录读取 Shapley 数组并评估错标检测 AUROC 等指标

只要按本 README 中的两条命令重新训练与评估，即可完整复现当前的一阶 Shapley 结果。如果后续你想再做一阶或二阶的改进，可以直接基于以上这些文件继续修改。

# "Ghost" Suites for Fast Gradient Information Calculation


## Introduction
Computing per-sample gradient information and pair-wise gradient similarity is often the computational bottleneck for data-centric research (e.g., data selection, synthetic data generation). A naive approach would require setting the batch size to 1, backpropagating on the loss of each training sample, and storing all the huge gradient vectors. Consequently, this approach would be computationally prohibitive for practical applications. 

In [Data Shapley in One Training Run](https://openreview.net/pdf?id=HD6bWcj87Y) (ICLR'25 Outstanding Paper Runner-up), we proposed a highly efficient method to obtain per-sample gradient information. It turns out that we can compute the gradient dot-product between every pair of data points within a large batch in just a single backpropagation. At high level, the technique exploits information that's already being computed during standard backpropagation with respect to the aggregated loss on a batch of data points. 

This repository provides a clean, drop-in implementation of "ghost"-based techniques for fast per-sample gradient information calculation. Our goal is to enable per-sample gradient computation and extraction with **minimal code changes**—often just a few lines added to your existing model training loop.


## Available Engines
- `GradDotProdEngine`
  - Purpose: Online computation of gradient similarities between validation loss and individual training samples in a single backprop pass.
  - Core idea: Reuse activations and output gradients already computed during backprop to obtain per‑parameter dot products without materializing model‑sized gradients; typically concatenates a small validation batch with the training batch.
  - Best for: computing pair-wise gradient similarities through the entire training process (e.g., online data selection, reweighting, curriculum learning, or analyzing training dynamics). 

- `GradProjLoRAEngine`
  - Purpose: Offline, corpus‑scale analysis by storing low‑dimensional per‑sample gradient projections to disk for later similarity analysis.
  - Core idea: Similar to `GradDotProdEngine`, we can reuse activations and output gradients already computed during backprop. Instead of directly computing gradient similarity, we store these per-sample info to disks. Specifically, we can apply a Kronecker‑structured random projection $P = P_i \otimes P_o$. This can be elegantly implemented through a zero‑impact [LoRA‑style side branch](https://arxiv.org/pdf/2405.13954); no changes to model behavior.
  - Best for: computing pair-wise gradient similarities for a large dataset w.r.t. a fixed model checkpoint.  

Logic and when to use which
- Both engines exploit the same gradient structure to avoid instantiating full gradients and add minimal training overhead.
- Use GradDotProd when you need on‑the‑fly similarities within a step (e.g., *online data selection or reweighting, curriculum learning, auditing training dynamics*).
- Use GradProjLoRA when you need reusable per‑sample representations across many batches or the whole corpus (e.g., *offline data selection, clustering, etc*). These projections preserve inner products up to JL distortion. 


## Installation
```bash
pip install -r requirements.txt
```


## Quick Start

In `examples/`, we provide three minimal examples for demonstrating core usage of GhostEngines:

- **`ghost_mlp.py`**: Basic GradDotProd usage for MLP models
  - Trains for 10 steps on synthetic data
  - Prints per-parameter gradient dot-products

- **`ghost_gradproj_mlp.py`**: Per-sample gradient projection computation and storage for MLP

- **`ghost_gradproj_lm.py`**: Per-sample gradient projection computation and storage for language models
  - Projects gradients for transformer layers
  - Demonstrates similarity computation from saved projections

### LLM pretraining with TorchTitan
TorchTitan (https://github.com/pytorch/torchtitan) is a PyTorch-native training stack for large-scale model development and experimentation.

Run the TorchTitan GradDotProd integration with the Llama 3 130M ghost config:

```bash
CONFIG_FILE="./examples/torchtitan/torchtitan/models/llama3/train_configs/llama3_130m_ghost.toml" ./examples/torchtitan/run_train_with_ghost.sh
```

The language-model examples under `examples/GradDotProd_LM/` and `examples/GradProj_LM/` are deprecated in v0.33 and will be fixed soon. If you need to run them, please use v0.2: https://github.com/Jiachen-T-Wang/GhostSuite/tree/v0.2


## How the Ghost Engines Work

### GradDotProd Engine
1. **Batch Concatenation**: Training and validation batches are concatenated for a single forward pass
2. **Gradient Computation**: During backpropagation, the engine computes:
   - Per-parameter gradient dot products between validation and training samples. 
   - Aggregated training gradients are recovered seperately and stored in `.grad` before optimizer step. 

### GradProj Engine
- Uses LoRA-style low-rank projection matrices
- Projects high-dimensional gradients to lower-dimensional space
- Enables efficient per-sample gradient storage without materializing full gradients
- Supports both MLP and attention layer projections

See individual example directories for detailed documentation and configuration options.



## Integrating Ghost Engine with Your Training Loop

The `GhostEngineManager` provides a convenient interface for integrating gradient computation engines into your training loop. Here's an overview of how to modify your training loop:

```python
from ghostEngines import GhostEngineManager

# 1. Initialize the Ghost Engine Manager
ghost_engine = GhostEngineManager(
    config=config,                    # Your training configuration
    model=model,                      # PyTorch model
    optimizer=optimizer,              # Model optimizer
    ddp_info={"master_process": is_master},  # Distributed info for logging/saving
    val_data=(X_val, Y_val),          # Validation data (required for GradDotProd)
)

# 2. Training loop with Ghost Engine integration
for iteration in range(max_steps):
    # Get training batch
    X_train, Y_train, batch_idx = get_batch()

    optimizer.zero_grad(set_to_none=True)

    # Attach batch information to engine
    ghost_engine.attach_train_batch(X_train, Y_train, iteration, batch_idx)

    # Prepare input (concatenates val data for GradDotProd method)
    X_forward, Y_forward = ghost_engine.prepare_forward_input(X_train, Y_train)
    
    # Forward and backward pass (capture saved tensors for GradDotProd)
    with ghost_engine.saved_tensors_context():
        outputs = model(input_ids=X_forward, labels=Y_forward)
        loss = outputs.loss
        loss.backward()

    # Ghost engine gradient processing
    ghost_engine.prepare_gradients()    # Move accumulated gradients to .grad

    # Optimizer step
    optimizer.step()

    # Ghost engine post-processing
    ghost_engine.aggregate_and_log()    # Compute and log gradient metrics
    ghost_engine.clear_gradients()      # Clean up stored gradients
    
    # Periodic metric saving
    if ghost_engine.should_save_metrics(iteration):
        ghost_engine.save_metrics(iteration)
```

Notes:
- `saved_tensors_context()` is required for `GradDotProd` and is a no-op for other methods.
- With gradient accumulation, call `aggregate_and_log()` after each microbatch and move `prepare_gradients()`/`optimizer.step()` to the end of the accumulation window.
- For no-grad evaluation, use `ghost_engine.detach_for_evaluation()` and `ghost_engine.reattach_after_evaluation()`.


## Citation

```bibtex
@article{wang2024data,
  title={Data shapley in one training run},
  author={Wang, Jiachen T and Mittal, Prateek and Song, Dawn and Jia, Ruoxi},
  journal={arXiv preprint arXiv:2406.11011},
  year={2024}
}
```
