## In-Run Data Shapley（CIFAR-10N / ResNet18，一阶版本）

这个精简后的仓库只保留了与 **In-Run Data Shapley 一阶 Shapley 计算** 直接相关的代码和脚本，
用于在 CIFAR-10N / ResNet18 设置下重现实验结果并方便后续复现与调试。

### 1. 环境准备

```bash
conda create -n shapley311 python=3.11 -y
conda activate shapley311
pip install -r requirements.txt
```

默认假设：
- CUDA 环境可用（用于加速 ResNet18 训练）
- 当前工作目录为 `GhostSuit_2` 根目录

### 2. 一阶 In-Run Shapley：CIFAR-10N / ResNet18 / 5000 步

**训练命令：**

```bash
cd /home/mengkahan3/GhostSuit_2

conda run -n shapley311 python examples/InRunShapley_LM/scripts/train_cifar10_resnet18_inrun_shapley.py \
  --data_root ./data/cifar10 \
  --output_dir ./results/cifar10_resnet18_full_5000_order1_opt \
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
- 训练 ResNet18 模型 5000 步
- 在训练过程中实时计算一阶 In-Run Shapley 值并周期性保存到  
  `./results/cifar10_resnet18_full_5000_order1_opt/grad_dotprods/`
- 保存真实错标索引 `mislabeled_indices.npy`

**评估命令（错标检测 AUROC）：**

```bash
conda run -n shapley311 python examples/InRunShapley_LM/scripts/evaluate_tasks.py \
  --result_dir ./results/cifar10_resnet18_full_5000_order1_opt \
  --mislabeled_indices ./results/cifar10_resnet18_full_5000_order1_opt/mislabeled_indices.npy
```

### 3. 本次一阶 Shapley 的关键结果

在上述配置下，本环境中得到的典型结果为（使用真实 CIFAR-10N 噪声标签）：

- **模型性能：**
  - 最佳测试准确率（ResNet18）：≈ **0.6104**

- **Mislabeled Data Detection（基于一阶 In-Run Shapley）：**
  - **AUROC: 0.7253**
  - PR-AUC: 0.1733
  - Accuracy: 0.5562
  - Shapley 值统计（最终迭代）：
    - mean ≈ −1.4e‑5
    - std ≈ 4.4e‑5
    - min ≈ −3.76e‑4
    - max ≈ 1.08e‑4

与论文 *“Data Shapley in One Training Run”* 中报告的一阶 In-Run Shapley AUROC=0.678 相比，
当前实现的 **一阶 Shapley 在同一任务上已经取得更优的错标检测效果（0.7253 > 0.678）**。

### 4. 与一阶 Shapley 直接相关的关键代码

- `ghostEngines/`
  - `graddotprod_engine.py`：梯度点积引擎（挂载到优化器）
  - `autograd_grad_sample_dotprod.py`：逐样本梯度 / dot-product 钩子实现
  - `supported_layers_grad_samplers_dotprod.py`：
    - `_compute_linear_dot_product`：线性层的一阶 `<g_i, g_val>` 计算（已做公共 `grad_val` 重用的等价优化）

- `examples/InRunShapley_LM/`
  - `src/inrun_shapley_engine.py`：In-Run Shapley 引擎（基于 dot-product 聚合并累积一阶 Shapley 值）
  - `scripts/train_cifar10_resnet18_inrun_shapley.py`：CIFAR-10N / ResNet18 训练 + In-Run Shapley 主脚本
  - `scripts/evaluate_tasks.py`：从 `grad_dotprods` 目录读取 Shapley 数组并评估错标检测 AUROC 等指标

如需在此基础上继续改进（例如引入二阶修正、更多 PSE 风格的公共子树缓存、或其它模型/数据集），
直接在上述脚本和引擎中修改即可，再按本 README 中的命令重新训练与评估。

