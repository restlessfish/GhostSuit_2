## 实验记录（CIFAR-10N / ResNet18 / In-Run Shapley）

### Table A：吞吐对齐（GPT2-Small，单卡，本机）

吞吐指标：`datapoints/s`（paper Table 1 同口径），报告为训练循环中打印的 `[THROUGHPUT] mean=...`（排除前 10 steps）。

| 配置 | Throughput (datapoints/s) | 备注 |
|---|---:|---|
| Regular（不算 Shapley） | 124.99 | 单卡，本机记录 |
| InRunShapley（热点层 top‑k） | 118.75 | 单卡，`include_layer_names_file=hotspot_layers_gpt2small.json` |
| InRunShapley（全层 hook） | 111.19 | 单卡，include 为空=全层 |
| InRunShapley（子集：h.6~h.11） | 111.65 | 单卡，`include_layer_names_file=gpt2_subset_h6_11.json` |

结论（本机/本配置）：热点层 top‑k 相对全层 hook 的吞吐提升约 **+6.8%**（118.75 / 111.19）。

### Table B：AUROC 任务侧的吞吐（CIFAR-10N / ResNet18，单卡，本机）

吞吐指标：`datapoints/s`（每秒图像样本数）。注意：该表用于“同任务内部”对比，不直接与 GPT2 Table A 或 paper Table 1 的绝对值对比。

| include 子集（ResNet18） | Throughput mean (datapoints/s) | AUROC（历史记录） |
|---|---:|---:|
| layer3+layer4+fc | 1505.71 | 0.7231 |
| layer4+fc | 1783.64 | 0.6939 |
| layer2+layer3+layer4+fc | 1308.10 | 0.7342 |
| 方向1：热点层 top‑k（bench→top‑k→train） | 2033.68 | 0.7317 |

### 子集层 + 跨步验证梯度缓存（save/eval 仅末尾）

配置（四组一致）：
- `tracIn_enable=false`（标准一阶 In-Run Shapley）
- `include_layer_names=layer2+layer3+layer4+fc`
- `num_steps=5000, batch_size=128, val_batch_size=32`
- `shapley_save_interval=5000, eval_interval=5000`
- `seed=0, CUDA_VISIBLE_DEVICES=1`

结果（iter=-1）：

| grad_val_cache_K | 训练时间 (min) | AUROC |
|---:|---:|---:|
| 0  | 4.99 | 0.7415 |
| 2  | 5.03 | 0.7430 |
| 5  | 5.04 | 0.7438 |
| 10 | 5.01 | 0.7389 |

结论（本机/本配置）：`K>0` **未带来稳定提速**（时间基本持平），AUROC 在 \( \pm 0.003 \) 量级波动；本轮最优是 `K=5`。

### 方向 1：dotprod 分层 profiling → 只对“热点层”挂 hook

先跑 `num_steps=300`（全量 hook）做 profiling，bench summary（avg_ms top）显示最慢的层主要集中在：
- `layer2.1.conv1`、`layer3.0.conv2`、`layer3.0.downsample.0`（其余层明显更快）

尝试仅对 top6 热点层挂 hook（`layer2.1.conv1, layer3.0.conv2, layer3.0.downsample.0, layer2.0.downsample.0, layer2.0.conv2, fc`）：

| 配置 | 训练时间 (min) | AUROC |
|---|---:|---:|
| 子集层 `layer2+layer3+layer4+fc`（基线） | 4.99 | 0.7415 |
| 热点层 top6 | **1.46** | 0.6940 |

结论：热点层方案能**大幅提速**，但当前选取 top6 会**明显掉 AUROC**（需要扩大层集合/改为前缀规则扫描，见方向 2）。

