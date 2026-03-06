# In-Run Data Shapley 实现说明

本文档说明了对GhostSuite-0.35的改进，实现了论文"Data Shapley in One Training Run"中的In-Run Data Shapley方法。

## 改进内容

### 1. 核心引擎实现 (`inrun_shapley_engine.py`)

创建了`InRunShapleyEngine`类，继承自`GradDotProdEngine`，添加了以下功能：

- **Shapley值计算**：从梯度点积计算Shapley值
- **累积机制**：在整个训练过程中累积Shapley值
- **一阶方法**：使用梯度点积（已实现）
- **二阶方法**：框架已准备，待实现梯度-海塞矩阵-梯度乘积

关键方法：
- `_compute_shapley_from_dot_product()`: 从梯度点积计算Shapley值
- `aggregate_and_log()`: 重写父类方法，添加Shapley值计算和累积
- `save_shapley_values()`: 保存累积的Shapley值
- `get_shapley_array()`: 获取Shapley值的numpy数组

### 2. 训练循环 (`training_loop.py`)

创建了`InRunShapleyTrainer`类，专门用于In-Run Shapley训练：

- 集成Shapley引擎
- 在每个训练步骤后保存Shapley值
- 记录Shapley统计信息到WandB（如果启用）

### 3. 配置管理 (`config_file.py`)

创建了`InRunShapleyConfig`类，包含：

- Shapley特定参数（order, accumulate_shapley, save_interval）
- 结果目录管理
- WandB配置

### 4. 引擎管理器更新 (`ghostEngines/engine_manager.py`)

更新了`GhostEngineManager`以支持InRunShapley方法：

- 添加`_initialize_inrun_shapley_engine()`方法
- 更新`should_save_metrics()`和`save_metrics()`方法
- 更新`prepare_forward_input()`以支持InRunShapley
- 更新清理逻辑以保存最终Shapley值

### 5. 评估脚本 (`evaluate_shapley.py`)

提供了评估功能：

- **Mislabeled data detection**: 使用Shapley值检测错误标签
- **Data selection**: 基于Shapley值选择数据
- **Distribution analysis**: 分析Shapley值分布

### 6. 主程序 (`main.py`)

提供了完整的训练入口点，集成了所有组件。

## 使用方法

### 基本训练

```bash
cd examples/InRunShapley_LM
python main.py \
    --method InRunShapley \
    --architecture GPT2-Small \
    --batch_size 16 \
    --val_batch_size 1 \
    --max_steps 10000 \
    --shapley_order 1 \
    --shapley_save_interval 10
```

### 评估

```bash
python evaluate_shapley.py \
    --shapley_dir ./results/inrun_shapley/.../grad_dotprods \
    --mislabeled_indices path/to/mislabeled_indices.npy \
    --output_file evaluation_results.json
```

## 输出文件

训练过程会生成以下文件：

1. `shapley_values_iter_{iter_num}.pt`: 累积的Shapley值字典
2. `shapley_array_iter_{iter_num}.npy`: Shapley值的numpy数组
3. `shapley_history_iter_{iter_num}.pt`: 每迭代的Shapley值历史
4. `dot_prod_log_iter_{iter_num}.pt`: 原始梯度点积

## 与论文的对应关系

### 论文中的方法

1. **一阶方法**（Section 4.2）：
   - 使用梯度点积 `<∇_θ L(z_i), ∇_θ L(D_val)>`
   - 已实现：`_compute_shapley_from_dot_product()`

2. **二阶方法**（Section 4.3）：
   - 使用梯度-海塞矩阵-梯度乘积
   - 框架已准备，待实现

3. **累积机制**（Section 3）：
   - 在整个训练过程中累积Shapley值
   - 已实现：`aggregate_and_log()`和`shapley_values`字典

## 待完成的工作

1. **二阶方法实现**：
   - 实现高效的Hessian-vector product计算
   - 更新`_compute_second_order_shapley()`方法

2. **性能优化**：
   - 优化内存使用
   - 支持更大的batch size

3. **更多评估指标**：
   - 添加更多评估任务
   - 与baseline方法对比

## 文件结构

```
examples/InRunShapley_LM/
├── main.py                    # 主程序入口
├── config_file.py             # 配置管理
├── training_loop.py           # 训练循环
├── inrun_shapley_engine.py    # Shapley引擎实现
├── evaluate_shapley.py        # 评估脚本
├── README.md                   # 使用说明
└── IMPLEMENTATION.md          # 本文档
```

## 参考文献

Wang, J. T., Mittal, P., Song, D., & Jia, R. (2025). Data Shapley in One Training Run. ICLR 2025.
