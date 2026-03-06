# 快速开始指南

## 🚀 一键运行完整流程

### 方法1: 使用完整流程脚本（推荐）

```bash
cd examples/InRunShapley_LM
bash run_complete_pipeline.sh
```

这将自动执行：
1. 数据集准备
2. 训练（500步）
3. 评估任务
4. 结果分析
5. 论文对比

### 方法2: 分步执行

#### 步骤1: 准备数据集
```bash
python prepare_cifar10.py \
    --output_dir ./data/cifar10_simple \
    --simple \
    --num_samples 5000
```

#### 步骤2: 运行训练
```bash
python train_inrun_shapley.py \
    --data_dir ./data/cifar10_simple \
    --output_dir ./results/cifar10_complete \
    --num_steps 500 \
    --batch_size 8 \
    --val_batch_size 2
```

#### 步骤3: 评估结果
```bash
python evaluate_tasks.py \
    --result_dir ./results/cifar10_complete \
    --mislabeled_indices ./data/cifar10_simple/mislabeled_indices.npy
```

#### 步骤4: 分析对比
```bash
python analyze_results.py --result_dir ./results/cifar10_complete
python compare_with_paper.py --result_dir ./results/cifar10_complete
```

## 📊 快速验证（5分钟）

### 最小测试
```bash
# 1. 准备小数据集
python prepare_dataset.py --dataset_type test --num_samples 1000

# 2. 快速训练（30步）
python train_inrun_shapley.py \
    --data_dir ./data/test_dataset \
    --num_steps 30 \
    --batch_size 4

# 3. 查看结果
python analyze_results.py --result_dir ./results/inrun_shapley_train
```

## 🎯 不同场景的使用

### 场景1: 快速功能验证
```bash
# 使用测试数据集，快速验证代码功能
python prepare_dataset.py --dataset_type test --num_samples 1000
python train_inrun_shapley.py --num_steps 30 --batch_size 4
```

### 场景2: 完整训练
```bash
# 使用更多数据，完整训练
python prepare_dataset.py --dataset_type test --num_samples 10000
python train_inrun_shapley.py --num_steps 500 --batch_size 8
```

### 场景3: 论文复现
```bash
# 使用CIFAR-10数据集，复现论文结果
python prepare_cifar10.py --output_dir ./data/cifar10
python train_inrun_shapley.py \
    --data_dir ./data/cifar10 \
    --num_steps 1000 \
    --batch_size 16
```

## 📝 常用命令

### 训练相关
```bash
# 基本训练
python train_inrun_shapley.py --num_steps 200 --batch_size 4

# 自定义参数
python train_inrun_shapley.py \
    --data_dir ./data/test_dataset \
    --output_dir ./results/my_run \
    --num_steps 500 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --shapley_save_interval 50
```

### 评估相关
```bash
# 基本评估
python evaluate_tasks.py --result_dir ./results/inrun_shapley_train

# 自定义评估
python evaluate_tasks.py \
    --result_dir ./results/inrun_shapley_train \
    --mislabeled_ratio 0.1 \
    --selection_budgets 0.2 0.4 0.6 0.8
```

### 分析相关
```bash
# 结果分析
python analyze_results.py --result_dir ./results/inrun_shapley_train

# 论文对比
python compare_with_paper.py --result_dir ./results/inrun_shapley_train
```

## 🔧 参数说明

### 训练参数
- `--num_steps`: 训练步数（建议：200-1000）
- `--batch_size`: 批次大小（建议：4-16）
- `--val_batch_size`: 验证批次大小（建议：1-4）
- `--block_size`: 序列长度（建议：512-1024）
- `--learning_rate`: 学习率（建议：3e-4）

### 评估参数
- `--mislabeled_ratio`: 错误标签比例（默认：0.1）
- `--selection_budgets`: 选择预算（默认：0.2, 0.4, 0.6, 0.8）

## 📁 输出文件说明

训练完成后，结果保存在 `output_dir` 中：

```
results/
├── grad_dotprods/
│   ├── shapley_values_iter_*.pt      # Shapley值字典
│   ├── shapley_array_iter_*.npy      # Shapley值数组
│   └── shapley_history_iter_*.pt     # 历史记录
├── training_stats.json                # 训练统计
├── training_config.json               # 训练配置
├── evaluation/                        # 评估结果
│   ├── mislabeled_detection_results.json
│   └── data_selection_results.json
└── comparison_with_paper.json         # 论文对比
```

## ⚡ 性能优化建议

### GPU内存优化
```bash
# 减小batch size
python train_inrun_shapley.py --batch_size 2 --val_batch_size 1

# 减小block size
python train_inrun_shapley.py --block_size 256
```

### 训练速度优化
```bash
# 增加batch size（如果内存允许）
python train_inrun_shapley.py --batch_size 16

# 减少保存频率
python train_inrun_shapley.py --shapley_save_interval 100
```

## 🐛 常见问题

### Q: 内存不足怎么办？
A: 减小batch_size和block_size

### Q: 训练太慢怎么办？
A: 减少num_steps或增加batch_size

### Q: 如何查看训练进度？
A: 训练会实时打印loss和Shapley统计

### Q: 结果文件在哪里？
A: 在 `--output_dir` 指定的目录中

## 📚 更多信息

- 详细文档: `README.md`
- 实现说明: `IMPLEMENTATION.md`
- 数据集指南: `DATASET_GUIDE.md`
- 训练结果: `TRAINING_RESULTS.md`

---

**快速开始**: 运行 `bash run_complete_pipeline.sh` 一键完成所有步骤！
