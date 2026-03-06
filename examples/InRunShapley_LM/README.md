# In-Run Data Shapley

基于论文"Data Shapley in One Training Run"的完整实现。

## 📁 项目结构

```
InRunShapley_LM/
├── src/                    # 核心实现模块
│   ├── __init__.py
│   ├── inrun_shapley_engine.py    # Shapley引擎核心实现
│   └── simple_dataloader.py      # 数据加载器
│
├── scripts/                # 可执行脚本
│   ├── __init__.py
│   ├── train_inrun_shapley.py    # 训练脚本
│   ├── evaluate_tasks.py          # 评估脚本
│   ├── analyze_results.py        # 结果分析
│   ├── compare_with_paper.py     # 论文对比
│   ├── prepare_dataset.py         # 数据集准备
│   ├── prepare_cifar10.py        # CIFAR-10准备
│   ├── evaluate_shapley.py        # Shapley评估
│   ├── run_with_local_data.py    # 本地数据训练
│   └── run_complete_pipeline.sh   # 完整流程脚本
│
├── tests/                  # 测试文件
│   ├── __init__.py
│   ├── test_shapley_logic.py      # 核心逻辑测试
│   └── test_inrun_shapley.py      # 集成测试
│
├── docs/                   # 文档目录
│   ├── README.md                  # 使用说明
│   ├── QUICK_START.md             # 快速开始
│   ├── IMPLEMENTATION.md          # 实现说明
│   ├── DATASET_GUIDE.md           # 数据集指南
│   ├── TRAINING_RESULTS.md        # 训练结果
│   ├── FINAL_REPORT.md            # 最终报告
│   ├── COMPLETE_SUMMARY.md        # 完整总结
│   ├── PROJECT_COMPLETE.md       # 项目完成报告
│   ├── RUN_SUCCESS.md            # 运行成功报告
│   ├── NEXT_STEPS.md             # 下一步计划
│   └── PROJECT_STRUCTURE.md      # 项目结构说明
│
├── config_file.py         # 配置文件（兼容性）
├── main.py                # 主程序入口（兼容性）
└── training_loop.py      # 训练循环（兼容性）
```

## 🚀 快速开始

### 一键运行（推荐）

```bash
cd examples/InRunShapley_LM
bash scripts/run_complete_pipeline.sh
```

### 分步执行

```bash
# 1. 准备数据集
python scripts/prepare_cifar10.py --output_dir ./data/cifar10_simple --simple

# 2. 训练
python scripts/train_inrun_shapley.py \
    --data_dir ./data/cifar10_simple \
    --output_dir ./results/cifar10_complete \
    --num_steps 500

# 3. 评估
python scripts/evaluate_tasks.py --result_dir ./results/cifar10_complete

# 4. 分析
python scripts/analyze_results.py --result_dir ./results/cifar10_complete
python scripts/compare_with_paper.py --result_dir ./results/cifar10_complete
```

## 📚 文档

详细文档请查看 `docs/` 目录：
- `docs/QUICK_START.md` - 快速开始指南
- `docs/IMPLEMENTATION.md` - 实现细节
- `docs/DATASET_GUIDE.md` - 数据集准备指南
- `docs/PROJECT_STRUCTURE.md` - 项目结构说明

## 🔧 核心模块

### src/inrun_shapley_engine.py
In-Run Data Shapley引擎核心实现

### src/simple_dataloader.py
简单数据加载器

## 📊 脚本说明

### 训练脚本
- `scripts/train_inrun_shapley.py` - 主训练脚本

### 评估脚本
- `scripts/evaluate_tasks.py` - 任务评估
- `scripts/analyze_results.py` - 结果分析
- `scripts/compare_with_paper.py` - 论文对比

### 数据准备
- `scripts/prepare_dataset.py` - 通用数据集准备
- `scripts/prepare_cifar10.py` - CIFAR-10准备

## 🧪 测试

运行测试：
```bash
python tests/test_shapley_logic.py
python tests/test_inrun_shapley.py
```

## 📖 更多信息

- 论文: "Data Shapley in One Training Run" (ICLR 2025)
- 详细文档: 见 `docs/` 目录
- 结果文件: `results/` 目录

---

**项目状态**: ✅ 完成并可用  
**最后更新**: 2026-03-02
