# AUROC问题分析和改进方案

## 问题描述

初始评估结果显示 AUROC = 0.5005，接近随机水平，说明Shapley值无法有效区分mislabeled数据。

## 问题诊断

### 发现的问题

1. **数组大小错误** (已修复)
   - `get_shapley_array()` 使用了错误的数组大小（包含所有token索引）
   - 导致99.8%的样本Shapley值为0
   - **修复**: 使用compact格式，只保存非零值及其索引

2. **评估样本不匹配** (已修复)
   - 评估时使用了包含所有token的大数组
   - 实际只有3,998个训练样本有Shapley值
   - **修复**: 评估脚本现在只使用有Shapley值的样本

### 修复后的结果

- **AUROC**: 0.5005 → **0.5364** ✅ (超过随机水平)
- **PR-AUC**: 0.0001 → **0.1168** ✅
- **Accuracy**: 0.0015 → **0.5058** ✅

## 当前性能分析

### 与论文对比

| 方法 | AUROC | 状态 |
|------|-------|------|
| 论文 - 1st Order In-Run Shapley | 0.678 | 目标 |
| 论文 - 2nd Order In-Run Shapley | 0.680 | 目标 |
| **我们的实现** | **0.5364** | ⚠️ 需要改进 |
| 随机基线 | 0.500 | - |

### 差距分析

**差距**: 0.678 - 0.5364 = **0.1416** (约21%的相对差距)

## 可能的原因

### 1. 数据问题
- **合成数据**: 使用的是简化的CIFAR-10数据，可能不够有代表性
- **Mislabeled数据**: 随机生成的mislabeled索引，可能不够真实
- **数据量**: 只有3,998个训练样本，可能不够

### 2. 模型问题
- **模型大小**: 使用的是GPT-2 small，可能不够大
- **训练步数**: 500步可能不够充分
- **学习率**: 可能需要调整

### 3. Shapley值计算
- **归一化**: 可能需要更好的归一化
- **累积方式**: 当前是简单累加，可能需要加权
- **验证集**: 验证集大小和选择可能影响结果

### 4. 评估方法
- **评估指标**: AUROC可能不是最佳指标
- **阈值选择**: 可能需要更好的阈值选择方法

## 改进方案

### 短期改进（快速验证）

1. **增加训练步数**
   ```bash
   python scripts/train_inrun_shapley.py --num_steps 2000
   ```

2. **调整验证集大小**
   ```bash
   python scripts/train_inrun_shapley.py --val_batch_size 8
   ```

3. **使用真实mislabeled数据**
   - 准备真实的CIFAR-10 mislabeled数据
   - 或使用已知的noisy label数据集

### 中期改进（需要更多工作）

1. **实现二阶方法**
   - 论文中二阶方法性能更好（0.680 vs 0.678）
   - 需要实现Hessian-vector product

2. **优化Shapley值计算**
   - 检查归一化是否正确
   - 考虑使用加权累积

3. **改进验证集选择**
   - 使用更大的验证集
   - 确保验证集有代表性

### 长期改进（研究方向）

1. **使用完整数据集**
   - 使用完整的CIFAR-10数据集
   - 或使用论文中使用的数据集

2. **模型架构**
   - 尝试不同的模型架构
   - 使用论文中使用的ResNet18

3. **超参数调优**
   - 系统性地调优学习率、batch size等
   - 使用grid search或random search

## 验证步骤

### 1. 检查Shapley值分布
```bash
python scripts/diagnose_shapley.py --result_dir results/cifar10_complete
```

### 2. 重新训练并评估
```bash
# 训练
python scripts/train_inrun_shapley.py \
    --data_dir ./data/cifar10_simple \
    --output_dir ./results/cifar10_v2 \
    --num_steps 2000 \
    --val_batch_size 8

# 评估
python scripts/evaluate_tasks.py --result_dir ./results/cifar10_v2
```

### 3. 对比分析
```bash
python scripts/compare_with_paper.py --result_dir ./results/cifar10_v2
```

## 结论

虽然当前AUROC (0.5364) 已经超过随机水平，但距离论文结果 (0.678) 还有差距。主要问题可能在于：

1. ✅ **已修复**: 数组大小和评估样本匹配问题
2. ⚠️ **需要改进**: Shapley值计算和归一化
3. ⚠️ **需要改进**: 数据质量和训练设置
4. ⚠️ **需要改进**: 验证集选择

**下一步**: 尝试增加训练步数、调整超参数，并考虑实现二阶方法。

---

**更新时间**: 2026-03-02  
**状态**: ⚠️ 部分修复，需要进一步改进
