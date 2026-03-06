# 数据集准备指南

本指南说明如何为In-Run Data Shapley准备数据集。

## 快速开始

### 1. 创建测试数据集（推荐用于快速验证）

```bash
cd examples/InRunShapley_LM

# 创建小型测试数据集（5000个样本）
python prepare_dataset.py \
    --dataset_type test \
    --output_dir ./data/test_dataset \
    --num_samples 5000 \
    --seq_length 1024
```

这将创建：
- `train.bin`: 训练数据（5,120,000 tokens）
- `val.bin`: 验证数据（102,400 tokens）
- `test.bin`: 测试数据（102,400 tokens）

### 2. 使用测试数据集运行训练

```bash
python run_with_local_data.py \
    --data_dir ./data/test_dataset \
    --output_dir ./results/test_run \
    --num_steps 20
```

## 数据集类型

### 测试数据集（Test Dataset）

**用途**: 快速验证代码功能

**特点**:
- 合成数据，无需下载
- 快速生成
- 适合功能测试

**创建命令**:
```bash
python prepare_dataset.py --dataset_type test \
    --output_dir ./data/test_dataset \
    --num_samples 5000 \
    --seq_length 1024
```

### Pile数据集（Pile Dataset）

**用途**: 完整训练，复现论文结果

**特点**:
- 真实数据
- 需要网络连接下载
- 数据量大

**创建命令**:
```bash
python prepare_dataset.py --dataset_type pile \
    --output_dir ./data/pile_dataset \
    --max_tokens 1000000000  # 限制为10亿tokens
```

**注意**: 需要安装 `datasets` 库：
```bash
pip install datasets
```

### 简单文本数据集（Simple Text Dataset）

**用途**: 使用自定义文本

**特点**:
- 可以自定义文本内容
- 适合小规模实验

**创建命令**:
```bash
python prepare_dataset.py --dataset_type simple \
    --output_dir ./data/simple_dataset
```

## 数据集格式

所有数据集都保存为二进制文件格式：

- **文件格式**: `.bin` 文件
- **数据类型**: `uint16` (numpy)
- **内容**: Tokenized文本数据（GPT-2 token IDs）

### 文件结构

```
data/
├── train.bin    # 训练数据
├── val.bin      # 验证数据
└── test.bin     # 测试数据
```

## 使用数据集

### 方法1: 使用SimpleDataLoader（推荐）

```python
from simple_dataloader import SimpleDataLoader

# 加载数据集
dataloader = SimpleDataLoader('./data/test_dataset', block_size=1024)

# 获取批次
X_train, Y_train = dataloader.get_batch('train', batch_size=16, device='cuda')
X_val, Y_val = dataloader.get_batch('val', batch_size=2, device='cuda')
```

### 方法2: 直接使用numpy memmap

```python
import numpy as np

# 加载数据
train_data = np.memmap('data/test_dataset/train.bin', dtype=np.uint16, mode='r')

# 提取序列
block_size = 1024
start_idx = 0
sequence = train_data[start_idx:start_idx+block_size]
```

## 数据集大小建议

### 快速测试
- **样本数**: 1,000 - 5,000
- **序列长度**: 512 - 1024
- **总tokens**: ~1M - 5M

### 小规模实验
- **样本数**: 10,000 - 50,000
- **序列长度**: 1024
- **总tokens**: ~10M - 50M

### 完整训练
- **样本数**: 无限制（使用Pile数据集）
- **序列长度**: 1024
- **总tokens**: 10B+ (10亿+)

## 验证数据集

检查数据集是否正确创建：

```bash
python -c "
import numpy as np
import os

data_dir = './data/test_dataset'
train_file = os.path.join(data_dir, 'train.bin')

if os.path.exists(train_file):
    data = np.memmap(train_file, dtype=np.uint16, mode='r')
    print(f'Train data: {len(data):,} tokens')
    print(f'Data range: [{data.min()}, {data.max()}]')
    print('✓ Dataset is valid')
else:
    print('✗ Dataset not found')
"
```

## 常见问题

### Q: 数据集文件太大怎么办？

A: 使用 `--max_tokens` 参数限制数据集大小：
```bash
python prepare_dataset.py --dataset_type pile --max_tokens 100000000
```

### Q: 如何创建自定义数据集？

A: 修改 `prepare_simple_text_dataset()` 函数，提供你自己的文本列表。

### Q: 数据集路径在哪里配置？

A: 数据集路径可以在运行时指定：
```bash
python run_with_local_data.py --data_dir /path/to/your/data
```

### Q: 支持其他数据集格式吗？

A: 当前支持二进制格式（.bin）。如果需要其他格式，可以修改 `simple_dataloader.py`。

## 下一步

1. ✅ 创建测试数据集
2. ✅ 运行快速验证
3. ⏭️ 准备完整数据集（如需要）
4. ⏭️ 运行完整训练
5. ⏭️ 评估结果
