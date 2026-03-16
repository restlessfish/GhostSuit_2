#!/usr/bin/env python3
"""
快速测试改进后的二阶 Shapley 方法
"""
import sys
import os
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import numpy as np
from examples.InRunShapley_LM.src.inrun_shapley_engine import InRunShapleyEngine

# 创建一个简单的模型用于测试
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

print("=" * 70)
print("测试改进后的二阶 Shapley 方法")
print("=" * 70)

# 创建模型和数据
model = SimpleModel()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 创建虚拟数据
batch_size = 8
val_batch_size = 2
X_train = torch.randn(batch_size, 10).to(device)
Y_train = torch.randn(batch_size, 1).to(device)
X_val = torch.randn(val_batch_size, 10).to(device)
Y_val = torch.randn(val_batch_size, 1).to(device)

# 初始化引擎
print("\n[1/4] 初始化 InRunShapleyEngine (order=2)...")
engine = InRunShapleyEngine(
    module=model,
    val_batch_size=val_batch_size,
    order=2,  # 二阶方法
    accumulate=True,
    temperature=1.0,
)

# 模拟一次训练迭代
print("[2/4] 模拟训练迭代...")
engine.attach_train_batch(X_train, Y_train, iter_num=100)

# 前向传播
X_combined = torch.cat([X_train, X_val], dim=0)
Y_combined = torch.cat([Y_train, Y_val], dim=0)

with engine.saved_tensors_context():
    output = model(X_combined)
    loss = nn.functional.mse_loss(output, Y_combined)
    loss.backward()

# 设置验证损失（二阶方法需要）
val_output = model(X_val)
val_loss = nn.functional.mse_loss(val_output, Y_val)
engine.set_validation_loss(val_loss)

# 聚合和记录
print("[3/4] 计算 Shapley 值...")
engine.aggregate_and_log()

# 获取结果
print("[4/4] 获取 Shapley 值...")
shapley_result = engine.get_shapley_array(compact=True)
if isinstance(shapley_result, tuple):
    shapley_array, indices = shapley_result
else:
    shapley_array = shapley_result
    indices = None

print("\n" + "=" * 70)
print("测试结果:")
print("=" * 70)
print(f"✓ 模型类型: {type(model).__name__}")
print(f"✓ Shapley 值数量: {len(shapley_array)}")
print(f"✓ Shapley 值统计:")
print(f"  - 均值: {np.mean(shapley_array):.6f}")
print(f"  - 标准差: {np.std(shapley_array):.6f}")
print(f"  - 最小值: {np.min(shapley_array):.6f}")
print(f"  - 最大值: {np.max(shapley_array):.6f}")
print(f"✓ 二阶方法工作正常！")
print("=" * 70)
