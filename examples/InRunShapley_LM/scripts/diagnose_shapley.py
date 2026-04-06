#!/usr/bin/env python3
"""
诊断Shapley值计算问题
"""

import os
import sys
import numpy as np
import json
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def diagnose_shapley_values(result_dir):
    """诊断Shapley值的问题"""
    grad_dotprods_dir = os.path.join(result_dir, 'grad_dotprods')
    
    # 加载最新的Shapley值
    shapley_files = list(Path(grad_dotprods_dir).glob('shapley_array_iter_*.npy'))
    if not shapley_files:
        print("未找到Shapley值文件")
        return
    
    latest_file = max(shapley_files, key=lambda p: int(p.stem.split('_')[-1]))
    iter_num = int(latest_file.stem.split('_')[-1])
    
    print(f"=== 诊断 Shapley值 (iteration {iter_num}) ===")
    print(f"文件: {latest_file}")
    
    shapley_array = np.load(latest_file)
    print(f"\n数组大小: {len(shapley_array):,}")
    
    # 统计
    non_zero_mask = shapley_array != 0
    non_zero_count = np.sum(non_zero_mask)
    zero_count = len(shapley_array) - non_zero_count
    
    print(f"\n值统计:")
    print(f"  非零值: {non_zero_count:,} ({non_zero_count/len(shapley_array)*100:.2f}%)")
    print(f"  零值: {zero_count:,} ({zero_count/len(shapley_array)*100:.2f}%)")
    
    if non_zero_count > 0:
        non_zero_values = shapley_array[non_zero_mask]
        print(f"\n非零值统计:")
        print(f"  均值: {np.mean(non_zero_values):.8f}")
        print(f"  标准差: {np.std(non_zero_values):.8f}")
        print(f"  最小值: {np.min(non_zero_values):.8f}")
        print(f"  最大值: {np.max(non_zero_values):.8f}")
        print(f"  正值: {np.sum(non_zero_values > 0):,}")
        print(f"  负值: {np.sum(non_zero_values < 0):,}")
    
    # 检查训练统计
    stats_file = os.path.join(result_dir, 'training_stats.json')
    if os.path.exists(stats_file):
        with open(stats_file) as f:
            stats = json.load(f)
        print(f"\n训练统计:")
        if 'shapley' in stats:
            s = stats['shapley']
            print(f"  Shapley样本数: {s.get('num_samples', 'N/A')}")
            print(f"  有值的样本: {s.get('positive_count', 0) + s.get('negative_count', 0)}")
    
    # 检查原始数据文件
    pt_file = os.path.join(grad_dotprods_dir, f"shapley_values_iter_{iter_num}.pt")
    if os.path.exists(pt_file):
        data = torch.load(pt_file, map_location='cpu')
        print(f"\n原始数据:")
        print(f"  保存的样本数: {data.get('num_samples', len(data.get('shapley_values', {})))}")
        shapley_dict = data.get('shapley_values', {})
        if shapley_dict:
            values = list(shapley_dict.values())
            print(f"  字典中的值: {len(values)}")
            print(f"  非零值: {sum(1 for v in values if v != 0)}")
    
    # 问题诊断
    print(f"\n=== 问题诊断 ===")
    if zero_count / len(shapley_array) > 0.9:
        print("⚠️  问题: 超过90%的样本Shapley值为0")
        print("   可能原因:")
        print("   1. get_shapley_array()使用了错误的数组大小")
        print("   2. 样本索引映射不正确")
        print("   3. Shapley值没有正确累积")
    
    if non_zero_count > 0 and np.std(non_zero_values) < 1e-6:
        print("⚠️  问题: Shapley值方差太小")
        print("   可能原因:")
        print("   1. Shapley值计算不正确")
        print("   2. 归一化过度")
    
    if non_zero_count == 0:
        print("❌ 严重问题: 所有Shapley值都是0")
        print("   可能原因:")
        print("   1. Shapley值计算逻辑错误")
        print("   2. 梯度点积计算错误")
        print("   3. 累积逻辑错误")

if __name__ == "__main__":
    from pathlib import Path
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str, required=True)
    args = parser.parse_args()
    
    diagnose_shapley_values(args.result_dir)
