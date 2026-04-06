#!/usr/bin/env python3
"""
改进的训练脚本 - 使用所有优化
"""

import os
import sys
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(os.path.dirname(parent_dir))
sys.path.append(parent_dir)
sys.path.append(root_dir)

from scripts.train_inrun_shapley import train_inrun_shapley


def main():
    parser = argparse.ArgumentParser(description='Improved In-Run Shapley Training')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing dataset files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save results')
    parser.add_argument('--num_steps', type=int, default=2000,
                       help='Number of training steps (increased for better results)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Training batch size')
    parser.add_argument('--val_batch_size', type=int, default=4,
                       help='Validation batch size (increased for stability)')
    parser.add_argument('--block_size', type=int, default=512,
                       help='Sequence length')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--shapley_save_interval', type=int, default=100,
                       help='How often to save Shapley values')
    parser.add_argument('--eval_interval', type=int, default=200,
                       help='How often to evaluate')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("IMPROVED In-Run Data Shapley Training")
    print("=" * 80)
    print("Improvements:")
    print("  1. Better Shapley value normalization")
    print("  2. Weighted accumulation (later iterations weighted more)")
    print("  3. Sample counting and averaging")
    print("  4. Learning rate scheduling with warmup")
    print("  5. Validation set rotation")
    print("  6. Increased training steps and validation batch size")
    print("=" * 80)
    
    train_inrun_shapley(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        block_size=args.block_size,
        learning_rate=args.learning_rate,
        shapley_save_interval=args.shapley_save_interval,
        eval_interval=args.eval_interval,
        device=args.device,
        warmup_steps=min(100, args.num_steps // 20),
        use_lr_schedule=True,
    )


if __name__ == "__main__":
    main()
