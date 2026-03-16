#!/usr/bin/env bash

# 一键复现 CIFAR-10N / ResNet18 / In-Run 二阶 Shapley (AUROC≈0.71)
# 依赖：
#   - 已安装 conda，并存在名为 shapley311 的环境（可在下方修改）
#   - CIFAR-10 数据会自动下载到 ./data/cifar10
#
# 运行方式：
#   chmod +x run_inrun_shapley_cifar10_second_order.sh
#   ./run_inrun_shapley_cifar10_second_order.sh

set -e

CONDA_ENV=${CONDA_ENV:-shapley311}
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

DATA_ROOT="./data/cifar10"
OUTPUT_DIR="./results/cifar10_resnet18_inrun_order2_hvp"

echo "============================================================"
echo " Step 1: 训练 ResNet18 + CIFAR-10N，计算二阶 In-Run Shapley"
echo "============================================================"
conda run -n "${CONDA_ENV}" python examples/InRunShapley_LM/scripts/train_cifar10_resnet18_inrun_shapley.py \
  --data_root "${DATA_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_steps 4000 \
  --batch_size 128 \
  --val_batch_size 32 \
  --learning_rate 1e-3 \
  --noise_type aggre_label \
  --shapley_order 2 \
  --shapley_save_interval 1000 \
  --eval_interval 400 \
  --num_workers 4 \
  --device cuda

echo
echo "============================================================"
echo " Step 2: 使用真实 CIFAR-10N 噪声标签评估 mislabeled detection AUROC"
echo "============================================================"
conda run -n "${CONDA_ENV}" python examples/InRunShapley_LM/scripts/evaluate_tasks.py \
  --result_dir "${OUTPUT_DIR}" \
  --mislabeled_indices "${OUTPUT_DIR}/mislabeled_indices.npy"

echo
echo "Done. 训练与评估结果保存在: ${OUTPUT_DIR}"

