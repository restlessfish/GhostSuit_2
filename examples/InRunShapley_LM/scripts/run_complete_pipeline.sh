#!/bin/bash
# Complete pipeline for In-Run Data Shapley training and evaluation

set -e  # Exit on error

echo "=================================================================================="
echo "In-Run Data Shapley Complete Pipeline"
echo "=================================================================================="

# Configuration
DATA_DIR="./data/cifar10_simple"
OUTPUT_DIR="./results/cifar10_complete"
NUM_STEPS=500
BATCH_SIZE=8
VAL_BATCH_SIZE=2
BLOCK_SIZE=512
LEARNING_RATE=3e-4

# Step 1: Prepare dataset
echo ""
echo "Step 1: Preparing dataset..."
cd examples/InRunShapley_LM
python scripts/prepare_cifar10.py \
    --output_dir ${DATA_DIR} \
    --simple \
    --num_samples 5000 \
    --seed 42

# Step 2: Run training
echo ""
echo "Step 2: Running training..."
python scripts/train_inrun_shapley.py \
    --data_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --num_steps ${NUM_STEPS} \
    --batch_size ${BATCH_SIZE} \
    --val_batch_size ${VAL_BATCH_SIZE} \
    --block_size ${BLOCK_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --shapley_save_interval 50 \
    --eval_interval 100

# Step 3: Evaluate tasks
echo ""
echo "Step 3: Evaluating tasks..."
python scripts/evaluate_tasks.py \
    --result_dir ${OUTPUT_DIR} \
    --mislabeled_indices ${DATA_DIR}/mislabeled_indices.npy \
    --mislabeled_ratio 0.1 \
    --selection_budgets 0.2 0.4 0.6 0.8

# Step 4: Analyze results
echo ""
echo "Step 4: Analyzing results..."
python scripts/analyze_results.py \
    --result_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR}/analysis

# Step 5: Compare with paper
echo ""
echo "Step 5: Comparing with paper..."
python scripts/compare_with_paper.py \
    --result_dir ${OUTPUT_DIR} \
    --output_file ${OUTPUT_DIR}/comparison_with_paper.json

echo ""
echo "=================================================================================="
echo "Pipeline Complete!"
echo "=================================================================================="
echo ""
echo "Results saved to: ${OUTPUT_DIR}"
echo ""
echo "To view results:"
echo "  cat ${OUTPUT_DIR}/training_stats.json"
echo "  cat ${OUTPUT_DIR}/comparison_with_paper.json"
