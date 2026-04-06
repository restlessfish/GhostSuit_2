#!/usr/bin/env python3
"""
Evaluate In-Run Data Shapley on standard tasks:
1. Mislabeled data detection
2. Data selection

This script implements the evaluation tasks from the paper.
"""

import os
import sys
import numpy as np
import torch
import argparse
from pathlib import Path
import json

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def load_shapley_values(result_dir):
    """Load Shapley values from result directory."""
    grad_dotprods_dir = os.path.join(result_dir, 'grad_dotprods')
    
    # Find latest Shapley array file
    shapley_files = list(Path(grad_dotprods_dir).glob('shapley_array_iter_*.npy'))
    if not shapley_files:
        raise FileNotFoundError(f"No Shapley array files found in {grad_dotprods_dir}")
    
    latest_file = max(shapley_files, key=lambda p: int(p.stem.split('_')[-1]))
    iter_num = int(latest_file.stem.split('_')[-1])
    
    print(f"Loading Shapley values from: {latest_file} (iteration {iter_num})")
    
    # Load indices file (required for compact format)
    indices_file = Path(grad_dotprods_dir) / f"shapley_indices_iter_{iter_num}.npy"
    if indices_file.exists():
        # Compact format: load values and indices separately
        shapley_array = np.load(latest_file)
        indices = np.load(indices_file)
        print(f"Loaded {len(shapley_array)} Shapley values for {len(indices)} samples")
        return shapley_array, iter_num, indices
    else:
        # Legacy format: try to load full array and filter
        shapley_array = np.load(latest_file)
        non_zero_mask = shapley_array != 0
        if np.any(non_zero_mask):
            indices = np.where(non_zero_mask)[0]
            shapley_array = shapley_array[non_zero_mask]
            print(f"Found {len(indices)} non-zero Shapley values (filtered from {len(shapley_array)})")
            return shapley_array, iter_num, indices
        else:
            print(f"Warning: All Shapley values are zero!")
            return shapley_array, iter_num, np.arange(len(shapley_array))


def mislabeled_data_detection(shapley_array, mislabeled_indices, output_dir=None):
    """
    Evaluate mislabeled data detection using Shapley values.
    
    Args:
        shapley_array: Array of Shapley values
        mislabeled_indices: Indices of mislabeled samples
        output_dir: Directory to save results
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("\n" + "=" * 80)
    print("Task 1: Mislabeled Data Detection")
    print("=" * 80)
    
    # Create binary labels: 1 for mislabeled, 0 for correctly labeled
    binary_labels = np.zeros(len(shapley_array))
    binary_labels[mislabeled_indices] = 1
    
    # Lower Shapley values should indicate mislabeled data
    # So we use negative Shapley values as scores
    scores = -shapley_array
    
    # Compute metrics
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, auc
    
    try:
        auroc = roc_auc_score(binary_labels, scores)
    except ValueError:
        auroc = 0.5  # Handle case where all labels are the same
    
    # Find optimal threshold (median)
    threshold = np.median(scores)
    predictions = (scores >= threshold).astype(int)
    accuracy = accuracy_score(binary_labels, predictions)
    
    # Precision-Recall AUC
    try:
        precision, recall, _ = precision_recall_curve(binary_labels, scores)
        pr_auc = auc(recall, precision)
    except:
        pr_auc = 0.0
    
    # Calculate true positive rate and false positive rate at different thresholds
    sorted_indices = np.argsort(scores)[::-1]
    num_mislabeled = len(mislabeled_indices)
    
    # Top-k accuracy
    top_k_accuracies = {}
    for k in [10, 20, 50, 100]:
        if k <= len(shapley_array):
            top_k_indices = sorted_indices[:k]
            top_k_mislabeled = np.sum(binary_labels[top_k_indices])
            top_k_accuracies[f'top_{k}'] = {
                'detected': int(top_k_mislabeled),
                'total': k,
                'recall': float(top_k_mislabeled / num_mislabeled) if num_mislabeled > 0 else 0.0
            }
    
    results = {
        'auroc': float(auroc),
        'pr_auc': float(pr_auc),
        'accuracy': float(accuracy),
        'threshold': float(threshold),
        'num_mislabeled': int(num_mislabeled),
        'num_total': len(shapley_array),
        'top_k_accuracies': top_k_accuracies
    }
    
    print(f"\nResults:")
    print(f"  AUROC: {auroc:.4f}")
    print(f"  PR-AUC: {pr_auc:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Optimal threshold: {threshold:.6f}")
    print(f"  Mislabeled samples: {num_mislabeled}/{len(shapley_array)}")
    
    print(f"\nTop-K Detection:")
    for k, metrics in top_k_accuracies.items():
        print(f"  {k}: {metrics['detected']}/{metrics['total']} detected "
              f"(recall: {metrics['recall']:.2%})")
    
    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        results_file = os.path.join(output_dir, 'mislabeled_detection_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Saved results to {results_file}")
    
    return results


def data_selection_evaluation(shapley_array, test_labels, selection_budgets, output_dir=None):
    """
    Evaluate data selection based on Shapley values.
    
    Args:
        shapley_array: Array of Shapley values
        test_labels: Test labels for evaluation (optional, for future use)
        selection_budgets: List of selection budgets (fractions)
        output_dir: Directory to save results
        
    Returns:
        Dictionary with selection results
    """
    print("\n" + "=" * 80)
    print("Task 2: Data Selection")
    print("=" * 80)
    
    # Sort indices by Shapley values (descending)
    sorted_indices = np.argsort(shapley_array)[::-1]
    
    results = {}
    
    print(f"\nSelection Results:")
    for budget in selection_budgets:
        num_select = int(len(shapley_array) * budget)
        selected_indices = sorted_indices[:num_select]
        
        # Calculate statistics for selected samples
        selected_shapley = shapley_array[selected_indices]
        
        results[f'budget_{budget}'] = {
            'num_selected': num_select,
            'selected_indices': selected_indices.tolist(),
            'mean_shapley': float(np.mean(selected_shapley)),
            'min_shapley': float(np.min(selected_shapley)),
            'max_shapley': float(np.max(selected_shapley)),
        }
        
        print(f"  Budget {budget:.0%}:")
        print(f"    Selected: {num_select:,} samples")
        print(f"    Mean Shapley: {np.mean(selected_shapley):.6f}")
        print(f"    Range: [{np.min(selected_shapley):.6f}, {np.max(selected_shapley):.6f}]")
    
    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        results_file = os.path.join(output_dir, 'data_selection_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Saved results to {results_file}")
    
    return results


def create_synthetic_mislabeled_indices(num_samples, mislabeled_ratio=0.1, seed=42):
    """
    Create synthetic mislabeled indices for testing.
    
    Args:
        num_samples: Total number of samples
        mislabeled_ratio: Ratio of mislabeled samples
        seed: Random seed
        
    Returns:
        Array of mislabeled indices
    """
    np.random.seed(seed)
    num_mislabeled = int(num_samples * mislabeled_ratio)
    mislabeled_indices = np.random.choice(num_samples, size=num_mislabeled, replace=False)
    return mislabeled_indices


def main():
    parser = argparse.ArgumentParser(description='Evaluate In-Run Data Shapley on standard tasks')
    parser.add_argument('--result_dir', type=str, required=True,
                       help='Directory containing training results')
    parser.add_argument('--mislabeled_indices', type=str, default=None,
                       help='Path to file with mislabeled indices (numpy array or json)')
    parser.add_argument('--mislabeled_ratio', type=float, default=0.1,
                       help='Ratio of mislabeled samples (if creating synthetic)')
    parser.add_argument('--selection_budgets', type=float, nargs='+', 
                       default=[0.2, 0.4, 0.6, 0.8],
                       help='Selection budgets (fractions)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.result_dir, 'evaluation')
    
    print("=" * 80)
    print("In-Run Data Shapley Evaluation")
    print("=" * 80)
    print(f"Result directory: {args.result_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Load Shapley values
    result = load_shapley_values(args.result_dir)
    if len(result) == 3:
        shapley_array, iter_num, sample_indices = result
    else:
        shapley_array, iter_num = result
        sample_indices = np.arange(len(shapley_array))
    
    print(f"Loaded Shapley values: {len(shapley_array):,} samples")
    print(f"Shapley value stats: mean={np.mean(shapley_array):.6f}, std={np.std(shapley_array):.6f}")
    print(f"  min={np.min(shapley_array):.6f}, max={np.max(shapley_array):.6f}")
    
    # Prepare mislabeled indices
    if args.mislabeled_indices:
        if args.mislabeled_indices.endswith('.npy'):
            mislabeled_indices_raw = np.load(args.mislabeled_indices)
        elif args.mislabeled_indices.endswith('.json'):
            with open(args.mislabeled_indices, 'r') as f:
                data = json.load(f)
                mislabeled_indices_raw = np.array(data['mislabeled_indices'])
        else:
            raise ValueError("Unsupported file format for mislabeled_indices")
        print(f"Loaded mislabeled indices: {len(mislabeled_indices_raw)} samples")
        
        # Map to Shapley array indices
        # Find which mislabeled indices correspond to samples with Shapley values
        mislabeled_indices = []
        for idx in mislabeled_indices_raw:
            if idx in sample_indices:
                pos = np.where(sample_indices == idx)[0]
                if len(pos) > 0:
                    mislabeled_indices.append(pos[0])
        mislabeled_indices = np.array(mislabeled_indices)
        print(f"Mapped to {len(mislabeled_indices)} mislabeled indices in Shapley array")
    else:
        # Create synthetic mislabeled indices within the actual sample range
        print(f"Creating synthetic mislabeled indices (ratio: {args.mislabeled_ratio})...")
        mislabeled_indices = create_synthetic_mislabeled_indices(
            len(shapley_array), args.mislabeled_ratio
        )
        print(f"Created {len(mislabeled_indices)} synthetic mislabeled indices")
    
    # Ensure indices are within range
    mislabeled_indices = mislabeled_indices[mislabeled_indices < len(shapley_array)]
    
    if len(mislabeled_indices) == 0:
        print("Warning: No mislabeled indices found in the Shapley array!")
        print("This might indicate a mismatch between training samples and evaluation samples.")
    
    # Task 1: Mislabeled data detection
    mislabeled_results = mislabeled_data_detection(
        shapley_array, mislabeled_indices, args.output_dir
    )
    
    # Task 2: Data selection
    selection_results = data_selection_evaluation(
        shapley_array, None, args.selection_budgets, args.output_dir
    )
    
    # Summary
    print("\n" + "=" * 80)
    print("Evaluation Summary")
    print("=" * 80)
    print(f"\nMislabeled Data Detection:")
    print(f"  AUROC: {mislabeled_results['auroc']:.4f}")
    print(f"  PR-AUC: {mislabeled_results['pr_auc']:.4f}")
    print(f"  Accuracy: {mislabeled_results['accuracy']:.4f}")
    
    print(f"\nData Selection:")
    for budget, metrics in selection_results.items():
        print(f"  {budget}: {metrics['num_selected']:,} samples selected")
    
    # Compare with paper results
    print("\n" + "=" * 80)
    print("Comparison with Paper (Table 4 & 5)")
    print("=" * 80)
    print("\nPaper Results (CIFAR-10, ResNet18):")
    print("  Mislabeled Detection AUROC:")
    print("    - 1st Order In-Run Data Shapley: 0.678")
    print("    - 2nd Order In-Run Data Shapley: 0.680")
    print("    - KNN-Shapley: 0.76 (best)")
    print("    - Influence Function: 0.654")
    
    print(f"\nOur Results:")
    print(f"  AUROC: {mislabeled_results['auroc']:.4f}")
    if mislabeled_results['auroc'] > 0.5:
        print("  ✓ Performance above random (0.5)")
    if mislabeled_results['auroc'] > 0.6:
        print("  ✓ Performance comparable to baseline methods")
    
    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
